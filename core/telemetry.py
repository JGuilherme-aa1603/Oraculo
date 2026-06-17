"""Telemetria de latência e throughput por turno (STT -> LLM -> TTS).

Mede, por turno, as latências de cada estágio e a taxa de geração do modelo:
STT, TTFT (time-to-first-token), tokens/s, tempo até a 1ª fala (TTFA) e o
wall-clock total. Opt-in via `config`:

  - TELEMETRY_CONSOLE: imprime um resumo de 1 linha no terminal.
  - TELEMETRY_ENABLED:  faz append de 1 objeto JSON em
                        ~/.oraculo/telemetry/<YYYY-MM-DD>.jsonl.

Com ambas as flags `False`, nada é escrito nem impresso (custo zero).
Stdlib apenas (+ rich, já usado). NUNCA levanta exceção para fora — telemetria
jamais deve quebrar um turno.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime

import config


def active() -> bool:
    """True se alguma saída de telemetria está ligada."""
    return bool(config.TELEMETRY_ENABLED or config.TELEMETRY_CONSOLE)


class TurnTelemetry:
    """Coleta métricas de um turno. Todos os tempos em time.monotonic()."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.mode = "texto"                 # "voz" | "texto" (definido pelo chamador)
        self.marks: dict[str, float] = {}   # elapsed (s) desde t0
        self.stages: dict[str, float] = {}  # duração (s) de blocos cronometrados
        self.llm: dict = {}                 # metadata bruto do Ollama

    def mark(self, key: str) -> None:
        """Grava o elapsed (s) desde o início do turno (primeira ocorrência vence)."""
        if key not in self.marks:
            self.marks[key] = time.monotonic() - self.t0

    def mark_at(self, key: str, monotonic_ts: float | None) -> None:
        """Grava um mark a partir de um carimbo monotônico absoluto (de outro
        thread, ex.: o instante da 1ª fala medido pelo StreamSpeaker)."""
        if monotonic_ts is not None and key not in self.marks:
            self.marks[key] = monotonic_ts - self.t0

    @contextmanager
    def stage(self, key: str):
        """Context manager que cronometra um bloco e grava sua duração em stages."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.stages[key] = time.monotonic() - start

    def set_stage(self, key: str, seconds: float | None) -> None:
        """Registra a duração de um estágio medido fora deste objeto (ex.: STT)."""
        if seconds is not None:
            self.stages[key] = seconds

    def set_llm(self, usage: dict | None = None, meta: dict | None = None) -> None:
        """Guarda o metadata do Ollama (usage_metadata + response_metadata)."""
        self.llm = {"usage": usage or {}, "meta": meta or {}}

    def finish(self) -> dict:
        """Calcula os derivados (tokens/s, totais) e devolve o registro do turno."""
        total = time.monotonic() - self.t0
        usage = self.llm.get("usage") or {}
        meta = self.llm.get("meta") or {}

        # Contagem de tokens: prefere usage_metadata, cai para response_metadata.
        out_tokens = usage.get("output_tokens")
        if out_tokens is None:
            out_tokens = meta.get("eval_count")
        in_tokens = usage.get("input_tokens")
        if in_tokens is None:
            in_tokens = meta.get("prompt_eval_count")

        # tokens/s preferido: métricas do Ollama (eval_duration vem em ns).
        tokens_per_s = None
        eval_count = meta.get("eval_count")
        eval_ns = meta.get("eval_duration")
        if eval_count and eval_ns:
            tokens_per_s = eval_count / (eval_ns / 1e9)
        elif out_tokens:
            # Fallback: tokens / tempo de geração medido por nós (do 1º token ao fim).
            ttft = self.marks.get("first_token")
            if ttft is not None and total > ttft:
                tokens_per_s = out_tokens / (total - ttft)

        record: dict = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "model": meta.get("model") or meta.get("model_name") or config.OLLAMA_MODEL,
            "turn_s": round(total, 3),
            "ttft_s": _round(self.marks.get("first_token")),
            "prompt_tokens": in_tokens,
            "output_tokens": out_tokens,
            "tokens_per_s": _round(tokens_per_s, 1),
        }
        if self.mode == "voz":
            record["stt_engine"] = config.STT_ENGINE
            record["tts_engine"] = config.TTS_ENGINE
            record["stt_s"] = _round(self.stages.get("stt"))
            record["ttfa_s"] = _round(self.marks.get("first_audio"))
        return record


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


def log_turn(record: dict) -> None:
    """Imprime o resumo (se TELEMETRY_CONSOLE) e/ou faz append em JSONL (se
    TELEMETRY_ENABLED). Best-effort: nunca levanta exceção."""
    try:
        if config.TELEMETRY_CONSOLE:
            _print_summary(record)
        if config.TELEMETRY_ENABLED:
            _append_jsonl(record)
    except Exception:  # noqa: BLE001 — telemetria jamais derruba o turno
        pass


def _print_summary(record: dict) -> None:
    from rich.console import Console

    parts = [f"turno {record['turn_s']:.1f}s"]
    if record.get("stt_s") is not None:
        parts.append(f"STT {record['stt_s']:.1f}s")
    if record.get("ttft_s") is not None:
        parts.append(f"TTFT {record['ttft_s']:.1f}s")
    if record.get("output_tokens"):
        tok = f"{record['output_tokens']} tok"
        if record.get("tokens_per_s"):
            tok += f" @ {record['tokens_per_s']:.0f} tok/s"
        parts.append(tok)
    if record.get("ttfa_s") is not None:
        parts.append(f"1a fala {record['ttfa_s']:.1f}s")
    Console().print("[dim cyan]telemetria[/] [dim]" + " · ".join(parts) + "[/]")


def _append_jsonl(record: dict) -> None:
    config.TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TELEMETRY_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
