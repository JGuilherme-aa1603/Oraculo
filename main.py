"""Oráculo — assistente de voz local.

Fase 1: chat de terminal com memória e Ollama.
Fase 2: entrada/saída de voz opcional (Whisper STT + Piper TTS), comandos e
        persistência de sessões. O modo texto continua sendo o padrão.
"""

import contextlib
import json
import sys
import threading
import time
import urllib.request

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

import config
from core import (
    commands,
    history as history_mod,
    keyboard,
    llm as llm_mod,
    speaker as speaker_mod,
    telemetry,
)
from core.chain import OraculoChain
from core.splash import show_splash

console = Console()

# Intervalo mínimo entre repaints do streaming (s). Casa com o refresh_per_second
# do Live; segura o uso da iGPU (compositor) durante a escrita da resposta.
_REFRESH_INTERVAL = 1 / 6


def _model_is_loaded(model_name: str) -> bool:
    """Consulta o Ollama (/api/ps) para saber se o modelo já está na VRAM."""
    try:
        with urllib.request.urlopen(
            f"{config.OLLAMA_BASE_URL}/api/ps", timeout=1
        ) as resp:
            data = json.loads(resp.read())
    except Exception:  # noqa: BLE001 — Ollama lento/indisponível: assume não carregado
        return False
    for m in data.get("models", []):
        loaded = m.get("model") or m.get("name") or ""
        if model_name in loaded or loaded in model_name:
            return True
    return False


class _ThinkingStatus:
    """Spinner de espera até a 1ª saída do modelo chegar.

    Decide o rótulo pelo estado real:
      - modelo ainda não carregado → "Carregando modelo..." (amarelo) e, em
        background, verifica o /api/ps até subir, então troca o rótulo;
      - carregado + thinking ligado → "Pensando..." (o modelo vai raciocinar);
      - carregado + thinking desligado → "Gerando..." (honesto: não há raciocínio).
    `first_token()` encerra a espera — daí o streaming assume o Live.
    """

    def __init__(self, live: Live, model_name: str, thinking: bool) -> None:
        self._live = live
        self._model = model_name
        self._thinking = thinking
        self._done = threading.Event()

    def start(self) -> None:
        if _model_is_loaded(self._model):
            self._show(*self._wait_label())
        else:
            self._show("Carregando modelo...", "yellow")
            threading.Thread(target=self._poll, daemon=True).start()

    def _wait_label(self) -> tuple[str, str]:
        return ("Pensando...", "cyan") if self._thinking else ("Gerando...", "cyan")

    def _poll(self) -> None:
        while not self._done.wait(timeout=0.4):
            if _model_is_loaded(self._model):
                if not self._done.is_set():
                    self._show(*self._wait_label())
                return

    def _show(self, label: str, color: str) -> None:
        with contextlib.suppress(Exception):
            self._live.update(Spinner("dots", text=Text(label, style=f"dim {color}")))

    def first_token(self) -> None:
        self._done.set()


def _thinking_view(show: bool, reasoning: str):
    """Renderable da fase de raciocínio: o texto real (Ctrl+O ligado) ou um
    spinner honesto "Pensando..." (Ctrl+O desligado)."""
    if show:
        shown = reasoning.strip()
        if len(shown) > 1200:        # mostra só a cauda para não estourar a tela
            shown = "..." + shown[-1200:]
        return Panel(
            Text(shown or "...", style="dim italic"),
            title="[cyan]Pensando[/]",
            subtitle="[dim]Ctrl+O: ocultar[/]",
            title_align="left",
            subtitle_align="right",
            border_style="dim cyan",
        )
    return Spinner(
        "dots",
        text=Text.from_markup("[cyan]Pensando...[/]  [dim](Ctrl+O: ver raciocínio)[/]"),
    )


def _speak_until_done(speaker: speaker_mod.StreamSpeaker, ctx: dict) -> Exception | None:
    """Aguarda a fala terminar permitindo barge-in: Esc interrompe na hora e
    devolve o controle para a próxima mensagem. Sem TTY, só aguarda o fim."""
    con = ctx["console"]
    if sys.stdin.isatty() and not ctx.get("esc_hint_shown"):
        con.print("[dim](Esc interrompe a fala)[/]")
        ctx["esc_hint_shown"] = True

    interrupted = threading.Event()

    def _on_esc() -> None:
        interrupted.set()
        speaker.stop()

    with keyboard.watch_key(keyboard.ESC, _on_esc):
        err = speaker.close()
    if interrupted.is_set():
        con.print("[dim](fala interrompida)[/]")
    return err


def _listen(ctx: dict) -> str | None:
    """Captura uma fala no modo voz (push-to-talk). Primeiro Enter inicia a
    gravação, segundo Enter encerra; texto digitado é usado diretamente como
    escape. Retorna o texto ou None se nada foi captado."""
    from core import audio, stt

    ctx["last_stt_seconds"] = None
    typed = console.input(
        "[dim][voz] Enter para começar a gravar (ou digite e Enter):[/] "
    ).strip()
    if typed:
        return typed

    try:
        console.print("[dim](gravando... Enter para parar)[/]")
        path = audio.record_ptt()
        console.print("[dim](transcrevendo...)[/]")
        _stt_t0 = time.monotonic()
        text = stt.transcribe(path)
        ctx["last_stt_seconds"] = time.monotonic() - _stt_t0
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/]")
        console.print("[yellow]Voltando ao modo texto.[/]")
        ctx["voice_mode"] = False
        return None

    if not text:
        console.print("[dim](não entendi nada — tente de novo)[/]")
        return None

    console.print(f"[dim](você disse: {text})[/]")
    return text


def main() -> None:
    try:
        chain = OraculoChain()
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Falha ao iniciar o {config.ASSISTANT_NAME}:[/] {exc}\n"
            "[yellow]O Ollama está rodando? Tente: ollama serve[/]"
        )
        sys.exit(1)

    show_splash(chain.model_name, recent_sessions=history_mod.load_recent())

    history = history_mod.SessionHistory()
    ctx = {
        "console": console,
        "chain": chain,
        "running": True,
        "voice_mode": config.VOICE_MODE_DEFAULT,
        "thinking": False,
        "show_thinking": config.SHOW_THINKING_DEFAULT,
    }
    # Liga o thinking por padrão só se o modelo realmente suportar.
    if config.THINKING_DEFAULT and llm_mod.supports_thinking(chain.model_name):
        ctx["thinking"] = True
        chain.set_thinking(True)

    while ctx["running"]:
        try:
            if ctx["voice_mode"]:
                user_input = _listen(ctx)
                if not user_input:
                    continue
            else:
                user_input = console.input("[bold cyan]Você:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Encerrando...[/]")
            break

        if not user_input:
            continue

        if commands.handle(user_input, ctx):
            continue

        history.record("user", user_input)
        stt_seconds = ctx.pop("last_stt_seconds", None)
        console.print(f"[bold bright_cyan]{config.ASSISTANT_NAME}:[/]")
        # No modo voz, a fala é sintetizada frase a frase JÁ DURANTE a geração,
        # sobreposta à escrita — não espera a resposta inteira terminar.
        speaker = speaker_mod.StreamSpeaker() if ctx["voice_mode"] else None
        # Telemetria do turno: t0 começa aqui (pós-STT) — o STT entra como estágio
        # próprio. Marcações e métricas são best-effort e não alteram o turno.
        tel = telemetry.TurnTelemetry()
        tel.mode = "voz" if speaker else "texto"
        try:
            chunks: list[str] = []
            reasoning: list[str] = []
            got_output = False
            answering = False
            # Preview ao vivo enquanto a resposta chega, depois render final.
            # transient=True + vertical_overflow="crop": o Live mostra só a
            # última tela e redesenha NO LUGAR (sem isso, resposta mais alta que
            # o terminal faz o Live reemitir tudo a cada frame e o texto repete
            # em cascata). Ao sair, o preview se apaga e imprimimos o Markdown
            # completo uma única vez — rola naturalmente, sem repetição.
            # O throttle (_REFRESH_INTERVAL) evita reparsear o Markdown a cada
            # token; menos repaints = menos uso da iGPU (compositor).
            last_render = 0.0

            def _toggle_thinking() -> None:
                ctx["show_thinking"] = not ctx.get("show_thinking", False)

            # Ctrl+O alterna a exibição do raciocínio ao vivo. Só observa o
            # teclado quando o thinking está ligado; cbreak preserva o Ctrl+C,
            # que continua interrompendo a resposta.
            if ctx.get("thinking") and sys.stdin.isatty():
                key_watch = keyboard.watch_key(
                    keyboard.CTRL_O, _toggle_thinking,
                    once=False, preserve_signals=True,
                )
            else:
                key_watch = contextlib.nullcontext()

            with Live(console=console, refresh_per_second=6, transient=True,
                      vertical_overflow="crop") as live, key_watch:
                status = _ThinkingStatus(live, chain.model_name, ctx.get("thinking", False))
                status.start()
                for kind, text in chain.stream(user_input):
                    now = time.monotonic()
                    if not got_output:
                        got_output = True
                        status.first_token()        # encerra o spinner de espera
                        tel.mark("first_token")
                    if kind == "think":
                        reasoning.append(text)
                        if now - last_render >= _REFRESH_INTERVAL:
                            live.update(_thinking_view(
                                ctx.get("show_thinking"), "".join(reasoning)))
                            last_render = now
                        continue
                    # resposta
                    if not answering:
                        answering = True
                        last_render = 0.0           # força limpar o raciocínio e renderizar
                    chunks.append(text)
                    if speaker:
                        speaker.feed(text)
                    if now - last_render >= _REFRESH_INTERVAL:
                        live.update(Markdown("".join(chunks)))
                        last_render = now
            response = "".join(chunks)
            console.print(Markdown(response))
            history.record("assistant", response)
            tel.set_llm(**chain.last_usage)
        except KeyboardInterrupt:
            console.print("\n[yellow](resposta interrompida)[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Erro ao responder:[/] {exc}")
        finally:
            if speaker:
                err = _speak_until_done(speaker, ctx)
                if err:
                    console.print(f"[yellow](voz indisponível: {err})[/]")
                tel.mark_at("first_audio", speaker.first_audio_at)
            # Telemetria nunca quebra o turno: tudo em try/except próprio.
            try:
                tel.set_stage("stt", stt_seconds)
                telemetry.log_turn(tel.finish())
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
