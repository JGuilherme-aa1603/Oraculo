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
    prompt as prompt_mod,
    speaker as speaker_mod,
    telemetry,
    tui,
    ui,
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
        ui.notice(con, "Esc interrompe a fala")
        ctx["esc_hint_shown"] = True

    interrupted = threading.Event()

    def _on_esc() -> None:
        interrupted.set()
        speaker.stop()

    with keyboard.watch_key(keyboard.ESC, _on_esc):
        err = speaker.close()
    if interrupted.is_set():
        ui.notice(con, "fala interrompida")
    return err


def _status(ctx: dict) -> dict:
    """Estado exibido na barra abaixo da caixa de entrada. Recalculado a cada
    repaint, então acompanha /modelo, /voz e /think sem precisar de callback."""
    chain = ctx.get("chain")
    flags = ["voz" if ctx.get("voice_mode") else "texto"]
    flags.append("think on" if ctx.get("thinking") else "think off")
    if chain is not None:
        # Memória em pares (pergunta+resposta), que é como a janela é cortada.
        with contextlib.suppress(Exception):
            mem = chain.memory
            flags.append(f"mem {len(mem.messages) // 2}/{mem.max_messages // 2}")
    return {
        "model": chain.model_name if chain is not None else config.OLLAMA_MODEL,
        "flags": flags,
    }


def _listen(ctx: dict, ask, wait_stop=None) -> str | None:
    """Captura uma fala no modo voz (push-to-talk). Primeiro Enter inicia a
    gravação, segundo Enter encerra; texto digitado é usado diretamente como
    escape. Retorna o texto ou None se nada foi captado."""
    from core import audio, stt

    console = ctx["console"]
    ctx["last_stt_seconds"] = None
    typed = ask("[dim][voz] Enter para gravar (ou digite e Enter):[/] ")
    if typed:
        return typed

    try:
        ui.notice(console, "gravando... Enter para parar")
        path = audio.record_ptt(wait_stop=wait_stop)
        ui.notice(console, "transcrevendo...")
        _stt_t0 = time.monotonic()
        text = stt.transcribe(path)
        ctx["last_stt_seconds"] = time.monotonic() - _stt_t0
    except RuntimeError as exc:
        ui.warn(console, str(exc))
        ui.warn(console, "Voltando ao modo texto.")
        ctx["voice_mode"] = False
        return None

    if not text:
        ui.notice(console, "não entendi nada — tente de novo")
        return None

    return text


def run_standalone(argv: list[str]) -> int:
    """Executa um comando direto do shell, sem abrir o chat.

    É o modo usado pelo wrapper `bin/oraculo`: `oraculo transcrever <arquivo>`.
    Não instancia o modelo — os comandos permitidos aqui não precisam do Ollama.
    A barra é opcional, e o resto da linha é remontado como veio (caminhos com
    espaço funcionam sem aspas). Devolve o código de saída do processo.
    """
    raw = " ".join(argv).strip()
    if not raw.startswith("/"):
        raw = f"/{raw}"

    cmd = raw.split(maxsplit=1)[0].lower()
    if cmd not in commands.STANDALONE_COMMANDS:
        console.print(
            f"[yellow]'{cmd.lstrip('/')}' só funciona dentro do "
            f"{config.ASSISTANT_NAME}.[/]\n"
            f"[dim]Rode 'oraculo' sem argumentos para abrir o chat, ou "
            f"'oraculo ajuda' para ver os comandos.[/]"
        )
        return 2

    ctx = {
        "console": console,
        "chain": None,
        "running": True,
        "voice_mode": False,
        "thinking": False,
        "show_thinking": config.SHOW_THINKING_DEFAULT,
    }
    commands.handle(raw, ctx)
    return 0


def main() -> None:
    # O modelo é instanciado antes de limpar a tela: se o Ollama estiver fora do
    # ar, a mensagem de erro fica visível em vez de ser apagada logo em seguida.
    try:
        chain = OraculoChain()
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold red]Falha ao iniciar o {config.ASSISTANT_NAME}:[/] {exc}\n"
            "[yellow]O Ollama está rodando? Tente: ollama serve[/]"
        )
        sys.exit(1)

    if tui.disponivel():
        _run_fullscreen(chain)
    else:
        _run_inline(chain)


def _novo_ctx(chain: OraculoChain, out) -> dict:
    ctx = {
        "console": out,
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
    return ctx


def _run_inline(chain: OraculoChain) -> None:
    """Modo clássico: desenha no buffer normal, rolagem nativa do terminal."""
    ui.clear_screen(console)
    show_splash(chain.model_name, recent_sessions=history_mod.load_recent())
    ctx = _novo_ctx(chain, console)
    box = prompt_mod.InputBox(console, lambda: _status(ctx))

    def _live():
        return Live(console=console, refresh_per_second=6, transient=True,
                    vertical_overflow="crop")

    def _watch_ctrl_o(toggle):
        # Modo raw só no inline: no fullscreen o prompt_toolkit é dono do teclado.
        if ctx.get("thinking") and sys.stdin.isatty():
            return keyboard.watch_key(keyboard.CTRL_O, toggle,
                                      once=False, preserve_signals=True)
        return contextlib.nullcontext()

    _chat_loop(chain, ctx, ask=lambda p="": box.ask(p), live_factory=_live,
               echo=box.rich, watch_ctrl_o=_watch_ctrl_o)


def _run_fullscreen(chain: OraculoChain) -> None:
    """Modo tela cheia: tela alternativa, caixa fixa e rolagem própria."""
    ctx: dict = {"chain": chain, "voice_mode": config.VOICE_MODE_DEFAULT,
                 "thinking": False, "running": True}

    def loop(sessao) -> None:
        # O ctx é preenchido aqui porque só agora existe o console do transcript;
        # a barra de status já pode ter sido desenhada com os valores iniciais.
        ctx.update(_novo_ctx(chain, sessao.console))
        sessao.on_toggle_thinking = lambda: ctx.update(
            show_thinking=not ctx.get("show_thinking", False))
        show_splash(chain.model_name, recent_sessions=history_mod.load_recent(),
                    out=sessao.console)

        def _ask(_prompt: str = "") -> str:
            return sessao.ask()

        _chat_loop(chain, ctx, ask=_ask, live_factory=sessao.live, echo=True,
                   watch_ctrl_o=lambda _toggle: contextlib.nullcontext(),
                   interrupt=sessao.interromper, wait_stop=sessao.wait_enter)

    tui.run(loop, lambda: _status(ctx))


def _chat_loop(chain: OraculoChain, ctx: dict, *, ask, live_factory, echo: bool,
               watch_ctrl_o, interrupt=None, wait_stop=None) -> None:
    """Laço de turnos, compartilhado pelos dois modos de desenho.

    O que muda entre eles é injetado: de onde vem a mensagem (`ask`), o que
    mostra o preview do streaming (`live_factory`), como o Ctrl+O é observado e
    como a interrupção chega (`interrupt`, um Event no modo tela cheia — lá o
    KeyboardInterrupt não sobe pela thread do laço).
    """
    console = ctx["console"]
    history = history_mod.SessionHistory()

    while ctx["running"]:
        try:
            if ctx["voice_mode"]:
                user_input = _listen(ctx, ask, wait_stop=wait_stop)
                if not user_input:
                    continue
            else:
                user_input = ask()
        except (EOFError, KeyboardInterrupt):
            ui.notice(console, "Encerrando...", style="cyan")
            break

        if not user_input:
            continue

        # A caixa se apaga ao enviar, então o transcript precisa do eco para
        # guardar a pergunta. No fallback do rich o texto digitado já ficou na
        # tela — ecoar de novo duplicaria.
        if echo:
            ui.user_echo(console, user_input)

        if commands.handle(user_input, ctx):
            ui.spacer(console)
            continue

        history.record("user", user_input)
        stt_seconds = ctx.pop("last_stt_seconds", None)
        ui.assistant_header(console)
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

            with live_factory() as live, watch_ctrl_o(_toggle_thinking):
                status = _ThinkingStatus(live, chain.model_name, ctx.get("thinking", False))
                status.start()
                for kind, text in chain.stream(user_input):
                    # No modo tela cheia o Ctrl+C não sobe como exceção nesta
                    # thread: ele marca o Event, e a checagem é aqui.
                    if interrupt is not None and interrupt.is_set():
                        raise KeyboardInterrupt
                    now = time.monotonic()
                    if not got_output:
                        got_output = True
                        status.first_token()        # encerra o spinner de espera
                        tel.mark("first_token")
                    if kind == "think":
                        reasoning.append(text)
                        if now - last_render >= _REFRESH_INTERVAL:
                            live.update(ui.indent(_thinking_view(
                                ctx.get("show_thinking"), "".join(reasoning))))
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
                        live.update(ui.indent(Markdown("".join(chunks))))
                        last_render = now
            response = "".join(chunks)
            ui.body(console, Markdown(response))
            history.record("assistant", response)
            tel.set_llm(**chain.last_usage)
        except KeyboardInterrupt:
            ui.warn(console, "resposta interrompida")
        except Exception as exc:  # noqa: BLE001
            ui.error(console, f"Erro ao responder: {exc}")
        finally:
            if speaker:
                err = _speak_until_done(speaker, ctx)
                if err:
                    ui.warn(console, f"voz indisponível: {err}")
                tel.mark_at("first_audio", speaker.first_audio_at)
            # Telemetria nunca quebra o turno: tudo em try/except próprio.
            try:
                tel.set_stage("stt", stt_seconds)
                record = tel.finish()
                if config.UI_SHOW_TURN_METRICS:
                    ui.turn_footer(console, telemetry.summary_line(record))
                telemetry.log_turn(record)
            except Exception:  # noqa: BLE001
                pass
            ui.spacer(console)


if __name__ == "__main__":
    # Com argumentos, roda o comando e sai; sem argumentos, abre o chat.
    if len(sys.argv) > 1:
        sys.exit(run_standalone(sys.argv[1:]))
    main()
