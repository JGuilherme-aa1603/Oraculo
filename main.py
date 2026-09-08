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

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

import config
from core import (
    commands,
    history as history_mod,
    keyboard,
    llm as llm_mod,
    prefs as prefs_mod,
    prompt as prompt_mod,
    speaker as speaker_mod,
    telemetry,
    title as title_mod,
    tui,
    ui,
)
from core.chain import OraculoChain
from core.splash import show_splash

console = ui.make_console()

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
    """Espera até a 1ª saída do modelo chegar, com a íris girando.

    Decide o rótulo pelo estado real:
      - modelo ainda não carregado → "Carregando modelo..." e, em background,
        verifica o /api/ps até subir, então troca o rótulo;
      - carregado + thinking ligado → "Pensando..." (o modelo vai raciocinar);
      - carregado + thinking desligado → "Gerando..." (honesto: não há raciocínio).
    `first_token()` encerra a espera — daí o streaming assume o Live.

    O relógio da linha começa em `_t0`, e não a cada troca de rótulo: o que
    interessa é há quanto tempo o turno está esperando, não há quanto tempo o
    texto na tela é aquele.
    """

    def __init__(self, live: Live, model_name: str, thinking: bool) -> None:
        self._live = live
        self._model = model_name
        self._thinking = thinking
        self._done = threading.Event()
        self._t0 = time.monotonic()

    def start(self) -> None:
        if _model_is_loaded(self._model):
            self._show(*self._wait_label())
        else:
            self._show("Carregando modelo...", config.UI_COLOR_DIM)
            threading.Thread(target=self._poll, daemon=True).start()

    def _wait_label(self) -> tuple[str, str]:
        rotulo = "Pensando..." if self._thinking else "Gerando..."
        return (rotulo, config.UI_COLOR_SOFT)

    def _poll(self) -> None:
        while not self._done.wait(timeout=0.4):
            if _model_is_loaded(self._model):
                if not self._done.is_set():
                    self._show(*self._wait_label())
                return

    def _show(self, label: str, style: str) -> None:
        with contextlib.suppress(Exception):
            # Sem `ui.indent()` por fora: o Padding esconderia a marca `ANIMADO`
            # e a íris congelaria na tela cheia. A `Waiting` traz o próprio recuo.
            self._live.update(ui.Waiting(label, "Ctrl+C corta a resposta",
                                         since=self._t0, style=style))

    def first_token(self) -> None:
        self._done.set()


def _thinking_view(show: bool, reasoning: str, since: float):
    """Renderable da fase de raciocínio: o texto real (Ctrl+O ligado) ou a linha
    de espera honesta "Pensando..." (Ctrl+O desligado).

    Com o texto à mostra o bloco ganha só uma barra na margem, não uma moldura:
    o raciocínio é um aparte do turno, e uma caixa fechada o deixaria mais
    pesado na tela do que a própria resposta.

    Devolve o renderable JÁ recuado. Só o ramo do texto passa pelo `indent()`:
    a `Waiting` traz o recuo dela e não pode ser embrulhada, senão o `ANIMADO`
    some e a íris congela na tela cheia.
    """
    if show:
        shown = reasoning.strip()
        if len(shown) > 1200:        # mostra só a cauda para não estourar a tela
            shown = "..." + shown[-1200:]
        cabecalho = Text("raciocínio", style=config.UI_COLOR_DIM)
        cabecalho.append("  · Ctrl+O oculta", style=config.UI_COLOR_FAINT)
        corpo = Text(shown or "...", style=f"italic {config.UI_COLOR_FAINT}")
        return ui.indent(ui.LeftRule(Group(cabecalho, corpo)))
    return ui.Waiting("Pensando...", "Ctrl+O mostra o raciocínio", since=since)


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

    title_mod.marcar("falando")
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
    # O microfone aberto precisa ficar visível, e vai no campo de estado (em
    # acento, à esquerda da dica): é a única pista de que a sala está sendo
    # ouvida, e escondê-la ou apagá-la junto com os flags seria a pior escolha
    # possível aqui.
    estado = (f'ouvindo "{config.WAKE_WORD}"'
              if ctx.get("voice_mode") and config.WAKE_ENABLED else "")
    return {
        "model": chain.model_name if chain is not None else config.OLLAMA_MODEL,
        "flags": flags,
        "state": estado,
    }


def _listen(ctx: dict, ask, wait_stop=None, ask_nowait=None,
            interrupt=None, escuta_ctx=None) -> str | None:
    """Captura uma fala no modo voz. Retorna o texto ou None se nada foi captado.

    Três caminhos, do mais automático ao mais manual:

    - **wake word ligada:** o microfone já está aberto; basta dizer "Oráculo".
    - **VAD ligado:** Enter abre o microfone e a gravação encerra sozinha.
    - **nenhum dos dois:** push-to-talk, um segundo Enter encerra.
    """
    from core import audio, stt, vad, wake

    console = ctx["console"]
    ctx["last_stt_seconds"] = None
    usando_wake = config.WAKE_ENABLED and wake.disponivel()
    # Microfone aberto tem que aparecer no título também: fora da janela é a
    # única pista de que a sala está sendo ouvida.
    title_mod.marcar("ouvindo")

    if not usando_wake:
        typed = ask(f"[{config.UI_COLOR_FAINT}][voz] Enter para falar "
                    f"(ou digite e Enter):[/] ")
        if typed:
            return typed

    try:
        if usando_wake:
            path, digitado = _escutar_wake(ctx, ask_nowait, interrupt,
                                           escuta_ctx)
            if digitado:                    # você preferiu digitar
                return digitado
            if path is None:                # Ctrl+C: sai da escuta, não do app
                config.WAKE_ENABLED = False
                ui.ok(console, f'escuta encerrada — diga /despertar para '
                               f'voltar a chamar por "{config.WAKE_WORD}".')
                return None
        elif config.VAD_ENABLED and vad.disponivel():
            path = _gravar_com_vad(console)
            if path is None:
                ui.notice(console, "não ouvi nada — tente de novo")
                return None
        else:
            ui.recording(console, "gravando", "Enter para parar")
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

    if usando_wake:
        text = _apos_acordar(ctx, text)

    if not text:
        ui.notice(console, "não entendi nada — tente de novo")
        return None

    return text


def _gravar_com_vad(console) -> str | None:
    """Grava com parada automática, narrando a transição para o usuário.

    Sem o aviso de "ouvindo" a interface fica muda entre o Enter e a primeira
    palavra, e não dá para saber se o microfone abriu.
    """
    from core import audio, vad

    def _on_state(estado: str) -> None:
        # Só a entrada em FALANDO interessa: o retorno a AGUARDANDO acontece
        # quando uma rajada curta é descartada, e anunciar isso viraria ruído.
        if estado == vad.FALANDO:
            ui.recording(console, "gravando", "o VAD para sozinho quando "
                                              "você parar de falar")

    ui.notice(console, "ouvindo...")
    return audio.record_vad(on_state=_on_state)


def _escutar_wake(ctx: dict, ask_nowait, interrupt, escuta_ctx=None):
    """Espera pela palavra de despertar. Devolve (caminho do WAV, texto digitado).

    Digitar continua sendo o escape imediato: a caixa é consultada sem bloquear a
    cada bloco de 80 ms. Bloquear nela em paralelo deixaria uma thread pendurada
    que engoliria a mensagem seguinte.
    """
    from core import audio

    console = ctx["console"]
    digitado: dict[str, str] = {}

    def _abortar() -> bool:
        if interrupt is not None and interrupt.is_set():
            return True
        if ask_nowait is not None:
            texto = ask_nowait()
            if texto:
                digitado["texto"] = texto
                return True
        return False

    def _on_state(estado: str) -> None:
        if estado == "ouvindo":
            ui.notice(console, f'ouvindo — diga "{config.WAKE_WORD}" '
                               f'(Ctrl+C encerra a escuta)')
        elif estado == "acordado":
            ui.ok(console, "sim?")

    contexto = escuta_ctx() if escuta_ctx is not None else contextlib.nullcontext()
    try:
        with contexto:
            path = audio.escutar_wake(on_state=_on_state, abortar=_abortar)
    except KeyboardInterrupt:
        return None, None
    return path, digitado.get("texto")


def _apos_acordar(ctx: dict, texto: str) -> str:
    """Tira o "Oráculo," da frente; se sobrou nada, abre a escuta de continuação.

    É o caso de quem chama o nome e só depois formula o pedido. Sem isso, dizer
    apenas "Oráculo" viraria um turno vazio e a chamada se perderia.
    """
    from core import audio, stt, wake

    console = ctx["console"]

    # Segunda etapa, de graça: o detector é acústico e não sabe onde a palavra
    # caiu na frase, então "consultei o oráculo de Delfos" dispara com razão. A
    # transcrição já existe; conferir que o nome está no COMEÇO custa nada e é o
    # que faz valer a promessa de só responder a quem chamou.
    if config.WAKE_CONFIRMA_TEXTO and not wake.contem_nome(texto):
        ui.notice(console, f'não era para mim (não começou com '
                           f'"{config.WAKE_WORD}") — continuo ouvindo')
        return ""

    pedido = wake.remove_nome(texto)
    if pedido:
        return pedido

    ui.notice(console, "estou ouvindo...")
    path = audio.record_vad(start_timeout=config.WAKE_FOLLOWUP_TIMEOUT)
    if path is None:
        return ""
    ui.notice(console, "transcrevendo...")
    t0 = time.monotonic()
    pedido = stt.transcribe(path)
    ctx["last_stt_seconds"] = (ctx.get("last_stt_seconds") or 0) + \
        (time.monotonic() - t0)
    return wake.remove_nome(pedido)


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
            f"[{config.UI_COLOR_ALERT}]'{cmd.lstrip('/')}' só funciona dentro "
            f"do {config.ASSISTANT_NAME}.[/]\n"
            f"[{config.UI_COLOR_FAINT}]Rode 'oraculo' sem argumentos para abrir "
            f"o chat, ou 'oraculo ajuda' para ver os comandos.[/]"
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
    # As preferências entram ANTES de tudo: o modelo guardado decide qual chain
    # instanciar, e os flags de config (STT/TTS/VAD) precisam estar valendo
    # antes de qualquer módulo lê-los.
    preferencias = prefs_mod.carregar()
    prefs_mod.aplicar_ao_config(preferencias)

    # A lista de modelos vai sendo buscada em paralelo com o arranque, para o
    # autocomplete de /modelo já estar pronto quando a caixa aparecer.
    commands.prefetch_modelos()

    # O modelo é instanciado antes de limpar a tela: se o Ollama estiver fora do
    # ar, a mensagem de erro fica visível em vez de ser apagada logo em seguida.
    try:
        chain = OraculoChain(preferencias.get("modelo") or config.OLLAMA_MODEL)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[bold {config.UI_COLOR_ALERT}]Falha ao iniciar o "
            f"{config.ASSISTANT_NAME}:[/] {exc}\n"
            f"[{config.UI_COLOR_DIM}]O Ollama está rodando? "
            f"Tente: ollama serve[/]"
        )
        sys.exit(1)

    # A faxina roda antes da splash para a lista de "Conversas recentes" já sair
    # refletindo o que sobrou — mostrar uma conversa que acabou de ser apagada
    # seria pior que não mostrar nada.
    removidas = history_mod.limpar_antigas()

    if tui.disponivel():
        _run_fullscreen(chain, preferencias, removidas)
    else:
        _run_inline(chain, preferencias, removidas)


def _aviso_faxina(out, removidas: int) -> None:
    """Conta o que a limpeza apagou. Apagar conversa em silêncio seria pior."""
    if not removidas:
        return
    plural = "s" if removidas > 1 else ""
    ui.notice(out, f"{removidas} conversa{plural} com mais de "
                   f"{config.SESSIONS_MAX_AGE_DAYS} dias removida{plural} "
                   f"(config.SESSIONS_MAX_AGE_DAYS).")


def _novo_ctx(chain: OraculoChain, out, preferencias: dict | None = None) -> dict:
    """Estado da sessão. As preferências guardadas vencem o padrão do config."""
    prefs = preferencias or {}
    ctx = {
        "console": out,
        "chain": chain,
        "running": True,
        "voice_mode": prefs.get("voz", config.VOICE_MODE_DEFAULT),
        "thinking": False,
        "show_thinking": prefs.get("mostrar_raciocinio",
                                   config.SHOW_THINKING_DEFAULT),
    }
    # A escuta pela palavra de despertar NÃO é lida das preferências, de
    # propósito: microfone aberto é escolha de cada sessão, nunca herdada de
    # ontem por um arquivo. Ver core/prefs.py.
    #
    # O thinking só liga se o modelo realmente suportar — a preferência é um
    # desejo, não uma garantia, e o modelo pode ter mudado desde então.
    quer_thinking = prefs.get("think", config.THINKING_DEFAULT)
    if quer_thinking and llm_mod.supports_thinking(chain.model_name):
        ctx["thinking"] = True
        chain.set_thinking(True)
    return ctx


def _run_inline(chain: OraculoChain, preferencias: dict | None = None,
                removidas: int = 0) -> None:
    """Modo clássico: desenha no buffer normal, rolagem nativa do terminal."""
    ui.clear_screen(console)
    show_splash(chain.model_name, recent_sessions=history_mod.load_recent(),
                fullscreen=False)
    _aviso_faxina(console, removidas)
    ctx = _novo_ctx(chain, console, preferencias)
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

    # No inline o título é escrito pela própria thread do laço, e só nos limites
    # do turno — nunca de dentro do Live. O rich repinta de uma thread própria,
    # e uma sequência OSC caindo no meio de um quadro sai como lixo na tela. O
    # preço é o olho não girar aqui: ele muda de estado, não de quadro. No
    # fullscreen, que é o padrão, o gancho de repaint dá a animação completa.
    title_mod.abrir()
    try:
        _chat_loop(chain, ctx, ask=lambda p="": box.ask(p), live_factory=_live,
                   echo=box.rich, watch_ctrl_o=_watch_ctrl_o,
                   tick_title=title_mod.desenhar)
    finally:
        title_mod.fechar()


def _alterna_raciocinio(ctx: dict) -> None:
    """Ctrl+O: mostra/oculta o texto do raciocínio, e lembra a escolha.

    Um só lugar para os dois modos de desenho — no fullscreen quem chama é um
    atalho do prompt_toolkit, no inline é o observador de tecla em modo raw, e
    duas cópias divergiriam na hora de gravar a preferência.
    """
    novo = not ctx.get("show_thinking", False)
    ctx["show_thinking"] = novo
    prefs_mod.gravar(mostrar_raciocinio=novo)


def _run_fullscreen(chain: OraculoChain, preferencias: dict | None = None,
                    removidas: int = 0) -> None:
    """Modo tela cheia: tela alternativa, caixa fixa e rolagem própria."""
    prefs = preferencias or {}
    # A barra de status pode ser desenhada antes do laço começar, então o ctx
    # inicial já nasce com as preferências — senão ela anuncia "texto" por um
    # instante numa sessão que vai abrir em voz.
    ctx: dict = {"chain": chain, "thinking": False, "running": True,
                 "voice_mode": prefs.get("voz", config.VOICE_MODE_DEFAULT)}

    def loop(sessao) -> None:
        # O ctx é preenchido aqui porque só agora existe o console do transcript.
        ctx.update(_novo_ctx(chain, sessao.console, prefs))
        sessao.on_toggle_thinking = lambda: _alterna_raciocinio(ctx)
        show_splash(chain.model_name, recent_sessions=history_mod.load_recent(),
                    out=sessao.console, fullscreen=True)
        _aviso_faxina(sessao.console, removidas)

        def _ask(_prompt: str = "") -> str:
            return sessao.ask()

        _chat_loop(chain, ctx, ask=_ask, live_factory=sessao.live, echo=True,
                   watch_ctrl_o=lambda _toggle: contextlib.nullcontext(),
                   interrupt=sessao.interromper, wait_stop=sessao.wait_enter,
                   ask_nowait=sessao.ask_nowait,
                   escuta_ctx=sessao.modo_escuta)

    tui.run(loop, lambda: _status(ctx))


def _chat_loop(chain: OraculoChain, ctx: dict, *, ask, live_factory, echo: bool,
               watch_ctrl_o, interrupt=None, wait_stop=None,
               ask_nowait=None, escuta_ctx=None, tick_title=lambda: None) -> None:
    """Laço de turnos, compartilhado pelos dois modos de desenho.

    O que muda entre eles é injetado: de onde vem a mensagem (`ask`), o que
    mostra o preview do streaming (`live_factory`), como o Ctrl+O é observado,
    como a interrupção chega (`interrupt`, um Event no modo tela cheia — lá o
    KeyboardInterrupt não sobe pela thread do laço) e quem escreve o título
    (`tick_title`).

    O laço só **marca** o estado do título; quem emite os bytes muda com o modo,
    porque muda quem é dono do stdout. No fullscreen o `tick_title` é vazio e o
    gancho de repaint faz o trabalho na thread certa (core/tui.py).
    """
    console = ctx["console"]
    history = history_mod.SessionHistory()
    # O /retomar troca o arquivo em que esta sessão escreve, então ele precisa
    # do objeto — e não de uma cópia do caminho, que ficaria desatualizada.
    ctx["history"] = history

    while ctx["running"]:
        title_mod.marcar()
        tick_title()
        try:
            if ctx["voice_mode"]:
                user_input = _listen(ctx, ask, wait_stop=wait_stop,
                                     ask_nowait=ask_nowait, interrupt=interrupt,
                                     escuta_ctx=escuta_ctx)
                if not user_input:
                    continue
            else:
                user_input = ask()
        except (EOFError, KeyboardInterrupt):
            ui.ok(console, "Encerrando...")
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
        title_mod.marcar("pensando" if ctx.get("thinking") else "gerando")
        tick_title()
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
            # Relógio do turno: a linha de espera e o bloco de raciocínio
            # mostram o mesmo contador, contado do início da geração.
            turno_t0 = time.monotonic()

            with live_factory() as live, watch_ctrl_o(
                    lambda: _alterna_raciocinio(ctx)):
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
                            live.update(_thinking_view(
                                ctx.get("show_thinking"), "".join(reasoning),
                                turno_t0))
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
                        live.update(ui.body_view(Markdown("".join(chunks))))
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
            title_mod.marcar()
            tick_title()


if __name__ == "__main__":
    # Com argumentos, roda o comando e sai; sem argumentos, abre o chat.
    if len(sys.argv) > 1:
        sys.exit(run_standalone(sys.argv[1:]))
    main()
