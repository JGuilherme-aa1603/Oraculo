"""Caixa de entrada do chat — borda, histórico e autocomplete dos /comandos.

Desenha uma caixa que reflui conforme você digita, com a barra de status logo
abaixo (modelo, modo, thinking, memória):

    ╭──────────────────────────────────────────╮
    │ > /tr                                    │
    ╰──────────────────────────────────────────╯
      gemma4:e4b · texto · think off · mem 8/10

Por que uma `Application` inline em vez do `PromptSession` padrão: o prompt comum
do prompt_toolkit não fecha a borda direita (a linha do texto tem largura variável),
o que deixa a caixa quebrada. Uma Application não-fullscreen com `Frame` desenha a
moldura completa, reflui em texto longo e apaga tudo ao enviar — o transcript fica
limpo, com o eco da mensagem sendo impresso pelo `core.ui`.

Degradação: sem prompt_toolkit instalado, sem TTY (pipe, redirecionamento) ou com
INPUT_RICH_EDITOR desligado, cai para `console.input()`. O modo texto nunca depende
desta camada para funcionar.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import config

# Enter envia; Alt+Enter quebra linha. Ctrl+C/Ctrl+D seguem o contrato do loop
# principal (interromper / encerrar), então são repassados como exceções.
_HINT = "Enter envia · Alt+Enter quebra linha · / comandos"


def _available() -> bool:
    """A caixa só entra se estiver ligada, houver TTY e a lib existir."""
    import sys

    if not config.INPUT_RICH_EDITOR or not sys.stdin.isatty():
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return True


def _trim_history() -> None:
    """Mantém o arquivo de histórico dentro de INPUT_HISTORY_MAX entradas.
    O FileHistory do prompt_toolkit cresce indefinidamente sozinho."""
    path = config.INPUT_HISTORY_FILE
    with contextlib.suppress(OSError):
        if not path.exists():
            return
        linhas = path.read_text(encoding="utf-8").splitlines(keepends=True)
        # Cada entrada é um bloco iniciado por "# <timestamp>".
        marcos = [i for i, ln in enumerate(linhas) if ln.startswith("# ")]
        if len(marcos) <= config.INPUT_HISTORY_MAX:
            return
        corte = marcos[len(marcos) - config.INPUT_HISTORY_MAX]
        path.write_text("".join(linhas[corte:]), encoding="utf-8")


def _precisa_argumento(texto: str) -> bool:
    """True se o comando digitado espera um argumento (tem dica em COMMAND_SPECS).
    Serve para o Enter decidir entre enviar na hora ou esperar o argumento."""
    from core import commands as commands_mod

    cmd = texto.strip().lower()
    for nome, arg, _ in commands_mod.COMMAND_SPECS:
        if nome == cmd:
            return bool(arg)
    return False


def _build_completer():
    """Autocomplete: /comandos no início da linha e caminhos para /transcrever."""
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    from core import commands as commands_mod

    paths = PathCompleter(expanduser=True)

    class OraculoCompleter(Completer):
        def get_completions(self, document, complete_event):
            texto = document.text_before_cursor
            # Argumento de /transcrever → completa caminhos de arquivo.
            if texto.startswith(("/transcrever ", "/transcricao ")):
                arg = texto.split(" ", 1)[1]
                yield from paths.get_completions(Document(arg, len(arg)),
                                                 complete_event)
                return
            # Argumento de /stt → completa os motores.
            if texto.startswith("/stt "):
                parcial = texto[len("/stt "):].lstrip()
                for eng in commands_mod.STT_ENGINES:
                    if eng.startswith(parcial):
                        yield Completion(eng, start_position=-len(parcial))
                return
            # Início da linha → completa os comandos.
            if texto.startswith("/") and " " not in texto:
                for cmd, arg, desc in commands_mod.COMMAND_SPECS:
                    if cmd.startswith(texto.lower()):
                        rotulo = f"{cmd} {arg}".strip()
                        yield Completion(
                            cmd,
                            start_position=-len(texto),
                            display=rotulo,
                            display_meta=desc,
                        )

    return OraculoCompleter()


def _janela_entrada(buffer, **kwargs):
    """Janela da caixa de entrada que rola sozinha ao arrastar contra a borda.

    Sem isto só dá para selecionar o que está visível: o `Window` do
    prompt_toolkit traduz a posição da tela para uma posição no documento e
    **prende ao último visível** (`y = min(max_y, y)`), então arrastar para fora
    da caixa não passa dali. A caixa cresce até 8 linhas e depois rola, então uma
    mensagem longa fica com a maior parte fora de alcance.

    O handler é interceptado ANTES dessa tradução, usando coordenadas de tela.
    No limite, o cursor anda uma linha visual e a seleção é estendida — é o
    cursor que puxa a rolagem, porque o Window recalcula o scroll a cada quadro
    para manter o cursor visível (mexer no scroll direto seria desfeito).
    """
    from prompt_toolkit.layout import Window
    from prompt_toolkit.mouse_events import MouseButton, MouseEventType

    class _Entrada(Window):
        arrastando = False

        def write_to_screen(self, screen, mouse_handlers, write_position,
                            parent_style, erase_bg, z_index) -> None:
            super().write_to_screen(screen, mouse_handlers, write_position,
                                    parent_style, erase_bg, z_index)
            wp = write_position
            if wp.height <= 0 or wp.width <= 0:
                return
            original = mouse_handlers.mouse_handlers[wp.ypos][wp.xpos]
            if not callable(original):
                return
            topo = wp.ypos
            base = wp.ypos + wp.height - 1
            largura = wp.width

            def _com_autoscroll(ev):
                if ev.event_type == MouseEventType.MOUSE_DOWN:
                    self.arrastando = True
                elif ev.event_type == MouseEventType.MOUSE_UP:
                    self.arrastando = False
                elif (ev.event_type == MouseEventType.MOUSE_MOVE
                        and ev.button == MouseButton.LEFT):
                    # Sem consultar top_visible/bottom_visible: eles raciocinam
                    # em linhas do DOCUMENTO, e uma mensagem digitada sem quebras
                    # é uma linha só — a primeira estaria "sempre visível" e a
                    # rolagem nunca aconteceria. Quem limita é o cursor, que já
                    # está preso aos extremos do texto.
                    if ev.position.y <= topo:
                        self._arrastar(-largura)
                        return None
                    if ev.position.y >= base:
                        self._arrastar(largura)
                        return None
                elif (ev.event_type == MouseEventType.MOUSE_MOVE
                        and ev.button == MouseButton.NONE):
                    # Soltou fora da janela: o terminal reporta o movimento, não
                    # a soltura. Sem isto o arrasto ficaria ativo para sempre.
                    self.arrastando = False
                # Fora da caixa não há posição de texto para o handler original
                # traduzir; ele prenderia o cursor na borda e desfaria a seleção.
                if not (topo <= ev.position.y <= base):
                    return None
                return original(ev)

            mouse_handlers.set_mouse_handler_for_range(
                wp.xpos, wp.xpos + wp.width, wp.ypos, wp.ypos + wp.height,
                _com_autoscroll)

            # Enquanto arrasta, a tela inteira responde a este handler: o
            # ponteiro sai da caixa (passa pela moldura, pela conversa) e os
            # eventos iriam para outras janelas, congelando a seleção.
            # O tamanho vem do output, não do `screen`: o renderer nunca atribui
            # `screen.width`/`height`, que ficam em 0 e dariam uma faixa vazia.
            if self.arrastando:
                from prompt_toolkit.application import get_app

                tam = get_app().output.get_size()
                mouse_handlers.set_mouse_handler_for_range(
                    0, tam.columns, 0, tam.rows, _com_autoscroll)

        def _arrastar(self, passo: int) -> None:
            """Estende a seleção uma linha visual acima/abaixo.

            O passo é a largura da caixa porque o texto digitado costuma ser uma
            única linha lógica quebrada na tela: mover "uma linha" pelo documento
            não sairia do lugar. É aproximado — ao voltar o ponteiro para dentro
            da caixa, o handler original recoloca o cursor na posição exata.
            """
            if buffer.selection_state is None:
                buffer.start_selection()
            destino = buffer.cursor_position + passo
            buffer.cursor_position = max(0, min(destino, len(buffer.text)))

    return _Entrada(**kwargs)


def _rounded_frame(body):
    """Moldura arredondada ao redor do corpo.

    O `Frame` do prompt_toolkit fixa os cantos em ┌┐└┘ (a classe Border tem os
    caracteres hard-coded, sem parâmetro), então a moldura é remontada aqui com
    os cantos arredondados para casar com a paleta do resto da interface.
    """
    from functools import partial

    from prompt_toolkit.layout import HSplit, VSplit, Window

    fill = partial(Window, style="class:frame.border")
    return HSplit([
        VSplit([fill(width=1, height=1, char="╭"),
                fill(char="─"),
                fill(width=1, height=1, char="╮")], height=1),
        VSplit([fill(width=1, char="│"), body, fill(width=1, char="│")]),
        VSplit([fill(width=1, height=1, char="╰"),
                fill(char="─"),
                fill(width=1, height=1, char="╯")], height=1),
    ])


def _status_fragments(status: dict, largura: int) -> list[tuple[str, str]]:
    """Barra de status abaixo da caixa, em fragmentos (estilo, texto).

    A dica de teclas é o primeiro item a cair quando o terminal é estreito —
    deixar a barra quebrar em duas linhas desalinharia tudo abaixo da caixa.
    """
    esquerda = f"  {status['model']}"
    partes: list[tuple[str, str]] = [("class:status.accent", esquerda)]
    usado = len(esquerda)
    for rotulo in status["flags"]:
        trecho = f"  ·  {rotulo}"
        if usado + len(trecho) > largura:
            return partes
        partes.append(("class:status", trecho))
        usado += len(trecho)
    # O modo tela cheia manda a própria dica (rolagem, F2), que não faz sentido
    # no inline.
    dica = status.get("hint") or _HINT
    if usado + len(dica) + 3 <= largura:
        partes.append(("class:status.hint", f"   {dica}"))
    return partes


def build_editor(status_fn: Callable[[], dict],
                 on_submit: Callable[[str], None]) -> dict:
    """Monta os widgets da entrada e devolve as peças soltas.

    Fábrica compartilhada: o modo inline (`InputBox`) embrulha isso numa
    Application própria, e o modo fullscreen (`core.tui`) embute as mesmas peças
    na app que desenha a tela inteira. Assim a caixa, o histórico, o autocomplete
    e as teclas são idênticos nos dois modos — só muda quem é dono da tela.

    `on_submit` recebe o texto quando o Enter conclui o envio.
    """
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import has_completions, has_selection
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.layout.processors import BeforeInput
    from prompt_toolkit.styles import Style

    config.INPUT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _trim_history()

    buffer = Buffer(
        multiline=True,
        history=FileHistory(str(config.INPUT_HISTORY_FILE)),
        completer=_build_completer(),
        complete_while_typing=True,
    )

    kb = KeyBindings()

    @kb.add("enter")
    def _enviar(event) -> None:
        buf = event.app.current_buffer
        estado = buf.complete_state
        if estado is not None:
            if estado.current_completion is not None:
                # Escolhido com Tab/setas: o texto já está no buffer. Fecha o
                # menu zerando o estado — `cancel_completion()` NÃO serve aqui,
                # ele faz go_to_completion(None) e desfaz o que o Tab inseriu.
                buf.complete_state = None
            elif (buf.text.startswith("/") and " " not in buf.text
                    and len(estado.completions) == 1):
                # Comando incompleto com uma única saída: Enter completa.
                buf.apply_completion(estado.completions[0])
                if _precisa_argumento(buf.text):
                    # Ainda falta o argumento (ex.: /transcrever <arquivo>):
                    # abre espaço e espera — o próximo Enter é que envia.
                    buf.insert_text(" ")
                    return
        on_submit(buf.text)

    @kb.add("escape", "enter")     # Alt+Enter
    def _nova_linha(event) -> None:
        event.app.current_buffer.insert_text("\n")

    # --- Seleção: digitar substitui, Backspace/Delete apaga -----------------
    # O prompt_toolkit não faz isso sozinho: `insert_text` e `delete_before_cursor`
    # ignoram a seleção, então o texto marcado sobrevivia ao que se digitava por
    # cima. `<any>` tem prioridade menor que as teclas específicas, então isto não
    # engole setas, Enter, Tab nem os atalhos.

    @kb.add("<any>", filter=has_selection)
    def _substituir_selecao(event) -> None:
        buf = event.current_buffer
        buf.cut_selection()
        buf.insert_text(event.data)

    @kb.add("backspace", filter=has_selection)
    @kb.add("delete", filter=has_selection)
    def _apagar_selecao(event) -> None:
        event.current_buffer.cut_selection()

    @kb.add("tab")
    def _proxima_sugestao(event) -> None:
        buf = event.app.current_buffer
        if buf.complete_state:
            buf.complete_next()
        else:
            buf.start_completion(select_first=True)

    @kb.add("s-tab")
    def _sugestao_anterior(event) -> None:
        buf = event.app.current_buffer
        if buf.complete_state:
            buf.complete_previous()

    entrada = _janela_entrada(
        buffer,
        content=BufferControl(
            buffer=buffer,
            input_processors=[BeforeInput(f"{config.UI_GLYPH_USER} ",
                                          style="class:prompt")],
        ),
        wrap_lines=True,
        height=Dimension(min=1, max=8),
        # Sem isto, no modo tela cheia o HSplit entrega a sobra vertical para a
        # caixa (ela aceita até 8 linhas) e ela abre linhas em branco. A sobra
        # tem que ir toda para o transcript.
        dont_extend_height=True,
    )
    moldura = _rounded_frame(entrada)

    def _status_texto():
        from prompt_toolkit.application import get_app

        return _status_fragments(status_fn(), get_app().output.get_size().columns)

    status = Window(FormattedTextControl(_status_texto), height=1)

    # O menu de completion entra no fluxo, abaixo da barra de status, em vez
    # de flutuar: numa Application não-fullscreen o float fica preso à altura
    # da própria app e acabaria desenhado por cima da borda inferior da caixa.
    menu = ConditionalContainer(
        CompletionsMenu(max_height=6, scroll_offset=1),
        filter=has_completions,
    )

    estilo = Style.from_dict({
        "frame.border": "#5f8787",
        "prompt": "#00d7d7 bold",
        "status": "#6c6c6c",
        "status.accent": "#00d7d7",
        "status.hint": "#444444",
        "completion-menu.completion": "bg:#1c1c1c #b2b2b2",
        "completion-menu.completion.current": "bg:#00afaf #000000",
        "completion-menu.meta.completion": "bg:#1c1c1c #6c6c6c",
        "completion-menu.meta.completion.current": "bg:#008787 #000000",
    })

    # Um único container: um mesmo objeto de layout não pode ser montado em dois
    # lugares da árvore do prompt_toolkit.
    return {
        "buffer": buffer,
        "entrada": entrada,
        "keys": kb,
        "estilo": estilo,
        "container": HSplit([moldura, status, menu]),
    }


class InputBox:
    """Caixa de entrada reutilizável entre turnos (preserva o histórico)."""

    def __init__(self, console, status_fn: Callable[[], dict]) -> None:
        self._console = console
        self._status_fn = status_fn
        self._app = None
        self._buffer = None
        if _available():
            with contextlib.suppress(Exception):
                self._build()

    # -- construção ------------------------------------------------------
    def _build(self) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.layout import Layout

        editor = build_editor(self._status_fn, on_submit=self._submit)
        self._buffer = editor["buffer"]

        # Ctrl+C/Ctrl+D ficam fora da fábrica: aqui eles encerram a Application
        # da vez (um turno de leitura), enquanto no modo fullscreen precisam
        # interromper a geração e encerrar a sessão inteira.
        kb = editor["keys"]

        @kb.add("c-c")
        def _interromper(event) -> None:
            event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

        @kb.add("c-d")
        def _encerrar(event) -> None:
            if not event.app.current_buffer.text:
                event.app.exit(exception=EOFError, style="class:aborting")

        self._app = Application(
            layout=Layout(editor["container"], focused_element=editor["entrada"]),
            key_bindings=kb,
            style=editor["estilo"],
            full_screen=False,
            erase_when_done=True,   # some ao enviar; o eco fica por conta do core.ui
        )

    def _submit(self, texto: str) -> None:
        from prompt_toolkit.application import get_app

        get_app().exit(result=texto)

    # -- uso -------------------------------------------------------------
    @property
    def rich(self) -> bool:
        """True se a caixa está ativa (False = fallback do rich)."""
        return self._app is not None

    def ask(self, fallback_prompt: str = "") -> str:
        """Lê uma mensagem. Propaga KeyboardInterrupt/EOFError como o input()."""
        if self._app is None:
            return self._console.input(fallback_prompt or
                                       f"[bold {config.UI_COLOR_USER}]>[/] ").strip()
        self._buffer.reset()
        texto = self._app.run()
        return (texto or "").strip()
