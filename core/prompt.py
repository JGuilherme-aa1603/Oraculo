"""Caixa de entrada do chat — borda, histórico e autocomplete dos /comandos.

Desenha uma caixa que reflui conforme você digita, com a barra de status logo
abaixo (modelo, modo, thinking, memória):

    ┌──────────────────────────────────────────┐
    │ > /tr                                    │
    └──────────────────────────────────────────┘
      gemma4:e4b · texto · think off · mem 8/10        Enter envia

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
import time
from collections.abc import Callable
from pathlib import Path

import config

# Comandos cujo argumento é um caminho de arquivo. O Enter tem tratamento
# próprio dentro deles: entrar numa pasta em vez de enviar.
_ARGUMENTOS_DE_CAMINHO = ("/transcrever ", "/transcricao ")

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


def _escolhe_sozinho(texto: str) -> bool:
    """True se o Enter pode aplicar a primeira sugestão sem você confirmar.

    Vale para nome de comando e para os argumentos de conjunto fechado (motor de
    STT, modelo do Ollama, número de sessão): ali as opções são todas conhecidas
    e a primeira da lista é a que está na sua frente na tela.

    **Não** vale para ARQUIVO. Um caminho que você digitou por inteiro pode ser
    prefixo de outro (`/tmp/nota.ogg` e `/tmp/nota.ogg.bak`), e escolher sozinho
    mandaria o comando para o arquivo errado — com o agravante de que
    `/transcrever` gasta minutos antes de você perceber. Pasta é outra história e
    tem caminho próprio (`_entra_na_pasta`): entrar numa pasta não decide nada,
    só mostra o que tem dentro.
    """
    from core import commands as commands_mod

    t = texto.lstrip()
    if not t.startswith("/"):
        return False
    if " " not in t:
        return True
    return t.split(" ", 1)[0].lower() in commands_mod.COMANDOS_COM_OPCOES_FECHADAS


def _argumento_de_caminho(texto: str) -> str | None:
    """O trecho do texto que é um caminho de arquivo, ou None."""
    for prefixo in _ARGUMENTOS_DE_CAMINHO:
        if texto.startswith(prefixo):
            return texto[len(prefixo):]
    return None


def _e_pasta(caminho: str) -> bool:
    if not caminho.strip():
        return False
    try:
        return Path(caminho.strip()).expanduser().is_dir()
    except OSError:      # caminho absurdo, permissão, link quebrado
        return False


def _build_completer(status_fn):
    """Autocomplete dos /comandos e dos seus argumentos.

    `status_fn` é a mesma função que alimenta a barra de status; daqui ela serve
    só para saber qual modelo está ativo e marcá-lo na lista.
    """
    import contextlib as _contextlib

    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    from core import commands as commands_mod

    paths = PathCompleter(expanduser=True)

    def _modelo_ativo() -> str:
        with _contextlib.suppress(Exception):
            return (status_fn() or {}).get("model", "") or ""
        return ""

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
            # Argumento de /modelo → completa os modelos instalados no Ollama.
            # A lista vem de um cache alimentado em segundo plano: uma consulta
            # HTTP aqui travaria a caixa a cada tecla digitada.
            if texto.startswith("/modelo "):
                parcial = texto[len("/modelo "):].lstrip()
                atual = _modelo_ativo()
                for nome in commands_mod.modelos_conhecidos():
                    if nome.startswith(parcial):
                        yield Completion(
                            nome,
                            start_position=-len(parcial),
                            display=nome,
                            display_meta="em uso" if nome == atual else "",
                        )
                return
            # Argumento de /retomar → os números da lista, com a conversa ao lado.
            # Sem isto o número é um índice cego e você teria de rodar /retomar
            # primeiro só para descobrir o que é o 3.
            if texto.startswith("/retomar "):
                parcial = texto[len("/retomar "):].lstrip()
                for i, s in enumerate(commands_mod.sessoes_recentes(), 1):
                    rotulo = str(i)
                    if rotulo.startswith(parcial):
                        yield Completion(
                            rotulo,
                            start_position=-len(parcial),
                            display=f"{rotulo}. {commands_mod.resumo_titulo(s)}",
                            display_meta=s.get("ago", ""),
                        )
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


def _frame(body):
    """Moldura de canto vivo ao redor do corpo.

    Remontada à mão em vez de usar o `Frame` do prompt_toolkit porque a classe
    `Border` dele tem os caracteres hard-coded, sem parâmetro — e é aqui que a
    moldura ganha a cor da paleta. Os cantos são retos de propósito: a moldura é
    estrutura, e o sistema visual do projeto não arredonda canto nenhum.
    """
    from functools import partial

    from prompt_toolkit.layout import HSplit, VSplit, Window

    fill = partial(Window, style="class:frame.border")
    return HSplit([
        VSplit([fill(width=1, height=1, char="┌"),
                fill(char="─"),
                fill(width=1, height=1, char="┐")], height=1),
        VSplit([fill(width=1, char="│"), body, fill(width=1, char="│")]),
        VSplit([fill(width=1, height=1, char="└"),
                fill(char="─"),
                fill(width=1, height=1, char="┘")], height=1),
    ])


def _status_fragments(status: dict, largura: int) -> list[tuple[str, str]]:
    """Barra de status abaixo da caixa, em fragmentos (estilo, texto).

    Três grupos, com pesos diferentes de propósito:

        gemma4:e4b · texto · think off · mem 3/10 · ouvindo            <dica>
        ^ acento     ^ estado permanente          ^ acontecendo agora  ^ direita

    A dica de teclas encosta na borda direita e é o primeiro item a cair quando
    o terminal é estreito — deixar a barra quebrar em duas linhas desalinharia
    tudo abaixo da caixa. O `state` cai depois dela, e os flags por último.
    """
    esquerda = f"  {status['model']}"
    partes: list[tuple[str, str]] = [("class:status.accent", esquerda)]
    usado = len(esquerda)

    def _separado(trecho: str, estilo: str) -> bool:
        """Acrescenta '  ·  trecho' se couber. False = não coube."""
        nonlocal usado
        if usado + len(trecho) + 5 > largura:
            return False
        partes.append(("class:status.sep", "  ·  "))
        partes.append((estilo, trecho))
        usado += len(trecho) + 5
        return True

    for rotulo in status["flags"]:
        if not _separado(rotulo, "class:status"):
            return partes

    # O que está acontecendo agora (ouvindo, rolagem pausada, copiado): é a
    # informação mais volátil da barra, então ganha o acento.
    estado = status.get("state")
    if estado:
        _separado(estado, "class:status.state")

    # O modo tela cheia manda a própria dica (rolagem, F2), que não faz sentido
    # no inline.
    dica = status.get("hint") or _HINT
    folga = largura - usado - len(dica) - 2
    if folga >= 3:
        partes.append(("class:status.hint", " " * folga + dica))
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

    completer = _build_completer(status_fn)
    buffer = Buffer(
        multiline=True,
        history=FileHistory(str(config.INPUT_HISTORY_FILE)),
        completer=completer,
        complete_while_typing=True,
    )

    def _primeira_sugestao(buf):
        """A primeira sugestão para o texto atual, calculada NA HORA.

        Não dá para confiar no `complete_state` aqui: com
        `complete_while_typing` o menu é preenchido de forma assíncrona, então
        quem digita rápido — ou cola a linha inteira — chega no Enter antes de
        existir sugestão nenhuma, e o comando abreviado seguia adiante como
        "comando desconhecido". O sintoma dependia da velocidade de digitação,
        que é o tipo de bug que não se reproduz quando se vai procurar.

        Recalcular custa pouco: estes completers leem cache e listas em memória,
        nunca a rede. O de caminho de arquivo, que toca o disco, nunca chega
        aqui — `_escolhe_sozinho` o mantém de fora.
        """
        from prompt_toolkit.completion import CompleteEvent

        for sugestao in completer.get_completions(buf.document, CompleteEvent()):
            return sugestao
        return None

    def _destacar_primeira(_buf=None) -> None:
        """Deixa a primeira sugestão já destacada assim que o menu abre.

        Mexe SÓ no `complete_index`. O caminho normal de seleção do
        prompt_toolkit é o `go_to_completion`, que além de destacar **reescreve
        o texto do buffer** com a sugestão — e aí digitar `/` viraria `/ajuda`
        na caixa, e a próxima tecla sairia como `/ajudat`. Como
        `current_completion` é derivado do índice, atribuí-lo direto destaca sem
        encostar no documento.

        Com a primeira já destacada, a seta para baixo vai para a **segunda**,
        que é o comportamento esperado de um menu que já mostra uma escolha
        feita. Quem aplica a primeira é o Enter (ou o Tab).
        """
        estado = buffer.complete_state
        if (estado is not None and estado.complete_index is None
                and estado.completions):
            estado.complete_index = 0

    buffer.on_completions_changed += _destacar_primeira

    def _destaque_nao_aplicado(buf) -> bool:
        """True se há sugestão destacada mas o texto ainda é o que você digitou.

        É o estado que `_destacar_primeira` cria e que não existe no
        prompt_toolkit padrão, onde destacar e inserir andam juntos.
        """
        estado = buf.complete_state
        return (estado is not None and estado.current_completion is not None
                and buf.text == estado.original_document.text)

    def _enter_no_caminho(buf) -> bool:
        """Enter dentro de um argumento de caminho — navegar pastas com Enter.

        A sugestão em foco (a destacada, ou a primeira) é aplicada e então:

        - **pasta** → ganha a barra e o menu reabre com o conteúdo dela. Nada é
          enviado: descer um nível não decide nada, só mostra o que tem dentro.
          É o que faz escolher um áudio virar navegação em vez de digitar o
          caminho inteiro de cabeça.
        - **arquivo** → o nome fica completo na caixa e o Enter segue para o
          envio, que é o que você queria ao apontar para ele.

        Escolher o arquivo em foco é seguro aqui porque ele está **destacado na
        tela**: você vê o que vai ser escolhido antes de confirmar. E o
        `PathCompleter` ordena por nome, então um `nota.ogg` digitado inteiro vem
        antes do `nota.ogg.bak` — o casamento exato nunca perde para uma extensão
        a mais.

        Devolve True quando tratou tudo (e aí não se envia nada).

        Detalhe que custou uma iteração: o `PathCompleter` põe a barra final da
        pasta só no `display`, nunca no texto da sugestão. Sem acrescentá-la
        aqui, o menu não reabre e a navegação para na primeira pasta.
        """
        arg = _argumento_de_caminho(buf.text)
        if arg is None:
            return False

        # Sem nada digitado não há a que apontar: deixa o comando seguir e
        # mostrar o próprio modo de usar, em vez de fisgar um arquivo qualquer
        # do diretório atual.
        if arg.strip():
            if _destaque_nao_aplicado(buf):
                sugestao = buf.complete_state.current_completion
            elif buf.complete_state is not None:
                sugestao = None     # navegou com a seta: o texto já está aqui
            else:
                sugestao = _primeira_sugestao(buf)
            if sugestao is not None:
                buf.complete_state = None
                buf.apply_completion(sugestao)
                arg = _argumento_de_caminho(buf.text) or ""

        if not _e_pasta(arg):
            return False            # arquivo (ou caminho livre): segue e envia
        if not arg.endswith("/"):
            buf.insert_text("/")
        buf.start_completion()
        return True

    kb = KeyBindings()

    @kb.add("enter")
    def _enviar(event) -> None:
        buf = event.app.current_buffer

        # Caminho de arquivo tem regra própria: pasta se entra, não se envia.
        if _enter_no_caminho(buf):
            return

        estado = buf.complete_state
        if (estado is not None and estado.current_completion is not None
                and not _destaque_nao_aplicado(buf)):
            # Escolhido com Tab/setas: o texto já está no buffer. Fecha o menu
            # zerando o estado — `cancel_completion()` NÃO serve aqui, ele faz
            # go_to_completion(None) e desfaz o que o Tab inseriu.
            buf.complete_state = None
        elif _escolhe_sozinho(buf.text):
            # Nada escolhido à mão: o Enter fica com a PRIMEIRA sugestão, que é
            # a que está no topo do menu, à sua frente. Antes isso só valia
            # quando havia exatamente uma opção; com duas era preciso descer com
            # a seta só para confirmar o que já estava visível.
            sugestao = _primeira_sugestao(buf)
            if sugestao is not None:
                # Zerado ANTES do apply: com o estado montado, `apply_completion`
                # começa por um `go_to_completion(None)` que devolve o documento
                # ao original — e a posição da nossa sugestão foi medida no
                # documento de agora.
                buf.complete_state = None
                antes = buf.text
                buf.apply_completion(sugestao)
                if buf.text != antes and _precisa_argumento(buf.text):
                    # Expandimos uma abreviação para um comando que espera
                    # argumento (`/tra` → `/transcrever`): abre espaço e espera
                    # — o próximo Enter é que envia.
                    #
                    # A comparação com o texto anterior é o que separa os dois
                    # casos. Quem digitou `/modelo` inteiro e apertou Enter quer
                    # RODAR o comando (que sem argumento lista os modelos), e
                    # antes precisava de dois Enters para isso: o primeiro só
                    # inseria um espaço.
                    buf.insert_text(" ")
                    return
        # O histórico é gravado AQUI porque este Enter substitui o
        # `validate_and_handle()` do prompt_toolkit, que é quem normalmente
        # chamaria isto. Sem a chamada, `~/.oraculo/input_history` nunca era
        # criado e a seta para cima percorria um histórico vazio — a caixa
        # tinha um `FileHistory` configurado desde sempre, só nunca escrito.
        buf.append_to_history()
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
        if not buf.complete_state:
            buf.start_completion(select_first=True)
            return
        if _destaque_nao_aplicado(buf):
            # A primeira já está destacada mas ainda não foi escrita na caixa:
            # o Tab escreve. Sem isto ele pularia para a segunda e a primeira
            # ficaria inalcançável pelo Tab.
            buf.go_to_completion(buf.complete_state.complete_index)
            return
        buf.complete_next()

    @kb.add("s-tab")
    def _sugestao_anterior(event) -> None:
        buf = event.app.current_buffer
        if buf.complete_state:
            buf.complete_previous()

    entrada = _janela_entrada(
        buffer,
        content=BufferControl(
            buffer=buffer,
            # Um espaço antes do glifo: sem ele o ">" encosta na borda esquerda
            # e a caixa fica apertada de um lado só.
            input_processors=[BeforeInput(f" {config.UI_GLYPH_USER} ",
                                          style="class:prompt")],
        ),
        wrap_lines=True,
        style="class:input",
        height=Dimension(min=1, max=8),
        # Sem isto, no modo tela cheia o HSplit entrega a sobra vertical para a
        # caixa (ela aceita até 8 linhas) e ela abre linhas em branco. A sobra
        # tem que ir toda para o transcript.
        dont_extend_height=True,
    )
    moldura = _frame(entrada)

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

    # Mesma paleta do transcript, vinda do config: o prompt_toolkit não fala
    # rich, então os hex são repassados diretamente. É por isso que a paleta é
    # hex e não nome de cor — nome do rich não teria tradução aqui.
    estilo = Style.from_dict({
        "frame.border": config.UI_COLOR_BORDER,
        "prompt": f"{config.UI_COLOR_PROMPT} bold",
        "input": config.UI_COLOR_BRIGHT,
        "status": config.UI_COLOR_DIM,
        "status.accent": f"{config.UI_COLOR_ACCENT} bold",
        "status.state": config.UI_COLOR_ACCENT,
        "status.sep": config.UI_COLOR_FAINT,
        "status.hint": config.UI_COLOR_FAINT,
        "completion-menu.completion": f"bg:#12143a {config.UI_COLOR_SOFT}",
        "completion-menu.completion.current":
            f"bg:{config.UI_COLOR_BORDER} #05060f",
        "completion-menu.meta.completion": f"bg:#12143a {config.UI_COLOR_FAINT}",
        "completion-menu.meta.completion.current":
            f"bg:{config.UI_COLOR_ACCENT} #05060f",
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

        # Instante do último Ctrl+C ocioso. Numa lista para o closure poder
        # escrever nela sem `nonlocal` espalhado pelos handlers.
        armado = [0.0]

        def _saida_armada() -> bool:
            return time.monotonic() - armado[0] < config.CTRL_C_EXIT_WINDOW

        def _status_com_saida() -> dict:
            estado = dict(self._status_fn())
            if _saida_armada():
                estado["state"] = "Ctrl+C de novo encerra"
                estado["hint"] = "qualquer tecla cancela"
            return estado

        editor = build_editor(_status_com_saida, on_submit=self._submit)
        self._buffer = editor["buffer"]
        self._buffer.on_text_changed += lambda _b: armado.__setitem__(0, 0.0)

        # Ctrl+C/Ctrl+D ficam fora da fábrica: aqui eles encerram a Application
        # da vez (um turno de leitura), enquanto no modo fullscreen precisam
        # interromper a geração e encerrar a sessão inteira.
        kb = editor["keys"]

        @kb.add("c-c")
        def _interromper(event) -> None:
            # O primeiro Ctrl+C limpa a caixa; só o segundo encerra. Mesmo
            # contrato do modo tela cheia (ver core/tui.py).
            buf = event.app.current_buffer
            if buf.text:
                buf.reset()
                armado[0] = time.monotonic()
                return
            if _saida_armada():
                event.app.exit(exception=KeyboardInterrupt,
                               style="class:aborting")
            else:
                armado[0] = time.monotonic()

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
