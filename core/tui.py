"""Interface em tela cheia — o transcript rola por conta própria.

Desenha na tela alternativa do terminal (a mesma do vim/htop), com a caixa de
entrada fixa no rodapé e a conversa rolando acima dela. Ao sair, o terminal volta
exatamente como estava.

Como a rolagem existe se a tela alternativa não tem scrollback: ela não vem do
terminal, vem daqui. O transcript é uma lista de blocos em memória; a cada quadro
só as linhas visíveis são desenhadas — é a mesma estratégia do vim, e do modo
fullscreen do Claude Code.

A ponte com o resto do projeto é o `TranscriptConsole`: um `rich.Console` que, em
vez de escrever no stdout, guarda o que foi impresso como um bloco do transcript.
Com isso `core.ui` e `core.commands` funcionam aqui sem alteração nenhuma — eles
continuam chamando `console.print()` como sempre.

Divisão de threads:
  - principal: a Application do prompt_toolkit, dona da tela e do teclado;
  - trabalhadora: o laço de conversa (LLM, TTS, comandos), que bloqueia numa fila
    esperando a próxima mensagem.
O laço nunca toca na tela: ele escreve no transcript e pede um repaint.
"""

from __future__ import annotations

import base64
import contextlib
import io
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

from rich.console import Console

import config

# Sequências ANSI, para extrair o texto puro de uma linha já renderizada.
_ANSI = re.compile(r"\x1b\[[0-9;:?]*[a-zA-Z]|\x1b[()][A-B0-2]")
# Caracteres que compõem uma "palavra" no duplo-clique.
_PALAVRA = re.compile(r"[\w\-/.:@]", re.UNICODE)


def texto_puro(linha: str) -> str:
    """Texto de uma linha renderizada, sem os códigos de cor."""
    return _ANSI.sub("", linha)


def copiar(texto: str) -> str:
    """Copia para a área de transferência. Devolve o método usado.

    Tenta as ferramentas do sistema e, se não houver nenhuma, cai no OSC 52 —
    uma sequência que pede ao próprio terminal para copiar. Não exige binário
    algum e funciona até por SSH; em compensação, alguns terminais a desativam
    por padrão.
    """
    dados = texto.encode("utf-8")
    for cmd in (["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"]):
        if shutil.which(cmd[0]):
            # O processo roda numa thread: esta função é chamada de dentro do
            # tratamento do mouse, na thread que desenha a tela. Um utilitário
            # de clipboard que demore a sair congelaria a interface inteira.
            # Sem shell=True, sempre lista de argumentos (regra do projeto).
            def _executa(cmd=cmd) -> None:
                with contextlib.suppress(Exception):
                    subprocess.run(cmd, input=dados, check=False, timeout=5)

            threading.Thread(target=_executa, daemon=True).start()
            return cmd[0]
    try:
        from prompt_toolkit.application import get_app

        seq = "\x1b]52;c;" + base64.b64encode(dados).decode("ascii") + "\x07"
        out = get_app().output
        out.write_raw(seq)
        out.flush()
        return "osc52"
    except Exception:  # noqa: BLE001
        return ""


def _limites_palavra(texto: str, col: int) -> tuple[int, int]:
    """Início e fim da palavra sob a coluna (usado no duplo-clique)."""
    if not texto:
        return (0, 0)
    col = max(0, min(col, len(texto) - 1))
    if not _PALAVRA.match(texto[col]):
        return (col, col + 1)
    ini = col
    while ini > 0 and _PALAVRA.match(texto[ini - 1]):
        ini -= 1
    fim = col
    while fim < len(texto) and _PALAVRA.match(texto[fim]):
        fim += 1
    return (ini, fim)


def _destacar(frags, ini: int, fim: int):
    """Aplica vídeo reverso às colunas [ini, fim) de uma linha de fragmentos.

    Os fragmentos vêm da conversão do ANSI e não respeitam as bordas da seleção,
    então cada um é cortado nos limites antes de receber o estilo.
    """
    saida = []
    col = 0
    for frag in frags:
        estilo, texto = frag[0], frag[1]
        if not texto:
            continue
        fim_frag = col + len(texto)
        cortes = sorted({col, fim_frag}
                        | {p for p in (ini, fim) if col < p < fim_frag})
        for a, b in zip(cortes, cortes[1:]):
            trecho = texto[a - col:b - col]
            dentro = ini <= a < fim
            saida.append(((estilo + " reverse").strip() if dentro else estilo, trecho))
        col = fim_frag
    return saida


class Selecao:
    """Trecho selecionado com o mouse, em coordenadas absolutas do transcript.

    Guardar a linha absoluta (e não a posição na tela) é o que faz a seleção
    continuar no mesmo texto quando a conversa rola ou cresce durante a geração.
    """

    def __init__(self) -> None:
        self.ancora: tuple[int, int] | None = None
        self.cursor: tuple[int, int] | None = None
        self.arrastando = False

    def limpar(self) -> None:
        self.ancora = self.cursor = None
        self.arrastando = False

    @property
    def vazia(self) -> bool:
        return (self.ancora is None or self.cursor is None
                or self.ancora == self.cursor)

    def ordenada(self) -> tuple[tuple[int, int], tuple[int, int]]:
        a, b = self.ancora, self.cursor
        return (a, b) if a <= b else (b, a)

    def intervalo(self, linha: int, tamanho: int) -> tuple[int, int] | None:
        """Colunas selecionadas nesta linha, ou None se ela está fora."""
        if self.vazia:
            return None
        (l1, c1), (l2, c2) = self.ordenada()
        if linha < l1 or linha > l2:
            return None
        ini = c1 if linha == l1 else 0
        fim = c2 if linha == l2 else tamanho
        return (min(ini, tamanho), min(fim, tamanho))

    def texto(self, linhas: list[str]) -> str:
        if self.vazia:
            return ""
        (l1, _), (l2, _) = self.ordenada()
        partes = []
        for i in range(l1, min(l2, len(linhas) - 1) + 1):
            puro = texto_puro(linhas[i])
            faixa = self.intervalo(i, len(puro))
            if faixa is None:
                continue
            partes.append(puro[faixa[0]:faixa[1]].rstrip())
        return "\n".join(partes)


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------

class _PrintBlock:
    """Uma chamada de `console.print` guardada para renderizar depois.

    Guardar os argumentos (e não o texto já renderizado) é o que permite
    reflowar tudo quando o terminal muda de largura.
    """

    __slots__ = ("args", "kwargs")

    def __init__(self, args: tuple, kwargs: dict) -> None:
        self.args = args
        self.kwargs = kwargs


class Transcript:
    """Blocos da conversa + viewport com rolagem própria."""

    def __init__(self) -> None:
        self._blocks: list[_PrintBlock] = []
        self._cache: list[list[str] | None] = []
        self._width = 80
        self._height = 24
        self._lines: list[str] | None = None      # achatado, invalidado a cada mudança
        self.scroll = 0                            # 1ª linha visível
        self.follow = True                         # colado no fim
        self._lock = threading.RLock()
        self.on_change: Callable[[], None] = lambda: None

    # -- escrita ---------------------------------------------------------
    def append(self, args: tuple, kwargs: dict) -> None:
        with self._lock:
            self._blocks.append(_PrintBlock(args, kwargs))
            self._cache.append(None)
            self._lines = None
        self.on_change()

    def replace_last(self, args: tuple, kwargs: dict) -> None:
        """Substitui o último bloco — é assim que a resposta cresce em streaming
        sem empilhar uma cópia por token."""
        with self._lock:
            if not self._blocks:
                self._blocks.append(_PrintBlock(args, kwargs))
                self._cache.append(None)
            else:
                self._blocks[-1] = _PrintBlock(args, kwargs)
                self._cache[-1] = None
            self._lines = None
        self.on_change()

    def drop_last(self) -> None:
        """Remove o último bloco (usado para apagar o preview do streaming antes
        de gravar a renderização final)."""
        with self._lock:
            if self._blocks:
                self._blocks.pop()
                self._cache.pop()
                self._lines = None
        self.on_change()

    # -- renderização ----------------------------------------------------
    def _render_block(self, block: _PrintBlock) -> list[str]:
        buf = io.StringIO()
        console = Console(file=buf, width=self._width, force_terminal=True,
                          color_system="truecolor", highlight=False, soft_wrap=False)
        console.print(*block.args, **block.kwargs)
        linhas = buf.getvalue().split("\n")
        if linhas and linhas[-1] == "":
            linhas.pop()
        return linhas

    def lines(self) -> list[str]:
        with self._lock:
            if self._lines is None:
                for i, bloco in enumerate(self._blocks):
                    if self._cache[i] is None:
                        self._cache[i] = self._render_block(bloco)
                self._lines = [ln for grupo in self._cache for ln in grupo]
            return self._lines

    # -- viewport --------------------------------------------------------
    @property
    def altura(self) -> int:
        """Altura da área visível do transcript, em linhas."""
        return self._height

    def set_viewport(self, width: int, height: int) -> None:
        with self._lock:
            if width != self._width:
                self._width = width
                self._cache = [None] * len(self._blocks)
                self._lines = None
            self._height = height
        self._clamp()

    def _clamp(self) -> None:
        total = len(self.lines())
        maximo = max(0, total - self._height)
        if self.follow:
            self.scroll = maximo
        else:
            self.scroll = max(0, min(self.scroll, maximo))

    def visible(self) -> list[str]:
        self._clamp()
        return self.lines()[self.scroll:self.scroll + self._height]

    # -- rolagem ---------------------------------------------------------
    def scroll_by(self, delta: int) -> None:
        total = len(self.lines())
        maximo = max(0, total - self._height)
        novo = max(0, min(self.scroll + delta, maximo))
        self.scroll = novo
        # Voltar ao fim religa o auto-follow; sair dele pausa, para a resposta
        # que continua chegando não arrastar a leitura de volta para baixo.
        self.follow = novo >= maximo
        self.on_change()

    def scroll_page(self, paginas: int) -> None:
        self.scroll_by(paginas * max(1, self._height - 1))

    def to_top(self) -> None:
        self.scroll = 0
        self.follow = False
        self.on_change()

    def to_bottom(self) -> None:
        self.follow = True
        self._clamp()
        self.on_change()

    @property
    def atrasado(self) -> bool:
        """True se o usuário está lendo acima do fim (auto-follow pausado)."""
        return not self.follow


class TranscriptConsole(Console):
    """`rich.Console` que imprime no transcript em vez do stdout.

    É o que permite reaproveitar `core.ui` e `core.commands` sem tocar neles:
    para esse código, isto continua sendo um Console comum.
    """

    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript
        super().__init__(file=io.StringIO(), force_terminal=True,
                         color_system="truecolor", highlight=False)

    def print(self, *args, **kwargs) -> None:  # noqa: A003
        self._transcript.append(args, kwargs)

    # `input()` não faz sentido aqui: a entrada é a caixa do rodapé. Se algum
    # caminho chamar isso, é melhor estourar do que travar a thread esperando
    # um stdin que ninguém está lendo.
    def input(self, *args, **kwargs):  # noqa: A003
        raise RuntimeError("TranscriptConsole não lê entrada; use a caixa do rodapé.")


class LiveBlock:
    """Substituto do `rich.Live` para o modo tela cheia.

    Mesma interface que o laço de conversa já usa (`with ... as live` +
    `live.update(renderable)`), mas em vez de repintar uma região do terminal,
    reescreve o último bloco do transcript. Ao sair, o preview é descartado — o
    laço grava a versão final logo depois.
    """

    def __init__(self, transcript: Transcript) -> None:
        self._t = transcript
        self._ativo = False

    def __enter__(self) -> LiveBlock:
        return self

    def __exit__(self, *exc) -> None:
        if self._ativo:
            self._t.drop_last()
            self._ativo = False

    def update(self, renderable) -> None:
        if self._ativo:
            self._t.replace_last((renderable,), {})
        else:
            self._t.append((renderable,), {})
            self._ativo = True


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------

class FullscreenSession:
    """A tela: transcript rolável em cima, caixa de entrada fixa embaixo."""

    def __init__(self, status_fn: Callable[[], dict]) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        from core import prompt as prompt_mod

        self.transcript = Transcript()
        self.console = TranscriptConsole(self.transcript)
        self._fila: queue.Queue[str | None] = queue.Queue()
        self._encerrando = threading.Event()
        # Sinaliza ao laço que o usuário pediu para interromper a resposta.
        self.interromper = threading.Event()
        self._ocupado = False
        # Escuta do wake word: a caixa aceita texto (o laço não está gerando
        # nada), mas o Ctrl+C tem que interromper a escuta, não encerrar o app.
        self._escutando = False
        # Definido antes do editor: a barra de status já lê este estado.
        self.mouse = bool(config.TUI_MOUSE)
        # Seleção com o mouse + detecção de duplo/triplo clique.
        self.selecao = Selecao()
        self._ultimo_clique = 0.0
        self._ultimo_clique_pos: tuple[int, int] | None = None
        self._cliques = 0
        self.copiado_em = 0.0
        self.copiado_metodo = ""
        # Preenchido pelo laço; Ctrl+O alterna a exibição do raciocínio.
        self.on_toggle_thinking: Callable[[], None] = lambda: None

        transcript = self.transcript
        sessao = self

        class _TranscriptControl(FormattedTextControl):
            """Só as linhas visíveis viram fragmentos a cada quadro.

            `create_content` é onde o prompt_toolkit informa o tamanho real da
            região, então é o ponto certo para ajustar o viewport antes de
            fatiar as linhas.
            """

            def create_content(self, width: int, height: int | None = None, **kw):
                # O prompt_toolkit também chama isto com height=None, só para
                # perguntar a altura preferida do conteúdo. Nessa passada não há
                # viewport para ajustar — mexer nele com None estoura.
                if height is not None:
                    transcript.set_viewport(width, height)
                return super().create_content(width, height, **kw)

            def mouse_handler(self, mouse_event):
                return sessao._mouse(mouse_event)

        def _fragmentos():
            from prompt_toolkit.formatted_text import ANSI, to_formatted_text

            saida: list[tuple[str, str]] = []
            base = transcript.scroll
            for i, linha in enumerate(transcript.visible()):
                if i:
                    saida.append(("", "\n"))
                frags = to_formatted_text(ANSI(linha))
                faixa = self.selecao.intervalo(base + i, len(texto_puro(linha)))
                if faixa is not None and faixa[1] > faixa[0]:
                    frags = _destacar(frags, *faixa)
                saida.extend(frags)
            return saida

        corpo = Window(_TranscriptControl(_fragmentos), wrap_lines=False)

        def _status_com_dica() -> dict:
            estado = dict(status_fn())
            # A confirmação da cópia aparece por alguns segundos e some sozinha
            # (o refresh_interval da app garante o repaint).
            if self.copiado_em and time.monotonic() - self.copiado_em < 3:
                # O método aparece no aviso: se o terminal ignorar o OSC 52, a
                # menção é a única pista de por que nada foi para o clipboard.
                estado["hint"] = ("copiado via terminal (OSC 52)"
                                  if self.copiado_metodo == "osc52"
                                  else "copiado para a área de transferência")
            elif self.transcript.atrasado:
                # Estado mais importante que qualquer dica: sem isto, a resposta
                # chegando fora da vista parece a interface travada.
                estado["hint"] = "rolagem pausada · Ctrl+End volta ao fim"
            elif not self.mouse:
                estado["hint"] = "mouse solto · F2 recaptura"
            else:
                estado["hint"] = ("arraste seleciona · 2x palavra · 3x linha · "
                                  "PgUp/PgDn rola")
            return estado

        editor = prompt_mod.build_editor(_status_com_dica, on_submit=self._submit)
        self._buffer = editor["buffer"]

        kb = editor["keys"]

        @kb.add("pageup")
        def _pgup(event) -> None:
            self.transcript.scroll_page(-1)

        @kb.add("pagedown")
        def _pgdn(event) -> None:
            self.transcript.scroll_page(1)

        @kb.add("c-home")
        def _topo(event) -> None:
            self.transcript.to_top()

        @kb.add("c-end")
        def _fim(event) -> None:
            self.transcript.to_bottom()

        @kb.add("c-c")
        def _interromper(event) -> None:
            # Gerando, Ctrl+C corta a resposta; escutando, encerra a escuta;
            # ocioso, encerra o Oráculo.
            if self._ocupado or self._escutando:
                self.interromper.set()
            else:
                self._encerrar()

        @kb.add("c-d")
        def _sair(event) -> None:
            if not event.app.current_buffer.text:
                self._encerrar()

        @kb.add("c-o")
        def _thinking(event) -> None:
            # No modo inline isto é feito por core.keyboard em modo raw; aqui o
            # prompt_toolkit já é dono do teclado, então vira só mais um atalho.
            self.on_toggle_thinking()

        @kb.add("f2")
        def _mouse(event) -> None:
            self.toggle_mouse()

        @kb.add("escape")
        def _limpar_selecao(event) -> None:
            # Sem `eager`: Alt+Enter chega como ("escape", "enter"), e um Esc
            # ansioso engoliria o prefixo e mataria a quebra de linha. O preço é
            # o Esc sozinho só agir depois do tempo de desambiguação.
            self.limpar_selecao()

        # `mouse_support` aceita um filtro, então a captura pode ser alternada em
        # tempo real: solta o mouse para selecionar/copiar e devolve depois, sem
        # reiniciar. Enquanto capturado, o terminal não vê o botão esquerdo e a
        # seleção nativa não funciona — é a mesma troca que o Claude Code faz.
        class _CapturaArrasto(HSplit):
            """Com o botão pressionado, a tela inteira responde ao arrasto.

            O prompt_toolkit entrega o evento ao controle sob o ponteiro. Sem
            isto, arrastar para fora do transcript (para a caixa de entrada, ou
            para além da borda) manda os eventos para outro controle: o arrasto
            congela no meio e o soltar nunca chega, deixando a seleção presa.
            """

            def write_to_screen(self, screen, mouse_handlers, write_position,
                                parent_style, erase_bg, z_index) -> None:
                super().write_to_screen(screen, mouse_handlers, write_position,
                                        parent_style, erase_bg, z_index)
                if sessao.selecao.arrastando:
                    mouse_handlers.set_mouse_handler_for_range(
                        write_position.xpos,
                        write_position.xpos + write_position.width,
                        write_position.ypos,
                        write_position.ypos + write_position.height,
                        sessao._mouse,
                    )

        self.app = Application(
            layout=Layout(_CapturaArrasto([corpo, editor["container"]]),
                          focused_element=editor["entrada"]),
            key_bindings=kb,
            style=editor["estilo"],
            full_screen=True,        # entra na tela alternativa
            mouse_support=Condition(lambda: self.mouse),
            refresh_interval=0.2,    # mantém o relógio do spinner andando
        )

        # Repaint pedido pela thread do laço: `invalidate` é seguro entre threads.
        self.transcript.on_change = self._invalidate

    # -- ponte com o laço ------------------------------------------------
    def _invalidate(self) -> None:
        try:
            self.app.invalidate()
        except Exception:  # noqa: BLE001 — app ainda não rodando/já encerrada
            pass

    def _submit(self, texto: str) -> None:
        if self._ocupado:
            return                    # ignora envio enquanto responde
        self._buffer.reset()
        # Enviar sempre volta ao fim. Se o usuário tinha rolado para cima (ou
        # selecionado um trecho, que também pausa o acompanhamento), a resposta
        # chegaria fora da área visível e a interface pareceria travada.
        self.selecao.limpar()
        self.transcript.to_bottom()
        self._fila.put(texto)

    def _encerrar(self) -> None:
        self._encerrando.set()
        self._fila.put(None)
        with_exit = getattr(self.app, "exit", None)
        if with_exit:
            try:
                self.app.exit()
            except Exception:  # noqa: BLE001
                pass

    def ask(self) -> str:
        """Bloqueia a thread do laço até a próxima mensagem.

        Levanta EOFError quando a sessão foi encerrada, que é o que o laço já
        trata como "acabou".
        """
        self._ocupado = False
        self._invalidate()
        item = self._fila.get()
        if item is None or self._encerrando.is_set():
            raise EOFError
        self._ocupado = True
        self.interromper.clear()
        self._invalidate()
        return item

    def modo_escuta(self):
        """Contexto da escuta pelo wake word.

        Solta o `_ocupado` para o `_submit` voltar a aceitar Enter — sem isso o
        texto digitado durante a escuta é engolido em silêncio e fica preso na
        caixa. Mas mantém o Ctrl+C ligado ao `interromper`, senão ele encerraria
        o Oráculo inteiro em vez de só sair da escuta.
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            anterior, self._ocupado = self._ocupado, False
            self._escutando = True
            self.interromper.clear()
            self._invalidate()
            try:
                yield
            finally:
                self._escutando = False
                self._ocupado = anterior
                self._invalidate()

        return _ctx()

    def ask_nowait(self) -> str | None:
        """Pega uma mensagem já digitada, ou None se não houver. Nunca bloqueia.

        É o que permite continuar digitando enquanto o wake word escuta. A versão
        bloqueante não serviria: uma thread parada no `_fila.get()` enquanto o
        microfone decide sozinho ficaria pendurada e engoliria a mensagem
        *seguinte* — o mesmo risco que manteve o `wait_stop` fora do VAD.
        """
        import queue as _queue

        try:
            item = self._fila.get_nowait()
        except _queue.Empty:
            return None
        if item is None or self._encerrando.is_set():
            raise EOFError
        return item

    # -- seleção com o mouse ---------------------------------------------
    def _mouse(self, ev):
        """Roda, arrasto de seleção, duplo e triplo clique.

        Devolver `NotImplemented` devolve o evento ao prompt_toolkit; devolver
        `None` significa "tratado aqui".
        """
        from prompt_toolkit.mouse_events import MouseButton, MouseEventType

        if ev.event_type == MouseEventType.SCROLL_UP:
            self.transcript.scroll_by(-config.TUI_SCROLL_LINES)
            return None
        if ev.event_type == MouseEventType.SCROLL_DOWN:
            self.transcript.scroll_by(config.TUI_SCROLL_LINES)
            return None

        # Durante o arrasto o handler vale para a tela inteira, então y pode cair
        # fora do transcript (na caixa de entrada, por exemplo): prende na área
        # visível para o ponto continuar fazendo sentido.
        altura = max(1, self.transcript.altura)
        y = max(0, min(ev.position.y, altura - 1))

        # Arrastar contra a borda rola: sem isto não dá para selecionar um trecho
        # maior que a tela.
        if self.selecao.arrastando and ev.event_type == MouseEventType.MOUSE_MOVE:
            if ev.position.y <= 0:
                self.transcript.scroll_by(-1)
            elif ev.position.y >= altura - 1:
                self.transcript.scroll_by(1)

        linhas = self.transcript.lines()
        linha = self.transcript.scroll + y
        if linha >= len(linhas):
            linha = max(0, len(linhas) - 1)
        puro = texto_puro(linhas[linha]) if linhas else ""
        col = min(ev.position.x, len(puro))

        if ev.event_type == MouseEventType.MOUSE_DOWN:
            agora = time.monotonic()
            perto = (self._ultimo_clique_pos is not None
                     and abs(self._ultimo_clique_pos[0] - linha) == 0
                     and abs(self._ultimo_clique_pos[1] - col) <= 1)
            if perto and agora - self._ultimo_clique < 0.5:
                self._cliques += 1
            else:
                self._cliques = 1
            self._ultimo_clique = agora
            self._ultimo_clique_pos = (linha, col)

            # Rolar durante o arrasto atrapalharia a leitura; pausa o
            # acompanhamento enquanto o usuário está selecionando.
            self.transcript.follow = False

            if self._cliques >= 3:                      # linha inteira
                self.selecao.ancora = (linha, 0)
                self.selecao.cursor = (linha, len(puro))
                self._finalizar_selecao()
            elif self._cliques == 2:                    # palavra
                ini, fim = _limites_palavra(puro, col)
                self.selecao.ancora = (linha, ini)
                self.selecao.cursor = (linha, fim)
                self._finalizar_selecao()
            else:
                self.selecao.ancora = (linha, col)
                self.selecao.cursor = (linha, col)
                self.selecao.arrastando = True
            self._invalidate()
            return None

        if ev.event_type == MouseEventType.MOUSE_MOVE:
            if self.selecao.arrastando:
                if ev.button == MouseButton.LEFT:
                    self.selecao.cursor = (linha, col)
                    self._invalidate()
                    return None
                # Movimento sem botão enquanto arrastávamos: o soltar aconteceu
                # fora da janela e o terminal não o reportou. Sem esta
                # recuperação a seleção fica presa em modo de arrasto para
                # sempre — e todo clique seguinte parece travado.
                self.selecao.arrastando = False
                self._finalizar_selecao()
                self._invalidate()
                return None
            return NotImplemented

        if ev.event_type == MouseEventType.MOUSE_UP:
            if self.selecao.arrastando:
                self.selecao.cursor = (linha, col)
                self.selecao.arrastando = False
                self._finalizar_selecao()
                self._invalidate()
                return None
            return NotImplemented

        return NotImplemented

    def _finalizar_selecao(self) -> None:
        """Copia o trecho selecionado, se houver."""
        texto = self.selecao.texto(self.transcript.lines())
        if not texto.strip():
            self.selecao.limpar()
            return
        metodo = copiar(texto)
        self.copiado_em = time.monotonic() if metodo else 0.0
        self.copiado_metodo = metodo

    def limpar_selecao(self) -> None:
        self.selecao.limpar()
        self._invalidate()

    def toggle_mouse(self) -> None:
        """Solta ou recaptura o mouse, para poder selecionar e copiar texto.

        Com a captura ligada os cliques vão para a aplicação e o terminal nunca
        vê o arrasto, então não há seleção nativa. Soltando, a seleção volta ao
        normal e a roda deixa de rolar o transcript (PgUp/PgDn continuam).
        """
        from core import ui

        self.mouse = not self.mouse
        if self.mouse:
            ui.notice(self.console,
                      "Mouse capturado — a roda rola a conversa. F2 solta para copiar.")
        else:
            ui.notice(self.console,
                      "Mouse solto — selecione e copie normalmente. "
                      "Role com PgUp/PgDn; F2 recaptura.")
        self._invalidate()

    def wait_enter(self) -> None:
        """Espera um Enter na caixa, sem consumir o texto como mensagem.

        É o sinal de parada do push-to-talk: no modo tela cheia o `input()` de
        `audio.record_ptt` brigaria com o prompt_toolkit pelo stdin.
        """
        anterior, self._ocupado = self._ocupado, False
        self._invalidate()
        try:
            item = self._fila.get()
            if item is None or self._encerrando.is_set():
                raise EOFError
        finally:
            self._ocupado = anterior
            self._invalidate()

    def live(self) -> LiveBlock:
        return LiveBlock(self.transcript)


def disponivel() -> bool:
    """True se dá para rodar em tela cheia (opção ligada, TTY e lib presente)."""
    import sys

    if config.TUI_MODE != "fullscreen" or not sys.stdin.isatty():
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return True


def run(loop: Callable[[FullscreenSession], None],
        status_fn: Callable[[], dict]) -> None:
    """Roda a sessão em tela cheia.

    `loop` é o laço de conversa; recebe a sessão e roda numa thread própria,
    porque a thread principal fica com a Application desenhando a tela.
    """
    sessao = FullscreenSession(status_fn)

    def _worker() -> None:
        try:
            loop(sessao)
        finally:
            sessao._encerrar()

    t = threading.Thread(target=_worker, daemon=True, name="oraculo-loop")
    t.start()
    try:
        sessao.app.run()
    finally:
        sessao._encerrando.set()
        sessao._fila.put(None)
        t.join(timeout=2.0)
