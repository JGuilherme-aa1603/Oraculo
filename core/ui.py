"""Desenho da área de conversa — a "calha" do transcript.

Todo turno é escrito na mesma estrutura vertical:

    > pergunta do usuário          (eco, recuado 2)

    ●  Oráculo                     (cabeçalho com glifo na margem)
       corpo da resposta...        (recuado UI_GUTTER, alinhado sob o nome)
       1.9s · 142 tok · 38 tok/s   (rodapé opcional, discreto)

Por que uma calha em vez de painéis por mensagem: a moldura custa 4 colunas de
largura por mensagem e vira ruído em respostas longas com código. O recuo dá a
mesma hierarquia de graça e o texto continua selecionável/copiável limpo.

A largura é limitada a UI_MAX_WIDTH — num terminal de 200 colunas a linha de
leitura ficaria longa demais para o olho acompanhar.

Toda cor vem de `config.UI_COLOR_*`, em hex. Nome de cor do rich ("cyan",
"grey42") resolve para a paleta do emulador, então o mesmo desenho sai diferente
em cada terminal — e é o hex, além disso, que o prompt_toolkit consegue comer
direto na barra de status (core/prompt.py), mantendo as duas metades da tela na
mesma paleta.
"""

import time

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.constrain import Constrain
from rich.padding import Padding
from rich.rule import Rule
from rich.segment import Segment
from rich.spinner import SPINNERS
from rich.style import Style
from rich.styled import Styled
from rich.text import Text
from rich.theme import Theme

import config

# Paleta do Markdown. A resposta do modelo é renderizada pelo rich, que traz o
# próprio tema — números de lista amarelos, código em ciano do terminal — e ele
# ignora solenemente a cor base que passamos no `body_view`. Sem sobrescrever
# aqui, toda resposta com lista ou código sai fora da paleta.
#
# Precisa ser aplicado em TODO Console que renderiza transcript: o modo tela
# cheia re-renderiza cada bloco num Console novo a cada mudança de largura
# (core/tui.py), e um Console sem o tema desfaz isso na primeira reflow.
THEME = Theme({
    "markdown.item.number": config.UI_COLOR_PROMPT,
    "markdown.item.bullet": config.UI_COLOR_PROMPT,
    "markdown.code": config.UI_COLOR_ACCENT,
    "markdown.code_block": config.UI_COLOR_SOFT,
    "markdown.link": config.UI_COLOR_ACCENT,
    "markdown.link_url": config.UI_COLOR_FAINT,
    "markdown.block_quote": config.UI_COLOR_FAINT,
    "markdown.hr": config.UI_COLOR_BORDER,
    "markdown.h1": f"bold {config.UI_COLOR_ACCENT}",
    "markdown.h2": f"bold {config.UI_COLOR_ACCENT}",
    "markdown.h3": f"bold {config.UI_COLOR_ACCENT}",
    "markdown.h4": f"bold {config.UI_COLOR_SOFT}",
    "markdown.h5": config.UI_COLOR_SOFT,
    "markdown.h6": config.UI_COLOR_SOFT,
})


# Coluna em que o glifo do turno (`●`) é desenhado — ver `assistant_header`. A
# linha de espera se ancora nela para o olho cair debaixo do ponto.
_COL_GLIFO = 2


def make_console(**kwargs) -> Console:
    """Console do Oráculo: o rich comum, já com o tema da paleta.

    `highlight=False` junto: o realce automático do rich pinta números e
    caminhos com cores próprias, o que reintroduz a paleta do rich por outra
    porta — e no transcript ele já estava desligado, então isto também alinha
    os dois modos de desenho.
    """
    kwargs.setdefault("highlight", False)
    return Console(theme=THEME, **kwargs)


# A íris registrada como spinner do rich, para poder ser usada em qualquer
# `Spinner("oraculo")`/`console.status`. Os quadros vêm do config — é lá que a
# identidade visual mora.
IRIS = "oraculo"
SPINNERS[IRIS] = {
    "interval": config.UI_IRIS_INTERVAL_MS,
    "frames": [f"({f})" for f in config.UI_IRIS_FRAMES],
}


class Waiting:
    """Linha de espera: a íris girando, um rótulo, o relógio e o que fazer.

        ●  Oráculo
       (✦) Pensando...  · 3s · Ctrl+C corta a resposta

    O quadro da íris e o contador saem do relógio a cada renderização — nada
    aqui guarda estado de animação, e ninguém precisa chamar `update()` só para
    mexer o ponteiro. O que falta é garantir que a renderização *aconteça*: o
    `rich.Live` do modo inline repinta sozinho, mas o transcript da tela cheia
    guarda o resultado de cada bloco em cache e só o refaz quando o bloco muda.
    Daí o `ANIMADO`: é a marca que `core/tui.py` procura para invalidar este
    bloco a cada quadro. Sem ela a íris congela no primeiro desenho — e congela
    *silenciosamente*, que foi como o bug passou.
    """

    #: Marca lida por core/tui.py — este renderable muda sozinho com o tempo.
    ANIMADO = True

    def __init__(self, label: str, hint: str = "", *,
                 since: float | None = None, style: str | None = None) -> None:
        self.label = label
        self.hint = hint
        self.since = time.monotonic() if since is None else since
        self.style = style or config.UI_COLOR_SOFT

    def __rich_console__(self, console: Console,
                         options: ConsoleOptions) -> RenderResult:
        decorrido = time.monotonic() - self.since
        quadros = config.UI_IRIS_FRAMES
        i = int(decorrido * 1000 / config.UI_IRIS_INTERVAL_MS) % len(quadros)
        # O recuo é próprio, não vem de um `indent()` por fora: embrulhar isto
        # num Padding esconderia o `ANIMADO` de quem procura a marca, e a linha
        # voltaria a congelar na tela cheia.
        #
        # A íris cai exatamente na coluna do `●` do cabeçalho, e o rótulo na
        # coluna do corpo — a linha de espera OCUPA o lugar da resposta que
        # ainda não chegou, então tem que se encaixar na mesma grade:
        #
        #     ●  Oráculo
        #    (◈) Pensando...
        #
        # Recuar pela calha inteira jogava o olho quatro colunas à direita do
        # ponto, e a coluna inteira do turno parecia torta.
        linha = Text(" " * max(0, _COL_GLIFO - 1))
        linha.append(f"({quadros[i]})", style=config.UI_COLOR_ACCENT)
        linha.append(" " * max(1, config.UI_GUTTER - _COL_GLIFO - 2))
        linha.append(self.label, style=self.style)
        linha.append(f"  · {int(decorrido)}s", style=config.UI_COLOR_FAINT)
        if self.hint:
            linha.append(f" · {self.hint}", style=config.UI_COLOR_FAINT)
        yield linha


class LeftRule:
    """Bloco com uma barra vertical na margem esquerda.

    É o desenho do raciocínio: em vez de uma moldura fechada (que custa 4 colunas
    e transforma um aparte discreto no elemento mais pesado da tela), só uma
    barra marca o bloco como subordinado ao turno.

    Renderiza o conteúdo na largura já descontada da barra e prefixa linha a
    linha — `Padding` sozinho não desenharia a barra nas linhas de continuação
    de um parágrafo quebrado.
    """

    def __init__(self, renderable: RenderableType,
                 style: str = config.UI_COLOR_BORDER) -> None:
        self.renderable = renderable
        self.style = style

    def __rich_console__(self, console: Console,
                         options: ConsoleOptions) -> RenderResult:
        barra = Segment(f"{config.UI_GLYPH_RULE} ", Style.parse(self.style))
        largura = max(1, options.max_width - 2)
        linhas = console.render_lines(self.renderable,
                                      options.update_width(largura), pad=False)
        for linha in linhas:
            yield barra
            yield from linha
            yield Segment("\n")


def clear_screen(console: Console) -> None:
    """Limpa a tela ao abrir, para a sessão começar num quadro limpo.

    Deliberadamente NÃO usa o buffer alternativo do terminal (o do vim/htop).
    Desenhando no buffer normal, a rolagem nativa do terminal continua valendo
    de graça — roda do mouse e Shift+PgUp percorrem a conversa — e o transcript
    permanece no terminal depois que o Oráculo encerra. É como o Claude Code se
    comporta. A tela alternativa daria uma saída sem rastro, mas custaria a
    rolagem: dentro dela o terminal não tem scrollback, e recuperá-la exigiria
    reimplementar um viewport próprio, como o vim faz.

    Só limpa a tela visível: o scrollback anterior do terminal é preservado, então
    nada do que você já tinha ali é destruído.
    """
    if config.CLEAR_ON_START and console.is_terminal:
        console.clear()


def indent(renderable: RenderableType) -> Padding:
    """Recua o renderable para a calha, respeitando o teto de leitura.

    Usado também no preview ao vivo do streaming: se o Live não usasse o mesmo
    recuo, o texto pularia de coluna no instante em que o preview é substituído
    pela renderização final.

    Com `UI_MAX_WIDTH = 0` não há teto e o texto ocupa o terminal inteiro.
    """
    if config.UI_MAX_WIDTH:
        largura = max(20, config.UI_MAX_WIDTH - config.UI_GUTTER)
        renderable = Constrain(renderable, largura)
    return Padding(renderable, (0, 0, 0, config.UI_GUTTER))


def user_echo(console: Console, text: str) -> None:
    """Ecoa a mensagem enviada. No modo texto o terminal já mostrou o que foi
    digitado, mas a caixa de entrada é apagada ao enviar — sem este eco o
    transcript perderia a pergunta."""
    line = Text("  ")
    line.append(f"{config.UI_GLYPH_USER} ", style=f"bold {config.UI_COLOR_PROMPT}")
    line.append(text, style=config.UI_COLOR_BRIGHT)
    console.print(line)
    console.print()


def assistant_header(console: Console) -> None:
    """Abre um turno do Oráculo: glifo na margem + nome."""
    head = Text(" " * _COL_GLIFO)
    head.append(config.UI_GLYPH_ASSISTANT, style=config.UI_COLOR_ACCENT)
    # O nome começa na calha, para o corpo da resposta alinhar debaixo dele.
    head.append(" " * max(1, config.UI_GUTTER - _COL_GLIFO - 1))
    head.append(config.ASSISTANT_NAME, style=f"bold {config.UI_COLOR_ACCENT}")
    console.print(head)


def body_view(renderable: RenderableType) -> Styled:
    """Corpo da resposta pronto para desenhar: recuo da calha + cor base.

    Existe separado de `body()` porque o preview ao vivo do streaming precisa do
    mesmo desenho sem passar pelo console — se o preview não trouxesse a cor
    base, o texto mudaria de tom no instante em que a renderização final o
    substitui, exatamente como mudaria de coluna sem o recuo comum.
    """
    return Styled(indent(renderable), config.UI_COLOR_BODY)


def body(console: Console, renderable: RenderableType) -> None:
    """Corpo da resposta, recuado para alinhar sob o nome."""
    console.print(body_view(renderable))


def turn_footer(console: Console, metrics: str | None) -> None:
    """Rodapé discreto com as métricas do turno. `None` ou vazio não imprime nada."""
    if not metrics:
        return
    console.print(Text(" " * config.UI_GUTTER + metrics, style=config.UI_COLOR_FAINT))


def sources(console: Console, fontes: list[str]) -> None:
    """Notas que entraram no contexto deste turno.

    Recuperar em silêncio é indistinguível de não recuperar nada — a mesma
    razão pela qual a faxina de sessões anuncia o que apagou e pela qual a
    verificação de voz avisa quando descarta uma fala. Sem esta linha, "o
    Oráculo respondeu errado" e "o Oráculo respondeu sem consultar as notas"
    parecem exatamente a mesma coisa na tela.
    """
    if not fontes:
        return
    line = Text(" " * config.UI_GUTTER)
    line.append("notas: ", style=config.UI_COLOR_FAINT)
    for i, fonte in enumerate(fontes):
        if i:
            line.append(" · ", style=config.UI_COLOR_FAINT)
        line.append(fonte, style=config.UI_COLOR_DIM)
    console.print(line)


def notice(console: Console, text: str, *, style: str | None = None) -> None:
    """Mensagem subordinada ao turno (gravando, transcrevendo, fala interrompida).
    Usa o glifo de continuação para não competir com o cabeçalho do turno.

    Vai dentro de um Padding para o aviso longo quebrar alinhado à calha; solto,
    a segunda linha voltaria para a coluna 0 e sairia da margem do transcript.
    """
    line = Text()
    # O glifo acompanha a cor do texto quando há uma: um "⎿" apagado na frente de
    # uma confirmação em ciano parte a linha em dois pesos sem motivo.
    line.append(f"{config.UI_GLYPH_NOTICE}  ", style=style or config.UI_COLOR_FAINT)
    line.append(text, style=style or config.UI_COLOR_DIM)
    console.print(Padding(line, (0, 0, 0, 2)))


def ok(console: Console, text: str) -> None:
    """Confirmação de algo que aconteceu (modo trocado, memória limpa)."""
    notice(console, text, style=config.UI_COLOR_ACCENT)


def warn(console: Console, text: str) -> None:
    """Aviso (voz indisponível, comando desconhecido) — mesma calha, cor de alerta."""
    notice(console, text, style=config.UI_COLOR_ALERT)


def error(console: Console, text: str) -> None:
    notice(console, text, style=f"bold {config.UI_COLOR_ALERT}")


def hint(console: Console, markup: str) -> None:
    """Dica subordinada que precisa destacar um comando dentro do texto.

    Existe além do `notice()` porque este aceita marcação do rich: em "Use
    /modelo <nome> para trocar" o que o olho procura é o comando, e ele precisa
    sair na cor de comando sem que a frase inteira mude de peso.
    """
    console.print(f"  [{config.UI_COLOR_FAINT}]{config.UI_GLYPH_NOTICE}  "
                  f"{markup}[/]")


def recording(console: Console, text: str, dica: str = "") -> None:
    """Indicador de microfone aberto.

    O ponto vermelho é o único lugar da interface onde a cor de alerta aparece
    sem haver erro nenhum: microfone aberto é exatamente o tipo de estado que
    não pode se confundir com o resto do transcript.

    Não há medidor de nível aqui de propósito — desenhar barras que não vêm da
    amplitude real seria enfeite mentindo sobre a captura, e é justamente onde
    um bug de áudio se esconderia.
    """
    line = Text("  ")
    line.append("●  ", style=config.UI_COLOR_ALERT)
    line.append(text, style=config.UI_COLOR_BRIGHT)
    if dica:
        line.append(f"   · {dica}", style=config.UI_COLOR_FAINT)
    console.print(line)


def heading(console: Console, text: str) -> None:
    """Título de um bloco de saída de comando (/ajuda, /modelo, /stt).

    Indenta 2 para casar com a calha do eco: blocos de lista não passam pelo
    `indent()` do corpo, então o alinhamento é feito aqui.
    """
    console.print(Text("  " + text, style=f"bold {config.UI_COLOR_ACCENT}"))


def divider(console: Console, text: str = "") -> None:
    """Régua fina com um rótulo, para marcar uma fronteira no transcript.

    Usada no `/retomar`, separando a conversa que voltou do disco do que vai ser
    dito agora. Sem uma fronteira visível, a conversa antiga e a nova viram um
    bloco só e não dá para saber onde uma acaba.
    """
    console.print(Rule(Text(text, style=config.UI_COLOR_DIM) if text else "",
                       style=config.UI_COLOR_BORDER, characters="─"))


def spacer(console: Console) -> None:
    """Respiro entre turnos. Um turno termina com uma linha em branco, sempre —
    sem isso o prompt seguinte cola no fim da resposta anterior."""
    console.print()
