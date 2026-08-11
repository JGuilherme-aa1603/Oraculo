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
"""

from rich.console import Console, RenderableType
from rich.constrain import Constrain
from rich.padding import Padding
from rich.text import Text

import config


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
    line.append(f"{config.UI_GLYPH_USER} ", style=f"bold {config.UI_COLOR_USER}")
    line.append(text, style="white")
    console.print(line)
    console.print()


def assistant_header(console: Console) -> None:
    """Abre um turno do Oráculo: glifo na margem + nome."""
    head = Text("  ")
    head.append(config.UI_GLYPH_ASSISTANT, style=config.UI_COLOR_ACCENT)
    head.append(f"  {config.ASSISTANT_NAME}", style=f"bold {config.UI_COLOR_ACCENT}")
    console.print(head)


def body(console: Console, renderable: RenderableType) -> None:
    """Corpo da resposta, recuado para alinhar sob o nome."""
    console.print(indent(renderable))


def turn_footer(console: Console, metrics: str | None) -> None:
    """Rodapé discreto com as métricas do turno. `None` ou vazio não imprime nada."""
    if not metrics:
        return
    console.print(Text(" " * config.UI_GUTTER + metrics, style=config.UI_COLOR_DIM))


def notice(console: Console, text: str, *, style: str | None = None) -> None:
    """Mensagem subordinada ao turno (gravando, transcrevendo, fala interrompida).
    Usa o glifo de continuação para não competir com o cabeçalho do turno.

    Vai dentro de um Padding para o aviso longo quebrar alinhado à calha; solto,
    a segunda linha voltaria para a coluna 0 e sairia da margem do transcript.
    """
    line = Text()
    line.append(f"{config.UI_GLYPH_NOTICE}  ", style=config.UI_COLOR_FAINT)
    line.append(text, style=style or config.UI_COLOR_DIM)
    console.print(Padding(line, (0, 0, 0, 2)))


def warn(console: Console, text: str) -> None:
    """Aviso (voz indisponível, comando desconhecido) — mesma calha, cor de alerta."""
    notice(console, text, style="yellow")


def error(console: Console, text: str) -> None:
    notice(console, text, style="bold red")


def spacer(console: Console) -> None:
    """Respiro entre turnos. Um turno termina com uma linha em branco, sempre —
    sem isso o prompt seguinte cola no fim da resposta anterior."""
    console.print()
