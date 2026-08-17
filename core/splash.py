"""Splash screen do Oráculo (v2) — layout de duas colunas estilo Claude Code.

Identidade à esquerda, comandos + conversas recentes à direita, dentro de uma
moldura ciano. As conversas recentes vêm de core.history.load_recent().
"""

import getpass
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from core import ui

console = ui.make_console()

_RULE = "─" * 22
# A íris parada. Na splash ela não gira — a splash é impressa uma vez e vira um
# bloco do transcript, não uma região repintada. É o mesmo glifo que gira depois,
# durante a resposta (ui.iris), então o olho é reconhecível nos dois estados.
_IRIS = config.UI_IRIS_FRAMES[0]


def _user_name() -> str:
    if config.USER_NAME:
        return config.USER_NAME
    return getpass.getuser().capitalize()


def _cwd_display() -> str:
    cwd = Path.cwd()
    home = Path.home()
    try:
        return "~/" + str(cwd.relative_to(home))
    except ValueError:
        return str(cwd)


def _truncate(text: str, width: int = 46) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def _join(lines: list[Text]) -> Text:
    out = Text()
    for i, line in enumerate(lines):
        if i:
            out.append("\n")
        out.append_text(line)
    return out


# Parênteses de três linhas (peças de parêntese grande, U+239B-23A0). No design
# o olho tem 60px — quatro vezes a altura do texto — e num terminal não existe
# corpo de fonte, só células: a única forma de um símbolo ficar grande é ocupar
# mais de uma linha. As peças foram feitas para empilhar, então as três linhas
# leem como um par de parênteses alto em volta da íris.
#
# Todas as linhas precisam ter a MESMA largura em células, senão a coluna
# centralizada da splash desloca uma delas e a pupila sai do eixo.
_OLHO = ("⎛     ⎞",
         "⎜  {}  ⎟",
         "⎝     ⎠")


def _iris_block() -> list[Text]:
    """O olho: a íris em ciano dentro de parênteses roxos altos."""
    linhas: list[Text] = []
    for modelo in _OLHO:
        line = Text()
        antes, _, depois = modelo.partition("{}")
        line.append(antes, style=f"bold {config.UI_COLOR_PROMPT}")
        if depois:
            line.append(_IRIS, style=f"bold {config.UI_COLOR_ACCENT}")
            line.append(depois, style=f"bold {config.UI_COLOR_PROMPT}")
        linhas.append(line)
    return linhas


def _identity(model: str, memory_active: bool) -> list[Text]:
    memoria = "ativa" if memory_active else "inativa"
    modelo = Text(model, style=config.UI_COLOR_ACCENT)
    modelo.append(" · offline · local", style=config.UI_COLOR_DIM)
    return [
        Text(""),
        Text(f"Bem-vindo, {_user_name()}!", style=f"bold {config.UI_COLOR_BRIGHT}"),
        Text(""),
        *_iris_block(),
        Text(""),
        modelo,
        Text(f"memória {memoria} · {config.DEVICE_LABEL}",
             style=config.UI_COLOR_DIM),
        Text(_cwd_display(), style=config.UI_COLOR_FAINT),
        Text(""),
    ]


def _section(titulo: str) -> list[Text]:
    return [
        Text(titulo, style=f"bold {config.UI_COLOR_ACCENT}"),
        Text(_RULE, style=config.UI_COLOR_BORDER),
    ]


def _ago_line(ago: str, messages: int) -> Text:
    """Meta de uma conversa recente: só a quantidade recebe destaque.

    O "há" fica apagado junto com o resto e o valor ("16 horas") sai em ciano —
    é o que o olho procura ao varrer a lista.
    """
    meta = Text("  ")
    prefixo, _, valor = ago.partition(" ")
    if valor and prefixo == "há":
        meta.append("há ", style=config.UI_COLOR_FAINT)
        meta.append(valor, style=config.UI_COLOR_ACCENT)
    else:
        meta.append(ago, style=config.UI_COLOR_ACCENT)
    meta.append(f" · {messages} mensagens", style=config.UI_COLOR_FAINT)
    return meta


def _info(recent_sessions: list[dict]) -> list[Text]:
    lines: list[Text] = _section("Comandos disponíveis")

    def cmd_line(prefix: str, command: str, suffix: str) -> Text:
        t = Text(prefix, style=config.UI_COLOR_SOFT)
        t.append(command, style=config.UI_COLOR_PROMPT)
        t.append(suffix, style=config.UI_COLOR_SOFT)
        return t

    lines.append(cmd_line("Digite ", "/ajuda", " para ver todos os comandos"))
    lines.append(cmd_line("Use ", "/modelo", " para trocar o modelo ativo"))
    lines.append(cmd_line("Use ", "/sair", " para encerrar o Oráculo"))
    lines.append(Text(""))
    lines += _section("Conversas recentes")

    if not recent_sessions:
        lines.append(Text("nenhuma conversa ainda", style=config.UI_COLOR_FAINT))
        return lines

    for s in recent_sessions:
        lines.append(Text(_truncate(s.get("title", "")),
                          style=config.UI_COLOR_SOFT))
        lines.append(_ago_line(s.get("ago", ""), s.get("messages", 0)))
    return lines


def _footer(fullscreen: bool) -> Text:
    """Linha abaixo da moldura: o que é esta sessão e como se anda nela.

    O modo de desenho aparece porque é ele que decide como rolar — no fullscreen
    o scrollback do terminal não existe, e quem não souber disso conclui que a
    conversa foi perdida.
    """
    line = Text("  ")
    line.append(f"{config.UI_GLYPH_NOTICE}  ", style=config.UI_COLOR_FAINT)
    line.append("sessão nova", style=config.UI_COLOR_FAINT)
    if fullscreen:
        line.append(" · tela cheia", style=config.UI_COLOR_DIM)
        line.append(" · PgUp/PgDn rola", style=config.UI_COLOR_FAINT)
    else:
        line.append(" · rolagem do terminal", style=config.UI_COLOR_DIM)
    return line


def show_splash(model: str, recent_sessions: list[dict] | None = None,
                memory_active: bool = True, out: Console | None = None,
                fullscreen: bool = False) -> None:
    """Renderiza a splash de duas colunas.

    `out` permite desenhar em outro Console — no modo tela cheia é o
    TranscriptConsole, para a splash virar o primeiro bloco do transcript em vez
    de ir para o stdout, que ali está sob controle do prompt_toolkit.
    """
    recent_sessions = recent_sessions or []
    out = out or console

    left = _identity(model, memory_active)
    right = _info(recent_sessions)

    height = max(len(left), len(right))
    # Centraliza verticalmente a coluna da esquerda em relação à direita.
    pad_top = (height - len(left)) // 2
    left = [Text("")] * pad_top + left
    left += [Text("")] * (height - len(left))
    right += [Text("")] * (height - len(right))

    # O respiro vertical vem de uma linha em branco DENTRO das colunas, não do
    # padding do Panel: o padding fica fora do grid, então o divisor pararia
    # antes das bordas e pareceria flutuar no meio da moldura.
    left = [Text("")] + left + [Text("")]
    right = [Text("")] + right + [Text("")]
    divider = _join([Text("│", style=config.UI_COLOR_BORDER)
                     for _ in range(height + 2)])

    grid = Table.grid(expand=True, padding=(0, 2))
    # no_wrap + elipse: garante que nenhuma célula ganhe linhas extras por quebra
    # de texto, o que desalinharia o divisor vertical.
    grid.add_column(justify="center", ratio=5, no_wrap=True, overflow="ellipsis")
    grid.add_column(justify="center", width=1, no_wrap=True)
    grid.add_column(justify="left", ratio=6, no_wrap=True, overflow="ellipsis")
    grid.add_row(_join(left), divider, _join(right))

    title = Text.assemble(
        (f"{config.ASSISTANT_NAME} ", f"bold {config.UI_COLOR_ACCENT}"),
        (f"v{config.APP_VERSION}", config.UI_COLOR_DIM))

    out.print()
    # Canto vivo, não arredondado: a moldura é estrutura, não enfeite — é a
    # mesma decisão do sistema visual de origem, onde nenhum raio é maior que 0.
    out.print(Panel(grid, title=title, title_align="left",
                    border_style=config.UI_COLOR_BORDER, box=box.SQUARE,
                    padding=(0, 2)))
    out.print(_footer(fullscreen))
    out.print()
