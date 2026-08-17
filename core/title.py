"""Título da janela do terminal — o olho do Oráculo visto de fora.

Enquanto uma resposta é gerada, a única pista de que o Oráculo está trabalhando
mora dentro da janela. Levando o olho para o título, a barra de tarefas e o
alternador de janelas passam a mostrar o estado sem precisar trazer o terminal
para a frente:

    (✦) Oráculo              ocioso
    (✧) Oráculo · pensando   gerando uma resposta (a íris gira)
    (✦) Oráculo · ouvindo    microfone aberto

Três decisões que o desenho carrega:

- **Quem escreve é sempre a thread dona da saída.** No modo tela cheia o
  prompt_toolkit é dono do stdout, e emitir a sequência da thread do laço
  interleava com o repaint — sai lixo na tela, não um título errado. Por isso
  `marcar()` (qualquer thread) só anota o estado, e `desenhar()` é que emite os
  bytes, chamado de onde é seguro: o gancho `before_render` no fullscreen, os
  limites do turno no inline.
- **O título anterior volta.** `abrir()` empurra o título atual na pilha do
  terminal (CSI 22 t) e `fechar()` restaura (CSI 23 t). Sem isso o terminal fica
  marcado "Oráculo" para sempre depois que ele encerra — inclusive na aba, que é
  onde mais incomoda.
- **Nada é escrito duas vezes.** `desenhar()` compara com o último título
  emitido e volta calado se nada mudou, então pode ser chamado a cada quadro.

Custo zero com `TITLE_ENABLED = False`: `abrir()` não faz nada e todo o resto
sai no primeiro `if`.
"""

from __future__ import annotations

import sys
import threading
import time

import config

# Pilha de títulos do terminal (XTerm; o kitty implementa em
# window.manipulate_title_stack). O `0` quer dizer ícone + janela.
_EMPILHA = "\x1b[22;0t"
_DESEMPILHA = "\x1b[23;0t"

_lock = threading.Lock()
_rotulo = ""            # estado pedido pelo laço; "" = ocioso
_escrito: str | None = None
_aberto = False


def disponivel() -> bool:
    """Só mexe no título se estiver ligado e houver um terminal de verdade."""
    if not config.TITLE_ENABLED:
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):  # stdout fechado ou substituído
        return False


def _emitir(seq: str) -> None:
    """Manda a sequência ao terminal pelo caminho de quem é dono da saída.

    Com a app do prompt_toolkit rodando, o stdout está sob controle dela e a
    escrita tem que passar pelo `output` — é o mesmo cuidado que o OSC 52 da
    cópia toma em core/tui.py. Fora dela, stdout direto.
    """
    try:
        from prompt_toolkit.application.current import get_app_or_none

        app = get_app_or_none()
    except Exception:  # noqa: BLE001 — sem prompt_toolkit instalado
        app = None

    if app is not None:
        try:
            app.output.write_raw(seq)
            app.output.flush()
            return
        except Exception:  # noqa: BLE001 — app encerrando; cai para o stdout
            pass
    try:
        sys.stdout.write(seq)
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 — título nunca pode derrubar a conversa
        pass


def _texto() -> str:
    """O título de agora, com o quadro da íris tirado do relógio.

    Mesma matemática de `ui.Waiting`: nada de estado de animação guardado, o
    quadro é função do tempo. Parado quando ocioso — um olho girando numa aba
    que não está fazendo nada seria ruído (e chamaria atenção à toa)."""
    with _lock:
        rotulo = _rotulo
    quadros = config.UI_IRIS_FRAMES
    if not rotulo:
        return f"({quadros[0]}) {config.ASSISTANT_NAME}"
    i = int(time.monotonic() * 1000 / config.UI_IRIS_INTERVAL_MS) % len(quadros)
    return f"({quadros[i]}) {config.ASSISTANT_NAME} · {rotulo}"


def marcar(rotulo: str = "") -> None:
    """Anota o que o Oráculo está fazendo. Seguro em qualquer thread."""
    global _rotulo
    with _lock:
        _rotulo = rotulo


def desenhar() -> None:
    """Emite o título, se mudou. Só chame da thread dona da saída."""
    global _escrito
    if not _aberto:
        return
    texto = _texto()
    if texto == _escrito:
        return
    _escrito = texto
    # BEL como terminador em vez de ST: é o que todo terminal aceita.
    _emitir(f"\x1b]2;{texto}\x07")


def abrir() -> None:
    """Guarda o título atual e assume a janela."""
    global _aberto
    if _aberto or not disponivel():
        return
    _emitir(_EMPILHA)
    _aberto = True
    desenhar()


def fechar() -> None:
    """Devolve ao terminal o título que ele tinha antes."""
    global _aberto, _escrito
    if not _aberto:
        return
    _aberto = False
    _escrito = None
    _emitir(_DESEMPILHA)
