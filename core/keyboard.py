"""Monitoramento de teclas no terminal (Linux), para interromper a fala.

Coloca o stdin em modo raw temporariamente e observa o teclado numa thread,
chamando um callback quando a tecla-alvo é pressionada. Restaura o terminal ao
sair, sempre. Sem TTY (pipe, redirecionamento, alguns IDEs) ou sem `termios`,
vira um no-op silencioso — o chamador continua funcionando sem barge-in.
"""

import contextlib
import os
import sys
import threading
from collections.abc import Callable, Iterator

ESC = b"\x1b"
CTRL_O = b"\x0f"


@contextlib.contextmanager
def watch_key(
    target: bytes,
    on_press: Callable[[], None],
    *,
    once: bool = True,
    preserve_signals: bool = False,
) -> Iterator[None]:
    """Enquanto o bloco roda, observa o stdin e chama `on_press()` a cada vez que
    a tecla `target` é pressionada.

    `once=True` para após o primeiro disparo (ex.: Esc interrompe a fala);
    `once=False` continua observando (ex.: Ctrl+O alterna a exibição do raciocínio).

    `preserve_signals=True` usa modo cbreak, que mantém os sinais do terminal —
    Ctrl+C continua gerando KeyboardInterrupt (essencial durante a geração).
    `False` usa modo raw, que engole tudo (Ctrl+C não encerra durante a fala).

    Sequências de escape de teclas especiais (setas, F1...) começam com ESC mas
    vêm seguidas de mais bytes; quando `target` é ESC, elas são drenadas e
    ignoradas, então só um ESC "puro" dispara o callback.
    """
    try:
        import select
        import termios
        import tty
    except ImportError:
        yield                      # plataforma sem termios → no-op
        return

    if not sys.stdin.isatty():
        yield                      # entrada não é um terminal → no-op
        return

    fd = sys.stdin.fileno()
    try:
        old_attr = termios.tcgetattr(fd)
    except (termios.error, ValueError):
        yield
        return

    set_mode = tty.setcbreak if preserve_signals else tty.setraw
    stop = threading.Event()

    def _loop() -> None:
        try:
            set_mode(fd, termios.TCSANOW)
            while not stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                ch = os.read(fd, 1)
                if ch != target:
                    continue
                if target == ESC:
                    more, _, _ = select.select([fd], [], [], 0.02)
                    if more:
                        os.read(fd, 16)   # drena o resto da sequência
                        continue
                on_press()
                if once:
                    return
        except Exception:  # noqa: BLE001 — qualquer falha de terminal: só não observa
            pass

    watcher = threading.Thread(target=_loop, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join(timeout=0.3)
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
