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


@contextlib.contextmanager
def watch_key(target: bytes, on_press: Callable[[], None]) -> Iterator[None]:
    """Enquanto o bloco roda, observa o stdin; ao receber `target`, chama
    `on_press()` (uma única vez) e para de observar.

    `target` é o byte da tecla (ex.: `ESC`). Sequências de escape de teclas
    especiais (setas, F1...) começam com ESC mas vêm seguidas de mais bytes —
    elas são drenadas e ignoradas, então só um ESC "puro" dispara o callback.
    """
    try:
        import select
        import termios
        import tty
    except ImportError:
        yield                      # plataforma sem termios → sem barge-in
        return

    if not sys.stdin.isatty():
        yield                      # entrada não é um terminal → sem barge-in
        return

    fd = sys.stdin.fileno()
    try:
        old_attr = termios.tcgetattr(fd)
    except (termios.error, ValueError):
        yield
        return

    stop = threading.Event()

    def _loop() -> None:
        try:
            tty.setraw(fd, termios.TCSANOW)
            while not stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                ch = os.read(fd, 1)
                if ch != target:
                    continue
                if target == ESC:
                    # ESC puro vs. início de sequência (seta etc.): se houver
                    # mais bytes imediatamente, é uma sequência — drena e ignora.
                    more, _, _ = select.select([fd], [], [], 0.02)
                    if more:
                        os.read(fd, 16)
                        continue
                on_press()
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
