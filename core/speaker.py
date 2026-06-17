"""Fala em streaming: sintetiza e toca frases conforme o texto é gerado.

Em vez de esperar a resposta inteira para começar a falar, o texto é alimentado
em pedaços (tokens) durante a geração do LLM. Assim que uma frase completa surge,
ela é sintetizada e tocada — a fala começa quase junto com a escrita.

Dois threads em pipeline:
  - síntese: lê frases, gera o PCM em memória (Kokoro/Piper) — etapa cara.
  - reprodução: toca os blocos de PCM em ordem.
Uma fila de tamanho 1 entre eles mantém a síntese no máximo uma frase à frente,
escondendo sua latência. O áudio nunca toca o disco no caminho do streaming.

Barge-in: `stop()` interrompe a fala na hora. Os dois loops fazem get/put com
timeout curto e checam a flag `_stopped`, então saem em até ~`_POLL` segundos
sem nenhum malabarismo de sentinela — robusto mesmo se um lado já tiver saído.
"""

import contextlib
import queue
import threading
import time

from core import text as textproc


class StreamSpeaker:
    """Fala incremental: alimente com `feed`, finalize com `close`.

    Frases completas são agrupadas em blocos antes de ir pro TTS: blocos muito
    curtos saem com prosódia chapada (soam robóticos/"europeus"); blocos com uma
    ou duas frases inteiras soam naturais. O primeiro bloco é menor de propósito,
    para a fala começar logo; os seguintes são maiores, priorizando naturalidade.
    """

    _FIRST_MIN = 45     # caracteres do 1º bloco (curto → começa a falar cedo)
    _CHUNK_MIN = 160    # caracteres dos blocos seguintes (longo → prosódia natural)
    _POLL = 0.1         # timeout dos get/put: de quanto em quanto os loops checam _stopped

    def __init__(self) -> None:
        self._buffer = ""        # tokens ainda sem frase completa
        self._pending = ""       # frases completas aguardando atingir o tamanho do bloco
        self._first = True
        self._sentences: queue.Queue = queue.Queue()
        self._wavs: queue.Queue = queue.Queue(maxsize=1)
        self._error: Exception | None = None
        self._stopped: bool = False
        # Instante monotônico em que o 1º bloco começou a tocar (telemetria/TTFA).
        self.first_audio_at: float | None = None

        self._synth = threading.Thread(target=self._synth_loop, daemon=True)
        self._play = threading.Thread(target=self._play_loop, daemon=True)
        self._synth.start()
        self._play.start()

    def feed(self, delta: str) -> None:
        """Acumula texto e enfileira um bloco quando ele atinge tamanho natural."""
        self._buffer += delta
        done, self._buffer = textproc.split_complete(self._buffer)
        for sentence in done:
            self._pending = f"{self._pending} {sentence}".strip()

        limit = self._FIRST_MIN if self._first else self._CHUNK_MIN
        if len(self._pending) >= limit:
            self._enqueue(self._pending)
            self._pending = ""
            self._first = False

    def close(self) -> Exception | None:
        """Despeja o resto, espera a fala terminar e devolve o erro, se houve.

        Se `stop()` foi chamado (barge-in), não despeja mais nada e aguarda os
        threads só por um instante — o áudio já foi cortado e o prompt volta logo.
        """
        if not self._stopped:
            leftover = f"{self._pending} {self._buffer.strip()}".strip()
            self._pending = self._buffer = ""
            if leftover:
                self._enqueue(leftover)
        self._sentences.put(None)  # sentinela de fim
        wait = self._POLL * 5 if self._stopped else None
        self._synth.join(timeout=wait)
        self._play.join(timeout=wait)
        return self._error

    def stop(self) -> None:
        """Interrompe a fala imediatamente, descartando o que estiver pendente.

        Apenas levanta a flag e corta a reprodução em curso; os loops percebem
        em até `_POLL` segundos e saem sozinhos. Idempotente e seguro de chamar
        concorrente com `close()`.
        """
        self._stopped = True
        with contextlib.suppress(Exception):
            import sounddevice as sd
            sd.stop()

    def _enqueue(self, raw: str) -> None:
        spoken = textproc.for_speech(raw)
        if spoken:
            self._sentences.put(spoken)

    def _put(self, q: queue.Queue, item) -> bool:
        """Enfileira `item`, abortando se `stop()` foi chamado. False se abortou."""
        while not self._stopped:
            try:
                q.put(item, timeout=self._POLL)
                return True
            except queue.Full:
                continue
        return False

    def _synth_loop(self) -> None:
        from core import tts

        while not self._stopped:
            try:
                item = self._sentences.get(timeout=self._POLL)
            except queue.Empty:
                continue
            if item is None:                 # fim normal
                break
            try:
                samples, sr = tts.synth(item)
            except Exception as exc:  # noqa: BLE001
                self._error = self._error or exc
                break
            if not self._put(self._wavs, (samples, sr)):
                return                       # stop() durante a entrega
        self._put(self._wavs, None)          # sinaliza fim ao _play_loop

    def _play_loop(self) -> None:
        from core import audio

        while not self._stopped:
            try:
                item = self._wavs.get(timeout=self._POLL)
            except queue.Empty:
                continue
            if item is None:                 # fim (normal ou erro de síntese)
                return
            if self._error is not None:
                continue  # drena a fila sem tocar, para a síntese não travar
            try:
                samples, sr = item
                if self.first_audio_at is None:
                    self.first_audio_at = time.monotonic()
                audio.play_array(samples, sr)
            except Exception as exc:  # noqa: BLE001
                self._error = self._error or exc
