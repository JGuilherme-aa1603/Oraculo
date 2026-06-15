"""Fala em streaming: sintetiza e toca frases conforme o texto é gerado.

Em vez de esperar a resposta inteira para começar a falar, o texto é alimentado
em pedaços (tokens) durante a geração do LLM. Assim que uma frase completa surge,
ela é sintetizada e tocada — a fala começa quase junto com a escrita.

Dois threads em pipeline:
  - síntese: lê frases, gera o WAV (Kokoro/Piper) — etapa cara.
  - reprodução: toca os WAVs em ordem.
Uma fila de tamanho 1 entre eles mantém a síntese no máximo uma frase à frente,
escondendo sua latência sem deixar dois WAVs disputarem o mesmo arquivo.
"""

import queue
import threading

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

    def __init__(self) -> None:
        self._buffer = ""        # tokens ainda sem frase completa
        self._pending = ""       # frases completas aguardando atingir o tamanho do bloco
        self._first = True
        self._sentences: queue.Queue = queue.Queue()
        self._wavs: queue.Queue = queue.Queue(maxsize=1)
        self._error: Exception | None = None

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
        """Despeja o resto, espera a fala terminar e devolve o erro, se houve."""
        leftover = f"{self._pending} {self._buffer.strip()}".strip()
        self._pending = self._buffer = ""
        if leftover:
            self._enqueue(leftover)
        self._sentences.put(None)  # sentinela de fim
        self._synth.join()
        self._play.join()
        return self._error

    def _enqueue(self, raw: str) -> None:
        spoken = textproc.for_speech(raw)
        if spoken:
            self._sentences.put(spoken)

    def _synth_loop(self) -> None:
        from core import tts

        n = 0
        while True:
            item = self._sentences.get()
            if item is None:
                self._wavs.put(None)
                return
            try:
                wav = tts.speak(item, output_path=f"/tmp/oraculo_tts_{n % 2}.wav")
                n += 1
                self._wavs.put(wav)
            except Exception as exc:  # noqa: BLE001
                self._error = self._error or exc
                self._wavs.put(None)
                return

    def _play_loop(self) -> None:
        from core import audio

        while True:
            wav = self._wavs.get()
            if wav is None:
                return
            if self._error is not None:
                continue  # drena a fila sem tocar, para a síntese não travar
            try:
                audio.play(wav)
            except Exception as exc:  # noqa: BLE001
                self._error = self._error or exc
