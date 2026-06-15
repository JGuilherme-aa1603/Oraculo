"""STT — transcrição de áudio para texto com faster-whisper (CTranslate2 + CUDA).

O modelo é carregado preguiçosamente na primeira transcrição e cacheado em memória.
Na primeira execução, o faster-whisper baixa o modelo para ~/.cache/huggingface.
"""

import config

_model = None


def get_model():
    """Carrega (uma vez) o modelo Whisper configurado."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper não instalado. Rode:\n"
                "  .venv/bin/python -m pip install faster-whisper"
            ) from exc

        try:
            _model = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
        except Exception as exc:  # noqa: BLE001
            # Tipicamente falta de libs CUDA (ex.: libcublas.so.12 / cuDNN).
            # Cai para CPU em vez de derrubar o modo voz.
            if config.WHISPER_DEVICE != "cpu":
                print(f"[stt] device='{config.WHISPER_DEVICE}' indisponível "
                      f"({exc}). Usando CPU.")
                _model = WhisperModel(config.WHISPER_MODEL,
                                      device="cpu", compute_type="int8")
            else:
                raise
    return _model


def transcribe(audio_path: str) -> str:
    """Transcreve um arquivo de áudio (pt-BR) para texto.

    Usa filtro VAD para descartar trechos sem fala — essencial na gravação fixa,
    onde boa parte dos 5s costuma ser silêncio e faz o modelo alucinar palavras.
    """
    segments, _ = get_model().transcribe(
        audio_path,
        language="pt",
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        temperature=0.0,
    )
    return " ".join(seg.text for seg in segments).strip()
