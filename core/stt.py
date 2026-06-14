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

        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(audio_path: str) -> str:
    """Transcreve um arquivo de áudio (pt-BR) para texto."""
    segments, _ = get_model().transcribe(audio_path, language="pt")
    return " ".join(seg.text for seg in segments).strip()
