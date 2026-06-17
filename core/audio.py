"""Captura de microfone e reprodução de áudio.

Depende de `sounddevice` (PortAudio) e `soundfile` (libsndfile), importados de forma
preguiçosa para que o modo texto funcione mesmo sem essas dependências instaladas.
"""

import config


def record(duration: float = config.RECORD_DURATION,
           samplerate: int = config.RECORD_SAMPLERATE,
           path: str = "/tmp/oraculo_in.wav") -> str:
    """Grava do microfone por `duration` segundos (modo de gravação fixa).

    Levanta RuntimeError com mensagem amigável se as dependências de áudio
    não estiverem disponíveis.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Áudio indisponível. Instale as dependências de voz:\n"
            "  sudo pacman -S portaudio libsndfile\n"
            "  .venv/bin/python -m pip install sounddevice soundfile"
        ) from exc

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(path, audio, samplerate)
    return path


def record_ptt(samplerate: int = config.RECORD_SAMPLERATE,
               path: str = "/tmp/oraculo_in.wav") -> str:
    """Push-to-talk: grava até o usuário pressionar Enter (sem duração fixa).

    O chamador deve avisar o usuário antes ("Enter para parar"). Bloqueia em
    input() enquanto captura o áudio do microfone em background.
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Áudio indisponível. Instale as dependências de voz:\n"
            "  sudo pacman -S portaudio libsndfile\n"
            "  .venv/bin/python -m pip install sounddevice soundfile"
        ) from exc

    frames: list = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
            input()  # bloqueia até Enter → encerra a gravação
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falha ao acessar o microfone: {exc}") from exc

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((1, 1), dtype="float32")
    audio = _normalize(audio)
    sf.write(path, audio, samplerate)
    return path


def _normalize(audio, target_peak: float = 0.95):
    """Normaliza o pico do áudio para um nível alto e constante.

    Microfone com ganho baixo grava em volume fraco, e ASR reconhece pior áudio
    fraco. Reescala para o pico chegar a `target_peak`, sem alterar nada se a
    gravação for praticamente silêncio (evita amplificar só ruído)."""
    import numpy as np

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-3:           # silêncio/ruído de fundo — não amplifica
        return audio
    return audio * (target_peak / peak)


def play(path: str) -> None:
    """Reproduz um arquivo WAV pelos alto-falantes."""
    try:
        import sounddevice as sd
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Áudio indisponível. Instale portaudio/libsndfile e sounddevice/soundfile."
        ) from exc

    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()


def play_array(samples, samplerate: int) -> None:
    """Reproduz PCM já em memória (float32 do Kokoro ou int16 do Piper)."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Áudio indisponível. Instale portaudio e sounddevice."
        ) from exc

    sd.play(samples, samplerate)
    sd.wait()
