"""TTS — síntese de texto para áudio, 100% local.

Dois motores selecionáveis por config.TTS_ENGINE:
  - "kokoro": voz feminina pt-BR natural (kokoro-onnx, roda na CPU)
  - "piper":  binário externo Piper (voz masculina, leve)

A interface pública `speak(text)` gera um WAV em disco e devolve o caminho.
"""

import os
import shutil
import subprocess

import config

_kokoro = None


# ----------------------------- Kokoro ----------------------------------------
def _get_kokoro():
    """Carrega (uma vez) o modelo Kokoro."""
    global _kokoro
    if _kokoro is None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx não instalado. Rode:\n"
                "  .venv/bin/python -m pip install kokoro-onnx"
            ) from exc

        for path in (config.KOKORO_MODEL, config.KOKORO_VOICES):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Modelo Kokoro não encontrado: {path}\n"
                    "Baixe kokoro-v1.0.onnx e voices-v1.0.bin dos releases de\n"
                    "thewh1teagle/kokoro-onnx e coloque na raiz do projeto."
                )
        _kokoro = Kokoro(config.KOKORO_MODEL, config.KOKORO_VOICES)
    return _kokoro


def _speak_kokoro(text: str, output_path: str) -> str:
    import soundfile as sf

    samples, sample_rate = _get_kokoro().create(
        text,
        voice=config.KOKORO_VOICE,
        speed=config.KOKORO_SPEED,
        lang=config.KOKORO_LANG,
    )
    sf.write(output_path, samples, sample_rate)
    return output_path


# ----------------------------- Piper -----------------------------------------
def _piper_available() -> bool:
    return shutil.which(config.PIPER_BIN) is not None


def _speak_piper(text: str, output_path: str) -> str:
    if not _piper_available():
        raise RuntimeError(
            f"Piper não encontrado ('{config.PIPER_BIN}'). Instale com:\n"
            "  yay -S piper-tts   (ou baixe o binário standalone)"
        )
    if not os.path.exists(config.PIPER_VOICE):
        raise RuntimeError(
            f"Modelo de voz não encontrado: {config.PIPER_VOICE}\n"
            "Baixe o .onnx e o .onnx.json (ex.: pt_BR-faber-medium) e coloque na\n"
            "raiz do projeto, ou ajuste PIPER_VOICE no config.py."
        )

    # stdout/stderr capturados para silenciar os logs de info do Piper;
    # só são exibidos se o processo falhar.
    proc = subprocess.run(
        [config.PIPER_BIN, "--model", config.PIPER_VOICE,
         "--output_file", output_path],
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Piper falhou ao sintetizar: {err}")
    return output_path


# ----------------------------- Interface -------------------------------------
def available() -> bool:
    """True se o motor TTS configurado está pronto para uso."""
    if config.TTS_ENGINE == "kokoro":
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            return False
        return (os.path.exists(config.KOKORO_MODEL)
                and os.path.exists(config.KOKORO_VOICES))
    return _piper_available()


def speak(text: str, output_path: str = "/tmp/oraculo_tts.wav") -> str:
    """Gera um WAV a partir do texto usando o motor configurado."""
    if config.TTS_ENGINE == "kokoro":
        return _speak_kokoro(text, output_path)
    return _speak_piper(text, output_path)
