"""TTS — síntese de texto para áudio com Piper (100% local).

Piper é um binário externo. O texto é enviado via stdin e o WAV é escrito em disco.
"""

import shutil
import subprocess

import config


def available() -> bool:
    """True se o binário do Piper estiver acessível no PATH/config."""
    return shutil.which(config.PIPER_BIN) is not None


def speak(text: str, output_path: str = "/tmp/oraculo_tts.wav") -> str:
    """Gera um WAV a partir do texto usando Piper. Retorna o caminho do arquivo."""
    if not available():
        raise RuntimeError(
            f"Piper não encontrado ('{config.PIPER_BIN}'). Instale com:\n"
            "  yay -S piper-tts   (ou baixe o binário standalone)\n"
            "E baixe uma voz pt-BR (ex.: pt_BR-faber-medium) do repositório\n"
            "rhasspy/piper-voices no HuggingFace."
        )

    subprocess.run(
        [config.PIPER_BIN, "--model", config.PIPER_VOICE,
         "--output_file", output_path],
        input=text.encode("utf-8"),
        check=True,
    )
    return output_path
