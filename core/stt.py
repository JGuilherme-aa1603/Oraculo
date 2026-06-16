"""STT — transcrição de áudio para texto, com motor selecionável.

Motores (config.STT_ENGINE):
  - "whisper":  faster-whisper (CTranslate2). VAD embutido, fallback CUDA→CPU,
                parâmetros ajustados (initial_prompt, beam_size, temperature, VAD).
  - "parakeet": NVIDIA Parakeet TDT 0.6b v3 via onnx-asr. Muito rápido na CPU,
                pontua/capitaliza sozinho. Limite de ~20-30s por clipe (sem VAD).

O backend onnx-asr é genérico (cache por nome de modelo), reutilizável por outros
modelos da mesma lib. A interface pública é transcribe(audio_path) -> str.
Os modelos são carregados preguiçosamente na 1ª transcrição e cacheados em memória;
os imports são preguiçosos para o modo texto nunca quebrar sem as dependências.
"""

import config

_whisper = None
_onnx_models: dict = {}   # nome do modelo onnx-asr → instância carregada


# ----------------------------- Whisper ---------------------------------------
def _enable_cuda_libs() -> None:
    """Carrega as libs CUDA 12 dos wheels nvidia-* (instalados no venv) no
    processo, para o ctranslate2 encontrá-las mesmo com o sistema em CUDA 13.

    Os .so ficam em site-packages/nvidia/*/lib, fora do caminho padrão do
    linker, então é preciso pré-carregá-los via ctypes RTLD_GLOBAL. cublas vem
    antes do cudnn (o cudnn depende de símbolos do cublas). Se os wheels não
    estiverem instalados, não faz nada — o fallback CPU assume."""
    import ctypes
    import glob
    import os

    try:
        import nvidia
    except ImportError:
        return

    # nvidia é namespace package (sem __file__); usar __path__.
    bases = list(getattr(nvidia, "__path__", []))
    for sub in ("cublas", "cudnn"):
        for base in bases:
            for so in sorted(glob.glob(os.path.join(base, sub, "lib", "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _get_whisper():
    """Carrega (uma vez) o modelo Whisper configurado, com fallback p/ CPU."""
    global _whisper
    if _whisper is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper não instalado. Rode:\n"
                "  .venv/bin/python -m pip install faster-whisper"
            ) from exc

        try:
            if config.WHISPER_DEVICE == "cuda":
                _enable_cuda_libs()
            _whisper = WhisperModel(
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
                _whisper = WhisperModel(config.WHISPER_MODEL,
                                        device="cpu", compute_type="int8")
            else:
                raise
    return _whisper


def _transcribe_whisper(audio_path: str) -> str:
    """Transcreve com faster-whisper. VAD descarta trechos sem fala (evita
    alucinação em silêncio); initial_prompt enviesa o domínio pt-BR."""
    segments, _ = _get_whisper().transcribe(
        audio_path,
        language="pt",
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        temperature=0.0,
    )
    return " ".join(seg.text for seg in segments).strip()


# ----------------------------- onnx-asr (Parakeet) ---------------------------
def _get_onnx_asr(model_name: str):
    """Carrega (uma vez por nome) um modelo onnx-asr, cacheado em memória."""
    if model_name not in _onnx_models:
        try:
            import onnx_asr
        except ImportError as exc:
            raise RuntimeError(
                "onnx-asr não instalado. Rode:\n"
                "  .venv/bin/python -m pip install onnx-asr onnxruntime"
            ) from exc
        # Baixa do Hugging Face na 1ª vez (~alguns minutos), depois cacheia.
        _onnx_models[model_name] = onnx_asr.load_model(model_name)
    return _onnx_models[model_name]


def _transcribe_onnx_asr(audio_path: str, model_name: str, language: str | None) -> str:
    """Transcreve com um modelo onnx-asr (Parakeet). recognize() devolve a
    string já pontuada.

    Fixa o idioma quando definido, em vez de deixar a detecção automática (que
    oscila entre idiomas e erra palavras). Espera WAV 16 kHz mono — o formato
    gravado por audio.record_ptt (RECORD_SAMPLERATE = 16000), sem conversão.
    Clipes acima de ~30s passam do limite do modelo (sem VAD — ver Fase 3)."""
    kwargs = {}
    if language:
        kwargs["language"] = language
    return _get_onnx_asr(model_name).recognize(audio_path, **kwargs).strip()


# ----------------------------- Interface -------------------------------------
_MIN_DURATION = 0.25   # s — clipes mais curtos não têm fala


def _too_short(audio_path: str) -> bool:
    """True se o áudio é curto demais para conter fala. Evita lixo/erro de
    divisão no pré-processador dos modelos onnx-asr (sem VAD) em gravações
    acidentais (Enter sem falar)."""
    try:
        import soundfile as sf

        info = sf.info(audio_path)
        return info.frames < _MIN_DURATION * info.samplerate
    except Exception:  # noqa: BLE001
        return False


def transcribe(audio_path: str) -> str:
    """Transcreve um WAV (pt-BR) para texto, usando o motor configurado."""
    if _too_short(audio_path):
        return ""
    if config.STT_ENGINE == "parakeet":
        return _transcribe_onnx_asr(audio_path, config.PARAKEET_MODEL,
                                    config.PARAKEET_LANGUAGE)
    return _transcribe_whisper(audio_path)


def available() -> bool:
    """True se o motor STT configurado está importável."""
    if config.STT_ENGINE == "parakeet":
        try:
            import onnx_asr  # noqa: F401
        except ImportError:
            return False
        return True
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True
