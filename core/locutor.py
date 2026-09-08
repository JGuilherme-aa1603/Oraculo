"""Verificação de locutor — o Oráculo só responde a quem ele conhece.

Por que existe: a wake word (core/wake.py) sabe que a *palavra* foi dita, não
*quem* disse. Uma televisão ligada, um vizinho ou um vídeo no YouTube abrem um
turno igualzinho ao dono. Este módulo fecha esse portão: transforma a fala num
vetor de 256 números que descreve o timbre, e compara com o perfil gravado por
tools/cadastrar_voz.py. Nada de texto, nada de conteúdo — só a voz.

**O que ele NÃO é:** autenticação. Um vetor de timbre é enganável por imitação
e por uma gravação sua tocada num alto-falante. Ele resolve o problema real
(a sala falando junto), não um adversário.

De onde vem o modelo: `voxceleb_resnet34_LM.onnx` do WeSpeaker (Apache-2.0,
26 MB), treinado no VoxCeleb. É o mesmo padrão do VAD e da wake word — ONNX
direto no `onnxruntime` que já está no venv, sem torch. Baixado uma única vez
pelo tools/cadastrar_voz.py, nunca no caminho da conversa (invariante 2).

**A parte que erra em silêncio são as features, não o modelo.** A rede espera
fbank de 80 bandas no dialeto do Kaldi, e cada convenção conta: a janela é
hamming simétrica, o DC sai antes da pré-ênfase (não depois), a pré-ênfase
repete a primeira amostra em vez de assumir zero, os triângulos são desenhados
em mel e a coluna de Nyquist é zero. Errar qualquer uma não levanta exceção —
só encolhe a separação entre as vozes, e o sintoma aparece como "o limiar não
acha um ponto bom". Por isso `_fbank` é conferida contra vozes conhecidas antes
de qualquer treino de perfil, e por isso a CMN (subtrair a média no tempo) vem
por último: além de tirar o efeito do microfone, ela anula qualquer erro de
escala do sinal, o que é a única armadilha desta lista que dá para relaxar.
"""

import config

# Constantes do modelo e do dialeto de features do WeSpeaker. Nenhuma é
# ajustável: mudar qualquer uma invalida o perfil já gravado.
SAMPLERATE = 16000
DIM = 256                  # tamanho do vetor de saída
MEL_BANDAS = 80
FRAME_LEN = 400            # 25 ms
FRAME_SHIFT = 160          # 10 ms
FFT = 512                  # próxima potência de 2 acima de FRAME_LEN
PREENFASE = 0.97
MEL_LOW_HZ = 20.0

MODELO = "voxceleb_resnet34_LM.onnx"
PERFIL = "perfil.npz"

# Baixado uma vez pelo tools/cadastrar_voz.py.
URL_MODELO = ("https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/"
              "resolve/main/voxceleb_resnet34_LM.onnx")

# Assinatura do ONNX. Conferida em _sessao(), não assumida — mesma regra do
# core/vad.py: melhor degradar com aviso do que estourar no meio de um turno.
_ENTRADA = "feats"
_SAIDA = "embs"

_sessao_cache = None
_banks_cache = None
_janela_cache = None
_perfil_cache = None


# ------------------------------ features -------------------------------------
def _mel(hz: float) -> float:
    """Escala mel do Kaldi (não é a do Slaney nem a do librosa)."""
    import math

    return 1127.0 * math.log(1.0 + hz / 700.0)


def _mel_banks():
    """Matriz (80, 257) de triângulos em mel. Calculada uma vez."""
    import numpy as np

    global _banks_cache
    if _banks_cache is not None:
        return _banks_cache

    n_fft_bins = FFT // 2                    # 256; a coluna de Nyquist fica zero
    largura = SAMPLERATE / FFT
    mel_low = _mel(MEL_LOW_HZ)
    mel_high = _mel(SAMPLERATE / 2)
    delta = (mel_high - mel_low) / (MEL_BANDAS + 1)

    freqs = np.arange(n_fft_bins) * largura
    mels = np.array([_mel(f) for f in freqs])

    banks = np.zeros((MEL_BANDAS, n_fft_bins + 1), dtype="float64")
    for i in range(MEL_BANDAS):
        esq = mel_low + i * delta
        centro = esq + delta
        dir_ = esq + 2 * delta
        dentro = (mels > esq) & (mels < dir_)
        subida = (mels - esq) / delta
        descida = (dir_ - mels) / delta
        banks[i, :n_fft_bins] = np.where(dentro,
                                         np.where(mels <= centro, subida, descida),
                                         0.0)
    _banks_cache = banks
    return banks


def _janela():
    """Hamming simétrica de 400 amostras (o `window_type='hamming'` do Kaldi)."""
    import numpy as np

    global _janela_cache
    if _janela_cache is None:
        n = np.arange(FRAME_LEN)
        _janela_cache = 0.54 - 0.46 * np.cos(2 * np.pi * n / (FRAME_LEN - 1))
    return _janela_cache


def fbank(wav):
    """fbank 80-d com CMN, no dialeto do Kaldi. `wav` é float32 mono em [-1, 1].

    Devolve (T, 80) float32, ou None se o áudio for curto demais para um frame.
    """
    import numpy as np

    x = np.asarray(wav, dtype="float64") * 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.shape[0] < FRAME_LEN:
        return None

    n_frames = 1 + (x.shape[0] - FRAME_LEN) // FRAME_SHIFT
    idx = (np.arange(FRAME_LEN)[None, :]
           + FRAME_SHIFT * np.arange(n_frames)[:, None])
    frames = x[idx]

    # A ordem é a do Kaldi: DC fora, depois pré-ênfase (repetindo a primeira
    # amostra do frame, não zero), depois a janela.
    frames = frames - frames.mean(axis=1, keepdims=True)
    anterior = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - PREENFASE * anterior
    frames = frames * _janela()

    espectro = np.abs(np.fft.rfft(frames, n=FFT)) ** 2
    energias = espectro @ _mel_banks().T
    piso = np.finfo(np.float32).eps         # o mesmo do Kaldi/torchaudio
    feats = np.log(np.maximum(energias, piso))
    feats = feats - feats.mean(axis=0, keepdims=True)   # CMN
    return feats.astype("float32")


# ------------------------------ modelo ---------------------------------------
def caminho_modelo():
    """Onde o ONNX do WeSpeaker mora."""
    return config.LOCUTOR_DIR / MODELO


def _sessao():
    """Sessão ONNX carregada uma vez. Mesmo cache global do core/vad.py."""
    global _sessao_cache
    if _sessao_cache is not None:
        return _sessao_cache

    alvo = caminho_modelo()
    if not alvo.exists():
        raise RuntimeError(
            f"verificação de voz indisponível: falta o {MODELO}.\n"
            f"  .venv/bin/python tools/cadastrar_voz.py --baixar"
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "verificação de voz indisponível: o onnxruntime não está instalado."
        ) from exc

    sessao = ort.InferenceSession(str(alvo), providers=["CPUExecutionProvider"])
    entradas = tuple(e.name for e in sessao.get_inputs())
    saidas = tuple(s.name for s in sessao.get_outputs())
    if entradas != (_ENTRADA,) or saidas != (_SAIDA,):
        raise RuntimeError(
            f"verificação de voz indisponível: o {MODELO} mudou de assinatura "
            f"(esperado {_ENTRADA}→{_SAIDA}, encontrado {entradas}→{saidas})."
        )
    _sessao_cache = sessao
    return _sessao_cache


def embedding(wav):
    """Vetor de 256 números (norma 1) que descreve o timbre de `wav`.

    Devolve None se o áudio for curto demais para render um frame de fbank.
    """
    import numpy as np

    feats = fbank(wav)
    if feats is None:
        return None
    saida = _sessao().run(None, {_ENTRADA: feats[None, :, :]})[0][0]
    norma = np.linalg.norm(saida)
    if norma == 0:
        return None
    return (saida / norma).astype("float32")


def similaridade(a, b) -> float:
    """Cosseno entre dois vetores já normalizados (−1 a 1, na prática 0 a 1)."""
    import numpy as np

    return float(np.dot(a, b))


# ------------------------------ perfil ---------------------------------------
def caminho_perfil():
    """Onde o perfil do dono mora."""
    return config.LOCUTOR_DIR / PERFIL


def gravar_perfil(vetores, limiar: float, fonte: str = "",
                  frases: int = 0) -> None:
    """Grava o centróide do dono e o limiar medido para ele.

    Guarda os vetores individuais junto: um perfil novo pode ser recalculado
    (limiar novo, gravação nova) sem pedir todas as frases de volta.

    `frases` é quantas gravações vieram do cadastro de VERDADE (`--gravar`), em
    oposição aos clipes de 1 s de "Oráculo" que o treinador do wake word deixou.
    A distinção não é estatística, é de significado: um perfil feito só dos
    clipes descreve uma PALAVRA, não uma voz. Quem decide com base nisso é
    `perfil_robusto()`, e é ele que sustenta o nível destrutivo da Fase 5.
    """
    import numpy as np

    vetores = np.asarray(vetores, dtype="float32")
    centro = vetores.mean(axis=0)
    centro = centro / np.linalg.norm(centro)
    config.LOCUTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(caminho_perfil(), centro=centro, vetores=vetores,
             limiar=np.float32(limiar), fonte=np.array(fonte),
             frases=np.int32(frases))
    global _perfil_cache
    _perfil_cache = None


def perfil():
    """(centro, limiar) do dono, ou None se ninguém se cadastrou ainda."""
    import numpy as np

    global _perfil_cache
    if _perfil_cache is not None:
        return _perfil_cache
    alvo = caminho_perfil()
    if not alvo.exists():
        return None
    dados = np.load(alvo, allow_pickle=False)
    # Mesmo contrato do WAKE_THRESHOLD: o limiar do config só vale se alguém o
    # tiver escrito lá; o padrão é confiar no que o cadastro mediu.
    limiar = config.LOCUTOR_THRESHOLD or float(dados["limiar"])
    _perfil_cache = (dados["centro"], limiar)
    return _perfil_cache


def perfil_robusto() -> bool:
    """True se o perfil veio de cadastro REAL, não do provisório.

    Um perfil sem a chave `frases` é, por construção, anterior a esta checagem —
    e o único que existe assim é o provisório, montado com os clipes de
    "Oráculo" do treinador do wake word. Tratá-lo como zero é o padrão seguro:
    um arquivo antigo **não** destrava o nível destrutivo por omissão.

    Isto é o que torna a regra estrutural em vez de um comentário: quem confere
    é o arquivo, não a boa vontade de quem escreveu o código acima.
    """
    import numpy as np

    alvo = caminho_perfil()
    if not alvo.exists():
        return False
    try:
        dados = np.load(alvo, allow_pickle=False)
        frases = int(dados["frases"]) if "frases" in dados.files else 0
    except (OSError, ValueError, KeyError):
        return False
    return frases >= config.ACOES_MIN_FRASES_PERFIL


def disponivel() -> bool:
    """True se dá para verificar agora: modelo no disco e perfil cadastrado.

    Espelha `vad.disponivel()`/`wake.disponivel()`: carrega o que precisa para
    que a incompatibilidade apareça aqui, não no meio de um turno.
    """
    if perfil() is None:
        return False
    try:
        _sessao()
    except Exception:  # noqa: BLE001
        return False
    return True


def motivo_indisponivel() -> str:
    """Explica por que `disponivel()` deu False, para o /dono poder dizer."""
    try:
        _sessao()
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    if perfil() is None:
        return ("ninguém cadastrou a voz ainda.\n"
                "  .venv/bin/python tools/cadastrar_voz.py --gravar 8\n"
                "  .venv/bin/python tools/cadastrar_voz.py")
    return ""


def esquecer() -> bool:
    """Apaga o perfil. Devolve True se havia algo para apagar."""
    global _perfil_cache
    _perfil_cache = None
    alvo = caminho_perfil()
    if not alvo.exists():
        return False
    alvo.unlink()
    return True


# ------------------------------ verificação ----------------------------------
# Resultados de verificar().
DONO = "dono"          # é você
ESTRANHO = "estranho"  # não é você — descartar a fala
CURTO = "curto"        # fala curta demais para julgar; passa assim mesmo


def verificar(wav) -> tuple[str, float]:
    """Julga uma fala. Devolve (resultado, similaridade).

    Fala mais curta que `LOCUTOR_MIN_SECONDS` devolve CURTO e **passa**: o vetor
    de timbre precisa de fala para se formar, e rejeitar por falta de evidência
    tornaria "sim", "não" e "para" inutilizáveis. Quem já filtrou a sala nesse
    caso foi a wake word.
    """
    import numpy as np

    dados = perfil()
    if dados is None:
        return CURTO, 0.0
    centro, limiar = dados

    wav = np.asarray(wav, dtype="float32")
    if wav.shape[0] < config.LOCUTOR_MIN_SECONDS * SAMPLERATE:
        return CURTO, 0.0

    vetor = embedding(wav)
    if vetor is None:
        return CURTO, 0.0
    score = similaridade(vetor, centro)
    return (DONO if score >= limiar else ESTRANHO), score


def verificar_arquivo(path: str) -> tuple[str, float]:
    """`verificar()` a partir de um WAV no disco (o que a gravação produz)."""
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if sr != SAMPLERATE:
        raise ValueError(f"esperado {SAMPLERATE} Hz, veio {sr}")
    return verificar(wav)
