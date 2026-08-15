"""Treina a cabeça do wake word "Oráculo". Roda uma vez, à mão.

    .venv/bin/python tools/treinar_wake.py --gravar 40
    .venv/bin/python tools/treinar_wake.py

Este script **não é importado pelo app** e não entra no caminho da conversa. É o
único lugar do projeto que usa scikit-learn e que baixa coisas — o runtime só lê
o .npz que sai daqui.

## Por que isto cabe no Python 3.14 sem torch

O openWakeWord tem fama de exigir torch/TF de 2022, mas isso vale para o script
de treino *deles*. A arquitetura é um extrator de características **congelado**
com uma cabeça minúscula por cima, e treinar essa cabeça é um MLP de 1536
entradas — sklearn dá conta. Os negativos nem precisam ser processados: o autor
publicou os embeddings já calculados.

## De onde vêm os dados

- **Positivos sintéticos:** "Oráculo" nas vozes pt-BR que já estão na máquina
  (Kokoro: pf_dora/pm_alex/pm_santa; Piper: faber, mais as que --baixar-vozes
  trouxer), com variação de velocidade, ganho, ruído e deslocamento.
- **Positivos reais (--gravar):** você dizendo a palavra. Valem muito mais que os
  sintéticos, porque 7 vozes de TTS têm timbre parecido demais e a cabeça tende a
  aprender "voz de robô" em vez de "a palavra". Ficam em ~/.oraculo/voice/.
- **Negativos difíceis:** palavras que rimam ou compartilham a cadeia tônica
  ("obstáculo", "espetáculo", "cálculo"...). São elas que decidem a taxa de falso
  positivo no uso real, muito mais que ruído genérico.
- **Negativos gerais:** embeddings pré-computados de ~2000 h de áudio real
  multilíngue (ACAV100M), baixados por range request — o .npy é contíguo depois
  do cabeçalho, então os primeiros N bytes já são as primeiras N/3072 janelas.
"""

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core import wake  # noqa: E402

HF = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
NEG_GRANDE = f"{HF}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
NEG_VALID = f"{HF}/validation_set_features.npy"
PIPER_HF = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/"
            "{nome}/{q}/pt_BR-{nome}-{q}.onnx")

# Frases positivas, sintetizadas INTEIRAS e com prosódia natural. Metade termina
# na palavra, metade continua — porque "Oráculo" dita no meio de uma frase soa
# diferente de "Oráculo." dita sozinha (vogal final encurtada, entonação que não
# cai, sem pausa), e um modelo que só viu a segunda ignora o uso mais comum.
#
# Colar dois clipes para simular isso NÃO funciona: a emenda tem a entonação de
# fim de frase no meio, que não existe na fala real. Por isso a frase é
# sintetizada de uma vez e o fim da palavra é localizado por alinhamento.
FRASES_POS = (
    "Oráculo.", "Oráculo,", "Ei, Oráculo.", "Ô Oráculo.",
    "Oráculo, que horas são?", "Oráculo, quanto é dois mais dois?",
    "Oráculo, me ajuda com uma coisa.", "Oráculo, tudo bem?",
    "Oráculo, como está o tempo hoje?", "Oráculo, resume isso para mim.",
    "Ei Oráculo, liga a luz da sala.", "Oráculo, preciso de uma ideia.",
)

# Modelo usado só para achar onde a palavra termina dentro da frase. Roda na CPU
# e uma vez por clipe base, fora do caminho da conversa.
MODELO_ALINHAMENTO = "small"

# Negativos difíceis: mesma terminação tônica ou início parecido. Sem isso o
# modelo dispara em "espetáculo", que é o tipo de erro que faz desistir do modo.
FRASES_NEG = (
    "obstáculo", "espetáculo", "tentáculo", "vernáculo", "receptáculo",
    "o cálculo", "ridículo", "articulo", "óculos", "ora pois",
    "oratória", "circulo", "vinculo", "musculo",
    "um cálculo rápido", "que espetáculo", "sem obstáculo nenhum",
    "ora, veja só", "oráquete", "aracnídeo", "oral",
)
# NÃO colocar aqui nada que contenha a palavra. "oráculos de Delfos" já esteve
# nesta lista e é literalmente "Oráculo" + /s/: o detector acertava ao disparar,
# e o rótulo é que estava errado. Quem cuida de "consultei o oráculo ontem" é a
# conferência de texto depois da transcrição, não o modelo acústico.

BLOCO = wake.BLOCO_SAMPLES
SR = wake.SAMPLERATE


# ----------------------------- utilidades ------------------------------------
def baixar(url: str, destino: Path, bytes_max: int | None = None) -> Path:
    """Baixa (uma vez) com barra simples. `bytes_max` pede só um prefixo."""
    if destino.exists() and bytes_max is None:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if bytes_max is not None:
        req.add_header("Range", f"bytes=0-{bytes_max - 1}")

    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp, \
            open(destino, "wb") as saida:
        total = int(resp.headers.get("Content-Length") or 0)
        lido = 0
        while True:
            pedaco = resp.read(1 << 20)
            if not pedaco:
                break
            saida.write(pedaco)
            lido += len(pedaco)
            # Só anima em terminal: num pipe o \r vira uma linha gigante de lixo.
            if total and sys.stdout.isatty():
                pct = 100 * lido / total
                print(f"\r  {destino.name}: {pct:5.1f}%  "
                      f"({lido/1e6:.0f}/{total/1e6:.0f} MB)", end="", flush=True)
    print(f"\r  {destino.name}: {lido/1e6:.0f} MB em "
          f"{time.monotonic()-t0:.0f}s{' '*20}")
    return destino


def baixar_modelos() -> None:
    """Os dois ONNX de característica (2,4 MB). Uma vez, nunca na conversa."""
    config.WAKE_DIR.mkdir(parents=True, exist_ok=True)
    for nome in (wake.MODELO_MEL, wake.MODELO_EMB):
        alvo = config.WAKE_DIR / nome
        if alvo.exists():
            print(f"  {nome}: já está aqui")
            continue
        baixar(wake.URL_MODELOS.format(nome), alvo)


def vozes_extras(baixar_faltantes: bool) -> list[str]:
    """Vozes pt-BR extras do Piper (63 MB cada). Mais timbres = melhor modelo.

    As que já estão no disco são usadas SEMPRE, com ou sem `--baixar-vozes`:
    o flag decide se busca as que faltam, não se aproveita as que existem.
    Ignorá-las seria treinar um modelo pior por acidente.
    """
    vozes = []
    destino = config.WAKE_DIR / "vozes"
    for nome, q in (("cadu", "medium"), ("jeff", "medium"), ("edresson", "low")):
        alvo = destino / f"pt_BR-{nome}-{q}.onnx"
        if not alvo.exists():
            if not baixar_faltantes:
                continue
            try:
                baixar(PIPER_HF.format(nome=nome, q=q), alvo)
                baixar(PIPER_HF.format(nome=nome, q=q) + ".json",
                       Path(str(alvo) + ".json"))
            except Exception as exc:  # noqa: BLE001
                print(f"  [aviso] não deu para baixar {nome}: {exc}")
                continue
        vozes.append(str(alvo))
    return vozes


def reamostra(sinal, de: int, para: int = SR):
    """Reamostragem linear. Boa o bastante: o mel logo adiante já é passa-baixa."""
    import numpy as np

    if de == para:
        return sinal.astype("float32")
    n = int(len(sinal) * para / de)
    return np.interp(np.linspace(0, len(sinal) - 1, n),
                     np.arange(len(sinal)), sinal).astype("float32")


# ----------------------------- síntese ---------------------------------------
def sintetiza_kokoro(frases, velocidades) -> list:
    import numpy as np
    from kokoro_onnx import Kokoro

    k = Kokoro(config.KOKORO_MODEL, config.KOKORO_VOICES)
    saida = []
    for voz in ("pf_dora", "pm_alex", "pm_santa"):
        for frase in frases:
            for vel in velocidades:
                try:
                    s, sr = k.create(frase, voice=voz, speed=vel, lang="pt-br")
                except Exception:  # noqa: BLE001
                    continue
                saida.append(reamostra(np.asarray(s, dtype="float32"), sr))
    return saida


def sintetiza_piper(frases, vozes) -> list:
    import subprocess

    import numpy as np

    saida = []
    for voz in vozes:
        if not os.path.exists(voz):
            continue
        for frase in frases:
            proc = subprocess.run(
                [config.PIPER_BIN, "--model", voz, "--output_raw"],
                input=frase.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            if proc.returncode != 0 or not proc.stdout:
                continue
            pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype("float32") / 32768
            saida.append(reamostra(pcm, 22050))
    return saida


def carrega_gravacoes() -> list:
    """As suas gravações da palavra, se existirem."""
    import numpy as np
    import soundfile as sf

    clipes = []
    if not config.VOICE_DIR.exists():
        return clipes
    for caminho in sorted(config.VOICE_DIR.glob("oraculo_*.wav")):
        dados, sr = sf.read(caminho, dtype="float32")
        if dados.ndim > 1:
            dados = dados[:, 0]
        clipes.append(reamostra(np.asarray(dados), sr))
    return clipes


# ----------------------------- características -------------------------------
# Silêncio à frente do clipe. Tem que ser maior que o aquecimento do extrator
# (~2,5 s), senão a janela que interessa — a que termina logo depois da palavra —
# ainda nem existe quando a palavra acaba, e o clipe inteiro é descartado.
PAD_INICIAL = 3.2
PAD_FINAL = 0.7


def janelas_do_clipe(extrator, clipe, rng, apenas_fim: bool, corte=None):
    """Roda a cadeia sobre ruído + clipe + ruído e devolve as janelas.

    `apenas_fim=True` fica só com as janelas cujo fim cai logo depois da palavra
    — são elas que o detector vê no instante do disparo. Pegar o clipe inteiro
    ensinaria a cabeça a disparar no meio da palavra.

    `corte` é onde a PALAVRA termina dentro do clipe, em amostras. O padrão (fim
    do clipe) só vale quando a palavra encerra a fala. Quando ela é seguida do
    pedido ("Oráculo, que horas são?"), o rótulo tem que ficar no fim da palavra
    — ancorar no fim da frase é o que faz o modelo ignorar exatamente o caso de
    uso mais comum.

    O preenchimento é ruído fraco, não zero: silêncio digital absoluto não existe
    no microfone, e treinar com ele ensinaria a cabeça a esperar um contexto que
    ela nunca vai encontrar.
    """
    import numpy as np

    def piso(segundos):
        return rng.normal(0, rng.uniform(1e-4, 3e-3),
                          int(segundos * SR)).astype("float32")

    frente = piso(PAD_INICIAL)
    fluxo = np.concatenate([frente, clipe, piso(PAD_FINAL)])
    fim_palavra = len(frente) + (len(clipe) if corte is None else corte)

    extrator.reset()
    janelas = []
    for i in range(0, len(fluxo) - BLOCO + 1, BLOCO):
        j = extrator.push(fluxo[i:i + BLOCO])
        if j is None:
            continue
        fim = i + BLOCO
        if apenas_fim and not (fim_palavra <= fim <= fim_palavra + int(0.4 * SR)):
            continue
        janelas.append(j.reshape(-1))
    return janelas


def aumenta(clipe, rng, n: int, corte=None):
    """Variações do mesmo clipe: ganho, ruído, velocidade e silêncio na frente.

    Devolve pares (áudio, corte). O corte acompanha as transformações que mexem
    no tempo — mudar a velocidade e não reescalar o rótulo desalinharia a janela
    positiva justamente nos clipes mais variados.
    """
    import numpy as np

    saida = [(clipe, corte)]
    for _ in range(n):
        x = clipe.copy()
        c = corte
        if rng.random() < 0.7:                       # velocidade
            fator = rng.uniform(0.88, 1.14)
            antes = len(x)
            x = reamostra(x, int(SR * fator), SR)
            if c is not None and antes:
                c = int(c * len(x) / antes)
        x = x * rng.uniform(0.35, 1.25)              # ganho
        if rng.random() < 0.8:                       # ruído
            snr = rng.uniform(8, 32)
            pot = float(np.mean(x ** 2)) or 1e-9
            x = x + rng.normal(0, (pot / (10 ** (snr / 10))) ** 0.5, len(x))
        deslocamento = int(rng.uniform(0, 0.05) * SR)  # desalinha o fim
        x = np.concatenate([np.zeros(deslocamento, dtype="float32"), x])
        if c is not None:
            c += deslocamento
        saida.append((np.clip(x, -1, 1).astype("float32"), c))
    return saida


def alinhador():
    """Devolve fim_da_palavra(clipe) -> amostra onde "Oráculo" termina, ou None.

    Usa os timestamps por palavra do faster-whisper, que já é dependência do
    projeto. É o jeito honesto de saber onde a palavra acaba dentro de uma frase
    natural — estimar pela duração de uma síntese separada erra, porque a
    prosódia muda quando a palavra deixa de encerrar a frase.
    """
    import soundfile as sf
    from faster_whisper import WhisperModel

    from core.wake import _normaliza, _parecido

    modelo = WhisperModel(MODELO_ALINHAMENTO, device="cpu", compute_type="int8")
    alvo = _normaliza(config.WAKE_WORD)
    tmp = "/tmp/oraculo_alinha.wav"

    def fim_da_palavra(clipe):
        sf.write(tmp, clipe, SR)
        segmentos, _ = modelo.transcribe(tmp, language="pt",
                                         word_timestamps=True)
        for seg in segmentos:
            for palavra in (seg.words or ()):
                if _parecido(_normaliza(palavra.word), alvo) >= config.WAKE_FUZZY:
                    return int(palavra.end * SR)
        return None

    return fim_da_palavra


def monta_positivos(rng, vozes_piper, n_aug: int):
    import numpy as np

    velocidades = (0.8, 0.9, 1.0, 1.1, 1.25)
    print("  sintetizando com Kokoro...")
    clipes = sintetiza_kokoro(FRASES_POS, velocidades)
    print(f"    {len(clipes)} clipes")

    piper = [config.PIPER_VOICE] + vozes_piper
    print("  sintetizando com Piper...")
    novos = sintetiza_piper(FRASES_POS, piper)
    print(f"    {len(novos)} clipes")
    clipes += novos

    gravados = carrega_gravacoes()
    if gravados:
        # As suas gravações valem mais: entram com mais aumentos para pesar
        # tanto quanto o time inteiro de TTS.
        print(f"  suas gravações: {len(gravados)} clipes (com 3x mais aumentos)")
        clipes += gravados * 3
    else:
        print("  [aviso] nenhuma gravação sua em ~/.oraculo/voice/.")
        print("          O modelo vai depender só de vozes sintéticas, que têm")
        print("          timbre parecido demais. Rode: --gravar 40")

    print("  localizando o fim da palavra em cada clipe...")
    fim_da_palavra = alinhador()
    cortes, alinhados, perdidos = [], [], 0
    for clipe in clipes:
        corte = fim_da_palavra(clipe)
        if corte is None:
            perdidos += 1          # a síntese saiu ininteligível; descarta
            continue
        alinhados.append(clipe)
        cortes.append(corte)
    clipes = alinhados
    print(f"    {len(clipes)} alinhados, {perdidos} descartados")
    if not clipes:
        return np.zeros((0, wake.EMB_JANELA * wake.EMB_DIM), "float32"), np.array([])

    extrator = wake.Extrator()
    janelas, grupos = [], []
    for i, (clipe, corte) in enumerate(zip(clipes, cortes)):
        for variante, c in aumenta(clipe, rng, n_aug, corte):
            novas = janelas_do_clipe(extrator, variante, rng,
                                     apenas_fim=True, corte=c)
            janelas += novas
            # Todas as janelas de um clipe (e de seus aumentos) pertencem ao
            # mesmo grupo: elas são quase duplicatas, e separar treino de
            # avaliação por janela deixaria a mesma fala dos dois lados.
            grupos += [i] * len(novas)
        if (i + 1) % 20 == 0 and sys.stdout.isatty():
            print(f"\r    características: {i+1}/{len(clipes)} clipes, "
                  f"{len(janelas)} janelas", end="", flush=True)
    print(f"\r    características: {len(clipes)} clipes, {len(janelas)} janelas")
    return np.array(janelas, dtype="float32"), np.array(grupos)


def monta_negativos_dificeis(rng, vozes_piper, n_aug: int):
    import numpy as np

    clipes = (sintetiza_kokoro(FRASES_NEG, (0.9, 1.05))
              + sintetiza_piper(FRASES_NEG, [config.PIPER_VOICE] + vozes_piper))
    extrator = wake.Extrator()
    janelas, grupos = [], []
    for i, clipe in enumerate(clipes):
        for variante, _ in aumenta(clipe, rng, n_aug):
            novas = janelas_do_clipe(extrator, variante, rng, apenas_fim=False)
            janelas += novas
            grupos += [i] * len(novas)
    print(f"  negativos difíceis: {len(clipes)} clipes, {len(janelas)} janelas")
    return np.array(janelas, dtype="float32"), np.array(grupos)


def carrega_negativos_gerais(gb: float):
    """Embeddings pré-computados: (N, 16, 96) do arquivo grande, achatados."""
    import numpy as np

    cache = config.WAKE_DIR / "negativos.npy"
    por_janela = wake.EMB_JANELA * wake.EMB_DIM * 2      # float16
    if not cache.exists():
        alvo = 128 + int(gb * 1e9) // por_janela * por_janela
        print(f"  baixando {alvo/1e9:.2f} GB de negativos "
              f"({(alvo-128)//por_janela} janelas)...")
        baixar(NEG_GRANDE, cache, bytes_max=alvo)

    # O cabeçalho é lido, não assumido em 128 bytes: o numpy alinha o tamanho e
    # ele muda com a versão do formato. Errar aqui desalinharia TODAS as janelas
    # por alguns valores — sem erro nenhum, só um modelo pior sem explicação.
    with open(cache, "rb") as fh:
        maior, menor = np.lib.format.read_magic(fh)
        leitor = getattr(np.lib.format,
                         f"read_array_header_{maior}_{menor}")
        leitor(fh)
        inicio = fh.tell()

    # memmap direto, SEM copiar: o arquivo tem gigabytes e materializá-lo na RAM
    # já custou um OOM. As conversões acontecem por lote, adiante.
    dados = np.memmap(cache, dtype="float16", mode="r", offset=inicio)
    n = len(dados) // (wake.EMB_JANELA * wake.EMB_DIM)
    print(f"  negativos gerais: {n} janelas "
          f"(~{n * BLOCO / SR / 3600:.1f} h de áudio, mapeados do disco)")
    return dados[:n * wake.EMB_JANELA * wake.EMB_DIM].reshape(n, -1)


# ----------------------------- treino ----------------------------------------
def treina(pos, neg_dificil, neg_geral, rng, semente: int):
    """Duas rodadas: uma normal, outra com os negativos mais difíceis minerados.

    A mineração é o que derruba o falso positivo. Treinar só com negativos
    sorteados deixa o modelo bom na média e ruim justamente onde importa — nos
    poucos trechos de áudio que se parecem com a palavra.
    """
    import numpy as np
    from sklearn.neural_network import MLPClassifier

    # Fatia de avaliação reservada: nunca treinada, nunca minerada. Quanto maior,
    # mais fina a medida de FP/hora — com 1 h só dá para dizer "menos de 1/h".
    # Fica como memmap: só é lida em lotes na calibração.
    n_aval = min(250_000, len(neg_geral) // 3)
    aval, pool = neg_geral[:n_aval], neg_geral[n_aval:]
    print(f"  reservado para avaliação: {n_aval} janelas "
          f"(~{n_aval * BLOCO / SR / 3600:.2f} h)")

    def fit(negativos, rotulo):
        # O MLPClassifier não aceita peso por classe. Sem replicar os positivos,
        # 1 positivo para cada 100 negativos faz o modelo aprender a responder
        # "não" sempre — o que dá 99% de acerto e recall zero.
        repete = max(1, min(40, len(negativos) // (10 * max(1, len(pos)))))
        positivos = np.tile(pos, (repete, 1)) if repete > 1 else pos
        X = np.concatenate([positivos, negativos]).astype("float32")
        y = np.concatenate([np.ones(len(positivos)), np.zeros(len(negativos))])
        media, desvio = X.mean(0), X.std(0) + 1e-6
        modelo = MLPClassifier(
            hidden_layer_sizes=(128, 32), activation="relu",
            alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
            max_iter=60, early_stopping=True, n_iter_no_change=6,
            validation_fraction=0.1, random_state=semente, verbose=False,
        )
        print(f"  treinando ({rotulo}): {len(pos)}x{repete} pos + "
              f"{len(negativos)} neg...")
        modelo.fit((X - media) / desvio, y)
        return modelo, media, desvio

    # Os tetos abaixo são de MEMÓRIA, não de qualidade: em float32 cada janela
    # ocupa 6 KB, e um treino com 260 mil negativos derrubou o processo por OOM
    # numa máquina de 16 GB. Mais negativos aqui rende pouco — quem faz o
    # trabalho pesado é a mineração logo abaixo.
    n1 = min(80_000, len(pool))
    sorteio = np.asarray(pool[rng.choice(len(pool), n1, replace=False)],
                         dtype="float32")
    modelo, media, desvio = fit(np.concatenate([neg_dificil, sorteio]),
                                "rodada 1")

    # Mineração: pontua o pool inteiro em lotes e fica com o topo. O pool é um
    # memmap, então o lote é o único pedaço que existe em RAM de cada vez.
    print("  minerando negativos difíceis...")
    LOTE = 50_000
    scores = np.concatenate([
        modelo.predict_proba(
            (np.asarray(pool[i:i + LOTE], dtype="float32") - media) / desvio)[:, 1]
        for i in range(0, len(pool), LOTE)
    ])
    n_duro = min(40_000, len(pool))
    indices = np.sort(np.argsort(scores)[-n_duro:])
    piores = np.asarray(pool[indices], dtype="float32")
    print(f"    {n_duro} piores; maior score do pool: {scores.max():.3f}")
    del scores

    modelo, media, desvio = fit(
        np.concatenate([neg_dificil, sorteio, piores]), "rodada 2")
    del sorteio, piores
    return modelo, media, desvio, aval


def calibra(modelo, media, desvio, pos, grupos, aval, alvo_fp_hora: float,
            recall_min: float, janela_min: float):
    """Escolhe o limiar pela taxa de falso positivo medida, não por chute.

    Um limiar chutado é o que dá a wake word a fama de disparar sozinha. Aqui o
    número sai de horas de áudio real que o modelo nunca viu, e o recall sai de
    clipes que também ficaram de fora do treino.
    """
    import numpy as np

    s_neg = np.concatenate([
        modelo.predict_proba(
            (np.asarray(aval[i:i + 50_000], dtype="float32") - media) / desvio)[:, 1]
        for i in range(0, len(aval), 50_000)
    ])
    s_pos = modelo.predict_proba((pos - media) / desvio)[:, 1]
    horas = len(aval) * BLOCO / SR / 3600

    def recall_por_fala(limiar):
        """Fração de FALAS reconhecidas, não de janelas.

        É o número que importa: uma fala rende ~5 janelas e basta uma passar do
        limiar para o detector disparar. O recall por janela subestima muito o
        comportamento real — 97% de janelas vira praticamente 100% de falas.
        """
        acima = s_pos >= limiar
        return float(np.mean([acima[grupos == g].any()
                              for g in np.unique(grupos)]))

    print(f"\n  {'limiar':>7s} {'FP/hora':>9s} {'janelas':>9s} {'falas':>8s}")
    # A folga importa tanto quanto o limite. Exigir só "0 falso positivo" empurra
    # a escolha para o extremo da tabela, onde uma fala legítima que pontua 0,998
    # é rejeitada por um limiar de 0,999. O piso de janelas garante que o limiar
    # fique numa região com margem dos dois lados.
    # Escolhe o MAIOR limiar que ainda reconhece as falas. Preferir precisão é
    # deliberado: um disparo à toa interrompe o usuário, um disparo perdido só
    # custa repetir a palavra. A regra oposta (maior recall sob um teto de FP)
    # escolhe limiares baixos e entrega um assistente que fala sozinho.
    escolhido = None
    for limiar in (0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999):
        fp_hora = float((s_neg >= limiar).sum()) / horas
        r_janela = float((s_pos >= limiar).mean())
        r_fala = recall_por_fala(limiar)
        marca = ""
        if (fp_hora <= alvo_fp_hora and r_fala >= recall_min
                and r_janela >= janela_min):
            escolhido, marca = limiar, "  <-- escolhido"
        print(f"  {limiar:7.3f} {fp_hora:9.2f} {r_janela:9.1%} "
              f"{r_fala:8.1%}{marca}")

    if escolhido is None:
        escolhido = 0.5
        print(f"\n  [aviso] nenhum limiar juntou {alvo_fp_hora} FP/hora com "
              f"{recall_min:.0%} das falas. Usando {escolhido} — confira os "
              f"números acima antes de confiar no modo.")
    return escolhido, s_pos, s_neg, horas, recall_por_fala(escolhido)


def salva(caminho, modelo, media, desvio, limiar) -> None:
    import numpy as np

    pesos = {}
    for i, (W, b) in enumerate(zip(modelo.coefs_, modelo.intercepts_)):
        pesos[f"W{i}"] = W.astype("float32")
        pesos[f"b{i}"] = b.astype("float32")
    caminho.parent.mkdir(parents=True, exist_ok=True)
    np.savez(caminho, media=media.astype("float32"),
             desvio=desvio.astype("float32"),
             limiar=np.float32(limiar), palavra=config.WAKE_WORD, **pesos)
    print(f"\n  gravado: {caminho} ({caminho.stat().st_size/1024:.0f} KB)")


# ----------------------------- gravação --------------------------------------
def testar_microfone(segundos: float = 3.0) -> int:
    """Grava um trecho de cada entrada plausível e diz qual serve.

    Existe porque "o microfone não funciona" tem causas que não dão erro nenhum:
    a fonte padrão pode ser um filtro que devolve silêncio, ou perder blocos. A
    única forma de saber é capturar e olhar o que veio.
    """
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    from core import audio, stt

    print("\n=== entradas disponíveis ===")
    audio.listar_entradas()
    print(f"\n  config.INPUT_DEVICE = {config.INPUT_DEVICE!r}")

    candidatos = [config.INPUT_DEVICE]
    padrao = sd.default.device[0]
    if config.INPUT_DEVICE is not None:
        candidatos.append(None)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and i != padrao:
            candidatos.append(i)

    print(f"\n=== falando por {segundos:.0f}s em cada entrada ===")
    print(f'  Diga "{config.WAKE_WORD}, teste de microfone" quando pedir.\n')

    melhor = None
    for dev in dict.fromkeys(candidatos):          # sem repetir, em ordem
        rotulo = "padrão do sistema" if dev is None else repr(dev)
        input(f"  Enter para testar {rotulo}... ")
        try:
            d = sd.rec(int(segundos * SR), samplerate=SR, channels=1,
                       dtype="float32", device=dev)
            sd.wait()
            d = d[:, 0]
        except Exception as exc:  # noqa: BLE001
            print(f"    ERRO: {str(exc)[:70]}\n")
            continue

        problema = audio.diagnostico_captura(d)
        pico, zeros = float(np.abs(d).max()), float((d == 0).mean())
        print(f"    pico={pico:.4f}  zeros={zeros*100:.1f}%  "
              f"{'OK' if not problema else 'PROBLEMA: ' + problema}")

        sf.write("/tmp/oraculo_mic.wav", audio._normalize(d), SR)
        try:
            texto = stt.transcribe("/tmp/oraculo_mic.wav")
            print(f"    transcrição: {texto!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"    transcrição falhou: {str(exc)[:60]}")
            texto = ""
        print()

        if not problema and texto and (melhor is None or zeros < melhor[1]):
            melhor = (dev, zeros)

    if melhor is None:
        print("  Nenhuma entrada devolveu fala limpa.")
        print(audio._aviso_dispositivo())
        return 1
    print(f"  Melhor entrada: {melhor[0]!r} — ponha isso em INPUT_DEVICE "
          f"no config.py.")
    return 0


def gravar(n: int) -> None:
    """Grava você dizendo a palavra, N vezes, com o VAD cortando cada uma.

    Confere cada amostra na hora e recusa as ruins. Guardar uma gravação
    quebrada é pior que não guardar nada: ela entra no treino como se fosse a
    sua voz e ensina o modelo a procurar o defeito.
    """
    import soundfile as sf

    from core import audio

    config.VOICE_DIR.mkdir(parents=True, exist_ok=True)
    existentes = len(list(config.VOICE_DIR.glob("oraculo_*.wav")))

    print("\n=== dispositivos de entrada ===")
    audio.listar_entradas()
    if config.INPUT_DEVICE is not None:
        print(f"  usando INPUT_DEVICE = {config.INPUT_DEVICE!r}")

    print(f"\nVou gravar {n} repetições de \"{config.WAKE_WORD}\".")
    print("Diga a palavra sozinha, do jeito que você diria de verdade.")
    print("Varie: mais perto, mais longe, mais rápido, mais baixo.")
    print("Espere o \"pode falar\" — o microfone leva ~1s para abrir.")
    print(f"Já existem {existentes} gravações. Ctrl+C para parar.\n")

    feitas, indice = 0, existentes
    try:
        for i in range(n):
            input(f"  [{i+1}/{n}] Enter, espere o aviso e diga "
                  f"\"{config.WAKE_WORD}\"... ")
            destino = config.VOICE_DIR / f"oraculo_{indice:03d}.wav"

            # O primeiro estado do segmentador chega no primeiro frame de áudio,
            # ou seja, quando o dispositivo já está entregando som de verdade —
            # é o único momento honesto para dizer "pode falar".
            pronto = False

            def _aviso(_estado) -> None:
                nonlocal pronto
                if not pronto:
                    pronto = True
                    print("        pode falar")

            try:
                caminho = audio.record_vad(path=str(destino), on_state=_aviso)
            except RuntimeError as exc:
                print(f"        [erro] {exc}")
                return

            if caminho is None:
                print("        não ouvi nada — repetindo")
                continue

            d, _ = sf.read(caminho, dtype="float32")
            dur = len(d) / config.RECORD_SAMPLERATE
            if dur < 0.35:
                print(f"        curta demais ({dur:.2f}s) — repetindo")
                os.remove(caminho)
                continue

            print(f"        ok ({dur:.2f}s)")
            feitas += 1
            indice += 1
    except KeyboardInterrupt:
        print("\n  interrompido.")

    print(f"\n  {feitas} gravações novas em {config.VOICE_DIR}")
    if feitas:
        print("  Agora rode o treino: .venv/bin/python tools/treinar_wake.py")


# ----------------------------- principal -------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gravar", type=int, metavar="N",
                   help="grava N repetições suas da palavra e sai")
    p.add_argument("--testar-microfone", action="store_true",
                   help="grava 3 s, mede a qualidade da captura, transcreve e sai")
    p.add_argument("--baixar-modelos", action="store_true",
                   help="baixa só os ONNX de característica e sai")
    p.add_argument("--baixar-vozes", action="store_true",
                   help="baixa 3 vozes pt-BR extras do Piper (189 MB)")
    p.add_argument("--negativos-gb", type=float, default=1.0,
                   help="quanto do banco de negativos baixar (padrão: 1.0)")
    p.add_argument("--aumentos", type=int, default=6,
                   help="variações por clipe sintetizado (padrão: 6)")
    p.add_argument("--fp-hora", type=float, default=0.3,
                   help="falso positivo por hora tolerado (padrão: 0.3)")
    p.add_argument("--recall-min", type=float, default=0.97,
                   help="fração mínima de FALAS reconhecidas (padrão: 0.97)")
    p.add_argument("--janela-min", type=float, default=0.85,
                   help="fração mínima de JANELAS positivas — é o que dá folga "
                        "ao limiar (padrão: 0.85)")
    p.add_argument("--semente", type=int, default=0)
    args = p.parse_args()

    if args.testar_microfone:
        return testar_microfone()

    if args.gravar:
        gravar(args.gravar)
        return 0

    print("\n=== 1. modelos de característica ===")
    baixar_modelos()
    if args.baixar_modelos:
        return 0

    import numpy as np

    rng = np.random.default_rng(args.semente)
    vozes = vozes_extras(args.baixar_vozes)
    print(f"  vozes Piper extras em uso: {len(vozes)}")

    print("\n=== 2. positivos ===")
    pos, g_pos = monta_positivos(rng, vozes, args.aumentos)
    if len(pos) < 200:
        print(f"\n[erro] só {len(pos)} janelas positivas — muito pouco para treinar.")
        return 1

    print("\n=== 3. negativos ===")
    neg_dificil, g_dif = monta_negativos_dificeis(
        rng, vozes, max(2, args.aumentos // 2))
    neg_geral = carrega_negativos_gerais(args.negativos_gb)

    # Separação por CLIPE, não por janela: as janelas de um mesmo clipe são
    # quase idênticas, e dividir por janela colocaria a mesma fala nos dois
    # lados — o recall sairia perto de 100% sem querer dizer nada.
    def parte(X, grupos, fracao=0.2):
        ids = np.unique(grupos)
        fora = set(rng.choice(ids, max(1, int(len(ids) * fracao)), replace=False))
        mascara = np.array([g in fora for g in grupos])
        return X[~mascara], X[mascara], grupos[mascara]

    pos_treino, pos_aval, g_pos_aval = parte(pos, g_pos)
    dif_treino, dif_aval, _ = parte(neg_dificil, g_dif)
    print(f"  reservado: {len(pos_aval)} janelas positivas e {len(dif_aval)} "
          f"difíceis, de clipes que o treino não vê")

    print("\n=== 4. treino ===")
    modelo, media, desvio, aval = treina(pos_treino, dif_treino, neg_geral,
                                         rng, args.semente)

    print("\n=== 5. calibração do limiar ===")
    limiar, s_pos, s_neg, horas, r_fala = calibra(
        modelo, media, desvio, pos_aval, g_pos_aval, aval,
        args.fp_hora, args.recall_min, args.janela_min)

    # Os negativos difíceis são o teste que importa: é neles que um modelo ruim
    # se entrega, e eles não aparecem na medida de FP/hora acima.
    s_dif = modelo.predict_proba((dif_aval - media) / desvio)[:, 1]
    print("\n  RESULTADO (tudo em clipes e áudio que o treino não viu)")
    print(f"    limiar escolhido: {limiar}")
    print(f"    falas reconhecidas: {r_fala:.1%}")
    print(f"    janelas positivas acima do limiar: {(s_pos >= limiar).mean():.1%} "
          f"({len(s_pos)} janelas)")
    print(f"    palavras parecidas que passariam: "
          f"{(s_dif >= limiar).sum()}/{len(s_dif)} (pior {s_dif.max():.3f})")
    print(f"    falso positivo em {horas:.2f} h de áudio real: "
          f"{float((s_neg >= limiar).sum()) / horas:.2f}/h")

    salva(config.WAKE_DIR / wake.CABECA, modelo, media, desvio, limiar)
    print("\nPronto. Ligue com /despertar dentro do Oráculo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
