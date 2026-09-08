"""Cadastra a sua voz para o Oráculo só responder a você. Roda uma vez, à mão.

    .venv/bin/python tools/cadastrar_voz.py --gravar 8
    .venv/bin/python tools/cadastrar_voz.py

Este script **não é importado pelo app** e não entra no caminho da conversa. Ele
é o único lugar que baixa coisas para a verificação de voz; o runtime só lê o
perfil.npz que sai daqui (ver core/locutor.py).

## O que ele faz

1. baixa o `voxceleb_resnet34_LM.onnx` do WeSpeaker (26 MB), uma vez;
2. junta as suas gravações — as frases do `--gravar` mais os clipes de "Oráculo"
   que o treinador do wake word já deixou em ~/.oraculo/voice/;
3. baixa fala de **outras pessoas** para medir o limiar (português real, do
   Multilingual LibriSpeech), guarda só os vetores e joga o áudio fora;
4. escolhe o limiar com folga e grava o perfil.

## Por que os negativos precisam ser gente de verdade

A primeira medição usou vozes de TTS como "outras pessoas" e deu uma folga
enorme — mentira: duas vozes sintéticas *diferentes* chegam a 0,61 de
similaridade entre si, mais do que muita gente de verdade. Sintético mede a
distância do robô para você, que não é a pergunta. Com locutores reais a conta
fechou em 0,43 (o mais parecido) contra 0,58 (a sua pior gravação).

## Por que o limiar fica no MEIO da folga

O erro do wake word foi escolher "o maior limiar com zero falso positivo", que
encosta no extremo da tabela e passa a rejeitar fala legítima. Aqui a regra é
explícita: `(pior negativo + pior positivo) / 2`, e se não houver folga nenhuma
o script recusa gravar o perfil em vez de entregar um número bonito.
"""

import argparse
import json
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from core import locutor  # noqa: E402
from treinar_wake import baixar  # noqa: E402  (mesmo baixador do wake word)

# Fala real em português, com locutor identificado, servida linha a linha pelo
# datasets-server do Hugging Face — dá para pegar 20 clipes sem baixar o corpus.
MLS = ("https://datasets-server.huggingface.co/rows?dataset=facebook%2F"
       "multilingual_librispeech&config=portuguese&split=test"
       "&offset={offset}&length={length}")

# Frases para ler no --gravar. Variadas de propósito: o modelo é independente do
# texto, mas um perfil feito só da palavra "Oráculo" descreve uma palavra, não
# uma voz. Cada uma dá ~4 s, que é a duração típica de um pedido de verdade.
FRASES = (
    "Oráculo, me lembra o que eu tinha combinado para hoje de manhã.",
    "Quero saber como está o tempo antes de sair de casa.",
    "Explica isso de novo, mas com um exemplo mais simples, por favor.",
    "Anota aí que amanhã eu preciso revisar o projeto inteiro.",
    "Qual é a diferença entre esses dois jeitos de resolver o problema?",
    "Hoje o dia foi longo e eu ainda tenho bastante coisa para terminar.",
    "Me ajuda a organizar essa lista em uma ordem que faça sentido.",
    "Deixa eu ver se entendi direito o que você acabou de falar.",
)

# Durações em que as janelas de negativo são cortadas. Cobrem o regime real de
# uso: um pedido curto tem ~1 s, um pedido comum tem 2 a 3 s.
JANELAS = (1.0, 2.0, 3.0)

# Piso de sanidade. Um limiar abaixo disto não está separando nada — melhor
# recusar e dizer o porquê do que deixar o portão aberto fingindo que fecha.
LIMIAR_MINIMO = 0.30


# ----------------------------- gravação --------------------------------------
def gravar(n: int) -> int:
    """Grava N frases suas, com parada automática pelo VAD."""
    from core import audio, vad

    if not vad.disponivel():
        print("  [aviso] VAD indisponível — cada frase vai parar pelo teto de "
              f"{config.VAD_MAX_SECONDS:.0f} s.")

    destino = config.LOCUTOR_DIR / "amostras"
    destino.mkdir(parents=True, exist_ok=True)
    ja = len(list(destino.glob("*.wav")))
    print(f"\n{n} frases. Fale no ritmo normal, como se estivesse pedindo algo.")
    print("A gravação para sozinha quando você silenciar.\n")

    gravadas = 0
    for i in range(n):
        frase = FRASES[(ja + i) % len(FRASES)]
        print(f"  [{i + 1}/{n}] {frase}")
        input("        Enter para gravar... ")
        alvo = destino / f"dono_{ja + i:03d}.wav"
        try:
            path = audio.record_vad(path=str(alvo), start_timeout=10.0)
        except RuntimeError as exc:
            print(f"        falhou: {exc}")
            return 1
        if path is None:
            print("        não ouvi nada — repetindo esta frase")
            continue
        import soundfile as sf

        info = sf.info(path)
        print(f"        ok, {info.duration:.1f} s\n")
        gravadas += 1

    print(f"{gravadas} frases em {destino}")
    print("Agora rode sem --gravar para montar o perfil.")
    return 0


def amostras_do_dono() -> tuple[list, list[str]]:
    """Todos os vetores seus: as frases do --gravar e os clipes do wake word.

    Devolve (vetores, descrição das fontes). As duas fontes entram juntas de
    propósito: as frases descrevem a voz em uso normal, e os clipes curtos de
    "Oráculo" descrevem o caso mais difícil — a fala de 1 s que a wake word abre.
    """
    import numpy as np
    import soundfile as sf

    fontes = [
        (config.LOCUTOR_DIR / "amostras", "frases (--gravar)"),
        (config.VOICE_DIR, 'clipes de "Oráculo" (treinador do wake word)'),
    ]
    vetores, descricao = [], []
    for pasta, rotulo in fontes:
        arquivos = sorted(pasta.glob("*.wav")) if pasta.exists() else []
        n_ok, segundos = 0, 0.0
        for arq in arquivos:
            wav, sr = sf.read(arq, dtype="float32")
            if sr != locutor.SAMPLERATE:
                print(f"  [aviso] {arq.name}: {sr} Hz, ignorado")
                continue
            vetor = locutor.embedding(wav)
            if vetor is None:
                continue
            vetores.append(vetor)
            n_ok += 1
            segundos += len(wav) / sr
        if n_ok:
            descricao.append(f"{n_ok} de {rotulo} ({segundos:.0f} s)")
    return (np.array(vetores) if vetores else np.zeros((0, locutor.DIM))), descricao


# ----------------------------- negativos -------------------------------------
def _linhas_mls(quantos: int) -> list[dict]:
    """Metadados de clipes do MLS, variando o offset para pegar mais locutores."""
    import numpy as np

    with urllib.request.urlopen(MLS.format(offset=0, length=1), timeout=60) as r:
        total = json.loads(r.read()).get("num_rows_total") or 800

    linhas = []
    for offset in [int(x) for x in np.linspace(0, max(0, total - 100), 9)]:
        try:
            with urllib.request.urlopen(
                    MLS.format(offset=offset, length=100), timeout=90) as r:
                linhas += json.loads(r.read())["rows"]
        except Exception as exc:  # noqa: BLE001
            print(f"  [aviso] offset {offset}: {exc}")
        if len({l["row"].get("speaker_id") for l in linhas}) >= 10 \
                and len(linhas) >= quantos * 3:
            break
    return linhas


def negativos(quantos: int, refazer: bool):
    """Vetores de outras pessoas falando. O áudio NÃO fica no disco.

    Só os embeddings são guardados (~1 KB por janela): eles bastam para remedir
    o limiar depois, e guardar a voz de terceiros para sempre seria coletar o
    que não é nosso. Mesma escolha do negativos.npy do wake word.
    """
    import numpy as np
    import soundfile as sf

    cache = config.LOCUTOR_DIR / "negativos.npz"
    if cache.exists() and not refazer:
        dados = np.load(cache, allow_pickle=False)
        print(f"  {len(dados['vetores'])} janelas de negativo em cache "
              f"({int(dados['locutores'])} locutores)")
        return dados["vetores"], int(dados["locutores"])

    print(f"  buscando {quantos} clipes de outros locutores (MLS português)...")
    linhas = _linhas_mls(quantos)
    if not linhas:
        return np.zeros((0, locutor.DIM), dtype="float32"), 0

    # Espalha entre os locutores em vez de esgotar o primeiro: um limiar medido
    # contra uma pessoa só não vale nada.
    por_locutor: dict = {}
    for linha in linhas:
        por_locutor.setdefault(linha["row"].get("speaker_id"), []).append(linha)
    ordem = [l for grupo in zip(*por_locutor.values()) for l in grupo]

    vetores = []
    with tempfile.TemporaryDirectory(prefix="oraculo-neg-") as tmp:
        for i, linha in enumerate(ordem[:quantos]):
            alvo = Path(tmp) / f"neg_{i:03d}.opus"
            try:
                with urllib.request.urlopen(
                        linha["row"]["audio"][0]["src"], timeout=120) as r:
                    alvo.write_bytes(r.read())
                wav, sr = sf.read(alvo, dtype="float32")
            except Exception as exc:  # noqa: BLE001
                print(f"  [aviso] clipe {i}: {exc}")
                continue
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != locutor.SAMPLERATE:
                print(f"  [aviso] clipe {i}: {sr} Hz, ignorado")
                continue
            for segundos in JANELAS:
                n = int(segundos * locutor.SAMPLERATE)
                for k in range(0, max(1, len(wav) - n + 1), n):
                    vetor = locutor.embedding(wav[k:k + n])
                    if vetor is not None:
                        vetores.append(vetor)
            alvo.unlink(missing_ok=True)
            if (i + 1) % 10 == 0:
                print(f"    {i + 1} clipes, {len(vetores)} janelas")

    vetores = np.array(vetores, dtype="float32")
    config.LOCUTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, vetores=vetores, locutores=np.int32(len(por_locutor)))
    print(f"  {len(vetores)} janelas de {len(por_locutor)} locutores "
          f"(o áudio foi descartado)")
    return vetores, len(por_locutor)


# ----------------------------- medição ---------------------------------------
def leave_one_out(vetores):
    """Similaridade de cada gravação sua com um perfil que NÃO a contém.

    Sem isso o número sai otimista de graça: o clipe está dentro do centróide
    contra o qual ele é medido. É a mesma armadilha do "separe por clipe, não
    por janela" do wake word, só que na avaliação.
    """
    import numpy as np

    scores = []
    for i in range(len(vetores)):
        centro = np.delete(vetores, i, axis=0).mean(axis=0)
        centro /= np.linalg.norm(centro)
        scores.append(float(vetores[i] @ centro))
    return np.array(scores)


def escolher_limiar(pos, neg) -> tuple[float, str]:
    """(limiar, explicação). Limiar no meio da folga; sem folga, devolve 0."""
    import numpy as np

    pior_pos = float(pos.min())
    pior_neg = float(neg.max()) if len(neg) else 0.0
    if pior_pos <= pior_neg:
        return 0.0, (f"sem folga: a sua pior gravação ({pior_pos:.3f}) pontua "
                     f"abaixo do negativo mais parecido ({pior_neg:.3f})")
    limiar = (pior_pos + pior_neg) / 2
    if limiar < LIMIAR_MINIMO:
        return 0.0, (f"folga baixa demais: o meio-termo daria {limiar:.3f}, "
                     f"abaixo do piso de {LIMIAR_MINIMO:.2f}")
    return limiar, (f"meio da folga [{pior_neg:.3f} .. {pior_pos:.3f}]")


def relatorio(pos, neg, limiar: float) -> None:
    """A tabela que justifica o número — e mostra o que ele custa."""
    import numpy as np

    print(f"\n  suas gravações (leave-one-out): n={len(pos)}  "
          f"média {pos.mean():.3f}  pior {pos.min():.3f}")
    if len(neg):
        print(f"  outros locutores:               n={len(neg)}  "
              f"média {neg.mean():.3f}  pior {neg.max():.3f}  "
              f"p99 {np.percentile(neg, 99):.3f}")
    print(f"\n  {'limiar':>8}  {'aceita você':>12}  {'aceita outros':>14}")
    candidatos = sorted({round(x, 2) for x in
                         list(np.arange(0.30, 0.75, 0.05)) + [limiar]})
    for cand in candidatos:
        recall = (pos >= cand).mean() * 100
        fp = (neg >= cand).mean() * 100 if len(neg) else float("nan")
        marca = "  <- escolhido" if abs(cand - limiar) < 1e-9 else ""
        print(f"  {cand:8.3f}  {recall:11.0f}%  {fp:13.2f}%{marca}")


# ----------------------------- main ------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gravar", type=int, metavar="N",
                   help="grava N frases suas e sai")
    p.add_argument("--baixar", action="store_true",
                   help="baixa só o modelo de timbre (26 MB) e sai")
    p.add_argument("--negativos", type=int, default=40, metavar="N",
                   help="clipes de outras pessoas para medir o limiar (padrão: 40)")
    p.add_argument("--refazer-negativos", action="store_true",
                   help="ignora o cache e baixa os negativos de novo")
    p.add_argument("--limiar", type=float, default=0.0,
                   help="força um limiar em vez do medido (não recomendado)")
    p.add_argument("--avaliar", action="store_true",
                   help="mede e mostra a tabela, sem gravar o perfil")
    p.add_argument("--esquecer", action="store_true",
                   help="apaga o perfil cadastrado e sai")
    args = p.parse_args()

    if args.esquecer:
        print("perfil apagado" if locutor.esquecer() else "não havia perfil")
        return 0

    print("\n=== 1. modelo de timbre ===")
    alvo = locutor.caminho_modelo()
    if alvo.exists():
        print(f"  {alvo.name}: já está aqui")
    else:
        alvo.parent.mkdir(parents=True, exist_ok=True)
        baixar(locutor.URL_MODELO, alvo)
    if args.baixar:
        return 0

    if args.gravar:
        return gravar(args.gravar)

    print("\n=== 2. suas gravações ===")
    pos_vetores, fontes = amostras_do_dono()
    for linha in fontes:
        print(f"  {linha}")
    if len(pos_vetores) < 8:
        print(f"\n  só {len(pos_vetores)} gravações — pouco para um perfil.")
        print("  .venv/bin/python tools/cadastrar_voz.py --gravar 8")
        return 1

    print("\n=== 3. outros locutores (para medir o limiar) ===")
    neg_vetores, n_locutores = negativos(args.negativos, args.refazer_negativos)
    if n_locutores < 4:
        print("  [aviso] poucos locutores — o limiar sai frouxo. Sem rede?")

    print("\n=== 4. limiar ===")
    import numpy as np

    pos = leave_one_out(pos_vetores)
    centro = pos_vetores.mean(axis=0)
    centro /= np.linalg.norm(centro)
    neg = neg_vetores @ centro if len(neg_vetores) else np.zeros(0)

    if args.limiar:
        limiar, porque = args.limiar, "forçado na linha de comando"
    else:
        limiar, porque = escolher_limiar(pos, neg)
    relatorio(pos, neg, limiar)

    if not limiar:
        print(f"\n  {porque}.")
        print("  Grave mais frases (--gravar 8) e tente de novo; se persistir, "
              "o microfone pode estar captando mal (audio.diagnostico_captura).")
        return 1
    print(f"\n  limiar {limiar:.3f} — {porque}")

    if args.avaliar:
        print("\n  (--avaliar: nada foi gravado)")
        return 0

    fonte = f"{len(pos_vetores)} gravações; {n_locutores} locutores de negativo"
    locutor.gravar_perfil(pos_vetores, limiar, fonte)
    print(f"\n  perfil gravado em {locutor.caminho_perfil()}")
    print("  Ligue com /dono dentro do Oráculo (ou LOCUTOR_ENABLED no config.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
