"""Indexa as notas do Obsidian para o Oráculo consultar. Roda à mão.

    .venv/bin/python tools/indexar_vault.py              # indexa o que mudou
    .venv/bin/python tools/indexar_vault.py --status     # o que está no índice
    .venv/bin/python tools/indexar_vault.py --buscar "como o VAD funciona"
    .venv/bin/python tools/indexar_vault.py --calibrar   # mede o RAG_MIN_SCORE

Este script **não é importado pelo app**: o runtime só lê o `indice.npz` que sai
daqui (ver core/rag.py). Dentro da conversa o mesmo trabalho está em `/indexar`.

## Por que o --buscar existe

É o instrumento antes da dedução. Quando o Oráculo responde mal com as notas
ligadas, há dois suspeitos muito diferentes — a busca trouxe o trecho errado, ou
trouxe o certo e o modelo não usou — e olhar a resposta não distingue os dois.
O `--buscar` mostra exatamente o que teria ido para o contexto, sem gastar o LLM.

## Por que o --calibrar não grava o limiar

O treinador do wake word e o cadastro de voz gravam o limiar que mediram, porque
lá existe um corpus de negativos REAIS (horas de fala que não é a palavra,
locutores que não são o dono). Aqui o negativo é "pergunta que o vault não
responde", e disso não há corpus: o que este comando monta é uma amostra
razoável, não uma medida com a mesma autoridade. Então ele reporta a folga e o
ponto médio, e quem decide é você, no `config.py`. Prometer mais precisão do que
a medição tem seria o mesmo erro dos negativos sintéticos da verificação de voz.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                    # noqa: E402
from core import rag                             # noqa: E402

def carregar_perguntas(origem: Path) -> tuple[list[str], list[str]]:
    """Lê o arquivo de calibração em duas listas: (positivas, negativas).

    As seções são marcadas por `[positivas]` e `[negativas]`. Sem marcador
    nenhum, tudo conta como positiva — é o formato antigo, e quebrar arquivos
    que já existem para introduzir uma seção seria trocar um incômodo por uma
    calibração silenciosamente errada.
    """
    positivas: list[str] = []
    negativas: list[str] = []
    atual = positivas
    for linha in origem.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        marcador = linha.lower().strip("[]") if linha.startswith("[") else ""
        if marcador in {"positivas", "negativas"}:
            atual = positivas if marcador == "positivas" else negativas
            continue
        atual.append(linha)
    return positivas, negativas


# Negativos embutidos: usados só quando o arquivo do usuário não traz seção
# própria. O melhor score de cada um é um falso positivo medido — é o que diz
# onde o piso precisa ficar.
FORA_DO_DOMINIO = (
    "qual a receita de bacalhau à Brás",
    "quem ganhou a Copa do Mundo de 1994",
    "como se conjuga o verbo hablar em espanhol",
    "qual a distância da Terra até Netuno",
    "quais os sintomas da dengue hemorrágica",
    "como trocar a correia dentada de um Fiat Uno",
    "o que é a Batalha de Stalingrado",
    "qual a diferença entre uma viola e um violoncelo",
    "qual a capital da Mongólia",
    "como se faz pão de queijo mineiro",
)


def _barra(feito: int, total: int) -> None:
    largura = 28
    cheio = int(largura * feito / total) if total else largura
    print(f"\r  [{'#' * cheio}{'.' * (largura - cheio)}] {feito}/{total}",
          end="", flush=True)


def indexar(forcar: bool) -> int:
    raiz = rag.vault_dir()
    print(f"Vault: {raiz}")
    if not forcar and Path(config.RAG_INDEX_FILE).exists():
        try:
            atual = rag.Indice.carregar()
        except rag.RagError:
            atual = None                          # índice velho/incompatível: refaz
        if atual is not None and not atual.desatualizado(raiz):
            print(f"Índice já está em dia: {atual.meta['trechos']} trechos de "
                  f"{atual.meta['notas']} notas ({atual.meta['criado_em']}).")
            print("Use --forcar para reindexar assim mesmo.")
            return 0

    notas = rag.listar_notas(raiz)
    print(f"Lendo {len(notas)} notas...")
    t0 = time.monotonic()
    indice = rag.construir(raiz, progresso=_barra)
    print()
    caminho = indice.gravar()
    tamanho = caminho.stat().st_size / 1024
    print(f"Indexadas {indice.meta['notas']} notas em "
          f"{indice.meta['trechos']} trechos "
          f"({time.monotonic() - t0:.1f}s, {tamanho:.0f} KB em {caminho}).")
    print(f"Ligue com /notas dentro da conversa "
          f"(ou RAG_ENABLED = True no config.py).")
    return 0


def status() -> int:
    try:
        indice = rag.Indice.carregar()
    except rag.RagError as exc:
        print(f"Sem índice utilizável: {exc}")
        return 1
    m = indice.meta
    print(f"Vault:    {m['vault']}")
    print(f"Modelo:   {m['modelo']}  ({m['dims']} dimensões)")
    print(f"Conteúdo: {m['trechos']} trechos de {m['notas']} notas")
    print(f"Criado:   {m['criado_em']}")
    print(f"Arquivo:  {config.RAG_INDEX_FILE} "
          f"({Path(config.RAG_INDEX_FILE).stat().st_size / 1024:.0f} KB)")
    if indice.desatualizado():
        print("\nO vault MUDOU desde a indexação — rode o comando sem argumentos.")
    else:
        print("\nEm dia com o vault.")
    return 0


def buscar(pergunta: str, k: int) -> int:
    indice = rag.Indice.carregar()
    t0 = time.monotonic()
    # minimo=-1 mostra também o que o limiar barraria: ver o trecho rejeitado
    # com 0,52 é o que diz se o limiar está apertado demais ou se a nota
    # simplesmente não existe.
    achados = indice.buscar(pergunta, k=k, minimo=-1.0)
    ms = (time.monotonic() - t0) * 1000
    print(f'"{pergunta}"  ({ms:.0f} ms, limiar atual {config.RAG_MIN_SCORE})\n')
    for achado in achados:
        marca = "  " if achado.score >= config.RAG_MIN_SCORE else "x "
        print(f"{marca}{achado.score:.3f}  {achado.trecho.fonte}")
        print(f"          {achado.trecho.caminho}")
        corpo = achado.trecho.texto.split("\n", 1)[-1].replace("\n", " ")
        print(f"          {corpo[:150]}...\n")
    if not achados:
        print("  (índice vazio)")
    return 0


def calibrar(caminho_perguntas: str | None) -> int:
    """Mede a folga entre perguntas que o vault responde e perguntas que não.

    ## Por que o positivo tem que ser uma PERGUNTA SUA

    A primeira versão deste comando montava os positivos sozinha: sorteava
    trechos e usava uma frase do corpo deles como consulta. Deu um número lindo
    e errado — mediana 0,808 —, porque uma frase copiada da nota é quase uma
    duplicata dela, e não se parece nada com a pergunta que uma pessoa digita.
    O limiar que saiu daquela medição (0,71) rejeitava "por que o VAD não roda
    no callback do PortAudio", que pontua 0,708 e é o caso de uso principal.

    É a mesma armadilha do wake word, onde separar treino e avaliação por
    janela em vez de por clipe dava 100% de recall e não queria dizer nada: um
    positivo fácil demais mede a facilidade, não o sistema. Por isso as
    perguntas vêm de um arquivo seu, e sem ele este comando não inventa um
    número — só mede o lado que ele consegue medir sozinho, o dos negativos.
    """
    import numpy as np

    indice = rag.Indice.carregar()

    def melhor(pergunta: str) -> tuple[float, str]:
        achados = indice.buscar(pergunta, k=1, minimo=-1.0)
        return (achados[0].score, achados[0].trecho.fonte) if achados else (0.0, "-")

    origem = Path(caminho_perguntas) if caminho_perguntas else config.RAG_PERGUNTAS_FILE
    perguntas: list[str] = []
    fora: list[str] = list(FORA_DO_DOMINIO)
    de_onde = "embutidos no script"
    if origem.exists():
        perguntas, do_arquivo = carregar_perguntas(origem)
        if do_arquivo:
            fora, de_onde = do_arquivo, origem.name

    # Rótulo repetido é a falha que envenena os dois lados de uma vez: a mesma
    # pergunta contando como positiva e negativa torna a folga insolúvel e o
    # relatório, mentira. Foi o erro de "oráculos de Delfos" no wake word.
    repetidas = set(map(str.lower, perguntas)) & set(map(str.lower, fora))
    if repetidas:
        print(f"Erro em {origem}: {len(repetidas)} pergunta(s) nas DUAS seções.")
        for p in sorted(repetidas):
            print(f"    \"{p}\"")
        print("Decida de que lado ela fica — o vault responde, ou não responde.")
        return 1

    negativos = [(*melhor(p), p) for p in fora]
    neg = np.array([s for s, _, _ in negativos])
    print(f"Negativos: {len(negativos)} perguntas fora do domínio ({de_onde})")
    print(f"  min {neg.min():.3f}   mediana {np.median(neg):.3f}   "
          f"MAX {neg.max():.3f}")
    for score, fonte, pergunta in sorted(negativos, reverse=True)[:3]:
        print(f"    {score:.3f}  \"{pergunta}\" → {fonte[:52]}")
    if len(negativos) < 25:
        # Medido neste projeto: com 10 negativos a folga dava +0,008 e zero
        # falso positivo; com 30 ela virou negativa. O aviso existe para o
        # próximo não repetir a conclusão otimista que essa amostra produz.
        print(f"  AVISO: {len(negativos)} negativos é pouco para enxergar a "
              f"cauda.\n  A folga que sair daqui será otimista — acrescente "
              f"perguntas fora do domínio.")

    if not perguntas:
        falta = "não existe" if not origem.exists() else "não tem perguntas positivas"
        print(f"\nSem o lado positivo: {origem} {falta}.")
        print("Escreva nele, sob [positivas], uma pergunta por linha — do jeito "
              "que você\nas digitaria, sobre coisas que as suas notas respondem. "
              "Sem elas não dá\npara saber onde o piso pode ficar sem cortar o "
              "que importa.")
        print(f"\nCom o que dá para medir agora: o limiar precisa ficar ACIMA "
              f"de {neg.max():.3f}.")
        return 1

    positivos = [(*melhor(p), p) for p in perguntas]
    pos = np.array([s for s, _, _ in positivos])
    print(f"\nPositivos: {len(positivos)} perguntas de {origem.name}")
    print(f"  MIN {pos.min():.3f}   mediana {np.median(pos):.3f}   "
          f"max {pos.max():.3f}")
    for score, fonte, pergunta in sorted(positivos)[:3]:
        print(f"    {score:.3f}  \"{pergunta}\" → {fonte[:52]}")

    teto_neg = float(neg.max())
    piso_pos = float(pos.min())
    print()
    if teto_neg >= piso_pos:
        # Sem folga não existe limiar honesto. Em vez de escolher um que
        # sacrifica um lado em silêncio, o comando mostra o custo dos dois.
        perdidas = int((pos <= teto_neg).sum())
        print(f"SEM FOLGA LIMPA: o pior negativo ({teto_neg:.3f}) alcança a "
              f"pior pergunta legítima ({piso_pos:.3f}).")
        print(f"Um piso acima de {teto_neg:.3f} descartaria {perdidas} das "
              f"{len(pos)} perguntas suas.")
        acima = pos[pos > teto_neg]
        if len(acima):
            print(f"Sugestão conservadora: RAG_MIN_SCORE = "
                  f"{(teto_neg + float(acima.min())) / 2:.2f}  "
                  f"(perde as {perdidas} de baixo, barra todos os negativos)")
        print(f"Config atual: RAG_MIN_SCORE = {config.RAG_MIN_SCORE}")
        return 1
    meio = (teto_neg + piso_pos) / 2
    print(f"Folga: {teto_neg:.3f} (pior negativo) → {piso_pos:.3f} (pior positivo)")
    print(f"Ponto médio: RAG_MIN_SCORE = {meio:.2f}")
    print(f"Config atual: RAG_MIN_SCORE = {config.RAG_MIN_SCORE}")
    print("\nO meio da folga, não o extremo: encostar no pior negativo passa a "
          "rejeitar\npergunta legítima, que foi o erro do primeiro limiar do "
          "wake word.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--forcar", action="store_true",
                   help="reindexa mesmo que o vault não tenha mudado")
    p.add_argument("--status", action="store_true",
                   help="mostra o que está no índice e se está em dia")
    p.add_argument("--buscar", metavar="PERGUNTA",
                   help="mostra os trechos que essa pergunta traria")
    p.add_argument("--k", type=int, default=8, metavar="N",
                   help="quantos trechos o --buscar mostra (padrão: 8)")
    p.add_argument("--calibrar", action="store_true",
                   help="mede a folga e sugere o RAG_MIN_SCORE")
    p.add_argument("--perguntas", metavar="ARQUIVO",
                   help=f"perguntas de verdade para a calibração, uma por "
                        f"linha (padrão: {config.RAG_PERGUNTAS_FILE})")
    args = p.parse_args()

    try:
        if args.status:
            return status()
        if args.buscar:
            return buscar(args.buscar, args.k)
        if args.calibrar:
            return calibrar(args.perguntas)
        return indexar(args.forcar)
    except rag.RagError as exc:
        print(f"Erro: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrompido.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
