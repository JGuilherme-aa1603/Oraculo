"""Wake word — acordar o Oráculo pelo nome, sem apertar nada.

Por que existe: o VAD (core/vad.py) já sabe quando a fala termina, mas ainda é
preciso um Enter para começar. Aqui o microfone fica aberto e só a palavra
"Oráculo" abre um turno. Tudo que for dito sem o nome é descartado no ato.

**A garantia que define este módulo:** antes do gatilho, o áudio não vai para o
disco e não é transcrito. O portão é um classificador que devolve um número entre
0 e 1 — ele não produz texto, então não existe transcrição de conversa alheia em
lugar nenhum. Quem acumula e grava é o core/audio.py, e só depois de disparar.

A cadeia (três ONNX pequenos, no onnxruntime que já está no venv):

    áudio 80 ms → melspectrogram → 5 frames de 32 bandas
                → janela de 76 frames → embedding → vetor de 96
                → janela de 16 vetores → cabeça → score

Os dois primeiros são do openWakeWord (Apache-2.0, 2,4 MB somados); o embedding é
o `speech_embedding` do Google, treinado em muito áudio e **congelado**. Só a
cabeça é nossa, e é ela que o tools/treinar_wake.py produz.

Por que a inferência é feita à mão em vez de usar o pacote `openwakeword`: ele não
instala no Python 3.14 deste venv (arrasta tflite-runtime), e o que ele faz é
exatamente a cadeia acima. Mesma decisão do core/vad.py, pelo mesmo motivo.

Por que a cabeça é `.npz` e não ONNX: são três matrizes e três vieses. Carregar
isso com numpy é uma linha; exportar para ONNX exigiria o `skl2onnx` como
dependência de runtime para ler 200 KB de números. O treino usa scikit-learn, mas
o **runtime não** — nada de sklearn no caminho da conversa.
"""

import time
import unicodedata

import config

# Constantes da cadeia. Nenhuma é ajustável: fazem parte dos modelos, e foram
# confirmadas por get_inputs()/get_outputs(), não lidas de documentação.
SAMPLERATE = 16000
BLOCO_SAMPLES = 1280      # 80 ms — o passo do wake word
MEL_POR_BLOCO = 5         # frames de mel que cada bloco produz
MEL_BANDAS = 32
MEL_JANELA = 76           # frames de mel que o embedding exige de uma vez
EMB_DIM = 96
EMB_JANELA = 16           # embeddings que a cabeça vê (≈1,28 s)

# Aquecimento: 76 frames de mel (16 blocos) até o 1º embedding, mais 15 blocos
# para encher a janela da cabeça. ~2,5 s depois de abrir o microfone o detector
# começa a pontuar. Os buffers NÃO são pré-preenchidos com silêncio sintético de
# propósito: o treinador usa este mesmo Extrator, e qualquer diferença entre
# treino e runtime aqui viraria um viés invisível.
BLOCOS_AQUECIMENTO = 16 + EMB_JANELA - 1

MODELO_MEL = "melspectrogram.onnx"
MODELO_EMB = "embedding_model.onnx"
CABECA = "oraculo.npz"

# URL dos dois modelos de característica. Baixados uma única vez pelo treinador,
# nunca no caminho da conversa (invariante 2).
URL_MODELOS = ("https://github.com/dscripka/openWakeWord/releases/download/"
               "v0.5.1/{}")

_sessoes_cache = None


def _sessoes():
    """Sessões ONNX (mel, embedding), carregadas uma vez. Padrão de vad._sessao()."""
    global _sessoes_cache
    if _sessoes_cache is not None:
        return _sessoes_cache

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "Wake word indisponível: onnxruntime não está instalado.\n"
            "  .venv/bin/python -m pip install onnxruntime"
        ) from exc

    caminhos = {}
    for nome in (MODELO_MEL, MODELO_EMB):
        caminho = config.WAKE_DIR / nome
        if not caminho.exists():
            raise RuntimeError(
                f"Wake word indisponível: falta {nome}.\n"
                f"  .venv/bin/python tools/treinar_wake.py --baixar-modelos"
            )
        caminhos[nome] = str(caminho)

    opcoes = ort.SessionOptions()
    # Um thread por sessão: o detector roda continuamente em segundo plano e
    # não pode disputar núcleos com o Ollama. São modelos de 1 MB — paralelizar
    # não acelera nada e só aumenta a contenção.
    opcoes.intra_op_num_threads = 1
    opcoes.inter_op_num_threads = 1

    mel = ort.InferenceSession(caminhos[MODELO_MEL], opcoes,
                               providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(caminhos[MODELO_EMB], opcoes,
                               providers=["CPUExecutionProvider"])

    # Conferido, não assumido — mesma disciplina do VAD, que já pagou dividendo.
    esperado = {mel: ("input",), emb: ("input_1",)}
    for sessao, nomes in esperado.items():
        reais = tuple(e.name for e in sessao.get_inputs())
        if reais != nomes:
            raise RuntimeError(
                f"Wake word indisponível: modelo mudou de assinatura "
                f"(esperado {nomes}, encontrado {reais})."
            )

    _sessoes_cache = (mel, emb)
    return _sessoes_cache


def disponivel() -> bool:
    """True se dá para usar o backend configurado.

    No backend "onnx" exige os dois modelos de característica **e** a cabeça
    treinada — sem ela não há o que detectar. Carrega tudo aqui para que a falta
    apareça no /despertar, e não no meio de uma escuta.
    """
    if config.WAKE_BACKEND == "transcricao":
        from core import stt, vad

        return stt.available() and vad.disponivel()
    try:
        _sessoes()
        _confere_palavra(Cabeca.carregar())
    except Exception:  # noqa: BLE001
        return False
    return True


def _confere_palavra(cabeca: "Cabeca") -> None:
    """Recusa uma cabeça treinada para outra palavra.

    Mudar `WAKE_WORD` no config não retreina nada. Sem esta conferência a
    interface anunciaria a palavra nova enquanto o detector continuaria ouvindo
    a antiga — e o usuário ficaria chamando um nome que ninguém escuta.
    """
    if _normaliza(cabeca.palavra) != _normaliza(config.WAKE_WORD):
        raise RuntimeError(
            f'O modelo treinado responde a "{cabeca.palavra}", mas o config '
            f'pede "{config.WAKE_WORD}".\n'
            f"  Treine de novo: .venv/bin/python tools/treinar_wake.py"
        )


def motivo_indisponivel() -> str:
    """Explica por que `disponivel()` deu False, para o /despertar poder dizer."""
    if config.WAKE_BACKEND == "transcricao":
        from core import stt, vad

        if not stt.available():
            return "o motor de transcrição não está disponível."
        if not vad.disponivel():
            return "o VAD não está disponível."
        return ""
    try:
        _sessoes()
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    try:
        _confere_palavra(Cabeca.carregar())
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return ""


class Extrator:
    """Áudio → janela de (16, 96) características. Compartilhado com o treinador.

    É de propósito que treino e runtime passem pelo mesmo código: qualquer
    diferença de janelamento ou de normalização entre os dois produziria um
    modelo que acerta no teste e erra no microfone.
    """

    def __init__(self) -> None:
        self._mel_s, self._emb_s = _sessoes()
        self.reset()

    def reset(self) -> None:
        """Esquece o áudio anterior. Chamar entre gravações independentes."""
        import numpy as np

        self._mel = np.zeros((0, MEL_BANDAS), dtype="float32")
        self._emb = np.zeros((EMB_JANELA, EMB_DIM), dtype="float32")
        self._n_emb = 0

    def push(self, bloco):
        """Consome 80 ms de áudio; devolve a janela (16, 96) ou None se aquecendo.

        `bloco` é float32 mono em [-1, 1] com BLOCO_SAMPLES amostras.
        """
        import numpy as np

        if bloco.shape[0] != BLOCO_SAMPLES:
            raise ValueError(
                f"bloco deve ter {BLOCO_SAMPLES} amostras, veio com {bloco.shape[0]}"
            )

        saida = self._mel_s.run(
            None, {"input": bloco[None, :].astype("float32")})[0]
        # A transformação (x/10)+2 é parte do contrato do openWakeWord: o
        # embedding foi treinado nessa escala. Sem ela o vetor de 96 sai sem
        # significado — e, pior, sem erro nenhum.
        frames = saida.reshape(-1, MEL_BANDAS) / 10.0 + 2.0
        self._mel = np.concatenate([self._mel, frames])[-MEL_JANELA:]

        if self._mel.shape[0] < MEL_JANELA:
            return None

        vetor = self._emb_s.run(
            None, {"input_1": self._mel[None, :, :, None].astype("float32")})[0]
        self._emb = np.roll(self._emb, -1, axis=0)
        self._emb[-1] = vetor.reshape(EMB_DIM)
        self._n_emb += 1

        if self._n_emb < EMB_JANELA:
            return None
        return self._emb.copy()


class Cabeca:
    """O classificador treinado: MLP pequeno, forward em numpy puro.

    Guardado como .npz (matrizes + normalização + limiar). Sem sklearn, sem
    ONNX — o treino que use o que quiser, o runtime só multiplica matrizes.
    """

    def __init__(self, dados) -> None:
        self._camadas = []
        i = 0
        while f"W{i}" in dados:
            self._camadas.append((dados[f"W{i}"], dados[f"b{i}"]))
            i += 1
        if not self._camadas:
            raise RuntimeError("Cabeça do wake word sem pesos (.npz corrompido?).")
        self.media = dados["media"]
        self.desvio = dados["desvio"]
        self.limiar = float(dados["limiar"])
        self.palavra = str(dados["palavra"]) if "palavra" in dados else config.WAKE_WORD

    @classmethod
    def carregar(cls, caminho=None):
        """Lê a cabeça de ~/.oraculo/wake/oraculo.npz."""
        import numpy as np

        caminho = caminho or (config.WAKE_DIR / CABECA)
        if not caminho.exists():
            raise RuntimeError(
                f"Wake word não treinada: falta {caminho}.\n"
                f"  .venv/bin/python tools/treinar_wake.py --gravar 40"
            )
        return cls(np.load(caminho))

    def prob(self, janela) -> float:
        """Probabilidade (0..1) de a janela conter a palavra de despertar."""
        import numpy as np

        x = ((janela.reshape(1, -1) - self.media) / self.desvio).astype("float32")
        for W, b in self._camadas[:-1]:
            x = np.maximum(0.0, x @ W + b)          # ReLU
        z = float((x @ self._camadas[-1][0] + self._camadas[-1][1])[0, 0])
        return 1.0 / (1.0 + np.exp(-z))             # sigmoide


class Detector:
    """Extrator + cabeça + refratário. Uma instância por sessão de escuta."""

    def __init__(self, cabeca: "Cabeca | None" = None) -> None:
        self.extrator = Extrator()
        self.cabeca = cabeca or Cabeca.carregar()
        self.limiar = config.WAKE_THRESHOLD or self.cabeca.limiar
        self._refratario = config.WAKE_REFRACTORY_MS / 1000.0
        self.reset()

    def reset(self) -> None:
        self.extrator.reset()
        self.score = 0.0
        self._ultimo_disparo = 0.0

    def push(self, bloco) -> bool:
        """Consome 80 ms; True quando a palavra acabou de ser reconhecida.

        O refratário existe porque a janela desliza: um único "Oráculo" fica
        acima do limiar por vários blocos seguidos e dispararia meia dúzia de
        vezes sem ele.
        """
        janela = self.extrator.push(bloco)
        if janela is None:
            self.score = 0.0
            return False

        self.score = self.cabeca.prob(janela)
        if self.score < self.limiar:
            return False

        agora = time.monotonic()
        if agora - self._ultimo_disparo < self._refratario:
            return False
        self._ultimo_disparo = agora
        return True


class Anel:
    """Buffer circular do áudio recente, em memória.

    Serve para o gatilho não comer o começo da frase: quando o detector dispara,
    a palavra já passou, e sem o pré-roll o recorte começaria depois dela.

    É este anel que sustenta a garantia de privacidade — o áudio de quem não
    chamou o nome vive aqui por ~2 s e é sobrescrito, sem nunca tocar o disco.
    """

    def __init__(self, ms: int | None = None) -> None:
        import numpy as np

        ms = config.WAKE_PREROLL_MS if ms is None else ms
        self._n = max(BLOCO_SAMPLES, int(SAMPLERATE * ms / 1000))
        self._buf = np.zeros(self._n, dtype="float32")
        self._cheio = False
        self._pos = 0

    def push(self, bloco) -> None:
        import numpy as np

        n = bloco.shape[0]
        fim = self._pos + n
        if fim <= self._n:
            self._buf[self._pos:fim] = bloco
            # `>=`, não `>`: fechar exatamente no fim do buffer também o enche.
            # Com `>`, `_pos` voltaria a 0 com `_cheio` False e `conteudo()`
            # devolveria vazio — o pré-roll sumiria sem erro nenhum.
            if fim >= self._n:
                self._cheio = True
        else:                                  # dá a volta
            corte = self._n - self._pos
            self._buf[self._pos:] = bloco[:corte]
            self._buf[:fim - self._n] = bloco[corte:]
            self._cheio = True
        self._pos = fim % self._n

    def conteudo(self):
        """O áudio guardado, em ordem cronológica."""
        import numpy as np

        if not self._cheio:
            return self._buf[:self._pos].copy()
        return np.concatenate([self._buf[self._pos:], self._buf[:self._pos]])


# ----------------------------- Texto -----------------------------------------
def _normaliza(texto: str) -> str:
    """Minúsculas, sem acento e sem pontuação — para comparar transcrição.

    O ASR devolve "Oráculo," com acento e vírgula, mas também "oraculo" sem
    acento dependendo do motor. Comparar a forma crua erraria por detalhe.
    """
    return "".join(c for c, _ in _normaliza_com_mapa(texto)).strip()


def _normaliza_com_mapa(texto: str):
    """Normaliza preservando a origem de cada caractere.

    Devolve pares (caractere normalizado, índice no texto cru). O mapa existe
    porque `remove_nome` precisa cortar o **texto original** — contar tokens no
    texto normalizado não serve: a pontuação vira espaço, então "Oráculo,que"
    é um token cru e dois normalizados.
    """
    saida = []
    for i, bruto in enumerate(texto):
        for c in unicodedata.normalize("NFKD", bruto):
            if unicodedata.combining(c):
                continue
            saida.append((c.lower() if c.isalnum() or c.isspace() else " ", i))
    return saida


def _tokens_com_fim(texto: str):
    """Tokens normalizados com o índice, no texto cru, logo após cada um."""
    pares = _normaliza_com_mapa(texto)
    tokens, atual = [], []
    for pos, (c, _origem) in enumerate(pares):
        if c.isspace():
            if atual:
                tokens.append(("".join(atual), pares[pos - 1][1] + 1))
                atual = []
        else:
            atual.append(c)
    if atual:
        tokens.append(("".join(atual), pares[-1][1] + 1))
    return tokens


def _parecido(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def posicao_nome(texto: str) -> int:
    """Índice, no texto cru, logo após a palavra de despertar. -1 se não achou.

    Procura só no começo (`WAKE_MAX_TOKENS` tokens): "Oráculo, que horas são"
    acorda, mas "eu consultei o oráculo ontem" não. Aceita a forma colada e a
    partida em dois ("o ráculo"), que é como o ASR às vezes quebra a palavra.

    Limite conhecido e aceito: "o oráculo de Delfos" acorda. Por texto ele é
    indistinguível de "Oráculo, de Delfos me fale" — a palavra ocupa a mesma
    posição —, e o "Ô" vocativo perde o acento na normalização, virando o mesmo
    token do artigo "o". Inventar regra para separar os dois quebraria a chamada
    legítima; quem quiser o comportamento estrito põe `WAKE_MAX_TOKENS = 1`, que
    exige o nome como primeira palavra (e aí "Ei Oráculo" deixa de valer).
    """
    alvo = _normaliza(config.WAKE_WORD)
    tokens = _tokens_com_fim(texto)
    limite = min(len(tokens), config.WAKE_MAX_TOKENS)

    # O plural é a colisão previsível em português: "os oráculos da Grécia" tem o
    # nome logo no começo e casa por similaridade (0,93 direto, 0,82 no bigrama).
    # Ninguém chama o assistente no plural, então a forma com -s é excluída antes
    # da comparação difusa — nos dois caminhos, senão o bigrama a deixa passar.
    def chama(texto_token: str) -> bool:
        if texto_token.endswith(alvo + "s"):
            return False
        return _parecido(texto_token, alvo) >= config.WAKE_FUZZY

    for i in range(limite):
        if chama(tokens[i][0]):
            return tokens[i][1]
        # Bigrama: existe só para o caso de o ASR PARTIR a palavra ("o ráculo").
        # Duas restrições, e ambas foram necessárias:
        #   - o par tem que caber na janela, senão "consultei o oráculo" casaria
        #     por "o"+"oraculo", alcançando um token fora do começo da frase;
        #   - o par tem que ter quase o comprimento da palavra, senão qualquer
        #     token colado a ela casa por conter a palavra inteira
        #     ("oraculos"+"da" = 0,82).
        if i + 1 >= limite:
            continue
        par = tokens[i][0] + tokens[i + 1][0]
        if abs(len(par) - len(alvo)) <= 1 and chama(par):
            return tokens[i + 1][1]
    return -1


def contem_nome(texto: str) -> bool:
    """True se a fala começa chamando o Oráculo."""
    return posicao_nome(texto) >= 0


def remove_nome(texto: str) -> str:
    """Devolve o pedido sem o "Oráculo," da frente.

    Preserva o texto original (acentos, pontuação, caixa): a normalização serve
    só para *achar* o nome, nunca para substituir o que vai ao modelo.
    """
    corte = posicao_nome(texto)
    if corte < 0:
        return texto.strip()
    return texto[corte:].lstrip(",.;:!?-— \t").strip()
