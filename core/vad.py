"""VAD — detecção de atividade de voz sobre o fluxo do microfone.

Por que existe: até a Fase 2 a gravação era push-to-talk puro — o usuário
sinalizava o fim com um segundo Enter. Com o VAD o Oráculo percebe sozinho que a
fala terminou, e o silêncio das pontas não vai para o STT.

De onde vem o modelo: o `faster-whisper`, que já é dependência do STT, empacota o
`silero_vad_v6.onnx` (1,2 MB) nos próprios assets. Usar esse arquivo mantém tudo
offline (nada é baixado, nem na primeira execução) e não acrescenta uma linha ao
`requirements.txt` — o que importa porque o venv está em Python 3.14, onde os
pacotes usuais de VAD (`silero-vad`, `webrtcvad`) não instalam.

Por que não usar o `SileroVADModel.__call__` do próprio faster-whisper: ele zera o
estado `h`/`c` da LSTM a cada chamada e exige o áudio inteiro de uma vez — é uma
API batch, feita para pós-processar gravação pronta. Aqui a decisão precisa sair
frame a frame com o microfone ainda aberto, então este módulo fala direto com a
`InferenceSession` e carrega `h`/`c` de um frame para o outro. Sem isso cada frame
seria julgado sem contexto e a probabilidade ficaria inútil.
"""

import time

import config

# O Silero v6 a 16 kHz trabalha em frames fixos de 512 amostras (32 ms), sempre
# precedidos pelas 64 últimas amostras do frame anterior. A entrada do ONNX é a
# concatenação dos dois: 576 amostras. Não são ajustáveis — fazem parte do modelo.
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
SAMPLERATE = 16000
FRAME_MS = FRAME_SAMPLES / SAMPLERATE * 1000.0   # 32.0

# Estados devolvidos por Segmentador.push().
AGUARDANDO = "aguardando"   # ainda não ouviu ninguém
FALANDO = "falando"         # fala em andamento
FIM = "fim"                 # fala terminou (ou bateu o teto) — pode transcrever
EXPIRADO = "expirado"       # ninguém falou dentro de VAD_START_TIMEOUT

# Nomes das entradas do silero_vad_v6.onnx. Conferidos em _sessao() em vez de
# assumidos: o v5 tinha outra assinatura (`state`/`sr`) e uma atualização do
# faster-whisper pode trocar o arquivo de novo. Melhor degradar com aviso claro
# do que estourar no meio de uma gravação.
_ENTRADAS = ("input", "h", "c")

_sessao_cache = None


def _sessao():
    """Sessão ONNX do Silero, carregada uma vez e reaproveitada.

    Mesmo padrão de cache global do `stt._get_whisper()`.
    """
    global _sessao_cache
    if _sessao_cache is not None:
        return _sessao_cache

    try:
        from faster_whisper.vad import get_vad_model
    except ImportError as exc:
        raise RuntimeError(
            "VAD indisponível: o faster-whisper não está instalado.\n"
            "  .venv/bin/python -m pip install faster-whisper"
        ) from exc

    sessao = get_vad_model().session
    nomes = tuple(entrada.name for entrada in sessao.get_inputs())
    if nomes != _ENTRADAS:
        raise RuntimeError(
            f"VAD indisponível: o modelo do faster-whisper mudou de assinatura "
            f"(esperado {_ENTRADAS}, encontrado {nomes}). Use /vad para desligar "
            f"e voltar ao push-to-talk."
        )
    _sessao_cache = sessao
    return _sessao_cache


def disponivel() -> bool:
    """True se o VAD pode ser usado. Espelha `stt.available()`.

    Carrega o modelo (barato: ~1 MB na CPU) para que uma incompatibilidade
    apareça aqui, e não no meio do turno.
    """
    try:
        _sessao()
    except Exception:  # noqa: BLE001
        return False
    return True


class Detector:
    """Probabilidade de fala por frame, com o estado da LSTM preservado."""

    def __init__(self) -> None:
        self._sessao = _sessao()
        self.reset()

    def reset(self) -> None:
        """Esquece a fala anterior. Chamar entre gravações independentes."""
        import numpy as np

        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._contexto = np.zeros(CONTEXT_SAMPLES, dtype="float32")

    def prob(self, frame) -> float:
        """Probabilidade de fala (0..1) de um frame de FRAME_SAMPLES amostras.

        `frame` é float32 mono em [-1, 1].
        """
        import numpy as np

        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(
                f"frame deve ter {FRAME_SAMPLES} amostras, veio com {frame.shape[0]}"
            )
        entrada = np.concatenate([self._contexto, frame])[None, :]
        probs, self._h, self._c = self._sessao.run(
            None, {"input": entrada, "h": self._h, "c": self._c}
        )
        self._contexto = frame[-CONTEXT_SAMPLES:].copy()
        return float(probs[0])


class Segmentador:
    """Máquina de estados que decide quando a fala começou e terminou.

    A histerese é o ponto todo: o limiar cru oscila entre palavras, e cortar na
    primeira amostra abaixo dele picotaria a frase em pedaços. Então só o
    silêncio *contínuo* de `VAD_SILENCE_MS` encerra, e uma rajada mais curta que
    `VAD_MIN_SPEECH_MS` (tosse, clique de tecla, batida na mesa) é descartada sem
    encerrar a escuta — volta a esperar em vez de mandar ruído para o STT.
    """

    def __init__(self, detector: Detector | None = None,
                 timeout_s: float | None = None) -> None:
        self.detector = detector or Detector()
        self.detector.reset()

        # Fotografados na construção: /vad e ajustes em runtime valem a partir da
        # próxima gravação, e não mudam no meio de uma.
        self._limiar = config.VAD_THRESHOLD
        self._frames_silencio_fim = max(1, round(config.VAD_SILENCE_MS / FRAME_MS))
        self._frames_fala_min = max(1, round(config.VAD_MIN_SPEECH_MS / FRAME_MS))
        self._frames_max = max(1, round(config.VAD_MAX_SECONDS * 1000 / FRAME_MS))
        # `timeout_s` sobrepõe a espera inicial. Serve à escuta de continuação do
        # wake word ("você disse só o nome"), que merece uma janela diferente da
        # gravação comum sem obrigar a mexer no config global.
        self._timeout_s = (config.VAD_START_TIMEOUT if timeout_s is None
                           else timeout_s)
        self._frames_timeout = max(1, round(self._timeout_s * 1000 / FRAME_MS))
        self._frames_pad = max(0, round(config.VAD_PAD_MS / FRAME_MS))

        self.estado = AGUARDANDO
        self._total = 0             # frames processados desde o início
        self._frames_fala = 0       # frames com fala na rajada atual
        self._silencio = 0          # frames de silêncio contínuo na rajada atual
        self._inicio = 0            # 1º frame da rajada atual
        self._fim = 0               # último frame com fala (exclusivo)
        self._t0 = None             # relógio, ligado no 1º frame recebido

    def esperando_ha(self) -> float:
        """Segundos de relógio desde o primeiro frame. 0 antes dele."""
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    def push(self, frame) -> str:
        """Consome um frame e devolve o estado resultante."""
        if self._t0 is None:
            # O relógio começa no primeiro frame, não na construção: abrir o
            # dispositivo de áudio pode levar ~1s na primeira vez, e esse tempo
            # não é do usuário pensando no que falar.
            self._t0 = time.monotonic()

        prob = self.detector.prob(frame)
        self._total += 1
        tem_fala = prob >= self._limiar

        if self.estado == AGUARDANDO:
            if tem_fala:
                self.estado = FALANDO
                self._inicio = self._total - 1
                self._fim = self._total
                self._frames_fala = 1
                self._silencio = 0
            elif self.expirou():
                self.estado = EXPIRADO
            return self.estado

        if self.estado == FALANDO:
            if tem_fala:
                self._frames_fala += 1
                self._silencio = 0
                self._fim = self._total
            else:
                self._silencio += 1
                if self._silencio >= self._frames_silencio_fim:
                    if self._frames_fala >= self._frames_fala_min:
                        self.estado = FIM
                    else:
                        # Rajada curta demais: provavelmente não foi fala.
                        # Descarta e continua ouvindo (o timeout segue correndo).
                        self.estado = AGUARDANDO
                        self._frames_fala = 0
                        self._silencio = 0
            # Teto de segurança: sem ele um ruído contínuo gravaria para sempre.
            if self.estado == FALANDO and self._total >= self._frames_max:
                self.estado = FIM

        return self.estado

    def tick(self) -> str:
        """Reavalia o timeout sem consumir frame.

        Serve para quem está lendo de uma fila e ficou sem blocos: o relógio de
        parede continua correndo mesmo quando o áudio não chega.
        """
        if self.estado == AGUARDANDO and self.expirou():
            self.estado = EXPIRADO
        return self.estado

    def expirou(self) -> bool:
        """True se ninguém começou a falar dentro de `VAD_START_TIMEOUT`.

        Dois relógios, e vale o que estourar primeiro. O de frames descreve o
        áudio e é determinístico (é ele que decide num teste offline, onde não se
        espera em tempo real). O de parede descreve o que o usuário sente: se o
        dispositivo entregar os blocos atrasado — a primeira abertura do PipeWire
        custa perto de um segundo —, contar só frames faria a espera de 8s durar
        12s na cara de quem está esperando.
        """
        if self.estado not in (AGUARDANDO, EXPIRADO):
            return False
        return (self._total >= self._frames_timeout
                or self.esperando_ha() >= self._timeout_s)

    @property
    def duracao_fala(self) -> float:
        """Segundos de fala efetivamente detectados (sem as pausas do fim)."""
        return max(0, self._fim - self._inicio) * FRAME_MS / 1000.0

    def recorte(self, total_amostras: int) -> tuple[int, int]:
        """Intervalo [início, fim) em amostras, com a margem de `VAD_PAD_MS`.

        A margem devolve o ataque da primeira sílaba e a cauda da última, que o
        modelo tende a marcar como silêncio — sem ela o STT recebe a palavra
        cortada. Os limites são presos ao tamanho real do buffer.
        """
        inicio = max(0, (self._inicio - self._frames_pad) * FRAME_SAMPLES)
        fim = min(total_amostras, (self._fim + self._frames_pad) * FRAME_SAMPLES)
        if fim <= inicio:
            return 0, total_amostras
        return inicio, fim
