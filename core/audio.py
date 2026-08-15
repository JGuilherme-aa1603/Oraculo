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

    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1,
                   device=config.INPUT_DEVICE)
    sd.wait()
    sf.write(path, audio, samplerate)
    return path


def record_ptt(samplerate: int = config.RECORD_SAMPLERATE,
               path: str = "/tmp/oraculo_in.wav",
               wait_stop=None) -> str:
    """Push-to-talk: grava até o usuário sinalizar o fim (sem duração fixa).

    O chamador deve avisar o usuário antes ("Enter para parar").

    `wait_stop` é a função que bloqueia até esse sinal chegar. O padrão é
    `input()`, que serve no modo inline. No modo tela cheia o stdin pertence ao
    prompt_toolkit — ler dali brigaria com ele —, então o chamador passa uma
    função que espera o Enter vindo da caixa de entrada.
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
        with sd.InputStream(samplerate=samplerate, channels=1,
                            device=config.INPUT_DEVICE, callback=callback):
            (wait_stop or input)()   # bloqueia até o sinal → encerra a gravação
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falha ao acessar o microfone: {exc}") from exc

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((1, 1), dtype="float32")
    audio = _normalize(audio)
    sf.write(path, audio, samplerate)
    return path


def record_vad(samplerate: int = config.RECORD_SAMPLERATE,
               path: str = "/tmp/oraculo_in.wav",
               on_state=None, start_timeout: float | None = None) -> str | None:
    """Grava até o VAD perceber que a fala terminou (sem Enter para parar).

    Devolve o caminho do WAV, ou None se ninguém falou dentro de
    `config.VAD_START_TIMEOUT` — nesse caso o chamador só avisa e volta ao prompt.

    A inferência roda nesta thread, não no callback do PortAudio: gastar ~1ms de
    ONNX dentro do callback atrasaria a entrega do próximo bloco e causaria
    estouro de buffer. O callback só empilha na fila; quem decide é o laço aqui.

    `on_state` (opcional) recebe cada mudança de estado do segmentador, para a UI
    poder mostrar "ouvindo" e depois "gravando".
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

    import queue
    import time

    from core import vad as vad_mod

    if samplerate != vad_mod.SAMPLERATE:
        raise RuntimeError(
            f"O VAD só opera a {vad_mod.SAMPLERATE} Hz (pedido: {samplerate} Hz). "
            f"Ajuste RECORD_SAMPLERATE ou desligue o VAD com /vad."
        )

    fila: queue.Queue = queue.Queue()

    def callback(indata, _frames, _time, _status):
        fila.put(indata[:, 0].copy())

    frames: list = []
    segmentador = vad_mod.Segmentador(timeout_s=start_timeout)
    estado_anterior = None

    _ESPERA_INICIAL = 5.0   # s sem receber bloco nenhum = dispositivo mudo

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                            device=config.INPUT_DEVICE,
                            blocksize=vad_mod.FRAME_SAMPLES, callback=callback):
            aberto_em = time.monotonic()
            while True:
                # O fim normal vem do VAD; este timeout só devolve o controle
                # para conferir os relógios, caso o dispositivo pare de entregar
                # blocos no meio da escuta (senão ficaríamos presos aqui).
                try:
                    frame = fila.get(timeout=0.5)
                except queue.Empty:
                    if not frames and time.monotonic() - aberto_em > _ESPERA_INICIAL:
                        raise RuntimeError(
                            "O microfone abriu mas não entregou áudio. "
                            "Verifique o dispositivo de entrada padrão."
                        )
                    estado = segmentador.tick()
                    if estado == vad_mod.EXPIRADO:
                        break
                    continue
                frames.append(frame)
                estado = segmentador.push(frame)
                if estado != estado_anterior:
                    estado_anterior = estado
                    if on_state is not None:
                        on_state(estado)
                if estado in (vad_mod.FIM, vad_mod.EXPIRADO):
                    break
    except RuntimeError:
        raise                     # mensagem já é clara; não embrulhar
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falha ao acessar o microfone: {exc}") from exc

    if segmentador.estado == vad_mod.EXPIRADO or not frames:
        return None

    audio = np.concatenate(frames)
    inicio, fim = segmentador.recorte(audio.shape[0])
    audio = audio[inicio:fim]
    # Conferido ANTES de normalizar: é o _normalize que amplifica um fragmento
    # fraco até 0,95 e faz uma captura quebrada parecer boa.
    problema = diagnostico_captura(audio)
    if problema:
        raise RuntimeError(f"Captura de áudio com problema: {problema}\n"
                           f"{_aviso_dispositivo()}")
    audio = _normalize(audio)
    sf.write(path, audio, samplerate)
    return path


def escutar_wake(samplerate: int = config.RECORD_SAMPLERATE,
                 path: str = "/tmp/oraculo_in.wav",
                 on_state=None, abortar=None) -> str | None:
    """Fica ouvindo até a palavra de despertar; então grava a frase e devolve o WAV.

    Devolve None se `abortar()` pediu para parar (texto digitado, Ctrl+C).

    **A garantia de privacidade mora aqui.** Antes do gatilho, cada bloco só passa
    pelo detector e por um anel em memória que se sobrescreve. Nada é acumulado,
    nada é escrito, nada é transcrito. O `frames` só começa a existir depois que a
    palavra é reconhecida — e é por isso que ele é criado dentro do `if`, e não no
    topo da função: um acumulador ligado desde o começo seria uma gravação
    contínua da sala esperando um bug para virar arquivo.

    Como no `record_vad`, a inferência roda nesta thread e o callback do PortAudio
    só empilha na fila.
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

    import queue

    from core import vad as vad_mod
    from core import wake as wake_mod

    if samplerate != wake_mod.SAMPLERATE:
        raise RuntimeError(
            f"O wake word só opera a {wake_mod.SAMPLERATE} Hz "
            f"(pedido: {samplerate} Hz). Ajuste RECORD_SAMPLERATE ou desligue "
            f"com /despertar."
        )

    detector = wake_mod.Detector()
    anel = wake_mod.Anel()
    segmentador = None
    frames = None                 # só nasce depois do gatilho — ver docstring
    n_preroll = 0
    resto = np.zeros(0, dtype="float32")   # sobra para casar 1280 com os 512 do VAD

    fila: queue.Queue = queue.Queue()

    def callback(indata, _frames, _time, _status):
        fila.put(indata[:, 0].copy())

    def avisa(estado):
        if on_state is not None:
            on_state(estado)

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                            device=config.INPUT_DEVICE,
                            blocksize=wake_mod.BLOCO_SAMPLES, callback=callback):
            avisa("ouvindo")
            while True:
                if abortar is not None and abortar():
                    return None
                try:
                    bloco = fila.get(timeout=0.2)
                except queue.Empty:
                    continue

                if segmentador is None:
                    anel.push(bloco)
                    if not detector.push(bloco):
                        continue
                    # Acordou. A partir daqui, e só a partir daqui, acumula.
                    avisa("acordado")
                    segmentador = vad_mod.Segmentador(
                        timeout_s=config.WAKE_FOLLOWUP_TIMEOUT)
                    # O pré-roll entra no arquivo (para não cortar a palavra),
                    # mas NÃO é julgado pelo VAD: se ele contiver fala anterior
                    # seguida de pausa, o VAD fecharia a frase no próprio
                    # instante do gatilho e gravaria o trecho errado. O VAD só
                    # opina sobre o que vem DEPOIS do nome.
                    preroll = anel.conteudo()
                    frames = [preroll]
                    n_preroll = preroll.shape[0]
                    continue
                frames.append(bloco)
                resto = np.concatenate([resto, bloco])

                # O VAD pensa em frames de 512 e o wake em blocos de 1280 — que
                # não são múltiplos. A sobra atravessa as iterações.
                n = vad_mod.FRAME_SAMPLES
                usados = 0
                estado = segmentador.estado
                while usados + n <= resto.shape[0]:
                    estado = segmentador.push(resto[usados:usados + n])
                    usados += n
                resto = resto[usados:]

                if estado in (vad_mod.FIM, vad_mod.EXPIRADO):
                    break
    except RuntimeError:
        raise                     # mensagem já é clara; não embrulhar
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falha ao acessar o microfone: {exc}") from exc

    if not frames:
        return None

    audio = np.concatenate(frames)
    # O corte do VAD está em coordenadas do áudio pós-gatilho; o pré-roll é
    # mantido inteiro na frente, então o fim precisa ser deslocado por ele.
    _, fim = segmentador.recorte(audio.shape[0] - n_preroll)
    audio = audio[:n_preroll + fim]
    problema = diagnostico_captura(audio)
    if problema:
        raise RuntimeError(f"Captura de áudio com problema: {problema}\n"
                           f"{_aviso_dispositivo()}")
    audio = _normalize(audio)
    sf.write(path, audio, samplerate)
    return path


def listar_entradas() -> None:
    """Imprime os dispositivos de entrada, marcando o padrão do sistema.

    Existe porque o padrão nem sempre é o microfone: num sistema com PipeWire e
    EasyEffects, o "default" do ALSA cai na fonte do EasyEffects, que pode estar
    atenuada ou muda. Com o índice em mãos, `config.INPUT_DEVICE` resolve.
    """
    import sounddevice as sd

    padrao = sd.default.device[0]
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        marca = "  <== padrão do sistema" if i == padrao else ""
        print(f"  [{i:2d}] {dev['name'][:48]:50s} "
              f"ch={dev['max_input_channels']:3d} "
              f"{int(dev['default_samplerate'])} Hz{marca}")


def diagnostico_captura(audio) -> str:
    """Descreve o que há de errado com uma captura, ou "" se estiver boa.

    Duas falhas que não levantam exceção nenhuma e por isso passavam direto:

    - **zeros exatos**: microfone real sempre tem ruído de fundo, então amostra
      exatamente zero é bloco perdido ou fonte muda no caminho. Um WAV com 58%
      de zeros já foi gravado sem uma linha de aviso.
    - **pico baixo demais**: som atenuado em -30 dB por um filtro no caminho.
      Pior, o `_normalize` amplificava o resto até 0,95 e escondia a evidência.
    """
    import numpy as np

    if audio.size == 0:
        return "a gravação saiu vazia."

    zeros = float((audio == 0).mean())
    pico = float(np.max(np.abs(audio)))

    if pico < config.CAPTURE_MIN_PEAK:
        return (f"o microfone não captou nada (pico {pico:.5f}). "
                f"A entrada em uso pode estar muda ou ser o dispositivo errado.")
    if zeros > config.CAPTURE_MAX_ZEROS:
        return (f"{zeros*100:.0f}% da gravação é silêncio digital exato — "
                f"há blocos se perdendo entre o microfone e o Oráculo.")
    return ""


def _aviso_dispositivo() -> str:
    """Texto de ajuda comum aos problemas de captura."""
    return (
        "  Liste as entradas com:\n"
        "    .venv/bin/python -c \"from core import audio; audio.listar_entradas()\"\n"
        "  e fixe a certa em INPUT_DEVICE, no config.py.\n"
        "  Se você usa EasyEffects ou algum filtro de microfone, confira se ele\n"
        "  não está no caminho da captura (ou baixando o volume da fonte)."
    )


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
