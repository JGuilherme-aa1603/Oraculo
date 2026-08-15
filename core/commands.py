"""Roteamento de comandos do terminal (/ajuda, /voz, /stt, /transcrever, /modelo,
/limpar, /sair).

O loop principal passa um dicionário de contexto mutável (`ctx`) com:
  - console: rich.Console
  - chain: OraculoChain (tem .memory, .model_name, .set_model)
  - running: bool   (setar False encerra o loop)
  - voice_mode: bool

`handle()` retorna True se o input era um comando (e portanto NÃO deve ir ao LLM).
"""

from pathlib import Path

import config
from core import llm as llm_mod, ui

STT_ENGINES = ("whisper", "parakeet")

# Comandos que também rodam fora do chat, via wrapper `oraculo <comando>`.
# Só entram aqui os que não dependem do LLM (ctx["chain"] é None nesse modo).
STANDALONE_COMMANDS = {"/transcrever", "/transcricao", "/ajuda", "/help", "/?"}

# Fonte única da verdade dos comandos: alimenta tanto o texto de /ajuda quanto o
# autocomplete da caixa de entrada (core/prompt.py). Ao adicionar um comando novo,
# basta registrá-lo aqui — os dois lugares acompanham sozinhos.
# (comando, dica de argumento, descrição de uma linha)
COMMAND_SPECS: tuple[tuple[str, str, str], ...] = (
    ("/ajuda", "", "mostra esta ajuda"),
    ("/voz", "", "alterna entre modo voz e modo texto"),
    ("/vad", "", "liga/desliga a parada automática da gravação"),
    ("/despertar", "", "liga/desliga a escuta pela palavra \"Oráculo\""),
    ("/think", "", "liga/desliga o raciocínio; Ctrl+O mostra o texto"),
    ("/stt", "<motor>", "lista ou troca o motor de transcrição"),
    ("/transcrever", "<arquivo>", "transcreve um áudio; --salvar grava um .md ao lado"),
    ("/modelo", "<nome>", "lista os modelos do Ollama ou troca o ativo"),
    ("/limpar", "", "apaga a memória da conversa atual"),
    ("/sair", "", "encerra o Oráculo"),
)


def _ajuda_text() -> str:
    """Monta o texto de /ajuda em duas colunas alinhadas: 'comando <arg>' à
    esquerda, descrição à direita. A largura vem do item mais longo."""
    assinaturas = {cmd: f"{cmd} {arg}".strip() for cmd, arg, _ in COMMAND_SPECS}
    largura = max(len(s) for s in assinaturas.values())
    # O recuo de 2 alinha o bloco com o eco da mensagem no transcript (core.ui).
    linhas = ["  [bold cyan]Comandos disponíveis[/]"]
    for cmd, arg, desc in COMMAND_SPECS:
        rotulo = f"[bright_cyan]{cmd}[/]" + (f" [dim]{arg}[/]" if arg else "")
        preenche = " " * (largura - len(assinaturas[cmd]))
        linhas.append(f"    {rotulo}{preenche}  {desc}")
    return "\n".join(linhas)


AJUDA_TEXT = _ajuda_text()


def _list_models() -> list[str]:
    """Consulta o Ollama pelos modelos instalados. Retorna [] em caso de falha."""
    try:
        import requests

        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def _handle_modelo(arg: str, ctx: dict) -> None:
    console = ctx["console"]
    chain = ctx["chain"]

    if not arg:
        models = _list_models()
        if not models:
            console.print("[yellow]Não consegui listar os modelos do Ollama.[/]")
            return
        console.print("  [bold cyan]Modelos disponíveis:[/]")
        for m in models:
            mark = "  [bright_white](atual)[/]" if m == chain.model_name else ""
            console.print(f"    • [bright_white]{m}[/]{mark}")
        console.print("  [dim]Use /modelo <nome> para trocar.[/]")
        return

    chain.set_model(arg)
    ui.notice(console, f"Modelo trocado para {arg}.", style="cyan")

    # set_model preserva o reasoning; se o novo modelo não suporta thinking,
    # desliga para o próximo turno não falhar com erro 400.
    if ctx.get("thinking") and not llm_mod.supports_thinking(arg):
        ctx["thinking"] = False
        chain.set_thinking(False)
        ui.warn(console, f"{arg} não suporta raciocínio — thinking desativado.")


def _handle_think(ctx: dict) -> None:
    console = ctx["console"]
    chain = ctx["chain"]
    want = not ctx.get("thinking", False)

    if want and not llm_mod.supports_thinking(chain.model_name):
        ui.warn(console, f"O modelo {chain.model_name} não suporta raciocínio (thinking).")
        return

    ctx["thinking"] = want
    chain.set_thinking(want)
    if want:
        ui.notice(console, "Raciocínio ativado — Ctrl+O mostra/oculta o texto.",
                  style="cyan")
    else:
        ui.notice(console, "Raciocínio (thinking) desativado.", style="cyan")


def _handle_vad(ctx: dict) -> None:
    """Alterna a parada automática da gravação (VAD) e o push-to-talk."""
    console = ctx["console"]
    quer = not config.VAD_ENABLED

    if quer:
        # Só liga se o modelo realmente carregar: melhor recusar aqui, com a
        # causa na tela, do que falhar no meio da próxima gravação.
        from core import vad

        if not vad.disponivel():
            ui.warn(console, "VAD indisponível — o push-to-talk continua ativo.")
            return

    config.VAD_ENABLED = quer
    if quer:
        ui.notice(console, "VAD ativado — a gravação para sozinha quando você "
                  "parar de falar.", style="cyan")
    else:
        ui.notice(console, "VAD desativado — Enter para gravar, Enter de novo "
                  "para parar.", style="cyan")


def _handle_despertar(ctx: dict) -> None:
    """Alterna a escuta pela palavra de despertar.

    Ao ligar, diz em voz alta o que isso implica: o microfone fica aberto. Essa
    frase não é enfeite — é o que faz do modo uma escolha informada.
    """
    console = ctx["console"]
    quer = not config.WAKE_ENABLED

    if quer:
        from core import wake

        if not wake.disponivel():
            ui.warn(console, "Escuta indisponível — o push-to-talk continua ativo.")
            motivo = wake.motivo_indisponivel()
            for linha in motivo.splitlines():
                if linha.strip():
                    ui.warn(console, f"  {linha.strip()}")
            return

    config.WAKE_ENABLED = quer
    if quer:
        ui.notice(console,
                  f'Escuta ativada — o microfone fica aberto e eu respondo '
                  f'quando você disser "{config.WAKE_WORD}".', style="cyan")
        ui.notice(console,
                  "  Nada é gravado nem transcrito antes disso. Ctrl+C encerra "
                  "a escuta.")
        if not ctx.get("voice_mode"):
            ui.notice(console, "  Só vale no modo voz — use /voz para entrar.")
    else:
        ui.notice(console, "Escuta desativada — o microfone só abre quando você "
                  "pedir.", style="cyan")


def _handle_stt(arg: str, ctx: dict) -> None:
    console = ctx["console"]
    arg = arg.lower()

    if not arg:
        console.print("  [bold cyan]Motores de transcrição (STT):[/]")
        for engine in STT_ENGINES:
            mark = "  [bright_white](atual)[/]" if engine == config.STT_ENGINE else ""
            console.print(f"    • [bright_white]{engine}[/]{mark}")
        console.print("  [dim]Use /stt <motor> para trocar.[/]")
        return

    if arg not in STT_ENGINES:
        ui.warn(console, f"Motor desconhecido: {arg} "
                f"(opções: {', '.join(STT_ENGINES)})")
        return

    # transcribe() lê config.STT_ENGINE a cada chamada, então sobrescrever aqui
    # já troca o motor da próxima transcrição — sem reiniciar.
    from core import stt

    config.STT_ENGINE = arg
    ui.notice(console, f"Motor de STT trocado para {arg}.", style="cyan")
    if not stt.available():
        ui.warn(console, f"Dependências de '{arg}' não instaladas — a transcrição "
                f"vai falhar até instalá-las.")


_SAVE_FLAGS = {"--salvar", "-s"}

TRANSCREVER_USO = (
    "[dim]Uso:[/] [bright_cyan]/transcrever <arquivo>[/] "
    "[dim][--salvar][/]\n"
    "[dim]  --salvar grava a transcrição em Markdown ao lado do áudio.[/]"
)


def _parse_alvo(arg: str) -> tuple[str, bool]:
    """Separa o caminho das flags.

    O caminho vem sem aspas na maioria das vezes e pode ter espaços (áudios do
    WhatsApp têm), então as flags só são reconhecidas no fim da linha e o resto
    inteiro é tratado como um caminho só."""
    salvar = False
    tokens = arg.split()
    while tokens and tokens[-1].lower() in _SAVE_FLAGS:
        salvar = True
        tokens.pop()

    caminho = " ".join(tokens)
    if len(caminho) > 1 and caminho[0] == caminho[-1] and caminho[0] in "\"'":
        caminho = caminho[1:-1]
    return caminho, salvar


def _com_progresso(segments, status, secs: float | None):
    """Repassa os segmentos atualizando o spinner com a posição no áudio."""
    from core.transcript import hms

    for seg in segments:
        total = f"/{hms(secs)}" if secs else ""
        status.update(f"[dim]Transcrevendo... {hms(seg[1])}{total}[/]")
        yield seg


def _handle_transcrever(arg: str, ctx: dict) -> None:
    console = ctx["console"]

    if not arg:
        console.print(TRANSCREVER_USO)
        return

    caminho, salvar = _parse_alvo(arg)
    path = Path(caminho).expanduser()
    if not path.is_file():
        console.print(f"[yellow]Arquivo não encontrado:[/] {path}")
        return

    from core import stt, transcript

    if not stt.available():
        console.print(f"[yellow]O motor '{config.STT_ENGINE}' não está "
                      f"instalado — veja /stt para trocar de motor.[/]")
        return

    if path.suffix.lower() not in config.TRANSCRIBE_EXTENSIONS:
        console.print(f"[yellow]'{path.suffix}' não parece um formato de áudio; "
                      f"vou tentar mesmo assim.[/]")

    secs = stt.duration(str(path))
    limite = config.TRANSCRIBE_PARAKEET_LIMIT
    if config.STT_ENGINE == "parakeet" and secs and secs > limite:
        console.print(f"[yellow]O parakeet trunca clipes acima de "
                      f"{limite:.0f}s. Para este áudio, use /stt whisper.[/]")

    dur = f"  [dim]({transcript.hms(secs)})[/]" if secs else ""
    console.print(f"[bold cyan]Transcrevendo[/] [bright_white]{path.name}[/]{dur}"
                  f"  [dim]· {transcript.engine_label()}[/]")

    paras: list[tuple[float, str]] = []
    interrompido = False
    try:
        with console.status("[dim]Carregando o motor de transcrição...[/]",
                            spinner="dots") as status:
            segments = _com_progresso(stt.transcribe_segments(str(path)),
                                      status, secs)
            for start, texto in transcript.paragraphs(segments):
                paras.append((start, texto))
                marca = f"[dim][{transcript.hms(start)}][/] " \
                    if config.TRANSCRIBE_TIMESTAMPS else ""
                console.print(f"{marca}{texto}")
    except KeyboardInterrupt:
        interrompido = True
        console.print("\n[yellow](transcrição interrompida)[/]")
    except RuntimeError as exc:      # dependência faltando
        console.print(f"[yellow]{exc}[/]")
        return
    except Exception as exc:         # noqa: BLE001 — áudio ilegível, disco, etc.
        console.print(f"[bold red]Erro ao transcrever:[/] {exc}")
        return

    if not paras:
        console.print("[dim](nada foi transcrito — o áudio tem fala?)[/]")
        return

    if salvar:
        try:
            destino = transcript.save(paras, path, secs)
            parcial = " [dim](parcial)[/]" if interrompido else ""
            console.print(f"[cyan]Transcrição salva em[/] "
                          f"[bright_white]{destino}[/]{parcial}")
        except OSError as exc:
            console.print(f"[bold red]Não consegui salvar:[/] {exc}")
    elif not interrompido:
        console.print("[dim](use --salvar para gravar em Markdown)[/]")


def handle(raw: str, ctx: dict) -> bool:
    raw = raw.strip()
    if not raw.startswith("/"):
        return False

    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    console = ctx["console"]

    if cmd in config.EXIT_COMMANDS:
        ctx["running"] = False
        ui.notice(console, "Encerrando...", style="cyan")
        return True

    if cmd in {"/ajuda", "/help", "/?"}:
        console.print(AJUDA_TEXT)
        return True

    if cmd == "/limpar":
        ctx["chain"].memory.clear()
        ui.notice(console, "Memória da sessão limpa.", style="cyan")
        return True

    if cmd == "/voz":
        ctx["voice_mode"] = not ctx["voice_mode"]
        estado = "ativado" if ctx["voice_mode"] else "desativado"
        ui.notice(console, f"Modo voz {estado}.", style="cyan")
        return True

    if cmd == "/vad":
        _handle_vad(ctx)
        return True

    if cmd == "/despertar":
        _handle_despertar(ctx)
        return True

    if cmd == "/think":
        _handle_think(ctx)
        return True

    if cmd == "/stt":
        _handle_stt(arg, ctx)
        return True

    if cmd in {"/transcrever", "/transcricao"}:
        _handle_transcrever(arg, ctx)
        return True

    if cmd == "/modelo":
        _handle_modelo(arg, ctx)
        return True

    ui.warn(console, f"Comando desconhecido: {cmd}  (veja /ajuda)")
    return True
