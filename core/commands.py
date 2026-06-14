"""Roteamento de comandos do terminal (/ajuda, /voz, /modelo, /limpar, /sair).

O loop principal passa um dicionário de contexto mutável (`ctx`) com:
  - console: rich.Console
  - chain: OraculoChain (tem .memory, .model_name, .set_model)
  - running: bool   (setar False encerra o loop)
  - voice_mode: bool

`handle()` retorna True se o input era um comando (e portanto NÃO deve ir ao LLM).
"""

import config

AJUDA_TEXT = """[bold cyan]Comandos disponíveis[/]
  [bright_cyan]/ajuda[/]   mostra esta ajuda
  [bright_cyan]/voz[/]     alterna entre modo voz e modo texto
  [bright_cyan]/modelo[/]  lista modelos do Ollama ou troca com [dim]/modelo <nome>[/]
  [bright_cyan]/limpar[/]  apaga a memória da conversa atual
  [bright_cyan]/sair[/]    encerra o Oráculo"""


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
        console.print("[bold cyan]Modelos disponíveis:[/]")
        for m in models:
            mark = "  [bright_white](atual)[/]" if m == chain.model_name else ""
            console.print(f"  • [bright_white]{m}[/]{mark}")
        console.print("[dim]Use /modelo <nome> para trocar.[/]")
        return

    chain.set_model(arg)
    console.print(f"[cyan]Modelo trocado para[/] [bright_white]{arg}[/].")


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
        console.print("[cyan]Encerrando...[/]")
        return True

    if cmd in {"/ajuda", "/help", "/?"}:
        console.print(AJUDA_TEXT)
        return True

    if cmd == "/limpar":
        ctx["chain"].memory.clear()
        console.print("[cyan]Memória da sessão limpa.[/]")
        return True

    if cmd == "/voz":
        ctx["voice_mode"] = not ctx["voice_mode"]
        estado = "ativado" if ctx["voice_mode"] else "desativado"
        console.print(f"[cyan]Modo voz {estado}.[/]")
        return True

    if cmd == "/modelo":
        _handle_modelo(arg, ctx)
        return True

    console.print(f"[yellow]Comando desconhecido:[/] {cmd}  [dim](veja /ajuda)[/]")
    return True
