"""Configuração do modelo Ollama via LangChain."""

import json
import urllib.request

from langchain_ollama import ChatOllama

import config


def build_llm(model: str = config.OLLAMA_MODEL, *, reasoning: bool | None = None) -> ChatOllama:
    """Instancia o modelo Ollama com os parâmetros do config.

    `reasoning`: True liga o raciocínio (thinking), False desliga explicitamente,
    None usa o padrão do modelo. Só é repassado ao ChatOllama quando não é None
    (mandar `reasoning=True` para um modelo sem suporte gera erro 400).
    """
    kwargs = dict(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_ctx=config.NUM_CTX,
        num_predict=config.MAX_TOKENS,
    )
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return ChatOllama(**kwargs)


def supports_thinking(model: str) -> bool:
    """True se o modelo declara a capacidade 'thinking' no /api/show do Ollama."""
    try:
        req = urllib.request.Request(
            f"{config.OLLAMA_BASE_URL}/api/show",
            data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:  # noqa: BLE001 — Ollama indisponível: assume sem suporte
        return False
    return "thinking" in (data.get("capabilities") or [])
