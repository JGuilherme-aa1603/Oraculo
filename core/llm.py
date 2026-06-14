"""Configuração do modelo Ollama via LangChain."""

from langchain_ollama import ChatOllama

import config


def build_llm(model: str = config.OLLAMA_MODEL) -> ChatOllama:
    """Instancia o modelo Ollama com os parâmetros do config."""
    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_ctx=config.NUM_CTX,
        num_predict=config.MAX_TOKENS,
    )
