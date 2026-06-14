# Oráculo

Assistente de voz local estilo Jarvis, rodando 100% offline. Desenvolvido em fases
incrementais. Esta é a **Fase 1 (MVP)**: um chatbot de terminal com memória de
conversação, usando um modelo local via [Ollama](https://ollama.com) orquestrado pelo
[LangChain](https://www.langchain.com).

## Requisitos

- Ollama rodando em `http://localhost:11434` (`ollama serve` se não estiver ativo)
- Modelo `qwen2.5:7b` baixado (`ollama pull qwen2.5:7b`)
- Python 3.10+

## Instalação

Este sistema (CachyOS) não tem `pip` global, então usamos um ambiente virtual:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Uso

```bash
.venv/bin/python main.py
```

Ou ative o venv primeiro (`source .venv/bin/activate.fish` no Fish) e rode
`python main.py`.

Comandos durante a conversa:

- `/sair`, `/exit` ou `/quit` — encerra o Oráculo

## Estrutura

```
oraculo/
├── main.py          # Entry point — loop de conversa no terminal
├── config.py        # Configurações centralizadas
├── core/
│   ├── llm.py       # Configuração do Ollama + LangChain
│   ├── memory.py    # Memória de conversação (janela deslizante)
│   ├── chain.py     # Pipeline: system prompt + memória + LLM
│   └── splash.py    # Splash screen (rich)
├── requirements.txt
└── README.md
```

## Configuração

Ajuste `config.py` para trocar o modelo, parâmetros de geração (temperatura,
`num_ctx`, máximo de tokens), tamanho da memória ou o system prompt.

## Roadmap

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 — MVP | Chat no terminal + memória + Ollama | ✅ Atual |
| 2 — Voz | Whisper (STT) + Piper (TTS) | ⏳ Futuro |
| 3 — Wake Word | OpenWakeWord + Resemblyzer | ⏳ Futuro |
| 4 — RAG | Indexar notas do Obsidian (nomic-embed-text) | ⏳ Futuro |
| 5 — Commands | Executar comandos do sistema com whitelist segura | ⏳ Futuro |
