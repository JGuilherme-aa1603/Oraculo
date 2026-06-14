# Oráculo

Assistente de voz local estilo Jarvis, rodando 100% offline. Desenvolvido em fases
incrementais. Usa um modelo local via [Ollama](https://ollama.com) orquestrado pelo
[LangChain](https://www.langchain.com).

- **Fase 1 (MVP):** chat de terminal com memória de conversação.
- **Fase 2 (Voz):** entrada/saída de voz opcional (Whisper STT + Piper TTS),
  roteamento de comandos e persistência de sessões. O modo texto continua padrão.

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

### Dependências de voz (Fase 2 — opcional)

A voz exige pacotes de sistema e um binário externo (Piper), além das libs Python:

```bash
# Bibliotecas de sistema (PortAudio para mic, libsndfile para WAV)
sudo pacman -S portaudio libsndfile

# Piper (TTS) via AUR — ou baixe o binário standalone
yay -S piper-tts

# Voz pt-BR do Piper (rhasspy/piper-voices no HuggingFace), ex.: pt_BR-faber-medium
#   .onnx + .onnx.json no diretório de trabalho ou ajuste PIPER_VOICE no config.py

# Libs Python (já incluídas no requirements.txt)
.venv/bin/python -m pip install faster-whisper sounddevice soundfile
```

Sem essas dependências, o modo texto funciona normalmente; ao tentar `/voz`, o
Oráculo avisa o que falta e volta ao texto.

## Uso

```bash
.venv/bin/python main.py
```

Ou ative o venv primeiro (`source .venv/bin/activate.fish` no Fish) e rode
`python main.py`.

### Comandos

- `/ajuda` — lista os comandos
- `/voz` — alterna entre modo voz e modo texto
- `/modelo` — lista os modelos do Ollama; `/modelo <nome>` troca o ativo
- `/limpar` — apaga a memória da conversa atual
- `/sair`, `/exit`, `/quit` — encerra o Oráculo

No modo voz (push-to-talk), pressione Enter para começar a gravar e Enter de novo
para parar; ou digite um texto e Enter como atalho. A transcrição usa o Whisper na
CPU (deixa a GPU livre para o LLM); a fala é sintetizada pelo Piper.

## Estrutura

```
oraculo/
├── main.py          # Entry point — loop de conversa (texto/voz)
├── config.py        # Configurações centralizadas
├── core/
│   ├── llm.py       # Configuração do Ollama + LangChain
│   ├── memory.py    # Memória de conversação (janela deslizante, em pares)
│   ├── chain.py     # Pipeline: system prompt + memória + LLM (troca de modelo)
│   ├── commands.py  # Roteamento de comandos (/ajuda, /voz, /modelo, /limpar, /sair)
│   ├── history.py   # Persistência de sessões em JSON (~/.oraculo/sessions)
│   ├── stt.py       # Whisper (faster-whisper) — áudio → texto
│   ├── tts.py       # Piper — texto → áudio
│   ├── audio.py     # Captura de microfone + reprodução
│   └── splash.py    # Splash screen de duas colunas (rich)
├── requirements.txt
└── README.md
```

As sessões são salvas em `~/.oraculo/sessions/*.json` e aparecem em "Conversas
recentes" na splash.

## Configuração

Ajuste `config.py` para trocar o modelo, parâmetros de geração (temperatura,
`num_ctx`, `MAX_TOKENS`/num_predict), tamanho da memória, system prompt, modelo do
Whisper, voz do Piper e o modo padrão (`VOICE_MODE_DEFAULT`).

## Roadmap

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 — MVP | Chat no terminal + memória + Ollama | ✅ Concluída |
| 2 — Voz | Whisper (STT) + Piper (TTS) + comandos + persistência | ✅ Concluída |
| 3 — Wake Word | OpenWakeWord + Resemblyzer (só sua voz) | ⏳ Futuro |
| 4 — RAG | Indexar notas do Obsidian (nomic-embed-text) | ⏳ Futuro |
| 5 — Commands | Executar comandos do sistema com whitelist segura | ⏳ Futuro |
