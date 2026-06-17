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

- **STT:** Whisper (`faster-whisper`) `large-v3` na GPU, padrão (cai para CPU
  automaticamente se faltar GPU). Alternativa opcional: **Parakeet V3**
  (`onnx-asr`), bem rápido na CPU — selecione com `STT_ENGINE = "parakeet"` no
  `config.py` (limite de ~30s por clipe).
- **TTS padrão:** Kokoro (`kokoro-onnx`), voz feminina pt-BR natural (`pf_dora`).
  Não precisa de sudo — o espeak-ng vem via `espeakng-loader`.

```bash
# Bibliotecas de sistema (PortAudio para mic, libsndfile para WAV)
sudo pacman -S portaudio libsndfile

# Libs Python de voz (já incluídas no requirements.txt)
.venv/bin/python -m pip install faster-whisper kokoro-onnx sounddevice soundfile

# Modelos do Kokoro (releases de thewh1teagle/kokoro-onnx) na raiz do projeto:
#   kokoro-v1.0.onnx   e   voices-v1.0.bin
```

Para usar a voz **masculina do Piper** em vez do Kokoro, defina `TTS_ENGINE = "piper"`
no `config.py`, instale `yay -S piper-tts` e baixe uma voz pt-BR (ex.: `pt_BR-faber-medium`
`.onnx` + `.onnx.json`) de rhasspy/piper-voices.

**STT na GPU (padrão):** o `large-v3` roda na RTX 4060 via `faster-whisper`. Como o
sistema está em CUDA 13 e o `ctranslate2` exige CUDA 12, as libs vêm em userspace
pelos wheels `nvidia-cublas-cu12` + `nvidia-cudnn-cu12` (já no requirements);
`stt._enable_cuda_libs()` as pré-carrega antes de instanciar o modelo. Com
`WHISPER_COMPUTE_TYPE = "int8_float16"` o modelo ocupa ~2 GB de VRAM, deixando
espaço para o Ollama nos 8 GB. Sem GPU/libs, o STT cai para CPU automaticamente
(ajuste `WHISPER_DEVICE`/`WHISPER_MODEL` no `config.py`).

Para usar o **Parakeet V3** em vez do Whisper, defina `STT_ENGINE = "parakeet"` no
`config.py` e instale `onnx-asr onnxruntime` (já no requirements). O modelo é baixado
do Hugging Face na primeira transcrição. É rápido na CPU e pontua sozinho, mas não
tem VAD e tem limite de ~30s por clipe (suficiente para falas curtas).

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
- `/stt` — lista os motores de transcrição; `/stt <motor>` troca (`whisper`/`parakeet`)
- `/modelo` — lista os modelos do Ollama; `/modelo <nome>` troca o ativo
- `/limpar` — apaga a memória da conversa atual
- `/sair`, `/exit`, `/quit` — encerra o Oráculo

No modo voz (push-to-talk), pressione Enter para começar a gravar e Enter de novo
para parar; ou digite um texto e Enter como atalho. A transcrição usa o Whisper
`large-v3` na GPU (quase em tempo real); a fala é sintetizada pelo Kokoro (voz feminina).
As respostas são renderizadas como Markdown no terminal; antes da síntese de voz a
marcação é removida para a fala não soletrar símbolos.

Enquanto a resposta não começa, um indicador mostra **"Carregando modelo..."** se o
Ollama ainda está subindo o modelo na VRAM (cold start) ou **"Pensando..."** quando
ele já está carregado e gerando.

**Interromper a fala (barge-in):** no modo voz, pressione **Esc** para o Oráculo parar
de falar na hora e liberar o prompt para a próxima mensagem — sem precisar esperar ele
terminar de ler a resposta.

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
│   ├── tts.py       # Kokoro/Piper — texto → áudio
│   ├── text.py      # Limpeza de texto (remove Markdown p/ voz, filtra CJK)
│   ├── audio.py     # Captura de microfone + reprodução
│   ├── keyboard.py  # Monitor de tecla no terminal (barge-in por Esc)
│   ├── telemetry.py # Latência por estágio + tokens/s (opt-in)
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

## Telemetria

Instrumentação opcional de cada turno para medir o pipeline (STT → LLM → TTS):
tempo de transcrição (STT), TTFT (time-to-first-token), tokens/s, tempo até a 1ª
fala (TTFA) e a duração total do turno. Desligada por padrão — **custo zero** com
ambas as flags em `False`.

Em `config.py`:

| Flag | Efeito |
|------|--------|
| `TELEMETRY_CONSOLE` | `True` → imprime um resumo de 1 linha por turno no terminal. Bom para desenvolvimento. |
| `TELEMETRY_ENABLED` | `True` → faz append de 1 objeto JSON por turno em `~/.oraculo/telemetry/<AAAA-MM-DD>.jsonl`. |

Resumo de console (os campos de voz são omitidos no modo texto):

```
telemetria turno 3.2s · STT 0.8s · TTFT 0.4s · 142 tok @ 38 tok/s · 1a fala 1.1s
```

A taxa de tokens/s vem das métricas do próprio Ollama (`eval_count`/`eval_duration`)
quando disponíveis; TTFT e o total são sempre medidos localmente. A telemetria é
best-effort: um erro de medição nunca interrompe a conversa, e nenhuma dependência
nova é exigida (apenas a biblioteca-padrão + `rich`).

## Roadmap

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 — MVP | Chat no terminal + memória + Ollama | ✅ Concluída |
| 2 — Voz | Whisper (STT) + Piper (TTS) + comandos + persistência | ✅ Concluída |
| 3 — Wake Word | OpenWakeWord + Resemblyzer (só sua voz) | ⏳ Futuro |
| 4 — RAG | Indexar notas do Obsidian (nomic-embed-text) | ⏳ Futuro |
| 5 — Commands | Executar comandos do sistema com whitelist segura | ⏳ Futuro |
