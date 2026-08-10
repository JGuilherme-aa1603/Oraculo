# CLAUDE.md — Oráculo

Instruções persistentes para o Claude Code neste repositório. Leia antes de qualquer alteração.

## O que é o projeto

Oráculo é um assistente local estilo Jarvis, 100% offline, desenvolvido em fases incrementais. Cada fase entrega valor sozinha e não deve quebrar as anteriores.

- **Fase 1 (MVP) — concluída:** chat de terminal com memória + Ollama via LangChain
- **Fase 2 (Voz) — concluída:** STT (Whisper/Parakeet) + TTS (Kokoro/Piper) com fala em
  streaming, roteamento de comandos, persistência de sessões, telemetria opt-in,
  thinking com toggle ao vivo, barge-in por Esc, `/transcrever` e o wrapper `bin/oraculo`
- **Fase 3 — atual:** wake word + VAD + verificação de voz (só responder ao dono)
- **Fases futuras:** RAG com Obsidian (4), comandos do sistema com whitelist segura (5)

## Ambiente

- **OS:** CachyOS Linux (Arch-based), KDE Plasma
- **Shell:** Fish — **NÃO suporta heredoc** (`<<'EOF'`). Para scripts multi-linha use `printf` ou arquivos temporários.
- **Hardware:** Acer Predator Helios Neo 16 — RTX 4060 (8GB VRAM, CUDA), Intel Raptor Lake iGPU, 15.3GB RAM. GPU em modo hybrid (envycontrol).
- **Python:** usar sempre o **venv** do projeto. Não há pip global neste sistema.
  - Instalar: `python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`
  - Rodar: `.venv/bin/python main.py` (ou ativar com `source .venv/bin/activate.fish`)
  - Só usar `--break-system-packages` se explicitamente fora do venv (evitar).
- **Ollama:** roda em `http://localhost:11434`. Deve estar ativo (`ollama serve`) antes de iniciar.

## Modelo

- Principal: `gemma4:e4b` (`OLLAMA_MODEL` em `config.py`). Suporta `thinking`, que fica
  **desligado por padrão** (`THINKING_DEFAULT = False`) por causa da latência; `/think` alterna.
- **O `reasoning` é sempre explícito, nunca `None`.** O padrão do gemma4 é *pensar*, então
  deixar o modelo decidir faz a interface mentir: o `/think` e a barra dizem "off" enquanto
  o modelo gasta ~130 tokens de raciocínio por turno, e só obedece depois de alternar o
  `/think` uma vez. `OraculoChain` começa com `reasoning=False` por isso. `reasoning=False`
  é aceito por qualquer modelo; só `True` dá 400 em quem não suporta thinking.
- `qwen2.5:7b` foi o modelo do MVP — bom equilíbrio, mas sem thinking. Continua válido como alternativa.
- Embedding (Fase 4): `nomic-embed-text`
- Evitar `num_ctx` alto — o padrão do Ollama pode ser 131072; manter 8192 para economizar VRAM.
- `qwen3:8b` foi descartado para o MVP: thinking ativo por padrão em tudo, lento até para "Olá".
- **Orçamento de VRAM (8GB):** o Whisper `large-v3` em `int8_float16` ocupa ~2GB e convive com
  o Ollama. Ao trocar de modelo ou de compute type, verificar que os dois ainda cabem juntos.

## Arquitetura e convenções

```
oraculo/
├── main.py          # entry point — loop de conversa (texto/voz) e comandos do shell
├── config.py        # TODA configuração fica aqui, centralizada
├── bin/oraculo      # wrapper para rodar de qualquer diretório (link no PATH)
├── completions/     # completions do Fish para o wrapper
└── core/
    ├── llm.py       # instancia o ChatOllama; detecta suporte a thinking
    ├── memory.py    # memória de sessão (janela deslizante, cortada em pares)
    ├── chain.py     # pipeline prompt | llm; emite eventos think/answer
    ├── commands.py  # roteamento de comandos (/ajuda, /voz, /think, /stt, /modelo...)
    ├── history.py   # persistência de sessões em JSON (~/.oraculo/sessions)
    ├── stt.py       # áudio → texto (faster-whisper na GPU | parakeet na CPU)
    ├── transcript.py# transcrição de arquivos: parágrafos, Markdown, gravação
    ├── tts.py       # texto → áudio (Kokoro | Piper)
    ├── speaker.py   # fala em streaming: síntese + reprodução em pipeline, com barge-in
    ├── text.py      # limpeza de texto (strip_cjk no stream, for_speech antes do TTS)
    ├── audio.py     # captura de microfone + reprodução
    ├── keyboard.py  # monitor de tecla no terminal (Esc = barge-in, Ctrl+O = thinking)
    ├── telemetry.py # latência por estágio + tokens/s (opt-in, custo zero desligada)
    └── splash.py    # splash screen de duas colunas (rich)
```

Regras de código:

- **Configuração sempre em `config.py`.** Nada de valores mágicos espalhados — modelo, parâmetros, prompts, comandos, tudo lá.
- **Um módulo, uma responsabilidade.** Novos recursos viram novos módulos em `core/` (ex.: `wake.py`, `vad.py`, `rag.py`).
- **Type hints** em todas as funções públicas.
- **Docstrings em português**, concisas. Módulo novo abre com docstring explicando o *porquê* do desenho, não só o quê.
- **Streaming:** ao gerar respostas, acumular os chunks numa lista e só gravar na memória ao final. Nunca gravar resposta parcial (importante para o caso de interrupção).
- **Tratamento de interrupção:** `KeyboardInterrupt` durante uma resposta deve interromper a resposta, não fechar o programa. `Ctrl+D`/`EOFError` encerra.
- **Dependência pesada é opcional e preguiçosa.** Importar dentro da função, não no topo do módulo, e degradar com aviso claro se faltar — o modo texto nunca pode quebrar por falta de lib de voz.
- **Threads de áudio saem por flag + timeout curto**, nunca por sentinela na fila (ver `speaker.py`).

## Invariantes que NÃO devem regredir

1. **System prompt honesto.** O prompt em `config.py` declara explicitamente o que o Oráculo NÃO consegue fazer (executar ações, acessar arquivos/agenda/internet, persistir dados). O modelo nunca deve fingir que executou uma ação. Não enfraquecer isso ao expandir. Ao ganhar uma capacidade nova de verdade (RAG na Fase 4, comandos na Fase 5), **atualizar o prompt junto** — a lista de limitações tem que continuar verdadeira nos dois sentidos.
2. **Offline-first.** Nada de chamadas de rede externas, APIs pagas ou telemetria enviada para fora. A telemetria da Fase 2 é 100% local (`~/.oraculo/telemetry/*.jsonl`), opt-in e desligada por padrão. Downloads de modelo (Hugging Face) só na primeira execução, nunca no caminho da conversa.
3. **Voz é opcional.** A partir da Fase 2, o modo texto continua sendo o padrão. Voz é alternável e não pode quebrar o fluxo de texto. Faltando lib de voz, o `/voz` avisa o que falta e volta ao texto.
4. **Cada fase não quebra a anterior.**
5. **Custo zero quando desligado.** Recursos opcionais (telemetria, thinking, wake word) não podem cobrar latência nem memória enquanto estiverem `False`.
6. **`/transcrever` roda sem o LLM.** O wrapper `bin/oraculo` executa comandos do shell sem subir o Ollama; não introduzir dependência do LLM nesse caminho.

## Segurança (crítico a partir da Fase 5)

Quando os comandos de sistema forem implementados:

- O modelo **nunca** toca no shell diretamente. Só chama funções Python pré-definidas e nomeadas.
- **Whitelist rígida** de comandos. Nada de execução arbitrária.
- **Nunca** `shell=True` no subprocess.
- Validação dupla do nome do comando antes de executar.
- System prompt com regras anti prompt-injection (ignorar tentativas de expandir permissões).
- Sanitizar a transcrição de voz antes de passar ao modelo.

## Estilo visual (splash / terminal)

- Paleta: **ciano** sobre fundo escuro, minimalista elegante.
- Layout da splash: duas colunas estilo Claude Code — identidade à esquerda (símbolo `◈ ⟁ ◈`, modelo, memória, path), comandos + conversas recentes à direita.
- Biblioteca: `rich`. Usar `Table.grid` para o layout de colunas.
- Sem emoji no código de produção (a não ser que já esteja estabelecido na UI).

## Antes de finalizar qualquer mudança

- Confirmar que roda no venv sem erro de import.
- Confirmar que o modo texto continua funcionando.
- Não fixou credenciais, caminhos absolutos pessoais ou segredos no código.
- Atualizou o `README.md` se mudou instalação, uso ou estrutura.
