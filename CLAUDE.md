# CLAUDE.md — Oráculo

Instruções persistentes para o Claude Code neste repositório. Leia antes de qualquer alteração.

## O que é o projeto

Oráculo é um assistente local estilo Jarvis, 100% offline, desenvolvido em fases incrementais. Cada fase entrega valor sozinha e não deve quebrar as anteriores.

- **Fase 1 (MVP) — concluída:** chat de terminal com memória + Ollama via LangChain
- **Fase 2 — atual:** voz (Whisper STT + Piper TTS) + roteamento de comandos
- **Fases futuras:** wake word + verificação de voz (3), RAG com Obsidian (4), comandos do sistema com whitelist segura (5)

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

- Principal: `qwen2.5:7b` (melhor equilíbrio qualidade/velocidade/VRAM neste hardware)
- Embedding (Fase 4): `nomic-embed-text`
- Evitar `num_ctx` alto — o padrão do Ollama pode ser 131072; manter 8192 para economizar VRAM.
- `qwen3:8b` foi descartado para o MVP: thinking ativo por padrão em tudo, lento até para "Olá".

## Arquitetura e convenções

```
oraculo/
├── main.py          # entry point — loop de conversa
├── config.py        # TODA configuração fica aqui, centralizada
└── core/
    ├── llm.py       # instancia o ChatOllama
    ├── memory.py    # memória de sessão (janela deslizante)
    ├── chain.py     # pipeline prompt | llm
    └── splash.py    # splash screen (rich)
```

Regras de código:

- **Configuração sempre em `config.py`.** Nada de valores mágicos espalhados — modelo, parâmetros, prompts, comandos, tudo lá.
- **Um módulo, uma responsabilidade.** Novos recursos viram novos módulos em `core/` (ex.: `stt.py`, `tts.py`, `audio.py`, `commands.py`).
- **Type hints** em todas as funções públicas.
- **Docstrings em português**, concisas.
- **Streaming:** ao gerar respostas, acumular os chunks numa lista e só gravar na memória ao final. Nunca gravar resposta parcial (importante para o caso de interrupção).
- **Tratamento de interrupção:** `KeyboardInterrupt` durante uma resposta deve interromper a resposta, não fechar o programa. `Ctrl+D`/`EOFError` encerra.

## Invariantes que NÃO devem regredir

1. **System prompt honesto.** O prompt em `config.py` declara explicitamente o que o Oráculo NÃO consegue fazer (executar ações, acessar arquivos/agenda/internet, persistir dados). O modelo nunca deve fingir que executou uma ação. Não enfraquecer isso ao expandir.
2. **Offline-first.** Nada de chamadas de rede externas, APIs pagas ou telemetria. Tudo local.
3. **Voz é opcional.** A partir da Fase 2, o modo texto continua sendo o padrão. Voz é alternável e não pode quebrar o fluxo de texto.
4. **Cada fase não quebra a anterior.**

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
