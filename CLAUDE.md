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
    ├── ui.py        # calha do transcript: eco, cabeçalho, corpo recuado, rodapé
    ├── prompt.py    # caixa de entrada (prompt_toolkit): borda, histórico, autocomplete
    ├── tui.py       # modo tela cheia: transcript rolável + caixa fixa no rodapé
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

**Calha do transcript (`core/ui.py`).** Todo turno segue a mesma estrutura vertical: eco
da pergunta recuado 2, cabeçalho `● Oráculo`, corpo recuado `UI_GUTTER`, rodapé com as
métricas, linha em branco. Saída nova no terminal passa por `ui.notice/warn/error` em vez
de `console.print` cru — é isso que mantém tudo na mesma margem. Blocos de lista (`/stt`,
`/modelo`) indentam manualmente para casar com a calha.

- Nada de painel por mensagem: a moldura custa 4 colunas por mensagem e vira ruído em
  resposta longa com código. O recuo dá a mesma hierarquia de graça.
- A largura de leitura é limitada por `UI_MAX_WIDTH` (`Constrain` do rich). Terminal largo
  com texto de ponta a ponta é ilegível.
- O preview do `Live` durante o streaming usa **o mesmo** `ui.indent()` da renderização
  final — sem isso o texto pula de coluna quando o preview é substituído.

**Dois modos de desenho (`TUI_MODE`).** É o mesmo par que o Claude Code oferece:

- `"fullscreen"` (padrão, `core/tui.py`) — tela alternativa do terminal, caixa fixa no
  rodapé, transcript com rolagem própria, saída sem rastro.
- `"inline"` (`core/ui.py` + `core/prompt.py`) — buffer normal, rolagem nativa do
  terminal, transcript permanece na tela ao encerrar. `CLEAR_ON_START` limpa no arranque
  usando só `2J`; nunca usar `3J`, que destruiria o scrollback de quem chamou.

Sem TTY ou sem prompt_toolkit, cai para `inline` sozinho.

**Como a rolagem existe na tela alternativa.** Ela não vem do terminal — lá não há
scrollback. Vem do `Transcript`: a conversa é uma lista de blocos em memória e, a cada
quadro, só as linhas visíveis viram fragmentos. Mesma estratégia do vim. Pontos que
custaram para acertar:

- `TranscriptConsole` é um `rich.Console` que guarda o que foi impresso em vez de escrever
  no stdout. **É o que faz `core/ui.py` e `core/commands.py` funcionarem nos dois modos sem
  alteração** — para eles, continua sendo um Console comum. Ao adicionar saída nova, use
  `console.print`/`ui.*` e ela aparece nos dois modos de graça.
- Guardar os *argumentos* do `print` (não o texto renderizado) é o que permite reflowar
  tudo quando o terminal muda de largura.
- `FormattedTextControl.create_content` também é chamado com `height=None`, só para
  perguntar a altura preferida. Mexer no viewport nessa passada estoura.
- A caixa precisa de `dont_extend_height=True`: senão o `HSplit` entrega a sobra vertical
  para ela (aceita até 8 linhas) e ela abre linhas em branco no lugar do transcript.
- No fullscreen o Ctrl+C **não** sobe como exceção na thread do laço — ele marca um
  `Event` que o laço confere a cada chunk do stream.
- `core/keyboard.py` (modo raw) só vale no inline. No fullscreen o prompt_toolkit é dono
  do teclado, então Ctrl+O é só mais um atalho da app. Pelo mesmo motivo `audio.record_ptt`
  recebe `wait_stop`: o `input()` dele brigaria pelo stdin.

- Capturar o mouse tira do terminal a seleção de texto. A saída **não** é soltar a captura,
  é implementar a seleção — que é o que o Claude Code faz. O `Selecao` guarda âncora e
  cursor em **linha absoluta do transcript**, não em posição de tela: assim o trecho
  selecionado continua no mesmo texto quando a conversa rola ou cresce durante a geração.
  O destaque sai em vídeo reverso, cortando os fragmentos nos limites da seleção.
  F2 (soltar a captura) sobra como escape hatch para quando o OSC 52 não passa.
- Cópia sem dependência: `wl-copy`/`xclip`/`xsel` se existirem, senão **OSC 52** (pede ao
  terminal para copiar). Subprocess sempre por lista de argumentos, nunca `shell=True`, e
  **numa thread**: a cópia é disparada de dentro do tratamento do mouse, que roda na
  thread que desenha — um utilitário lento ali congelaria a interface inteira.
- Arrasto que sai da área do transcript exige duas defesas, senão a seleção *trava*:
  (1) enquanto `selecao.arrastando`, o container raiz reivindica a tela toda em
  `write_to_screen`, porque o prompt_toolkit entrega o evento ao controle sob o ponteiro;
  (2) movimento **sem botão** durante um arrasto significa que o soltar aconteceu fora da
  janela e o terminal não o reportou — sem essa recuperação a seleção fica presa em modo
  de arrasto para sempre.
- Rolar para cima (inclusive por arrasto) pausa o auto-follow. Enviar mensagem **tem** que
  voltar ao fim, senão a resposta chega fora da vista e a interface parece travada. O
  estado pausado precisa ficar visível na barra pelo mesmo motivo.
- Ao ligar Esc a alguma ação, **não** use `eager=True`: Alt+Enter chega como
  `("escape", "enter")` e um Esc ansioso engole o prefixo, matando a quebra de linha.

**Arrasto de mouse no prompt_toolkit — três armadilhas** (todas custaram tempo, e valem
tanto para o transcript quanto para a caixa de entrada em `_janela_entrada`):

1. O `Window` traduz a posição da tela para posição no documento e **prende ao último
   visível** (`y = min(max_y, y)`). Para rolar ao arrastar contra a borda é preciso
   interceptar o handler **antes** dessa tradução, usando coordenadas de tela.
2. `screen.width`/`screen.height` são **sempre 0** dentro de `write_to_screen` — o
   renderer nunca os atribui. Para reivindicar a tela inteira use
   `get_app().output.get_size()`, senão a faixa sai vazia e nada é registrado.
3. `top_visible`/`bottom_visible` e `get_cursor_up_position()` raciocinam em linhas do
   **documento**. Uma mensagem digitada sem quebras é uma linha só: a primeira está
   "sempre visível" e "subir uma linha" não sai do lugar. Para conteúdo com `wrap_lines`
   ande por linha *visual* (a largura da janela em caracteres) e deixe o cursor, preso
   aos extremos do texto, ser o limite.

Também: é o **cursor** que puxa a rolagem. Mexer em `vertical_scroll` direto é desfeito,
porque o Window recalcula o scroll a cada quadro para manter o cursor visível.

Ao inspecionar o desenho num pty, **não** tire os ANSI do stream: o prompt_toolkit faz
repaint posicional e o resultado parece corrompido mesmo estando certo. Reproduza o stream
num emulador (`pyte`) e olhe a tela. E defina o tamanho do pty por `TIOCSWINSZ`: o
prompt_toolkit lê o tamanho por ioctl, não por `$COLUMNS`, então sem isso ele desenha a 80
colunas e o teste mede a largura errada.

**Caixa de entrada (`core/prompt.py`).** `Application` inline do prompt_toolkit, não
`PromptSession`: o prompt padrão não fecha a borda direita. A moldura arredondada é
remontada à mão porque a classe `Border` do prompt_toolkit tem os cantos hard-coded. O
menu de completion entra no fluxo abaixo da barra de status — como `Float` ele seria
desenhado por cima da borda, já que numa app não-fullscreen o float não escapa da altura
da própria app. Atenção: `Buffer.cancel_completion()` **reverte** o texto ao original;
para só fechar o menu preservando o que o Tab inseriu, zere `buf.complete_state`.

## Antes de finalizar qualquer mudança

- Confirmar que roda no venv sem erro de import.
- Confirmar que o modo texto continua funcionando.
- Não fixou credenciais, caminhos absolutos pessoais ou segredos no código.
- Atualizou o `README.md` se mudou instalação, uso ou estrutura.
