# CLAUDE.md — Oráculo

Instruções persistentes para o Claude Code neste repositório. Leia antes de qualquer alteração.

## O que é o projeto

Oráculo é um assistente local estilo Jarvis, 100% offline, desenvolvido em fases incrementais. Cada fase entrega valor sozinha e não deve quebrar as anteriores.

- **Fase 1 (MVP) — concluída:** chat de terminal com memória + Ollama via LangChain
- **Fase 2 (Voz) — concluída:** STT (Whisper/Parakeet) + TTS (Kokoro/Piper) com fala em
  streaming, roteamento de comandos, persistência de sessões, telemetria opt-in,
  thinking com toggle ao vivo, barge-in por Esc, `/transcrever` e o wrapper `bin/oraculo`
- **Fase 3 (Sempre ouvindo) — concluída:** VAD (`core/vad.py`: a gravação para sozinha
  quando você para de falar, `/vad` alterna), wake word (`core/wake.py` +
  `tools/treinar_wake.py`: o microfone fica aberto e só "Oráculo" abre um turno,
  `/despertar` alterna) e verificação de voz (`core/locutor.py` +
  `tools/cadastrar_voz.py`: fala que não é do dono é descartada antes do STT, `/dono`
  alterna)
- **Fase 4 — próxima:** RAG com Obsidian
- **Fase futura:** comandos do sistema com whitelist segura (5)

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
    ├── history.py   # persistência de sessões em JSON (~/.oraculo/sessions) + /retomar
    ├── prefs.py     # preferências que sobrevivem à sessão (~/.oraculo/prefs.json)
    ├── stt.py       # áudio → texto (faster-whisper na GPU | parakeet na CPU)
    ├── vad.py       # detecção de atividade de voz (Silero v6, streaming frame a frame)
    ├── wake.py      # palavra de despertar: mel + embedding (openWakeWord) + cabeça .npz
    ├── locutor.py   # verificação de voz: fbank Kaldi + WeSpeaker → só responde ao dono
    ├── transcript.py# transcrição de arquivos: parágrafos, Markdown, gravação
    ├── tts.py       # texto → áudio (Kokoro | Piper)
    ├── speaker.py   # fala em streaming: síntese + reprodução em pipeline, com barge-in
    ├── text.py      # limpeza de texto (strip_cjk no stream, for_speech antes do TTS)
    ├── audio.py     # captura de microfone + reprodução
    ├── keyboard.py  # monitor de tecla no terminal (Esc = barge-in, Ctrl+O = thinking)
    ├── telemetry.py # latência por estágio + tokens/s (opt-in, custo zero desligada)
    ├── title.py     # título da janela: o olho na aba/barra de tarefas
    ├── ui.py        # calha do transcript + paleta aplicada (tema, íris, avisos)
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

**VAD (`core/vad.py`) — o que custou tempo.**

- **O modelo já está no disco.** O `faster-whisper` empacota o `silero_vad_v6.onnx` em
  `assets/`, exposto por `faster_whisper.vad.get_vad_model()`. Nada a instalar, nada a
  baixar. Antes de adicionar dependência de áudio nova, **procure dentro do que já
  existe** — o venv é Python 3.14, onde `silero-vad` (arrasta torch) e `webrtcvad`
  (extensão C sem wheel) não entram. A regra para o resto da Fase 3 é a mesma: ONNX
  direto no `onnxruntime` que já está aqui.
- **`SileroVADModel.__call__` do faster-whisper não serve para streaming**: ele zera o
  estado `h`/`c` da LSTM a cada chamada e exige o áudio inteiro de uma vez. Chamá-lo por
  chunk de 512 amostras julga cada frame sem contexto. Por isso `core/vad.py` fala direto
  com a `InferenceSession` e carrega `h`/`c` de um frame para o outro. Os nomes das
  entradas (`input`/`h`/`c`) são **conferidos**, não assumidos — o v5 usava outra
  assinatura e o arquivo pode trocar numa atualização.
- **Inferência nunca no callback do PortAudio.** O callback só empilha na fila; quem
  decide é o laço na thread chamadora. É barato (~0,06 ms/frame, orçamento de 32 ms), mas
  bloquear no callback causa estouro de buffer.
- **Timeout de início em dois relógios, vale o que estourar primeiro.** O de frames
  descreve o áudio e mantém o teste offline determinístico; o de parede descreve o que o
  usuário sente. A primeira abertura do dispositivo custa perto de 1 s e os blocos chegam
  atrasados: contar só frames fez uma espera de 4 s durar 7,4 s na tela.
- **O `wait_stop` não entra no caminho do VAD, de propósito.** Chamar `sessao.wait_enter`
  em paralelo deixaria uma thread presa num `_fila.get()` depois que o VAD encerrasse
  sozinho, e ela engoliria a *próxima* mensagem digitada.
- **Antes de "consertar" um falso disparo, confirme que é falso.** O VAD disparando numa
  sala "silenciosa" parecia sensibilidade demais e quase virou um portão de energia (que
  teria abafado fala baixa de verdade). Era fala mesmo: o Whisper transcreveu palavras do
  ruído, e o espectro tinha 44% da energia em 1–3 kHz contra 1% abaixo de 150 Hz —
  ventoinha seria o inverso. **Transcreva o "ruído" e olhe o espectro** antes de mexer no
  limiar.

**Wake word (`core/wake.py`, `tools/treinar_wake.py`) — o que custou tempo.**

- **"Precisa de torch/TF de 2022" é falso, e quase custou o passo inteiro.** Essa fama
  vale para o *script de treino* do openWakeWord, não para a matemática. A arquitetura é
  um extrator **congelado** com uma cabeça minúscula por cima; treinar a cabeça é um MLP
  de 1536 entradas, que o sklearn faz no Python 3.14 sem GPU. Eu cheguei a recomendar um
  portão por transcrição por acreditar no contrário. **Antes de aceitar que uma
  dependência é impossível, olhe o que ela realmente faz.**
- **Os negativos já existem prontos.** `davidscripka/openwakeword_features` publica
  ~2000 h como *embeddings* já calculados. O `.npy` é contíguo depois do cabeçalho, então
  um range request nos primeiros N bytes dá as primeiras N/3072 janelas — não é preciso
  baixar 17 GB nem processar áudio nenhum.
- **Valide a cadeia contra um modelo pronto antes de treinar o seu.** Reimplementar
  mel→embedding→cabeça à mão tem várias chances de errar em silêncio (a transformação
  `(x/10)+2` é a mais traiçoeira: sem ela nada falha, o vetor só perde sentido). Sintetizar
  "hey jarvis" e rodar o `hey_jarvis_v0.1.onnx` oficial provou a cadeia em minutos: 0,998
  nos positivos, ≤0,31 nos negativos. Sem esse teste, um erro aqui viraria "o treino não
  converge" e a caçada seria no lugar errado.
- **O preenchimento de silêncio tem que passar do aquecimento.** O extrator só produz a
  primeira janela após ~2,5 s (76 frames de mel + 16 embeddings). Com 1 s de padding na
  frente, a janela que interessa — a que termina logo depois da palavra — ainda não existe
  quando a palavra acaba, e o clipe inteiro é descartado sem erro nenhum. E o padding é
  **ruído fraco, não zero**: silêncio digital absoluto não existe no microfone.
- **Separe treino de avaliação por CLIPE, nunca por janela.** As janelas de um mesmo clipe
  (e de seus aumentos) são quase duplicatas. Dividir por janela dá recall de 100% que não
  quer dizer nada. Eu publiquei esse número errado antes de perceber.
- **A janela positiva se ancora no fim da PALAVRA, não no fim do clipe.** Sintetizar
  "Oráculo, me ajuda." e rotular a janela do fim do clipe ensina o modelo a disparar no fim
  de "me ajuda". O sintoma foi cruel: 100% de recall na avaliação do treinador e falha
  exatamente no caso de uso principal — "Oráculo, que horas são?" pontuava 0,76 contra um
  limiar de 0,999. Por isso as frases positivas **terminam** na palavra, e o caso "chamou e
  já pediu" é montado colando uma continuação depois, com o corte no ponto que nós mesmos
  emendamos. Um teste offline independente do treinador é o que pega isso.
- **O `MLPClassifier` não aceita peso por classe.** Com 1 positivo para cada 100
  negativos ele aprende a responder "não" sempre — 99% de acerto e recall zero. Os
  positivos precisam ser replicados até uma proporção sã.
- **`.npy` grande se abre com `np.memmap`, nunca com `.tobytes()`.** Materializar os 2,5 GB
  de negativos na RAM derrubou o treino por OOM numa máquina de 16 GB (6,2 GB de RSS
  anônimo). Com memmap são 40 MB, e as conversões acontecem por lote. E o cabeçalho do
  `.npy` deve ser **lido** (`read_array_header_*`), não assumido em 128 bytes: errar isso
  desalinha todas as janelas em silêncio.
- **Limiar se mede, não se chuta — mas a regra precisa exigir FOLGA.** O treinador reserva
  horas de negativo que nunca viu e escolhe pelo falso positivo medido ali. Só que "o maior
  limiar com 0 falso positivo" leva ao extremo da tabela: com 0,999 uma fala legítima que
  pontua 0,998 é rejeitada, enquanto os negativos ficavam todos abaixo de 0,28. Daí o piso
  de *janelas* positivas (`--janela-min`) — o recall por fala é 100% em todo limiar e não
  discrimina nada sozinho.
- **Nada que contenha a palavra pode entrar na lista de negativos difíceis.** "oráculos de
  Delfos" esteve lá: é "Oráculo" + /s/, o detector acertava ao disparar e o rótulo é que
  estava errado — e isso envenena tanto o treino quanto a métrica. Quem trata "consultei o
  oráculo ontem" é a conferência de texto pós-transcrição (`WAKE_CONFIRMA_TEXTO`), porque o
  modelo acústico não tem como saber a posição da palavra na frase.
- **Nunca bloqueie na caixa de entrada em paralelo com a escuta.** É a mesma armadilha que
  manteve o `wait_stop` fora do VAD: uma thread parada em `_fila.get()` enquanto o
  microfone decide sozinho fica pendurada e engole a mensagem *seguinte*. Daí o
  `tui.ask_nowait()`, consultado a cada bloco de 80 ms.
- **O acumulador de áudio só nasce depois do gatilho**, dentro do `if`, nunca no topo da
  função. Um `frames = []` ligado desde o começo é uma gravação contínua da sala esperando
  um bug para virar arquivo. A garantia de privacidade é estrutural, não uma promessa.
- VAD e wake word usam tamanhos de frame diferentes (512 e 1280) que **não são
  múltiplos** — a sobra precisa atravessar as iterações do laço.
- **O dispositivo padrão do sistema não é necessariamente o microfone, e falhar assim não
  levanta exceção nenhuma.** Com PipeWire + EasyEffects, o `default`/`pulse` do PortAudio
  caía na *Easy Effects Source* (atenuada em −33 dB) e devolvia de 7% a 100% de silêncio
  digital; o ALSA cru do mic (`hw:1,7`) ficava em 0,1%. Daí `config.INPUT_DEVICE`, fixado
  por **nome** e não por índice — índice muda quando um fone entra.
- **Amostra exatamente zero denuncia captura quebrada.** Microfone real sempre tem ruído de
  fundo; zero exato é bloco perdido ou fonte muda. `audio.diagnostico_captura()` roda
  **antes** do `_normalize` — de propósito, porque era o próprio `_normalize` que
  amplificava o fragmento sobrevivente até 0,95 e fazia o arquivo quebrado parecer bom. Um
  WAV com 58% de zeros já foi gravado e guardado sem um aviso sequer.
- **Áudio de sala é o melhor teste de falso positivo que existe aqui.** 90 s de vídeo
  falando em português, com o detector escutando, sem um disparo — vale mais que qualquer
  negativo sintetizado.

**Verificação de voz (`core/locutor.py`, `tools/cadastrar_voz.py`) — o que custou tempo.**

- **PENDENTE: refazer o perfil com frases de verdade.** O perfil que está no disco foi
  montado só com os 40 clipes de 1 s de "Oráculo" que sobraram do treinador do wake word —
  ele descreve uma palavra mais do que uma voz. Falta rodar, na máquina do usuário:

      .venv/bin/python tools/cadastrar_voz.py --gravar 8
      .venv/bin/python tools/cadastrar_voz.py

  E depois o teste que realmente vale, pela regra do próprio projeto: deixar `/dono` ligado
  com um vídeo em português tocando na sala e conferir que nada passa. O corpus de negativos
  do cadastro tem só 10 locutores; áudio de sala é o melhor teste de falso positivo que
  existe aqui.

- **O que erra em silêncio aqui são as FEATURES, não o modelo.** A rede espera fbank de 80
  bandas no dialeto do Kaldi, e cada convenção conta: janela hamming simétrica, DC fora
  **antes** da pré-ênfase, pré-ênfase repetindo a primeira amostra (não zero), triângulos
  desenhados em mel, coluna de Nyquist zerada. Errar qualquer uma não levanta exceção — só
  encolhe a separação, e o sintoma chega disfarçado de "o limiar não acha um ponto bom".
  Confira a cadeia contra vozes conhecidas **antes** de medir limiar: o dono contra ele
  mesmo tem que dar alto, o mesmo clipe a 30% do volume tem que dar ~1,000 (é a CMN
  funcionando), e ruído branco tem que ficar longe do perfil. Foi o que provou a
  implementação em minutos, do mesmo jeito que o "hey jarvis" oficial provou a do wake word.
- **Negativo sintético mente, e mente para o lado otimista.** A primeira medição usou vozes
  de TTS como "outras pessoas" e deu uma folga enorme; só que duas vozes sintéticas
  *diferentes* chegam a 0,61 de similaridade entre si — mais do que muita gente de verdade.
  Sintético mede a distância do robô até você, que não é a pergunta. Com locutores reais
  (Multilingual LibriSpeech em português, servido linha a linha pelo datasets-server do HF)
  a conta fechou em 0,41 no pior negativo contra 0,58 na pior gravação do dono.
- **A avaliação do positivo precisa ser leave-one-out.** Medir um clipe contra um centróide
  que o contém é a mesma armadilha do "separe por clipe, não por janela" do wake word, só
  que na avaliação: o número sai otimista de graça.
- **O limiar fica no MEIO da folga**, não no extremo. "O maior limiar com zero falso
  positivo" foi o erro do wake word e teria dado 0,55+ aqui; o meio-termo (0,494) aceita
  100% do dono e nenhum dos 1038 negativos. Sem folga nenhuma o cadastro **recusa** gravar
  o perfil em vez de entregar um número bonito.
- **Fala curta demais PASSA, de propósito.** Abaixo de `LOCUTOR_MIN_SECONDS` o vetor de
  timbre ainda não se formou (8 dos 40 clipes de "Oráculo" caem aí). Rejeitar por falta de
  evidência tornaria "sim", "não" e "para" inutilizáveis; quem filtrou a sala nesse caso foi
  a wake word.
- **O portão fica antes do STT e apaga o WAV ao rejeitar.** É o único ponto por onde os três
  caminhos de captura passam, fala de outra pessoa não deve nem virar texto, e reter no
  `/tmp` a voz de quem não pediu nada seria guardar o que não é nosso. Pelo mesmo motivo o
  cadastro guarda só os *vetores* dos negativos e descarta o áudio de terceiros.
- **Verificação quebrada não pode calar o Oráculo.** Erro na conferência avisa, desliga a
  verificação e deixa o turno passar. O contrário — rejeitar tudo em silêncio — é o modo de
  falha que ninguém consegue diagnosticar.
- **O `/dono` é lembrado nas preferências; a wake word não.** Não é inconsistência: ele
  *fecha* um portão em vez de abrir o microfone, então herdá-lo ligado só restringe quem o
  Oráculo atende. Ver a nota no topo de `core/prefs.py`.
- **Isto não é autenticação, e a UI não pode sugerir que seja.** Um vetor de timbre é
  enganável por imitação e por uma gravação sua num alto-falante. Ele resolve a sala falando
  junto — prometer mais seria a mesma desonestidade que o system prompt evita.

**Sessões e preferências (`core/history.py`, `core/prefs.py`).**

- **`/retomar` redesenha a conversa inteira no transcript**, pela mesma calha de um turno ao
  vivo. A primeira versão mostrava só uma prévia de duas linhas e estava errada: retomar é
  voltar para a conversa, e sem ela na tela não há como saber o que foi dito ali. Sem
  rodapé de métricas (não houve turno agora) e sem raciocínio (nunca foi gravado); uma
  régua (`ui.divider`) marca onde o passado acaba.
- **`/retomar` continua o MESMO arquivo**, não abre outro: partir a conversa em dois deixa
  dois registros pela metade na lista de recentes, cada um parecendo ter morrido cedo.
- **O nome do arquivo de sessão precisava ser único.** O carimbo tem resolução de segundo,
  então duas sessões abertas no mesmo segundo (fechar e reabrir, dois terminais) caíam no
  mesmo arquivo e a segunda apagava a primeira em silêncio. Passava despercebido enquanto
  a sessão era só enfeite da splash; com o `/retomar` virou perda de dado.
- **A lista numerada do `/retomar` e a do autocomplete têm que sair do MESMO cache**
  (`commands.sessoes_recentes`). Reler o disco em cada lugar faria o "3" que você viu no
  menu abrir outra conversa se uma sessão fosse gravada no meio.
- **Preferência nasce só quando o usuário troca algo.** Um `prefs.json` vazio deixa o
  `config.py` mandar; se gravássemos tudo no primeiro arranque, editar o config depois não
  teria efeito nenhum e a causa seria invisível. `/padroes limpar` é a saída.
- **A wake word NUNCA é persistida**, e isso é decisão, não esquecimento: o projeto tem
  escrito que microfone aberto "é uma escolha do usuário, nunca um padrão herdado", e
  herdar por arquivo é herdar. Ver a nota no topo de `core/prefs.py`.
- **`/modelo` valida o nome antes de trocar e gravar.** O `ChatOllama` não valida na
  construção, então um modelo inexistente só falharia no meio do turno seguinte — e teria
  ido para as preferências, quebrando todas as sessões futuras. Aceita prefixo único.
- **A faxina de sessões tem duas travas**: idade (`SESSIONS_MAX_AGE_DAYS`, 0 desliga) e um
  piso de `SESSIONS_KEEP_MIN` mais recentes que ficam sempre. Sem o piso, voltar de três
  meses fora encontraria o histórico varrido. O que sai é anunciado — apagar conversa em
  silêncio seria pior.

**Autocomplete (`core/prompt.py`) — o Enter faz mais do que enviar.**

- **A primeira sugestão vem destacada, mexendo SÓ no `complete_index`.** O caminho normal
  do prompt_toolkit é `go_to_completion`, que além de destacar reescreve o texto do buffer:
  digitar `/` viraria `/ajuda` na caixa e a tecla seguinte sairia como `/ajudat`. Como
  `current_completion` é derivado do índice, atribuí-lo direto destaca sem tocar no
  documento. Isso cria um estado que o prompt_toolkit não tem — destacado mas não aplicado
  — e é por isso que existe `_destaque_nao_aplicado()`: Enter e Tab precisam distinguir
  "você escolheu com a seta" de "eu destaquei para você".
- **Destacar por conta própria obriga a blindar a leitura do índice.**
  `CompletionState.current_completion` indexa a lista **sem checar o tamanho** — o que se
  sustenta enquanto só o prompt_toolkit mexe no índice. Com o destaque sendo nosso, um
  índice sobrevive à troca de lista que o completer assíncrono faz e passa a apontar para
  fora dela. O `IndexError` subia de dentro do tratador do Enter, era engolido, e a
  mensagem **ficava parada na caixa sem erro nenhum na tela**: `/retomar 1` enviava e
  `/retomar 2` não, conforme quantas sugestões havia e quando o completer terminava. Daí
  `_sugestao_em_foco()` (nunca estoura) e o conserto do índice defasado em
  `_destacar_primeira()`. **Nunca leia `current_completion` direto neste arquivo.**
- **Não dá para confiar no `complete_state` no Enter.** Com `complete_while_typing` o menu
  é preenchido de forma assíncrona: quem digita rápido, ou cola a linha, chega no Enter
  antes de existir sugestão. Daí `_primeira_sugestao()` recalcular na hora — o sintoma
  dependia da velocidade de digitação, que é o tipo de bug que some quando se vai procurar.
- **Expandir abreviação ≠ digitar o comando inteiro.** Só quando o Enter *mudou* o texto é
  que ele abre espaço e espera o argumento; `/modelo` digitado por inteiro executa e lista.
  Antes eram dois Enters para isso.
- **Pasta se entra, arquivo se escolhe.** No argumento de caminho o Enter aplica a sugestão
  em foco: pasta ganha barra e o menu reabre com o conteúdo (não envia nada); arquivo
  completa e envia. Escolher arquivo só é seguro porque ele está **destacado na tela**. E o
  `PathCompleter` põe a barra final só no `display`, nunca no texto — sem acrescentá-la, o
  menu não reabre e a navegação para na primeira pasta.
- **O histórico só existe porque o Enter o grava.** Este Enter substitui o
  `validate_and_handle()` do prompt_toolkit, que é quem normalmente chama
  `append_to_history()`. Sem a chamada, `~/.oraculo/input_history` nunca era criado e a
  seta para cima percorria um histórico vazio — a caixa tinha `FileHistory` configurado
  desde sempre, só nunca escrito.
- **Ctrl+C ocioso limpa a caixa; só o segundo encerra** (`CTRL_C_EXIT_WINDOW`). O reflexo
  vindo do shell é usar Ctrl+C para apagar a linha, e isso fechava o Oráculo levando a
  mensagem junto. Digitar qualquer coisa cancela o armado, e a barra de status diz que o
  próximo toque encerra. Durante a geração e durante a escuta o Ctrl+C não muda de sentido.

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

- **Paleta em `config.py` (`UI_COLOR_*`), em hexadecimal, nunca nome de cor do rich.**
  Nome ("cyan", "grey42") resolve para a paleta do *emulador*: o mesmo código sai verde
  num terminal e azul noutro, e "grey30" some em fundo claro. O hex também é o que o
  prompt_toolkit come direto — é ele que mantém a caixa de entrada e a barra de status na
  mesma paleta do transcript, que é desenhado pelo rich. Cor nova entra no config; nenhum
  módulo inventa a sua.
- Papéis: ciano `ACCENT` (Oráculo, títulos, estado ativo), roxo `PROMPT` (`>` e nomes de
  comando), verde `BORDER` (molduras, divisores, a barra do raciocínio), `BRIGHT` (o que o
  usuário escreveu), `BODY` (resposta), `SOFT` (descrições), `DIM`/`FAINT` (rótulos,
  métricas, dicas), vermelho `ALERT` (avisos, erros e o ponto de gravação).
- **O rich traz o próprio tema de Markdown e ele ignora a cor base.** `ui.THEME` sobrescreve
  (número de lista, código, links, títulos) e `ui.make_console()` já o aplica — todo Console
  que renderiza transcript precisa dele, inclusive o Console descartável que `core/tui.py`
  cria por bloco ao reflowar. Sem isso, a primeira resposta com lista sai fora da paleta.
- **Canto vivo em toda moldura.** Splash (`box.SQUARE`) e caixa de entrada usam cantos
  retos: a moldura é estrutura, não enfeite.
- A íris `(✦ ✧ ✜ ✧)` é o olho do Oráculo: parada na splash, girando em `ui.Waiting` enquanto
  ele pensa, fala ou transcreve, e registrada como spinner do rich (`ui.IRIS`) para o
  `console.status`. `ui.Waiting` recalcula quadro e contador a cada renderização, sem thread
  e sem estado de animação guardado.
- Na splash o olho é `( ✦ )`, numa linha só. Já teve três, com as peças de parêntese grande
  (U+239B-23A0) empilhadas — num terminal não há corpo de fonte, então símbolo grande só
  existe ocupando várias linhas. Ficou pesado para o que é: uma marca de identidade no meio
  da coluna, não o assunto da tela.
- **A linha de espera se ancora na grade do turno, não na calha.** A íris cai na coluna do `●`
  (`ui._COL_GLIFO`) e o rótulo na coluna do corpo — ela ocupa o lugar da resposta que ainda não
  chegou. Recuar pela calha inteira jogava o olho quatro colunas à direita do ponto e a coluna
  do turno parecia torta.

**Título da janela (`core/title.py`).** O olho também vai para a aba e para a barra de
tarefas (`(✦) Oráculo · pensando`), girando enquanto o Oráculo trabalha — é o estado dele
visto de fora da janela. Duas regras:

- **Quem emite os bytes é a thread dona do stdout.** `marcar()` só anota o estado e é
  seguro em qualquer thread; `desenhar()` escreve. No fullscreen quem chama é o gancho
  `before_render` da app (`core/tui.py`), que roda na thread certa e já vem no ritmo do
  `refresh_interval`; no inline é a própria thread do laço, e **só nos limites do turno** —
  uma sequência OSC caindo dentro de um quadro do `rich.Live`, que repinta de outra thread,
  sai como lixo na tela. Por isso o olho não gira no inline: lá ele muda de estado, não de
  quadro. O laço nunca escreve direto; recebe `tick_title` injetado, como os outros
  ganchos por modo.
- **O título anterior volta.** `abrir()` empurra na pilha do terminal (CSI 22 t) e
  `fechar()` restaura (CSI 23 t). Sem isso a aba fica marcada "Oráculo" para sempre.

**A íris custou dois bugs, os dois silenciosos. Valem para qualquer glifo ou animação nova.**

- **Glifo dentro de faixa redirecionada do terminal some sem avisar.** `✦ ✧ ✜` (U+271C,
  U+2726-U+2727) são Dingbats e saíram **invisíveis** — um par de parênteses oco. O
  `~/.config/kitty/kitty.conf` tem `symbol_map U+2700-U+276D ... Noto Color Emoji`, e a Noto
  Color Emoji não contém Dingbats: o terminal procura o glifo lá, não acha e desenha nada. Sem
  erro, sem fallback, sem aviso. O conserto é uma linha `symbol_map` **posterior** no
  kitty.conf devolvendo esses codepoints à MesloLGL Nerd Font (em faixas sobrepostas, a última
  definição vence — `kitty.fonts.render.coalesce_symbol_maps` resolve por ordem de inserção).
  Ao adotar um glifo novo, **confira as faixas de `symbol_map` antes**; quando quiser algo que
  funcione em qualquer terminal sem ajuste, use U+25xx (Formas Geométricas), a faixa do `●`.
  Testar em `pyte` não pega isso: pyte é buffer de texto, não desenha fonte — o caractere
  *está* no stream, só não vira pixel.
- **O transcript da tela cheia cacheia o bloco renderizado.** `Transcript.lines()` só refaz um
  bloco quando alguém escreve nele, o que é justamente o ponto do cache — então um renderable
  que muda com o *relógio* congela no primeiro desenho. No inline o `rich.Live` repinta sozinho
  e o bug não aparece, o que o torna fácil de atribuir à fonte e parar de procurar. A saída é
  a marca `ANIMADO` em `ui.Waiting`, que `_animado()` procura no `append`/`replace_last` para
  invalidar o bloco a cada quadro. Ela é lida **no primeiro nível dos args**: passar a
  `Waiting` dentro de um `Padding`/`Group` esconde a marca e recongela a íris — por isso ela
  traz o próprio recuo e nunca passa por `ui.indent()`. `UI_IRIS_INTERVAL_MS` casa com o
  `refresh_interval` da app (200 ms): animar mais rápido que o repaint só pula quadros.
- Layout da splash: duas colunas estilo Claude Code — identidade à esquerda (a íris entre
  parênteses, modelo, memória, path), comandos + conversas recentes à direita.
- Biblioteca: `rich`. Usar `Table.grid` para o layout de colunas.
- Sem emoji no código de produção (a não ser que já esteja estabelecido na UI).
- **Nada de medidor de nível falso na gravação.** `ui.recording` mostra o ponto vermelho e
  o texto, sem barras de áudio: barra que não vem da amplitude real é enfeite mentindo
  sobre a captura, e é exatamente onde um bug de microfone se esconderia (ver a nota do
  `diagnostico_captura`).

**Calha do transcript (`core/ui.py`).** Todo turno segue a mesma estrutura vertical: eco
da pergunta recuado 2, cabeçalho `● Oráculo`, corpo recuado `UI_GUTTER`, rodapé com as
métricas, linha em branco. Saída nova no terminal passa pelos ajudantes do `ui` em vez de
`console.print` cru — é isso que mantém tudo na mesma margem *e* na mesma paleta:
`notice` (aparte), `ok` (confirmação, em acento), `warn`/`error` (alerta), `hint` (aparte
que precisa destacar um comando dentro do texto, aceita marcação), `heading` (título de
bloco de comando) e `recording` (microfone aberto). Blocos de lista (`/stt`, `/modelo`)
indentam manualmente para casar com a calha.

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

**Caixa de entrada (`core/prompt.py`).** `Application` inline do prompt_toolkit, não
`PromptSession`: o prompt padrão não fecha a borda direita. A moldura é remontada à mão
porque a classe `Border` do prompt_toolkit tem os cantos hard-coded — e é essa remontagem
que dá a ela a cor da paleta. A barra de status tem três grupos com pesos diferentes:
modelo em acento, flags permanentes (`texto`, `think off`, `mem 3/10`), o `state` do que
está acontecendo agora (`ouvindo "Oráculo"`, `rolagem pausada`, `seleção copiada`) também
em acento, e a dica de teclas encostada na borda **direita**. Quem enche o `state` é o
`_status` do main (wake word) e o `_status_com_dica` do tui (transiente) — separá-lo da
dica é o que impede a ajuda de teclado de piscar a cada seleção. Em terminal estreito cai
primeiro a dica, depois o `state`, depois os flags: a barra nunca pode quebrar em duas
linhas, isso desalinharia tudo abaixo da caixa. O
menu de completion entra no fluxo abaixo da barra de status — como `Float` ele seria
desenhado por cima da borda, já que numa app não-fullscreen o float não escapa da altura
da própria app. Atenção: `Buffer.cancel_completion()` **reverte** o texto ao original;
para só fechar o menu preservando o que o Tab inseriu, zere `buf.complete_state`.

## Como testar a interface

Não dá para conferir o desenho a olho num pipe — a interface só existe com TTY. O método
que funciona:

1. Suba o app num pty (`pty.fork`) e **defina o tamanho por `TIOCSWINSZ`**. O
   prompt_toolkit lê o tamanho por ioctl, não por `$COLUMNS`; sem isso ele desenha a 80
   colunas e o teste mede a largura errada.
2. Reproduza o stream num emulador de terminal e olhe a **tela**, não os bytes. `pyte`
   serve bem: `screen.display` dá o texto e `screen.buffer[y][x]` dá os atributos — é
   assim que se confere a seleção, que sai em vídeo reverso.
3. **Nunca** tire os ANSI do stream para "ler melhor": o repaint é posicional e o
   resultado parece corrompido mesmo estando certo. Isso já custou uma caçada a um bug
   inexistente.
4. Eventos de mouse entram no protocolo SGR: `\x1b[<{botão};{x};{y}M` (`m` solta,
   botão `32` = arrasto).

`pyte` é **só para teste** — não entra no `requirements.txt`. Instale à parte quando
precisar: `.venv/bin/python -m pip install pyte`.

Quando o comportamento não reproduz, **instrumente o handler** (escrever num arquivo
temporário já basta) em vez de deduzir pelo código. As armadilhas do prompt_toolkit
listadas acima só apareceram assim.

## Antes de finalizar qualquer mudança

- Confirmar que roda no venv sem erro de import.
- Confirmar que o modo texto continua funcionando.
- Não fixou credenciais, caminhos absolutos pessoais ou segredos no código.
- Atualizou o `README.md` se mudou instalação, uso ou estrutura.
