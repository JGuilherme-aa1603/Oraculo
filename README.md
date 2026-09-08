# Oráculo

Assistente de voz local estilo Jarvis, rodando 100% offline. Desenvolvido em fases
incrementais. Usa um modelo local via [Ollama](https://ollama.com) orquestrado pelo
[LangChain](https://www.langchain.com).

- **Fase 1 (MVP):** chat de terminal com memória de conversação.
- **Fase 2 (Voz):** entrada/saída de voz opcional (Whisper STT + Kokoro TTS) com fala em
  streaming e barge-in, roteamento de comandos, transcrição de arquivos, telemetria
  opt-in e persistência de sessões. O modo texto continua padrão.
- **Fase 3 (Sempre ouvindo):** parada automática por VAD, wake word "Oráculo" treinada na
  sua voz e verificação de locutor — o Oráculo responde só a você.
- **Fase 4 (Notas):** consulta ao seu vault do Obsidian antes de responder — busca
  híbrida (vetor + termos) sobre um índice local, com as notas consultadas visíveis
  no rodapé de cada turno.

## Requisitos

- Ollama rodando em `http://localhost:11434` (`ollama serve` se não estiver ativo)
- Modelo `gemma4:e4b` baixado (`ollama pull gemma4:e4b`) — ou outro, ajustando
  `OLLAMA_MODEL` no `config.py`
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
- `/vad` — liga/desliga a parada automática da gravação (desligado = push-to-talk)
- `/despertar` — liga/desliga a escuta pela palavra "Oráculo" (microfone sempre aberto)
- `/dono` — liga/desliga responder só à sua voz (exige o perfil cadastrado)
- `/notas` — liga/desliga a consulta às suas notas do Obsidian
- `/indexar` — (re)indexa o vault
- `/buscar <pergunta>` — mostra os trechos que a busca traria, sem gastar o LLM
- `/think` — liga/desliga o raciocínio (thinking) do modelo
- `/stt` — lista os motores de transcrição; `/stt <motor>` troca (`whisper`/`parakeet`)
- `/transcrever <arquivo> [--salvar]` — transcreve um arquivo de áudio
- `/modelo` — lista os modelos do Ollama; `/modelo <nome>` troca o ativo (basta o começo do nome)
- `/retomar` — lista as conversas anteriores; `/retomar <n>` continua uma delas
- `/padroes` — mostra as preferências guardadas; `/padroes limpar` esquece tudo
- `/limpar` — apaga a memória da conversa atual
- `/sair`, `/exit`, `/quit` — encerra o Oráculo

### Retomar uma conversa

`/retomar` lista as últimas conversas numeradas e `/retomar 3` continua a terceira de onde
ela parou. **A conversa inteira volta para a tela**, no mesmo formato de sempre, para você
ler e rolar; abaixo dela uma régua marca onde a conversa antiga acaba e a nova começa. As
mensagens voltam também para a memória do modelo (as últimas que couberem na janela de
contexto — o Oráculo avisa quando o arquivo tem mais do que cabe), e os turnos novos são
anexados ao **mesmo arquivo**, em vez de abrir um segundo registro pela metade.

### Preferências lembradas

O que você troca em conversa (`/modelo`, `/think`, `/stt`, `/vad`, `/voz`, `/dono` e o Ctrl+O) volta
igual na próxima sessão, gravado em `~/.oraculo/prefs.json`. O `config.py` é o padrão de
fábrica; o arquivo só ganha uma chave quando você troca aquilo em conversa, e aí ele vence.
`/padroes` mostra o que está guardado e `/padroes limpar` devolve o comando ao `config.py`.

**A escuta pela palavra "Oráculo" nunca é lembrada.** Manter o microfone aberto é escolha de
cada sessão, nunca herdada de ontem por um arquivo; quem quiser a escuta ligada de fábrica
muda `WAKE_ENABLED` no `config.py`, que é um ato deliberado e visível. O `/dono`, ao
contrário, **é** lembrado: ele fecha um portão em vez de abrir o microfone, então herdá-lo
ligado só restringe quem o Oráculo atende.

### Limpeza do histórico

No arranque, conversas com mais de `SESSIONS_MAX_AGE_DAYS` (90) são apagadas — mas as
`SESSIONS_KEEP_MIN` (20) mais recentes ficam sempre, por mais antigas que sejam, para que
voltar de uma temporada longe não encontre o histórico varrido. O que sai é anunciado na
tela. `SESSIONS_MAX_AGE_DAYS = 0` desliga a limpeza.

A conversa é desenhada numa calha: a pergunta aparece recuada, a resposta abre com
`● Oráculo` e o corpo fica alinhado numa coluna, com um rodapé discreto trazendo o
tempo do turno e a taxa de tokens/s.

### Modos de tela

O Oráculo desenha de dois jeitos, escolhidos por `TUI_MODE` no `config.py`:

| `TUI_MODE` | Comportamento |
|---|---|
| `"fullscreen"` (padrão) | Abre na tela alternativa do terminal (como `vim`/`htop`). A caixa de entrada fica **fixa no rodapé** e a conversa rola acima dela. Ao sair, o terminal volta como estava, sem rastro. |
| `"inline"` | Desenha no buffer normal: a rolagem nativa do terminal vale, e o transcript permanece na tela depois de encerrar. `CLEAR_ON_START` limpa a tela no arranque. |

No modo fullscreen a rolagem é da própria aplicação (a tela alternativa não tem
scrollback do terminal):

- **PgUp / PgDn** — sobe e desce uma página
- **Ctrl+Home / Ctrl+End** — início e fim da conversa
- **Roda do mouse** — rola o transcript (`TUI_SCROLL_LINES` ajusta a velocidade)

Rolar para cima pausa o acompanhamento automático, para a resposta que continua
chegando não arrastar a leitura de volta; voltar ao fim religa. Durante a geração a
caixa não sai do lugar, e **Ctrl+C** corta a resposta sem encerrar a sessão.

### Selecionar e copiar

A seleção é da própria aplicação, então funciona com o mouse capturado — sem alternar
nada:

- **Arrastar** seleciona um trecho; arrastar contra a borda rola junto, então dá para
  pegar um texto maior que a tela
- **Duplo clique** seleciona a palavra; **triplo clique**, a linha
- **Esc** limpa a seleção

Selecionar pausa o acompanhamento automático, para o texto não escorregar sob o
ponteiro. A barra de status avisa (**rolagem pausada**); Ctrl+End volta ao fim, e
enviar uma mensagem nova também.

O texto vai para a área de transferência assim que você solta o botão, e a barra de
status confirma. Se houver `wl-copy`, `xclip` ou `xsel` instalado, a cópia usa a
ferramenta; senão o Oráculo emite **OSC 52**, uma sequência que pede ao próprio
terminal para copiar — não exige nada instalado, mas alguns terminais a desativam por
padrão. Se a cópia não chegar ao clipboard, instale o `wl-clipboard`
(`sudo pacman -S wl-clipboard` no Wayland) ou o `xclip` no X11.

Como último recurso, **F2** solta o mouse e devolve a seleção nativa do terminal (a
rolagem passa a ser só por PgUp/PgDn); F2 de novo recaptura. Para começar assim,
`TUI_MOUSE = False` no `config.py`.

Sem terminal interativo (pipe, redirecionamento) ou sem `prompt_toolkit`, o modo
`inline` entra sozinho.

**Caixa de entrada:** o texto é digitado numa caixa com borda, e abaixo dela uma barra
mostra o modelo, o modo (texto/voz), o estado do thinking e a ocupação da memória.

- **↑/↓** navegam o histórico, que persiste entre sessões (`~/.oraculo/input_history`)
- **`/`** abre o autocomplete dos comandos, **já com a primeira opção destacada** — a seta
  para baixo vai para a segunda, e **Enter** aceita a que está em foco
- **Tab** completa sem enviar
- **Alt+Enter** quebra linha sem enviar (mensagens de várias linhas)
- **Ctrl+D** encerra; **Ctrl+C** limpa a caixa, e um segundo Ctrl+C encerra

Você não precisa digitar o comando inteiro: `/th` + Enter roda `/think`, e `/modelo qwen2`
acha o `qwen2.5:7b`. Comandos que esperam argumento (`/modelo`, `/transcrever`, `/stt`,
`/retomar`) abrem espaço e aguardam quando você **abrevia**; digitados por inteiro, o Enter
executa direto — `/modelo` sozinho lista os modelos.

Em `/transcrever`, **Enter numa pasta entra nela** e mostra o que há dentro, um nível de
cada vez, em vez de exigir o caminho completo de cabeça. Enter num arquivo o escolhe e
envia. Arquivo nunca é escolhido às cegas: o que vai ser aceito está destacado na tela.

**Ctrl+C** ocioso limpa a caixa de entrada e arma a saída por `CTRL_C_EXIT_WINDOW` segundos
(5); o segundo Ctrl+C dentro dessa janela encerra, e qualquer tecla cancela. Durante uma
resposta ele continua cortando a geração, e durante a escuta encerra só a escuta.

A caixa cresce até 8 linhas e depois rola. Para selecionar uma mensagem longa com o
mouse, arraste contra a borda de cima ou de baixo: a caixa rola junto e a seleção
acompanha até o começo (ou o fim) do texto. Com um trecho selecionado, digitar
substitui e **Backspace**/**Delete** apagam, como em qualquer editor.

`/tr` + Enter completa para `/transcrever` e espera o arquivo; comandos sem argumento
(como `/limpar`) são enviados na hora. Com o autocomplete de caminhos, **Tab** completa
o nome do arquivo a transcrever.

Sem `prompt_toolkit` instalado, ou rodando sem terminal interativo (pipe, redirecionamento),
a entrada cai automaticamente para o prompt simples do rich — nada deixa de funcionar.
Para desligar a caixa de propósito, `INPUT_RICH_EDITOR = False` no `config.py`.

No modo voz, pressione Enter para abrir o microfone e simplesmente fale: o VAD
percebe quando você parou e encerra a gravação sozinho (não há segundo Enter). Se
ninguém falar em 8 segundos, ele desiste e devolve o prompt. Digitar um texto e dar
Enter continua servindo de atalho. A transcrição usa o Whisper
`large-v3` na GPU (quase em tempo real); a fala é sintetizada pelo Kokoro (voz feminina).

**Parada automática (VAD):** `/vad` liga e desliga. Desligado, volta ao push-to-talk
clássico — Enter para gravar, Enter de novo para parar. O modelo (Silero v6) já vem
dentro do `faster-whisper`, então não há dependência nem download extra. Se a gravação
estiver cortando o fim das suas frases, aumente `VAD_SILENCE_MS` no `config.py`; se
estiver disparando com ruído de fundo, suba o `VAD_THRESHOLD`.
As respostas são renderizadas como Markdown no terminal; antes da síntese de voz a
marcação é removida para a fala não soletrar símbolos.

**Chamar pelo nome (wake word):** com `/despertar`, o Enter também some — o microfone
fica aberto e basta dizer **"Oráculo, que horas são?"**. O nome é removido antes de a
frase chegar ao modelo. Dizendo só "Oráculo", ele responde e abre uma escuta de
continuação para você formular o pedido. **Ctrl+C** encerra a escuta sem sair do
Oráculo.

No modo tela cheia dá para simplesmente **digitar** enquanto ele escuta — a mensagem tem
preferência e a escuta é abandonada. No modo inline isso não vale (não há leitura não
bloqueante do terminal): use Ctrl+C primeiro e depois digite.

O detector é acústico e não sabe onde a palavra caiu na frase, então "consultei o **oráculo**
de Delfos" acorda ele com razão. Depois de transcrever, o Oráculo confere que o nome está no
começo e, se não estiver, descarta o turno e volta a ouvir (`WAKE_CONFIRMA_TEXTO`).

O que acontece com o que é dito **sem** o nome:

- não vai para o disco — o áudio vive num anel em memória de 2 s que se sobrescreve;
- não é transcrito — o portão é um classificador que devolve um número, não texto;
- não chega ao modelo nem ao transcript.

Enquanto a escuta está ligada, a barra de status mostra `ouvindo "Oráculo"`, para nunca
haver dúvida sobre o microfone estar aberto. O modo nasce **desligado** e só liga com
`/despertar`.

**Custo medido** (Raptor Lake, sala silenciosa, cinco medições de 45 s): **3% a 5% de um
núcleo** — cerca de 1% é o microfone aberto e o resto é o detector, que gasta ~2,3 ms por
bloco de 80 ms (quase tudo no modelo de embedding). A variação vem da carga de fundo. Some
~0,5 s de arranque, uma vez, para carregar os modelos. A memória é o anel de pré-roll:
88 KB.

Antes do primeiro uso é preciso treinar a cabeça do detector — uma vez, na sua máquina,
sem GPU e sem torch:

```fish
.venv/bin/python -m pip install scikit-learn   # só para treinar
.venv/bin/python tools/treinar_wake.py --gravar 40   # você dizendo "Oráculo" 40x
.venv/bin/python tools/treinar_wake.py --baixar-vozes
```

**Se o microfone não parecer funcionar**, teste antes de gravar qualquer coisa:

```fish
.venv/bin/python tools/treinar_wake.py --testar-microfone
```

Ele grava um trecho em cada entrada, mede a captura, transcreve e diz qual serve. O padrão
do sistema **nem sempre é o microfone**: nesta máquina, com PipeWire e EasyEffects no
caminho, as rotas `default`/`pulse`/`pipewire` perdiam de 7% a 20% dos blocos (e às vezes
devolviam 100% de silêncio digital), enquanto o dispositivo ALSA do microfone ficava em
0,1%. Daí `INPUT_DEVICE = "hw:1,7"` no `config.py` — troque pelo que o teste indicar.

Amostra exatamente zero não existe em microfone real (sempre há ruído de fundo), então o
Oráculo trata isso como captura quebrada e recusa a gravação com a causa na tela, em vez de
guardar um arquivo ruim. Repare que abrir o dispositivo ALSA direto **contorna o PipeWire**,
o que também dispensa qualquer cancelamento de eco que estivesse configurado lá — se o
Oráculo passar a se ouvir falando, é por aí.

O `--gravar` é opcional mas faz muita diferença: as vozes sintéticas pt-BR disponíveis
têm timbre parecido demais, e sem amostras suas o modelo aprende "voz de robô" em vez de
"a palavra". O treinador imprime a taxa de falso positivo por hora medida em áudio real
que ele nunca viu, e escolhe o limiar a partir dela — se o número sair ruim, ele diz.

**Responder só a você (verificação de voz):** com `/dono`, cada fala captada é comparada
com o seu perfil de timbre antes de virar texto. Se não for a sua voz, o áudio é
descartado ali mesmo — sem transcrição, sem resposta, e o WAV temporário é apagado. É o
que impede uma televisão ligada, um vídeo ou uma visita de abrir um turno.

Como funciona: o `voxceleb_resnet34_LM` do WeSpeaker (26 MB, ONNX no `onnxruntime` que já
está aqui) transforma a fala num vetor de 256 números que descreve o timbre, e o cosseno
com o seu perfil decide. Nada de texto e nada de conteúdo entram nessa conta.

**Isto não é autenticação.** Um vetor de timbre é enganável por imitação e por uma
gravação sua tocada num alto-falante. Ele resolve o problema real (a sala falando junto),
não um adversário.

Falas com menos de `LOCUTOR_MIN_SECONDS` (1 s) **passam sem julgamento**: em meio segundo
o vetor ainda não se formou, e rejeitar por falta de evidência tornaria "sim", "não" e
"para" inutilizáveis. Nesse caso quem filtrou a sala foi a wake word.

Cadastre a sua voz uma vez — sem GPU e sem torch:

```fish
.venv/bin/python tools/cadastrar_voz.py --gravar 8   # 8 frases suas, ~4 s cada
.venv/bin/python tools/cadastrar_voz.py              # mede o limiar e grava o perfil
```

O cadastro aproveita também os clipes de "Oráculo" que o treinador do wake word deixou em
`~/.oraculo/voice/`. Para medir o limiar ele baixa fala de **outras pessoas** (português
real, do Multilingual LibriSpeech), guarda só os vetores e descarta o áudio; o limiar sai
no **meio da folga** entre a sua pior gravação e o negativo mais parecido, e o script
recusa gravar o perfil se folga não houver. `--avaliar` mostra a tabela sem gravar nada,
e `--esquecer` apaga o perfil.

Medição nesta máquina (40 clipes seus, 1038 janelas de 10 locutores): você entre 0,58 e
0,71, outras pessoas no máximo 0,41 — limiar 0,494, com 100% de aceitação sua e nenhum
falso positivo. Enquanto a verificação está ligada, a barra de status mostra `só você`.

Enquanto a resposta não começa, um indicador mostra **"Carregando modelo..."** se o
Ollama ainda está subindo o modelo na VRAM (cold start) ou **"Pensando..."** quando
ele já está carregado e gerando.

**Interromper a fala (barge-in):** no modo voz, pressione **Esc** para o Oráculo parar
de falar na hora e liberar o prompt para a próxima mensagem — sem precisar esperar ele
terminar de ler a resposta.

**Transcrever arquivos:** `/transcrever <arquivo>` transcreve um áudio ou vídeo já
gravado (ogg/opus do WhatsApp, mp3, m4a, wav, mp4...) usando o motor STT ativo. O
caminho pode conter espaços e não precisa de aspas; `~` é expandido. O texto sai no
terminal em parágrafos com timestamps, conforme é transcrito, e **Ctrl+C** interrompe
sem sair do Oráculo. Com `--salvar` (ou `-s`), grava um `.md` com cabeçalho de metadados
ao lado do áudio — ou em `TRANSCRIBE_OUTPUT_DIR`, se configurado.

```
/transcrever ~/Downloads/WhatsApp Ptt 2026-08-01 at 8.14.38 PM.ogg --salvar
```

Também funciona direto do shell, sem abrir o chat (ver [Comando global](#comando-global)):

```fish
oraculo transcrever ~/Downloads/audio.ogg --salvar
```

Não é preciso converter o formato: o faster-whisper decodifica via PyAV. Para áudios
longos use o `whisper` — o `parakeet` trunca acima de ~30s (o comando avisa).

**Raciocínio (thinking):** modelos com a capacidade `thinking` (gemma4, qwen3...) podem
raciocinar antes de responder. Ligue/desligue com `/think` (desligado por padrão, pois
adiciona latência). Quando ligado, o indicador mostra **"Pensando..."** apenas enquanto
há raciocínio real acontecendo; pressione **Ctrl+O** durante a resposta para mostrar/ocultar
o texto do raciocínio ao vivo. Modelos sem suporte são detectados e o `/think` avisa.

### Consultar as suas notas do Obsidian (Fase 4)

Com `/notas`, cada pergunta passa antes por uma busca no seu vault e os trechos mais
relevantes chegam ao modelo junto com a pergunta. As notas que entraram aparecem no
rodapé do turno — sem isso, "respondeu errado" e "respondeu sem consultar" seriam
indistinguíveis na tela.

Aponte `RAG_VAULT` para o seu vault no `config.py` e indexe uma vez:

```bash
.venv/bin/python tools/indexar_vault.py            # ~11s para 258 notas
```

Depois é só ligar com `/notas` dentro da conversa; a escolha é lembrada. Quando você
escrever notas novas, `/indexar` refaz o índice (o comando avisa sozinho quando o vault
mudou desde a última vez).

**Como a busca funciona.** O vetor (`nomic-embed-text`, no mesmo Ollama) encontra o
assunto; um BM25 sobre os mesmos trechos encontra o *termo*. Os dois rankings são
fundidos, e é essa metade léxica que resolve nomes próprios e jargão — "PortAudio",
"Kokoro", "VAD" — que o embedding dilui. O índice mora em `~/.oraculo/rag/indice.npz`,
fora do vault: ele é derivado, e escrever um arquivo nosso no meio das suas notas
sujaria o vault e a sincronização.

**Quando a resposta sair errada, use `/buscar`.** Ele mostra exatamente o que teria ido
para o contexto, sem gastar o LLM — e marca com `x` o que o limiar barrou, com o score.
É o que distingue "a busca trouxe o trecho errado" de "trouxe o certo e o modelo
ignorou", que a resposta sozinha não distingue.

**Sobre o limiar.** `RAG_MIN_SCORE` é medido, não chutado:

```bash
.venv/bin/python tools/indexar_vault.py --calibrar
```

A calibração compara perguntas que o vault responde — as **suas**, em
`~/.oraculo/rag/perguntas.txt` — com perguntas fora do domínio. Neste vault a folga é de
oito milésimos (0,668 contra 0,676), então o número é frágil por natureza: vale
acrescentar perguntas ao arquivo e remedir depois de escrever notas novas. E o comando
se recusa a inventar o lado positivo sozinho, porque uma frase copiada da nota mede a
facilidade, não a busca.

**O system prompt muda junto com a capacidade.** Com `/notas` ligado o Oráculo declara
que lê as suas notas indexadas; desligado, ele volta a declarar que não acessa arquivo
nenhum. As duas metades precisam ser verdadeiras ao mesmo tempo, então elas mudam
juntas — e ele nunca lê nada além dos trechos que a busca trouxe.

### Comando global

O wrapper `bin/oraculo` roda o projeto de qualquer diretório usando o venv, sem
precisar ativar nada. Instale com um link simbólico em algum diretório do `PATH`:

```fish
ln -s (pwd)/bin/oraculo ~/.local/bin/oraculo
ln -s (pwd)/completions/oraculo.fish ~/.config/fish/completions/  # completions (opcional)
```

Se `~/.local/bin` não estiver no `PATH`: `fish_add_path ~/.local/bin`.

```fish
oraculo                              # abre o chat
oraculo transcrever audio.ogg -s     # transcreve e sai (sem carregar o Ollama)
oraculo ajuda                        # lista os comandos
```

Com argumentos, o Oráculo executa o comando e sai; a barra é opcional. Só os
comandos que não dependem do LLM rodam assim (`transcrever`, `ajuda`) — os demais
avisam que só funcionam dentro do chat.

## Estrutura

```
oraculo/
├── main.py          # Entry point — loop de conversa (texto/voz) e comandos do shell
├── bin/oraculo      # Wrapper para rodar de qualquer diretório (link no PATH)
├── config.py        # Configurações centralizadas
├── core/
│   ├── llm.py       # Configuração do Ollama + LangChain
│   ├── memory.py    # Memória de conversação (janela deslizante, em pares)
│   ├── chain.py     # Pipeline: system prompt + memória + LLM (eventos think/answer)
│   ├── commands.py  # Roteamento de comandos (/ajuda, /voz, /think, /modelo, ...)
│   ├── history.py   # Persistência de sessões em JSON (~/.oraculo/sessions)
│   ├── stt.py       # Whisper (faster-whisper) — áudio → texto
│   ├── vad.py       # Detecção de atividade de voz (Silero) — sabe quando você parou
│   ├── wake.py      # Palavra de despertar "Oráculo" (openWakeWord + cabeça própria)
│   ├── locutor.py   # Verificação de voz: só responde ao dono (WeSpeaker)
│   ├── rag.py       # Notas do Obsidian: trechos, vetores e busca híbrida
│   ├── transcript.py# Transcrição de arquivos: parágrafos, Markdown, gravação
│   ├── tts.py       # Kokoro/Piper — texto → áudio
│   ├── text.py      # Limpeza de texto (remove Markdown p/ voz, filtra CJK)
│   ├── audio.py     # Captura de microfone (push-to-talk e VAD) + reprodução
│   ├── keyboard.py  # Monitor de tecla no terminal (barge-in por Esc)
│   ├── telemetry.py # Latência por estágio + tokens/s (opt-in)
│   ├── title.py     # Título da janela: o olho na aba e na barra de tarefas
│   ├── ui.py        # Calha do transcript (eco, cabeçalho, corpo, rodapé)
│   ├── prompt.py    # Caixa de entrada: borda, histórico e autocomplete
│   ├── tui.py       # Modo tela cheia: transcript rolável + caixa fixa no rodapé
│   └── splash.py    # Splash screen de duas colunas (rich)
├── tools/
│   ├── treinar_wake.py  # Treina a cabeça do wake word (roda uma vez, fora do app)
│   ├── cadastrar_voz.py # Cadastra a sua voz e mede o limiar (roda uma vez)
│   └── indexar_vault.py # Indexa o vault, inspeciona a busca e mede o limiar
├── requirements.txt
└── README.md
```

As sessões são salvas em `~/.oraculo/sessions/*.json` e aparecem em "Conversas
recentes" na splash.

## Configuração

Ajuste `config.py` para trocar o modelo, parâmetros de geração (temperatura,
`num_ctx`, `MAX_TOKENS`/num_predict), tamanho da memória, system prompt, motor e
modelo de STT (`STT_ENGINE`, `WHISPER_MODEL`), motor e voz de TTS (`TTS_ENGINE`,
`KOKORO_VOICE`), o raciocínio padrão (`THINKING_DEFAULT`) e o modo padrão
(`VOICE_MODE_DEFAULT`).

A aparência do terminal fica no bloco **Interface do terminal**: `UI_MAX_WIDTH` (teto da
coluna de leitura; `0` usa o terminal inteiro), `UI_GUTTER` (recuo do corpo), os glifos,
`UI_SHOW_TURN_METRICS` (rodapé com as métricas do turno), `CLEAR_ON_START` (limpar a
tela ao abrir), `TITLE_ENABLED` (o olho no título da janela) e `INPUT_RICH_EDITOR`.

Com `TITLE_ENABLED`, a aba e a barra de tarefas mostram `(✦) Oráculo`, com o estado ao lado
(`· pensando`, `· ouvindo`, `· falando`) e a íris girando enquanto ele trabalha — dá para
saber que a resposta ainda está vindo sem trazer o terminal para a frente. O título que o
terminal tinha antes é restaurado ao sair.

A paleta são os `UI_COLOR_*`, em hexadecimal: ciano (`ACCENT`, o Oráculo e os títulos),
roxo (`PROMPT`, o `>` e os comandos), verde (`BORDER`, molduras e divisores), três tons
de texto (`BRIGHT`/`BODY`/`SOFT`), dois de apoio (`DIM`/`FAINT`) e um vermelho de alerta.
Mudar uma dessas linhas repinta o transcript, a splash, a caixa de entrada e a barra de
status de uma vez — o rich e o prompt_toolkit leem os mesmos valores. É hexadecimal, e não
nome de cor, porque nome resolve para a paleta do emulador: o mesmo desenho sairia
diferente em cada terminal. `UI_IRIS_FRAMES` são os quadros do olho que gira enquanto o
Oráculo pensa, fala ou transcreve.

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
| 3 — Wake Word | VAD (Silero) ✅ · wake word "Oráculo" ✅ · verificação de voz ✅ | ✅ Concluída |
| 4 — RAG | Notas do Obsidian: busca híbrida sobre índice local ✅ | ✅ Concluída |
| 5 — Commands | Executar comandos do sistema com whitelist segura | ⏳ Futuro |
