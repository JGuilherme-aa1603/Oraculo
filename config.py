"""Configurações centralizadas do Oráculo."""

import os
from pathlib import Path

# --- Identidade / versão ---
ASSISTANT_NAME = "Oráculo"
APP_VERSION = "1.0.0"
USER_NAME = None  # None → detecta pelo usuário do sistema (getpass.getuser)

# --- Modelo Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"

# --- Parâmetros do modelo ---
TEMPERATURE = 0.7
NUM_CTX = 8192       # Mantido baixo para economizar VRAM (padrão do Ollama pode ser 131072)
MAX_TOKENS = 2000    # num_predict — limite de tokens na resposta.
                     # Ajustável: respostas mais longas podem exigir um valor maior;
                     # para conversa normal, 2000 é folgado.

# --- Raciocínio (thinking) ---
# Modelos com capacidade "thinking" (gemma4, qwen3...) podem raciocinar antes de
# responder. Custa latência, então o padrão é desligado; /think alterna na sessão.
# (No futuro este flag pode virar um modo "auto".)
THINKING_DEFAULT = False         # liga o raciocínio por padrão
SHOW_THINKING_DEFAULT = False    # exibe o texto do raciocínio ao vivo (Ctrl+O alterna)

# --- Memória ---
MAX_HISTORY_MESSAGES = 20  # Mantém as últimas N mensagens (sempre cortando em pares user/assistant)

# --- Voz / STT ---
STT_ENGINE = "whisper"          # whisper | parakeet
# whisper:  faster-whisper, VAD embutido, fallback CUDA→CPU. Bom p/ clipes longos.
# parakeet: NVIDIA Parakeet TDT 0.6b v3 (onnx-asr). Muito rápido na CPU e pontua
#           sozinho, mas tem limite de ~20-30s por clipe. Na conversa isso não
#           incomoda (o VAD recorta a fala e VAD_MAX_SECONDS a limita); em
#           /transcrever de arquivo longo, degrada — prefira o whisper.

# Whisper (faster-whisper) — backend padrão.
WHISPER_MODEL = "large-v3"          # base | small | medium | large-v3 (maior = mais preciso)
WHISPER_DEVICE = "cuda"             # cpu | cuda (cuda exige os wheels nvidia-*-cu12; ver nota)
WHISPER_COMPUTE_TYPE = "int8_float16"  # cuda→float16/int8_float16 | cpu→int8
WHISPER_BEAM_SIZE = 5           # busca em feixe: mais alto = mais preciso, um pouco mais lento
# Contexto inicial dado ao Whisper para enviesar a transcrição ao domínio da
# conversa (reduz erros como "software"→"sótua"). Não é texto a transcrever.
WHISPER_INITIAL_PROMPT = (
    "Conversa em português brasileiro sobre tecnologia, programação, "
    "desenvolvimento de software e o assistente Oráculo."
)
# GPU: o ctranslate2 exige CUDA 12 (libcublas.so.12 + cuDNN 9), mas o sistema tem
# CUDA 13. Contornado instalando as libs userspace no venv:
#   .venv/bin/python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
# O stt._enable_cuda_libs() pré-carrega esses .so antes de criar o modelo. Com
# 'int8_float16' o large-v3 ocupa ~2GB de VRAM, deixando espaço para o Ollama nos
# 8GB da RTX 4060. Se a GPU/libs faltarem, há fallback automático para CPU (int8).

# Parakeet (onnx-asr) — motor alternativo, rápido na CPU. Multilíngue; não tem
# initial_prompt nem VAD. Modelo ONNX baixado do Hugging Face na 1ª execução e
# cacheado em ~/.cache/huggingface.
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"
# Fixa o idioma em vez de deixar a detecção automática (que erra/oscila e piora
# o reconhecimento de palavras). None volta ao auto-detect.
PARAKEET_LANGUAGE = "pt"

RECORD_DURATION = 5.0           # segundos (modo gravação fixa)
RECORD_SAMPLERATE = 16000
# Dispositivo de entrada. None usa o padrão do sistema — que nem sempre é o
# microfone: num sistema com PipeWire + EasyEffects, o "default" do ALSA cai na
# fonte do EasyEffects, e ela pode entregar silêncio digital ou áudio atenuado
# em -33 dB. Aceita índice (int) ou parte do nome (str); veja a lista com
#   .venv/bin/python -c "from core import audio; audio.listar_entradas()"
#
# Aqui está fixado no microfone digital pelo nome ALSA, e não pelo índice: os
# índices mudam quando um fone entra ou sai. Medido nesta máquina, as rotas via
# PipeWire perdiam de 7% a 20% dos blocos, enquanto esta fica em 0,1%.
# Volte para None se trocar de hardware ou se o EasyEffects sair do caminho.
INPUT_DEVICE = "hw:1,7"
# Fração de amostras exatamente zero acima da qual a captura é considerada
# quebrada. Microfone real nunca dá zero exato — sempre há ruído de fundo —,
# então zeros significam blocos perdidos ou uma fonte muda no caminho.
CAPTURE_MAX_ZEROS = 0.05
CAPTURE_MIN_PEAK = 0.005        # pico abaixo disso é mudo, não é fala baixa

# --- Voz / VAD (Fase 3) ---
# Detecção de atividade de voz: a gravação para sozinha quando você para de
# falar, em vez de exigir um segundo Enter. O modelo (Silero v6, ONNX) já vem
# dentro do faster-whisper — nenhuma dependência ou download novo. Ver core/vad.py.
VAD_ENABLED = True              # False → volta ao push-to-talk ("Enter para parar")
VAD_THRESHOLD = 0.5             # probabilidade mínima para o frame contar como fala
# Silêncio contínuo que encerra a fala. Abaixo de ~500ms a frase é cortada na
# pausa entre orações; acima de ~1.2s a conversa fica lenta.
VAD_SILENCE_MS = 800
VAD_MIN_SPEECH_MS = 250         # rajada menor que isso é descartada (tosse, clique)
VAD_START_TIMEOUT = 8.0         # s sem ninguém falar → desiste e volta ao prompt
VAD_MAX_SECONDS = 30.0          # teto de segurança contra ruído contínuo
VAD_PAD_MS = 300                # margem antes/depois do recorte (salva a 1ª sílaba)

# --- Wake word (Fase 3) ---
# Modo "sempre ouvindo": o microfone fica aberto e só a palavra abaixo abre um
# turno. Ver core/wake.py para a cadeia e para as garantias de privacidade.
#
# DESLIGADO por padrão, e é assim que tem que ficar: manter o microfone aberto é
# uma escolha do usuário, nunca um padrão herdado. Com WAKE_ENABLED = False nada
# é carregado e o microfone não abre (invariante 5).
WAKE_ENABLED = False
WAKE_WORD = "Oráculo"
# "onnx"        → classificador dedicado; não transcreve nada antes de acordar.
# "transcricao" → sem modelo treinado: transcreve TODA fala da sala para conferir
#                 o nome. Funciona, mas custa ~0.5 s de CPU por fala e faz passar
#                 pelo ASR o que foi dito sem o nome. Opt-in consciente.
WAKE_BACKEND = "onnx"
# 0.0 → usa o limiar que o treinador escolheu e gravou no .npz (recomendado: ele
# foi calibrado contra negativos reais). Um valor aqui sobrepõe.
WAKE_THRESHOLD = 0.0
WAKE_REFRACTORY_MS = 1500       # ignora novos disparos logo após um (janela desliza)
# Áudio guardado em memória antes do gatilho. Precisa cobrir a palavra inteira
# (~0,7 s) mais o atraso do disparo (até 0,4 s), sem esticar tanto a ponto de
# engolir a frase anterior de quem estava conversando na sala.
WAKE_PREROLL_MS = 1400
WAKE_FOLLOWUP_TIMEOUT = 6.0     # s de escuta extra quando você diz só "Oráculo"
# Depois de acordar, confere na transcrição que o nome está mesmo no começo. O
# detector é acústico e não sabe a posição na frase, então "consultei o oráculo
# de Delfos" dispara com razão — e é esta conferência que descarta o turno.
WAKE_CONFIRMA_TEXTO = True
WAKE_MAX_TOKENS = 3             # só procura o nome nos N primeiros tokens da fala
WAKE_FUZZY = 0.82               # similaridade mínima ("oraculo" vs "oráculo")
# WAKE_DIR fica na seção de persistência, junto com DATA_DIR (definido lá).

# --- Transcrição de arquivos (/transcrever) ---
# Extensões reconhecidas como áudio/vídeo. Serve só para avisar quando o caminho
# não parece mídia — o whisper decodifica via PyAV e aceita bem mais formatos.
TRANSCRIBE_EXTENSIONS = (
    ".ogg", ".opus", ".mp3", ".m4a", ".wav", ".flac", ".aac", ".wma",
    ".webm", ".mp4", ".mkv", ".mov",
)
TRANSCRIBE_PARAGRAPH_CHARS = 400  # tamanho mínimo de um parágrafo agrupado
TRANSCRIBE_TIMESTAMPS = True      # prefixa cada parágrafo com [mm:ss]
TRANSCRIBE_OUTPUT_DIR = None      # None → grava o .md ao lado do áudio
# Acima desta duração o parakeet degrada/trunca; o comando avisa e sugere /stt
# whisper. (O VAD da conversa não ajuda aqui: /transcrever ainda manda o arquivo
# inteiro de uma vez — segmentar arquivo longo é um passo à parte.)
TRANSCRIBE_PARAKEET_LIMIT = 30.0  # segundos

# --- Voz / TTS ---
# Raiz do projeto, para ancorar caminhos de modelos no projeto (não no cwd).
PROJECT_ROOT = Path(__file__).resolve().parent

TTS_ENGINE = "kokoro"           # kokoro (feminina, natural) | piper (masculina, leve)

# Kokoro (kokoro-onnx) — voz feminina pt-BR natural, roda na CPU.
KOKORO_MODEL = str(PROJECT_ROOT / "kokoro-v1.0.onnx")
KOKORO_VOICES = str(PROJECT_ROOT / "voices-v1.0.bin")
KOKORO_VOICE = "pf_dora"        # pf_dora (feminina) | pm_alex, pm_santa (masculinas)
KOKORO_LANG = "pt-br"
KOKORO_SPEED = 1.0

# Piper (alternativa) — binário do pacote AUR piper-tts-bin. Caminho ABSOLUTO do
# modelo, ancorado na raiz do projeto, para funcionar de qualquer diretório.
PIPER_BIN = "piper-tts"
PIPER_VOICE = str(PROJECT_ROOT / "pt_BR-faber-medium.onnx")

# --- Modo padrão ---
VOICE_MODE_DEFAULT = False      # começa em texto, /voz alterna

# --- Persistência de sessões ---
DATA_DIR = Path(os.path.expanduser("~/.oraculo"))
SESSIONS_DIR = DATA_DIR / "sessions"
# Modelos de característica do wake word + a cabeça treinada (ver core/wake.py).
WAKE_DIR = DATA_DIR / "wake"
# Gravações do dono dizendo o nome. Nascem no treinador do wake word e são a
# semente da verificação de voz (passo 3c).
VOICE_DIR = DATA_DIR / "voice"
RECENT_SESSIONS_ON_SPLASH = 3

# --- Telemetria ---
# Defaults False → custo zero (nada escrito nem impresso). Para desenvolvimento,
# ligue TELEMETRY_CONSOLE para ver um resumo de 1 linha por turno.
# Também leem variáveis de ambiente (qualquer valor não-vazio = True):
#   TELEMETRY_CONSOLE=1 python main.py
#   TELEMETRY_ENABLED=1 python main.py
TELEMETRY_ENABLED = bool(os.environ.get("TELEMETRY_ENABLED"))
TELEMETRY_CONSOLE = bool(os.environ.get("TELEMETRY_CONSOLE"))
TELEMETRY_DIR = DATA_DIR / "telemetry"

# --- Rótulo de hardware exibido na splash (informativo) ---
DEVICE_LABEL = "CUDA RTX 4060"

# --- Interface do terminal ---
# Largura máxima da coluna de leitura, em colunas. 0 = sem limite (usa o terminal
# inteiro). Um teto entre 90 e 110 deixa a linha mais confortável de ler em monitor
# largo, ao custo de deixar espaço vazio à direita.
UI_MAX_WIDTH = 0
UI_GUTTER = 5                   # recuo do corpo da resposta (alinha sob o nome)
UI_GLYPH_ASSISTANT = "●"        # marca o início de um turno do Oráculo
UI_GLYPH_USER = ">"             # eco da mensagem enviada
UI_GLYPH_NOTICE = "⎿"           # avisos/resultados subordinados ao turno

# Paleta (nomes de cor do rich; a barra de status traduz para prompt_toolkit)
UI_COLOR_ACCENT = "bright_cyan"  # Oráculo, glifos ativos
UI_COLOR_USER = "cyan"           # eco do usuário
UI_COLOR_DIM = "grey42"          # métricas, rodapés
UI_COLOR_FAINT = "grey30"        # dicas de tecla

# Rodapé por turno com as métricas da telemetria (latência, tokens/s). Independe
# de TELEMETRY_ENABLED: aqui é só exibição, nada é gravado em disco.
UI_SHOW_TURN_METRICS = True

# Modo de desenho da interface:
#   "fullscreen" — tela alternativa do terminal (como vim/htop e como o Claude
#                  Code desenha por padrão): caixa de entrada fixa no rodapé,
#                  transcript com rolagem própria (PgUp/PgDn/Home/End e roda do
#                  mouse) e, ao sair, o terminal volta como estava.
#   "inline"     — desenha no buffer normal, rolagem nativa do terminal e o
#                  transcript permanece na tela depois de encerrar.
# Sem TTY ou sem prompt_toolkit, cai para "inline" automaticamente.
TUI_MODE = "fullscreen"

# Só vale no modo "inline": limpa a tela visível ao abrir (preserva o scrollback
# anterior — nunca usar 3J aqui, destruiria o histórico do terminal de quem chamou).
CLEAR_ON_START = True

# Linhas roladas por evento da roda do mouse no modo fullscreen.
TUI_SCROLL_LINES = 3

# Captura do mouse no modo fullscreen. Ligada, a roda rola o transcript; desligada,
# o terminal volta a tratar o mouse e a seleção com o botão esquerdo funciona
# normalmente (a rolagem fica por conta de PgUp/PgDn). Alternável em tempo real
# com F2 — não é preciso reiniciar para copiar um trecho.
TUI_MOUSE = True

# Entrada: caixa com borda, histórico entre sessões e autocomplete dos /comandos.
# Requer prompt_toolkit; sem ele (ou sem TTY) cai para um prompt simples do rich.
INPUT_RICH_EDITOR = True
INPUT_HISTORY_FILE = DATA_DIR / "input_history"
INPUT_HISTORY_MAX = 500

# --- System Prompt ---
SYSTEM_PROMPT = """Você é o Oráculo, um assistente pessoal local rodando 100% offline.
Você é direto, útil e responde sempre em português brasileiro.
Você tem memória da conversa atual e usa esse contexto para responder.

O QUE VOCÊ CONSEGUE FAZER:
- Conversar, responder perguntas, explicar, raciocinar e ajudar com texto.
- Lembrar do que foi dito NESTA conversa (a memória some ao encerrar a sessão).

O QUE VOCÊ NÃO CONSEGUE FAZER (seja honesto sobre isso):
- Você NÃO executa ações no computador, NÃO acessa arquivos, agenda, calendário,
  e-mail, lembretes ou qualquer sistema externo. Você só gera texto.
- Você NÃO armazena informação em lugar nenhum além do histórico desta conversa.
- Você NÃO acessa a internet.

REGRAS:
- Escreva SEMPRE e somente em português brasileiro, usando apenas o alfabeto
  latino. NUNCA inclua caracteres chineses, japoneses, coreanos ou de qualquer
  outro sistema de escrita.
- Responda de forma concisa, sem prolixidade desnecessária.
- Se não souber algo, diga claramente.
- NUNCA finja que executou uma ação (agendar, salvar, enviar, lembrar depois).
  Se pedirem algo que exige agir no mundo real, explique que você ainda não tem
  essa capacidade e, se útil, ajude apenas com o conteúdo (ex.: redigir o texto
  da reunião, sugerir como organizar), deixando claro que não foi salvo."""

# --- Comandos do terminal ---
EXIT_COMMANDS = {"/sair", "/exit", "/quit"}
