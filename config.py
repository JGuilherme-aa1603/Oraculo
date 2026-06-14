"""Configurações centralizadas do Oráculo."""

import os
from pathlib import Path

# --- Identidade / versão ---
ASSISTANT_NAME = "Oráculo"
APP_VERSION = "1.0.0"
USER_NAME = None  # None → detecta pelo usuário do sistema (getpass.getuser)

# --- Modelo Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# --- Parâmetros do modelo ---
TEMPERATURE = 0.7
NUM_CTX = 8192       # Mantido baixo para economizar VRAM (padrão do Ollama pode ser 131072)
MAX_TOKENS = 2000    # num_predict — limite de tokens na resposta.
                     # Ajustável: respostas mais longas podem exigir um valor maior;
                     # para conversa normal, 2000 é folgado.

# --- Memória ---
MAX_HISTORY_MESSAGES = 20  # Mantém as últimas N mensagens (sempre cortando em pares user/assistant)

# --- Voz / STT (Whisper) ---
WHISPER_MODEL = "base"          # base | small (small é mais preciso, ~3x mais lento)
WHISPER_DEVICE = "cpu"          # cpu | cuda
WHISPER_COMPUTE_TYPE = "int8"   # cpu→int8 | cuda→float16
# Nota: o ctranslate2 (backend do faster-whisper) exige CUDA 12 (libcublas.so.12)
# + cuDNN 9. Este sistema tem CUDA 13 e o Ollama usa o CUDA dele próprio, então o
# Whisper roda na CPU (rápido para clipes curtos) e a GPU fica livre para o LLM.
# Para usar GPU seria preciso instalar os wheels nvidia-cublas-cu12 e nvidia-cudnn-cu12.
RECORD_DURATION = 5.0           # segundos (modo gravação fixa)
RECORD_SAMPLERATE = 16000

# --- Voz / TTS (Piper) ---
PIPER_BIN = "piper-tts"         # binário do pacote AUR piper-tts-bin (em /usr/bin)
# Caminho ABSOLUTO do modelo de voz, ancorado na raiz do projeto (não no cwd),
# para que o Piper ache o .onnx independente de onde o app foi iniciado.
PROJECT_ROOT = Path(__file__).resolve().parent
PIPER_VOICE = str(PROJECT_ROOT / "pt_BR-faber-medium.onnx")

# --- Modo padrão ---
VOICE_MODE_DEFAULT = False      # começa em texto, /voz alterna

# --- Persistência de sessões ---
DATA_DIR = Path(os.path.expanduser("~/.oraculo"))
SESSIONS_DIR = DATA_DIR / "sessions"
RECENT_SESSIONS_ON_SPLASH = 3

# --- Rótulo de hardware exibido na splash (informativo) ---
DEVICE_LABEL = "CUDA RTX 4060"

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
- Responda de forma concisa, sem prolixidade desnecessária.
- Se não souber algo, diga claramente.
- NUNCA finja que executou uma ação (agendar, salvar, enviar, lembrar depois).
  Se pedirem algo que exige agir no mundo real, explique que você ainda não tem
  essa capacidade e, se útil, ajude apenas com o conteúdo (ex.: redigir o texto
  da reunião, sugerir como organizar), deixando claro que não foi salvo."""

# --- Comandos do terminal ---
EXIT_COMMANDS = {"/sair", "/exit", "/quit"}
