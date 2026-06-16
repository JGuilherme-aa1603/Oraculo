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

# --- Voz / STT ---
STT_ENGINE = "whisper"          # whisper | parakeet
# whisper:  faster-whisper, VAD embutido, fallback CUDA→CPU. Bom p/ clipes longos.
# parakeet: NVIDIA Parakeet TDT 0.6b v3 (onnx-asr). Muito rápido na CPU e pontua
#           sozinho, mas tem limite de ~20-30s por clipe (sem VAD; ver Fase 3).

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
