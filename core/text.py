"""Limpeza de texto para exibição e síntese de voz.

Dois usos:
  - strip_cjk: remove caracteres CJK que o modelo (qwen) às vezes vaza no meio
    da resposta em português. Aplicado no stream, afeta terminal, memória e voz.
  - for_speech: tira a marcação Markdown antes do TTS, para a voz não soletrar
    "asterisco", "cerquilha" etc. O terminal mantém o Markdown (renderizado).
"""

import re

# Faixas Unicode de CJK (chinês/japonês/coreano) e pontuação associada.
_CJK = re.compile(
    r"[　-〿぀-ヿ㄀-ㄯ㐀-䶿一-鿿"
    r"가-힯豈-﫿＀-￯]"
)

# Padrões de Markdown removidos para a fala (ordem importa).
_MD_PATTERNS = [
    (re.compile(r"```.*?```", re.DOTALL), " "),     # blocos de código
    (re.compile(r"`([^`]*)`"), r"\1"),               # código inline
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),    # cabeçalhos
    (re.compile(r"(\*\*|__)(.*?)\1"), r"\2"),        # negrito
    (re.compile(r"(\*|_)(.*?)\1"), r"\2"),           # itálico
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),         # marcadores de lista
    (re.compile(r"^\s*\d+\.\s+", re.M), ""),         # listas numeradas
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),   # links → só o texto
]


def strip_cjk(text: str) -> str:
    """Remove caracteres CJK que o modelo eventualmente emite por engano."""
    return _CJK.sub("", text)


def for_speech(text: str) -> str:
    """Texto pronto para o TTS: sem Markdown nem CJK, espaços normalizados."""
    text = strip_cjk(text)
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    text = text.replace("*", "").replace("#", "").replace("`", "")
    text = re.sub(r"\n{2,}", ". ", text)   # parágrafos viram pausa
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# Quebra em fim de frase (.!?…) ou nova linha.
_SENT_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text: str, min_len: int = 60) -> list[str]:
    """Divide o texto em blocos para falar incrementalmente (menor latência).

    Agrupa fragmentos curtos até `min_len` caracteres para a fala não sair
    picotada. O primeiro bloco é pequeno de propósito: começa a tocar antes de
    a resposta inteira ser sintetizada.
    """
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    buf = ""
    for part in (p.strip() for p in _SENT_END.split(text) if p.strip()):
        buf = f"{buf} {part}".strip() if buf else part
        if len(buf) >= min_len:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def split_complete(buffer: str) -> tuple[list[str], str]:
    """Separa frases já completas de um buffer em construção (streaming).

    Devolve (frases_completas, resto_incompleto). O resto é o pedaço após o
    último fim de frase, que ainda pode crescer com os próximos tokens.
    """
    matches = list(_SENT_END.finditer(buffer))
    if not matches:
        return [], buffer
    cut = matches[-1].end()
    head, tail = buffer[:cut], buffer[cut:]
    parts = [p.strip() for p in _SENT_END.split(head) if p.strip()]
    return parts, tail
