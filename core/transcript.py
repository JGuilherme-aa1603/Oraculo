"""Transcrição de arquivos de áudio: agrupamento, formatação e gravação.

Complementa o core/stt.py (que fala com os motores) cuidando do que vem depois:
juntar os segmentos em parágrafos legíveis, formatar em Markdown com timestamps
e gravar em disco. Usado pelo comando /transcrever.

Nada aqui carrega modelo nem conhece motor — recebe segmentos e devolve texto.
"""

from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import config
from core.stt import Segment

# Fim de frase: só quebra parágrafo em ponto final, para não cortar no meio.
_SENT_END = (".", "!", "?", "…", ":")

# (início, texto) — um parágrafo já agrupado.
Paragraph = tuple[float, str]


def hms(seconds: float) -> str:
    """Segundos → mm:ss (ou hh:mm:ss em áudios de mais de uma hora)."""
    total = int(seconds)
    h, m, s = total // 3600, (total // 60) % 60, total % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def paragraphs(segments: Iterable[Segment],
               min_chars: int | None = None) -> Iterator[Paragraph]:
    """Agrupa segmentos em parágrafos, preservando o tempo de início de cada um.

    O whisper corta em trechos de poucos segundos, o que vira uma parede de
    linhas soltas. Aqui os trechos se acumulam até `min_chars` E o fim de uma
    frase, para o parágrafo nunca quebrar no meio de uma ideia.
    """
    if min_chars is None:
        min_chars = config.TRANSCRIBE_PARAGRAPH_CHARS

    buf: list[str] = []
    start: float | None = None
    for seg_start, _seg_end, text in segments:
        if start is None:
            start = seg_start
        buf.append(text)
        joined = " ".join(buf)
        if len(joined) >= min_chars and joined.endswith(_SENT_END):
            yield start, joined
            buf, start = [], None
    if buf and start is not None:
        yield start, " ".join(buf)


def engine_label() -> str:
    """Descrição do motor STT ativo, para o cabeçalho do arquivo gerado."""
    if config.STT_ENGINE == "parakeet":
        return f"parakeet ({config.PARAKEET_MODEL})"
    return (f"whisper {config.WHISPER_MODEL} "
            f"({config.WHISPER_DEVICE}, beam {config.WHISPER_BEAM_SIZE})")


def as_markdown(paras: Iterable[Paragraph], audio_path: Path,
                secs: float | None = None, timestamps: bool | None = None) -> str:
    """Monta o documento Markdown: cabeçalho com metadados + parágrafos.

    Os metadados dizem de onde o texto veio e com que motor foi gerado — sem
    isso, meses depois não dá para saber se um trecho estranho é fala ou erro
    de transcrição."""
    if timestamps is None:
        timestamps = config.TRANSCRIBE_TIMESTAMPS

    linhas = [
        f"# Transcrição — {audio_path.stem}",
        "",
        f"- **Arquivo:** `{audio_path.name}`",
    ]
    if secs:
        linhas.append(f"- **Duração:** {hms(secs)}")
    linhas += [
        f"- **Motor:** {engine_label()}",
        f"- **Gerado em:** {datetime.now():%d/%m/%Y %H:%M}",
        "",
        "---",
        "",
    ]
    for start, texto in paras:
        linhas.append(f"**[{hms(start)}]** {texto}" if timestamps else texto)
        linhas.append("")
    return "\n".join(linhas).rstrip() + "\n"


def output_path(audio_path: Path) -> Path:
    """Onde gravar o .md: em TRANSCRIBE_OUTPUT_DIR ou ao lado do áudio."""
    nome = f"{audio_path.stem}.md"
    if config.TRANSCRIBE_OUTPUT_DIR:
        destino = Path(config.TRANSCRIBE_OUTPUT_DIR).expanduser()
        destino.mkdir(parents=True, exist_ok=True)
        return destino / nome
    return audio_path.with_name(nome)


def save(paras: Iterable[Paragraph], audio_path: Path,
         secs: float | None = None) -> Path:
    """Grava a transcrição em Markdown e devolve o caminho do arquivo.

    Não sobrescreve: se o nome já existe, acrescenta um sufixo numérico."""
    destino = output_path(audio_path)
    n = 2
    while destino.exists():
        destino = destino.with_name(f"{audio_path.stem}-{n}.md")
        n += 1
    destino.write_text(as_markdown(paras, audio_path, secs), encoding="utf-8")
    return destino
