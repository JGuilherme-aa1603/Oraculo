"""Ações que o Oráculo executa no computador — a whitelist e o executor.

Este é o módulo mais perigoso do projeto: é o único que faz o assistente sair do
texto e mexer na máquina. O desenho inteiro segue de uma frase:

    **O modelo ESCOLHE de um conjunto; ele nunca COMPÕE uma string.**

A whitelist sozinha protege o comando e não protege nada mais. O risco de
verdade está nos argumentos: eles vêm do modelo, que os tirou de texto seu — que
pode ter sido transcrito de voz, e que pode ter chegado perto de uma nota
recuperada pelo RAG. Por isso todo parâmetro passa por um validador que
**confere contra a realidade** (um nome de aplicativo tem que casar com um
`.desktop` instalado) ou contra um conjunto fechado (faixa numérica, enum).
Nada que o modelo escreve é usado como veio.

As camadas, de fora para dentro:

1. **Tool calling** — o modelo emite `nome + argumentos`, nunca uma linha de
   comando. Ele não vê um shell em momento nenhum.
2. **Registro** (`ACOES`) — nome conhecido, ou nada acontece.
3. **Validadores** — cada argumento vira um valor de um domínio fechado.
4. **Executor** — `subprocess.run` com **lista de argumentos**, `shell=False`
   sempre, timeout curto, e o binário conferido contra `_BINARIOS`.

A validação do nome é **dupla** (`despachar` e de novo em `_executar`), como
manda o CLAUDE.md. Não é redundância boba: a segunda existe para que um bug
futuro na primeira — um caminho novo que chame o executor direto, um refactor
distraído — não chegue ao `exec`.

## O nível destrutivo

`desligar` e `reiniciar` existem, mas atrás de um portão próprio (ver
`pode_destrutiva`). E o portão trata a verificação de voz pelo que ela é: um
**filtro de sala**, não uma autenticação. O projeto tem escrito que um vetor de
timbre é enganável por imitação e por uma gravação sua num alto-falante; promovê-
lo a senha seria a desonestidade que o system prompt evita. Então:

- a voz **impede a sala** de disparar um desligamento;
- a confirmação **autoriza**, e ela exige alguém no teclado;
- e nada disso vale enquanto o perfil for o provisório, o que é **conferido no
  arquivo** (`locutor.perfil_robusto`), não prometido num comentário.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import config


class AcaoErro(RuntimeError):
    """Falha que o usuário precisa ver (argumento inválido, binário ausente)."""


class AcaoBloqueada(AcaoErro):
    """A ação existe mas o portão a barrou. Mensagem explica o que falta."""


# --- Validadores ------------------------------------------------------------
#
# Cada um transforma o que o modelo escreveu num valor de domínio fechado, ou
# levanta. Eles nunca "consertam" silenciosamente: um volume de 300 é um erro
# que o usuário deve ver, não um 100 que ele não pediu.


class Validador:
    descricao = ""

    def limpar(self, valor):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Inteiro(Validador):
    minimo: int
    maximo: int

    @property
    def descricao(self) -> str:
        return f"inteiro de {self.minimo} a {self.maximo}"

    def limpar(self, valor) -> int:
        try:
            n = int(str(valor).strip())
        except (TypeError, ValueError):
            raise AcaoErro(f"esperava um número, veio {valor!r}") from None
        if not self.minimo <= n <= self.maximo:
            raise AcaoErro(f"{n} está fora de {self.minimo}-{self.maximo}")
        return n


@dataclass(frozen=True)
class Opcao(Validador):
    valores: tuple[str, ...]

    @property
    def descricao(self) -> str:
        return "um de: " + ", ".join(self.valores)

    def limpar(self, valor) -> str:
        v = str(valor).strip().lower()
        if v not in self.valores:
            raise AcaoErro(f"{valor!r} não é um de {', '.join(self.valores)}")
        return v


@dataclass(frozen=True)
class Booleano(Validador):
    descricao = "sim ou não"
    _VERDADE = {"true", "1", "sim", "on", "ligado", "yes"}
    _FALSO = {"false", "0", "não", "nao", "off", "desligado", "no"}

    def limpar(self, valor) -> bool:
        if isinstance(valor, bool):
            return valor
        v = str(valor).strip().lower()
        if v in self._VERDADE:
            return True
        if v in self._FALSO:
            return False
        raise AcaoErro(f"esperava sim/não, veio {valor!r}")


@dataclass(frozen=True)
class Texto(Validador):
    """Texto livre — o único domínio aberto, e por isso o mais cuidadoso.

    Só é usado onde o valor vira **conteúdo** (o corpo de uma notificação),
    nunca onde vira alvo de uma operação. Mesmo assim: cortado no tamanho e
    limpo de caracteres de controle, que é o que atravessa terminal e
    notificação fazendo estrago visual.
    """

    maximo: int = 200

    @property
    def descricao(self) -> str:
        return f"texto de até {self.maximo} caracteres"

    def limpar(self, valor) -> str:
        v = re.sub(r"[\x00-\x1f\x7f]", " ", str(valor)).strip()
        if not v:
            raise AcaoErro("texto vazio")
        return v[:self.maximo]


class Aplicativo(Validador):
    """Nome de app, conferido contra os `.desktop` REALMENTE instalados.

    É o validador que melhor mostra a regra do módulo: o modelo não informa o
    que executar, ele escolhe entre o que existe. Se o nome não casar com um
    aplicativo instalado, nada roda — e o erro lista o que havia, em vez de
    deixar o usuário adivinhando.
    """

    descricao = "nome de um aplicativo instalado"

    def limpar(self, valor) -> str:
        alvo = str(valor).strip().lower()
        if not alvo:
            raise AcaoErro("nome de aplicativo vazio")
        apps = aplicativos()
        if alvo in apps:
            return apps[alvo]
        # Casamento por prefixo/contido, mas só quando for INEQUÍVOCO: dois
        # candidatos e a escolha volta para o usuário, nunca para o acaso.
        candidatos = sorted({d for nome, d in apps.items() if alvo in nome})
        if len(candidatos) == 1:
            return candidatos[0]
        if not candidatos:
            raise AcaoErro(f"não achei nenhum aplicativo chamado {valor!r}")
        raise AcaoErro(f"{valor!r} é ambíguo: {', '.join(candidatos[:6])}")


# --- Aplicativos instalados -------------------------------------------------

_APPS_CACHE: dict[str, str] | None = None


def aplicativos(recarregar: bool = False) -> dict[str, str]:
    """`{nome em minúsculas: id do .desktop}` dos aplicativos visíveis.

    Lê os `.desktop` do sistema e do usuário, ignorando `NoDisplay`/`Hidden` —
    entradas ocultas não são coisas que alguém pede para abrir pelo nome. O
    resultado é cacheado: são ~85 arquivos, e reler a cada turno seria I/O à toa.
    """
    global _APPS_CACHE
    if _APPS_CACHE is not None and not recarregar:
        return _APPS_CACHE
    encontrados: dict[str, str] = {}
    for pasta in config.ACOES_APP_DIRS:
        raiz = Path(os.path.expanduser(pasta))
        if not raiz.is_dir():
            continue
        for arquivo in sorted(raiz.glob("*.desktop")):
            try:
                texto = arquivo.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(r"^(NoDisplay|Hidden)\s*=\s*true", texto, re.M | re.I):
                continue
            nome = re.search(r"^Name\s*=\s*(.+)$", texto, re.M)
            if not nome:
                continue
            encontrados.setdefault(nome.group(1).strip().lower(), arquivo.stem)
            # O próprio id serve como nome ("firefox"), que é como as pessoas
            # costumam pedir — e como o modelo tende a responder.
            encontrados.setdefault(arquivo.stem.lower(), arquivo.stem)
    _APPS_CACHE = encontrados
    return encontrados


# --- Executor ---------------------------------------------------------------
#
# Binários permitidos, conferidos imediatamente antes do exec. É a última linha:
# mesmo que um bug monte uma argv inesperada, ela não sai daqui se o programa
# não estiver nesta lista.
_BINARIOS = frozenset({
    "gtk-launch", "wpctl", "spectacle", "loginctl", "notify-send", "systemctl",
})


def _executar(argv: list[str], *, nome: str, timeout: float | None = None) -> str:
    """Roda um comando. Lista de argumentos, `shell` NUNCA, timeout sempre.

    `shell=True` aqui transformaria cada validador acima em decoração: bastaria
    um argumento com `;` para o domínio fechado deixar de importar. Com lista,
    o argumento é um argumento — não vira sintaxe.
    """
    # Segunda validação do nome (a primeira está em `despachar`). Redundante
    # hoje, de propósito: um caminho novo que chegue aqui direto morre aqui.
    if nome not in ACOES:
        raise AcaoErro(f"ação desconhecida: {nome!r}")
    if not argv or argv[0] not in _BINARIOS:
        raise AcaoErro(f"binário não permitido: {argv[0] if argv else '(vazio)'}")
    if shutil.which(argv[0]) is None:
        raise AcaoErro(f"{argv[0]} não está instalado nesta máquina")
    try:
        r = subprocess.run(  # noqa: S603 - argv fixa, shell=False, sem input
            argv, shell=False, capture_output=True, text=True,
            timeout=timeout or config.ACOES_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise AcaoErro(f"{argv[0]} não respondeu em "
                       f"{timeout or config.ACOES_TIMEOUT:.0f}s") from None
    except OSError as exc:
        raise AcaoErro(f"falha ao executar {argv[0]}: {exc}") from exc
    if r.returncode != 0:
        detalhe = (r.stderr or r.stdout or "").strip().splitlines()
        raise AcaoErro(f"{argv[0]} falhou: "
                       f"{detalhe[0][:120] if detalhe else r.returncode}")
    return (r.stdout or "").strip()


# --- As ações ---------------------------------------------------------------


@dataclass(frozen=True)
class Acao:
    nome: str
    descricao: str
    executar: Callable[..., str]
    parametros: dict[str, Validador] = field(default_factory=dict)
    destrutiva: bool = False


def _abrir_aplicativo(nome: str) -> str:
    _executar(["gtk-launch", f"{nome}.desktop"], nome="abrir_aplicativo")
    return f"abri {nome}"


def _ajustar_volume(nivel: int) -> str:
    _executar(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{nivel}%"],
              nome="ajustar_volume")
    return f"volume em {nivel}%"


def _silenciar(ligado: bool) -> str:
    _executar(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@",
               "1" if ligado else "0"], nome="silenciar")
    return "som mudo" if ligado else "som de volta"


def _tirar_screenshot(area: str) -> str:
    # -b roda sem abrir a janela do Spectacle, -n sem notificação própria.
    flag = {"tela": "-f", "janela": "-a", "regiao": "-r"}[area]
    _executar(["spectacle", "-b", "-n", flag], nome="tirar_screenshot",
              timeout=config.ACOES_TIMEOUT_LONGO)
    return f"screenshot ({area}) salvo em Imagens"


def _bloquear_tela() -> str:
    _executar(["loginctl", "lock-session"], nome="bloquear_tela")
    return "tela bloqueada"


def _notificar(texto: str) -> str:
    _executar(["notify-send", config.ASSISTANT_NAME, texto], nome="notificar")
    return "notificação enviada"


def _desligar() -> str:
    _executar(["systemctl", "poweroff"], nome="desligar")
    return "desligando"


def _reiniciar() -> str:
    _executar(["systemctl", "reboot"], nome="reiniciar")
    return "reiniciando"


ACOES: dict[str, Acao] = {
    a.nome: a for a in (
        Acao("abrir_aplicativo",
             "Abre um aplicativo instalado no computador do usuário.",
             _abrir_aplicativo, {"nome": Aplicativo()}),
        Acao("ajustar_volume",
             "Ajusta o volume do sistema para um valor de 0 a 100.",
             _ajustar_volume, {"nivel": Inteiro(0, 100)}),
        Acao("silenciar",
             "Muta ou desmuta o som do sistema.",
             _silenciar, {"ligado": Booleano()}),
        Acao("tirar_screenshot",
             "Captura a tela inteira, a janela ativa ou uma região.",
             _tirar_screenshot, {"area": Opcao(("tela", "janela", "regiao"))}),
        Acao("bloquear_tela",
             "Bloqueia a sessão, exigindo a senha para voltar.",
             _bloquear_tela),
        Acao("notificar",
             "Mostra uma notificação na área de trabalho.",
             _notificar, {"texto": Texto(200)}),
        Acao("desligar", "Desliga o computador.", _desligar, destrutiva=True),
        Acao("reiniciar", "Reinicia o computador.", _reiniciar, destrutiva=True),
    )
}


# --- Portão do nível destrutivo ---------------------------------------------


def pode_destrutiva(ctx: dict) -> tuple[bool, str]:
    """(liberado, motivo). O motivo é para mostrar ao usuário quando barrar.

    Três condições, e a ordem importa para a mensagem sair útil:

    1. o perfil de voz precisa ser de cadastro REAL, não o provisório montado
       com os clipes de "Oráculo" do wake word — aquele descreve uma palavra;
    2. no modo voz, a fala deste turno precisa ter sido reconhecida como do dono
       (`DONO`, nunca `CURTO`: "sim" não tem timbre suficiente para julgar, e é
       exatamente o que alguém diria para confirmar);
    3. no modo texto não há voz nenhuma — lá quem autoriza é a confirmação, que
       exige alguém no teclado e é evidência mais forte que um vetor de timbre.

    Em nenhum caso isto substitui a confirmação: a voz filtra a sala, ela não
    autentica ninguém.
    """
    from core import locutor

    if not locutor.perfil_robusto():
        return False, ("o perfil de voz ainda é o provisório (feito com os "
                       "clipes de \"Oráculo\" do wake word). Refaça com "
                       "'tools/cadastrar_voz.py --gravar 8' para liberar")
    if ctx.get("voice_mode"):
        if not config.LOCUTOR_ENABLED:
            return False, "no modo voz isto exige a verificação de voz (/dono)"
        if not ctx.get("voz_do_dono"):
            return False, ("não reconheci a sua voz nesta fala — repita mais "
                           "devagar, ou peça pelo teclado")
    return True, ""


# --- Despacho ---------------------------------------------------------------


def descricao_para_modelo() -> list[dict]:
    """As ações no formato que o `bind_tools` do LangChain consome."""
    esquemas = []
    for acao in ACOES.values():
        props = {
            nome: {"type": "string", "description": val.descricao}
            for nome, val in acao.parametros.items()
        }
        esquemas.append({
            "type": "function",
            "function": {
                "name": acao.nome,
                "description": acao.descricao,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                },
            },
        })
    return esquemas


def validar(nome: str, argumentos: dict) -> dict:
    """Confere o nome e limpa os argumentos. Levanta AcaoErro se algo não bate."""
    acao = ACOES.get(nome)
    if acao is None:
        raise AcaoErro(f"ação desconhecida: {nome!r}")
    limpos = {}
    for chave, validador in acao.parametros.items():
        if chave not in argumentos:
            raise AcaoErro(f"{nome}: falta o argumento '{chave}' "
                           f"({validador.descricao})")
        limpos[chave] = validador.limpar(argumentos[chave])
    # Argumento a mais é sinal de que o modelo inventou parâmetro; descartar em
    # silêncio esconderia isso, e passar adiante estouraria no executar().
    sobra = set(argumentos) - set(acao.parametros)
    if sobra:
        raise AcaoErro(f"{nome}: argumento(s) que não existem: "
                       f"{', '.join(sorted(sobra))}")
    return limpos


def despachar(nome: str, argumentos: dict) -> str:
    """Valida e executa. Devolve a frase de resultado que vai para a tela."""
    limpos = validar(nome, argumentos)
    return ACOES[nome].executar(**limpos)
