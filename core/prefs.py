"""Preferências do usuário que sobrevivem ao fim da sessão.

O que você troca em conversa (`/modelo`, `/think`, `/stt`, `/vad`, `/voz`) volta
igual na próxima vez, em vez de precisar ser trocado de novo todo dia. Mora em
`~/.oraculo/prefs.json`.

**A wake word NÃO entra aqui, e isso é uma decisão, não um esquecimento.**
Lembrar `WAKE_ENABLED = True` faria o microfone abrir sozinho no arranque
seguinte — e o projeto tem escrito que manter o microfone aberto "é uma escolha
do usuário, nunca um padrão herdado". Herdar por arquivo é exatamente herdar.
Quem quiser a escuta ligada de fábrica muda o `config.py`, que é um ato
deliberado e visível; o `/despertar` continua valendo só para a sessão da vez.

**O `/notas` (Fase 4) ENTRA, e pelo mesmo critério.** Ele não abre o microfone
para a sala: lê notas suas, na sua máquina, que você mandou indexar. É da
família do `/vad` e do `/stt`, não da wake word. Mas ele é aplicado pelo main, e
não por `_ESPELHA_CONFIG`, porque ligar depende de o índice existir e carregar —
escrever `RAG_ENABLED = True` sem o índice na mão poria o Oráculo anunciando no
system prompt uma capacidade que ele não tem, que é justamente o que o
invariante 1 proíbe.

**O `/comandos` (Fase 5) também NÃO entra**, e pelo critério da wake word, não
pelo do `/notas`. Ler as suas notas é passivo; executar ações mexe na máquina, e
uma sessão que já nasce podendo desligar o computador é exatamente o padrão
herdado que este projeto recusa. Ligar isso é um ato desta sessão, sempre.

Duas regras que o desenho segue:

- **Lista branca, com tipo.** Nada de despejar o `ctx` inteiro: só as chaves de
  `_CAMPOS` são lidas e gravadas, e um valor com o tipo errado é descartado.
  O arquivo é editável à mão, então ele é entrada não-confiável como qualquer
  outra — um `modelo: 12` não pode derrubar o arranque.
- **A chave só nasce quando você troca algo.** Um prefs.json vazio deixa o
  `config.py` mandar, que é o esperado de quem abriu o arquivo para editar. Se
  gravássemos tudo no primeiro arranque, mexer no config.py depois não teria
  efeito nenhum e a causa seria invisível.

Tudo aqui é best-effort: preferência é conforto, e nunca pode derrubar a
conversa. Erro de disco ou JSON quebrado vira "sem preferências".
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from typing import Any

import config

# chave no arquivo -> tipo aceito. A ordem é a que o `/padroes` exibe.
_CAMPOS: dict[str, type] = {
    "modelo": str,
    "think": bool,
    "mostrar_raciocinio": bool,
    "voz": bool,
    "stt": str,
    "tts": str,
    "vad": bool,
    "dono": bool,
    "notas": bool,
}

# Preferências que apenas espelham um flag do config. As outras (modelo, think,
# voz...) são aplicadas pelo main, que precisa do chain para isso.
_ESPELHA_CONFIG = {
    "stt": "STT_ENGINE",
    "tts": "TTS_ENGINE",
    "vad": "VAD_ENABLED",
    # A verificação de voz entra aqui, ao contrário da wake word, porque ela
    # FECHA um portão em vez de abrir o microfone: herdá-la ligada só restringe
    # quem o Oráculo atende, nunca amplia o que ele escuta.
    "dono": "LOCUTOR_ENABLED",
}


def _valida(dados: Any) -> dict:
    """Fica só com as chaves conhecidas e do tipo certo."""
    if not isinstance(dados, dict):
        return {}
    limpo: dict = {}
    for chave, tipo in _CAMPOS.items():
        valor = dados.get(chave)
        # bool é subclasse de int; `isinstance(True, str)` é False, então a
        # checagem direta basta — mas um bool onde se espera str tem que cair.
        if isinstance(valor, tipo) and not (tipo is str and not valor.strip()):
            limpo[chave] = valor
    return limpo


def carregar() -> dict:
    """Lê as preferências gravadas. `{}` se não houver, estiver desligado ou
    o arquivo estiver ilegível."""
    if not config.PREFS_ENABLED:
        return {}
    try:
        bruto = json.loads(config.PREFS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _valida(bruto)


def gravar(**mudancas: Any) -> None:
    """Junta as mudanças ao que já estava gravado e persiste.

    Escrita atômica (arquivo temporário + `os.replace`): um Ctrl+C no meio do
    `write` deixaria um JSON pela metade, e aí a próxima sessão perderia TODAS
    as preferências por causa de uma só. O replace é indivisível no mesmo
    sistema de arquivos, então ou vale o conteúdo velho ou o novo.
    """
    if not config.PREFS_ENABLED:
        return
    atual = carregar()
    atual.update(_valida(mudancas))
    try:
        config.PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=config.PREFS_FILE.parent,
            prefix=".prefs-", suffix=".tmp", delete=False,
        ) as tmp:
            json.dump(atual, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            temporario = tmp.name
        os.replace(temporario, config.PREFS_FILE)
    except OSError:
        # Preferência é conforto: disco cheio ou permissão não pode derrubar
        # a conversa. Falha calada e a sessão segue com os valores em memória.
        with contextlib.suppress(OSError, UnboundLocalError):
            os.unlink(temporario)


def esquecer() -> bool:
    """Apaga o arquivo, devolvendo o comando ao `config.py`. True se havia algo."""
    try:
        config.PREFS_FILE.unlink()
        return True
    except OSError:
        return False


def aplicar_ao_config(prefs: dict) -> None:
    """Escreve no `config` as preferências que são espelho direto de um flag.

    As outras (modelo, think, voz) dependem do chain e do ctx, e quem as aplica
    é o main — este módulo não conhece nenhum dos dois de propósito.
    """
    for chave, atributo in _ESPELHA_CONFIG.items():
        if chave in prefs:
            setattr(config, atributo, prefs[chave])


def campos() -> tuple[str, ...]:
    """Nomes das preferências, na ordem de exibição do `/padroes`."""
    return tuple(_CAMPOS)
