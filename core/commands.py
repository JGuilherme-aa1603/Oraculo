"""Roteamento de comandos do terminal (/ajuda, /voz, /stt, /transcrever, /modelo,
/limpar, /sair).

O loop principal passa um dicionário de contexto mutável (`ctx`) com:
  - console: rich.Console
  - chain: OraculoChain (tem .memory, .model_name, .set_model)
  - running: bool   (setar False encerra o loop)
  - voice_mode: bool

`handle()` retorna True se o input era um comando (e portanto NÃO deve ir ao LLM).
"""

import json
import threading
import time
from pathlib import Path

from rich.text import Text

import config
from core import history as history_mod, llm as llm_mod, prefs as prefs_mod, ui

STT_ENGINES = ("whisper", "parakeet")

# Comandos que também rodam fora do chat, via wrapper `oraculo <comando>`.
# Só entram aqui os que não dependem do LLM (ctx["chain"] é None nesse modo).
STANDALONE_COMMANDS = {"/transcrever", "/transcricao", "/ajuda", "/help", "/?"}

# Fonte única da verdade dos comandos: alimenta tanto o texto de /ajuda quanto o
# autocomplete da caixa de entrada (core/prompt.py). Ao adicionar um comando novo,
# basta registrá-lo aqui — os dois lugares acompanham sozinhos.
# (comando, dica de argumento, descrição de uma linha)
COMMAND_SPECS: tuple[tuple[str, str, str], ...] = (
    ("/ajuda", "", "mostra esta ajuda"),
    ("/voz", "", "alterna entre modo voz e modo texto"),
    ("/vad", "", "liga/desliga a parada automática da gravação"),
    ("/despertar", "", "liga/desliga a escuta pela palavra \"Oráculo\""),
    ("/dono", "", "liga/desliga responder só à sua voz"),
    ("/comandos", "", "liga/desliga as ações no computador (não é lembrado)"),
    ("/notas", "", "liga/desliga a consulta às suas notas do Obsidian"),
    ("/indexar", "", "reindexa o vault para a consulta às notas"),
    ("/buscar", "<pergunta>", "mostra os trechos que a busca traria, sem o LLM"),
    ("/think", "", "liga/desliga o raciocínio; Ctrl+O mostra o texto"),
    ("/stt", "<motor>", "lista ou troca o motor de transcrição"),
    ("/transcrever", "<arquivo>", "transcreve um áudio; --salvar grava um .md ao lado"),
    ("/modelo", "<nome>", "lista os modelos do Ollama ou troca o ativo"),
    ("/retomar", "<n>", "continua uma conversa anterior de onde parou"),
    ("/padroes", "", "mostra as preferências guardadas; 'limpar' esquece"),
    ("/limpar", "", "apaga a memória da conversa atual"),
    ("/sair", "", "encerra o Oráculo"),
)

# Comandos cujo argumento é um conjunto FECHADO e conhecido (nome de comando,
# motor de STT, modelo do Ollama). É neles que o Enter pode escolher sozinho a
# primeira sugestão — ver core/prompt.py. Caminho de arquivo fica de fora: lá o
# que você digitou pode ser um caminho válido que por acaso é prefixo de outro,
# e completar por conta própria mandaria o comando para o arquivo errado.
COMANDOS_COM_OPCOES_FECHADAS = ("/stt", "/modelo", "/retomar")


def _ajuda_text() -> str:
    """Monta o texto de /ajuda em duas colunas alinhadas: 'comando <arg>' à
    esquerda, descrição à direita. A largura vem do item mais longo."""
    assinaturas = {cmd: f"{cmd} {arg}".strip() for cmd, arg, _ in COMMAND_SPECS}
    largura = max(len(s) for s in assinaturas.values())
    # O recuo de 2 alinha o bloco com o eco da mensagem no transcript (core.ui).
    linhas = [f"  [bold {config.UI_COLOR_ACCENT}]Comandos disponíveis[/]"]
    for cmd, arg, desc in COMMAND_SPECS:
        rotulo = f"[{config.UI_COLOR_PROMPT}]{cmd}[/]"
        if arg:
            rotulo += f" [{config.UI_COLOR_FAINT}]{arg}[/]"
        preenche = " " * (largura - len(assinaturas[cmd]))
        linhas.append(f"    {rotulo}{preenche}  "
                      f"[{config.UI_COLOR_SOFT}]{desc}[/]")
    # A ajuda termina contando como se digita um comando — é a informação que
    # falta a quem acabou de descobrir que existem comandos.
    linhas.append(f"  [{config.UI_COLOR_FAINT}]{config.UI_GLYPH_NOTICE}  "
                  f"[{config.UI_COLOR_PROMPT}]/[/] abre o autocomplete · "
                  f"Tab escolhe · Alt+Enter quebra linha[/]")
    return "\n".join(linhas)


AJUDA_TEXT = _ajuda_text()


def _list_models() -> list[str]:
    """Consulta o Ollama pelos modelos instalados. Retorna [] em caso de falha."""
    try:
        import requests

        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


# --- Cache dos modelos, para o autocomplete de /modelo ---------------------
# O completer roda a cada tecla digitada. Uma chamada HTTP ali dentro travaria
# a caixa de entrada por até 5 s no pior caso — e o Ollama nem sempre responde
# rápido quando está carregando um modelo. Então o autocomplete lê SÓ este cache
# e a busca acontece numa thread; enquanto ele estiver vazio o autocomplete
# apenas não sugere nada, que é degradação aceitável.
_MODELOS_TTL = 60.0
_modelos: tuple[str, ...] = ()
_modelos_em = 0.0
_buscando = threading.Event()


def _buscar_modelos() -> None:
    global _modelos, _modelos_em
    try:
        lista = tuple(_list_models())
        if lista:
            # Lista vazia = Ollama fora do ar. Manter o cache anterior é melhor
            # que apagar as sugestões por causa de um soluço da rede local.
            _modelos = lista
    finally:
        _modelos_em = time.monotonic()
        _buscando.clear()


def prefetch_modelos() -> None:
    """Dispara a busca dos modelos em segundo plano. Nunca bloqueia.

    Chamado no arranque (o cache já está quente quando você digita `/modelo `)
    e depois de listar/trocar o modelo.
    """
    if _buscando.is_set():
        return
    _buscando.set()
    threading.Thread(target=_buscar_modelos, daemon=True,
                     name="oraculo-modelos").start()


def modelos_conhecidos() -> tuple[str, ...]:
    """Modelos do cache, para o autocomplete. Nunca bloqueia; pode vir vazio."""
    if time.monotonic() - _modelos_em > _MODELOS_TTL:
        prefetch_modelos()
    return _modelos


# --- Cache das sessões, para o autocomplete de /retomar --------------------
# `load_recent` abre e parseia um JSON por sessão, e um JSON de conversa longa
# passa fácil de centenas de KB. Reler os dez a cada tecla digitada seria
# megabytes por caractere; com o cache curto, no máximo uma releitura por
# dezena de segundos — e a lista de conversas antigas não muda enquanto você
# digita o número.
_SESSOES_TTL = 15.0
_sessoes: list[dict] = []
_sessoes_em = 0.0


def sessoes_recentes() -> list[dict]:
    """Sessões retomáveis, para o autocomplete. Cache de alguns segundos."""
    global _sessoes, _sessoes_em
    agora = time.monotonic()
    if agora - _sessoes_em > _SESSOES_TTL:
        _sessoes = history_mod.load_recent(limit=config.RESUME_LIST_LIMIT)
        _sessoes_em = agora
    return _sessoes


def invalidar_sessoes() -> None:
    """Força a próxima leitura da lista (depois de retomar ou de limpar)."""
    global _sessoes_em
    _sessoes_em = 0.0


def resumo_titulo(sessao: dict, largura: int = 44) -> str:
    """Título de uma sessão encurtado para caber no menu de autocomplete."""
    return _resumo(sessao.get("title", ""), largura)


def _guardar_modelos(models: list[str]) -> None:
    """Realimenta o cache do autocomplete com uma lista recém-buscada."""
    global _modelos, _modelos_em
    if models:
        _modelos = tuple(models)
    _modelos_em = time.monotonic()


def _resolver_modelo(arg: str, console) -> str | None:
    """Traduz o que foi digitado num nome de modelo real. None = não deu.

    Aceita prefixo: `/modelo qwen2` acha `qwen2.5:7b` se ele for o único que
    começa assim. Digitar o nome inteiro (com a tag) era uma exigência boba,
    ainda mais com nomes como `hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL`.

    Recusar o desconhecido importa mais do que parece: o `ChatOllama` não valida
    o nome na construção, então um modelo inexistente só falharia no meio do
    turno seguinte — e, pior, teria sido gravado nas preferências, quebrando
    todas as sessões futuras até alguém descobrir por quê.
    """
    modelos = modelos_conhecidos() or tuple(_list_models())
    if not modelos:
        # Ollama fora do ar: sem lista não dá para validar, e recusar aqui
        # impediria de trocar de modelo justamente quando ele voltar.
        return arg
    if arg in modelos:
        return arg
    candidatos = [m for m in modelos if m.startswith(arg)]
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        ui.warn(console, f"Modelo desconhecido: {arg}")
    else:
        ui.warn(console, f"'{arg}' casa com {len(candidatos)} modelos: "
                         f"{', '.join(candidatos)}")
    ui.hint(console, f"Use [{config.UI_COLOR_PROMPT}]/modelo[/] para ver a lista.")
    return None


def _handle_modelo(arg: str, ctx: dict) -> None:
    console = ctx["console"]
    chain = ctx["chain"]

    if not arg:
        # Aqui a chamada bloqueante é a certa: você pediu a lista e quer a de
        # agora, não a que estava em cache. O resultado realimenta o cache do
        # autocomplete de graça.
        models = _list_models()
        if not models:
            ui.warn(console, "Não consegui listar os modelos do Ollama.")
            return
        _guardar_modelos(models)
        ui.heading(console, "Modelos disponíveis:")
        for m in models:
            mark = (f"  [{config.UI_COLOR_ACCENT}](atual)[/]"
                    if m == chain.model_name else "")
            console.print(f"    • [{config.UI_COLOR_BRIGHT}]{m}[/]{mark}")
        ui.hint(console, f"Use [{config.UI_COLOR_PROMPT}]/modelo <nome>[/] "
                         f"para trocar — basta o começo do nome, e o Tab completa.")
        return

    nome = _resolver_modelo(arg, console)
    if nome is None:
        return

    chain.set_model(nome)
    prefs_mod.gravar(modelo=nome)
    ui.ok(console, f"Modelo trocado para {nome}.")

    # set_model preserva o reasoning; se o novo modelo não suporta thinking,
    # desliga para o próximo turno não falhar com erro 400.
    if ctx.get("thinking") and not llm_mod.supports_thinking(arg):
        ctx["thinking"] = False
        chain.set_thinking(False)
        ui.warn(console, f"{arg} não suporta raciocínio — thinking desativado.")


def _resumo(texto: str, largura: int = 68) -> str:
    """Uma linha do conteúdo de uma mensagem, para a prévia da retomada."""
    limpo = " ".join((texto or "").split())
    return limpo if len(limpo) <= largura else limpo[: largura - 1].rstrip() + "…"


def _redesenhar_conversa(console, mensagens: list[dict]) -> None:
    """Reescreve a conversa retomada no transcript, no formato de sempre.

    Retomar é voltar para a conversa, então ela precisa estar NA TELA: para ler,
    rolar e lembrar do que já foi dito. Uma prévia de duas linhas — que foi a
    primeira tentativa — dizia o título e mais nada; quem retomava continuava
    sem saber o que tinha conversado ali.

    Os turnos saem pela mesma calha de um turno ao vivo (`ui.user_echo`,
    `ui.assistant_header`, `ui.body`), e não num formato "de arquivo": é a mesma
    conversa, e dar a ela outra aparência só faria procurar diferença onde não
    há. O que muda é o que não existe: sem rodapé de métricas, porque não houve
    turno agora, e sem o raciocínio, que nunca foi gravado.
    """
    from rich.markdown import Markdown

    for m in mensagens:
        conteudo = (m.get("content") or "").strip()
        if not conteudo:
            continue
        papel = m.get("role")
        if papel == "user":
            ui.user_echo(console, conteudo)
        elif papel == "assistant":
            ui.assistant_header(console)
            ui.body(console, Markdown(conteudo))
            ui.spacer(console)


def _handle_retomar(arg: str, ctx: dict) -> None:
    """Continua uma conversa anterior de onde ela parou."""
    console = ctx["console"]
    # A MESMA lista que o autocomplete numerou. Se o comando relesse o disco por
    # conta própria, uma sessão nova gravada no meio mudaria a ordem e o "3" que
    # você viu no menu abriria outra conversa.
    sessoes = sessoes_recentes()

    if not sessoes:
        ui.notice(console, "nenhuma conversa anterior para retomar.")
        return

    if not arg:
        ui.heading(console, "Conversas anteriores:")
        for i, s in enumerate(sessoes, 1):
            console.print(
                f"    [{config.UI_COLOR_PROMPT}]{i}.[/] "
                f"[{config.UI_COLOR_BRIGHT}]{_resumo(s['title'], 56)}[/]"
                f"  [{config.UI_COLOR_FAINT}]{s['ago']} · "
                f"{s['messages']} mensagens[/]")
        ui.hint(console, f"Use [{config.UI_COLOR_PROMPT}]/retomar <n>[/] "
                         f"para continuar uma delas.")
        return

    try:
        n = int(arg.strip())
    except ValueError:
        ui.warn(console, f"'{arg}' não é um número — use /retomar <n>.")
        return
    if not 1 <= n <= len(sessoes):
        ui.warn(console, f"Escolha entre 1 e {len(sessoes)} "
                         f"(veja a lista com /retomar).")
        return

    escolhida = sessoes[n - 1]
    history = ctx.get("history")
    chain = ctx.get("chain")
    if history is None or chain is None:
        ui.warn(console, "/retomar só funciona dentro do chat.")
        return

    try:
        mensagens = history.retomar(escolhida["path"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ui.error(console, f"Não consegui abrir a sessão: {exc}")
        return

    na_memoria = chain.memory.carregar(mensagens)
    invalidar_sessoes()      # a partir daqui esta sessão é a mais recente

    ui.ok(console, f'Retomando "{_resumo(escolhida["title"], 48)}" '
                   f'— {escolhida["ago"]} · {len(mensagens)} mensagens.')
    ui.spacer(console)
    _redesenhar_conversa(console, mensagens)

    # A janela de contexto é menor que o arquivo, e dizer isso evita a surpresa
    # de perguntar sobre o começo de uma conversa longa e não ser entendido: o
    # texto está todo na tela, mas o modelo só recebe a parte que coube.
    if na_memoria < len(mensagens):
        ui.notice(console, f"desta conversa, as {na_memoria} últimas mensagens "
                           f"entraram na memória do modelo; o resto está acima, "
                           f"só para você.")
    ui.divider(console, " continuando daqui ")
    ui.spacer(console)


def _handle_padroes(arg: str, ctx: dict) -> None:
    """Mostra (ou esquece) as preferências que sobrevivem entre sessões."""
    console = ctx["console"]

    if arg.strip().lower() in {"limpar", "esquecer", "apagar", "reset"}:
        if prefs_mod.esquecer():
            ui.ok(console, "Preferências esquecidas — na próxima sessão vale "
                           "o que estiver no config.py.")
        else:
            ui.notice(console, "não havia preferências gravadas.")
        return

    guardadas = prefs_mod.carregar()
    if not guardadas:
        ui.notice(console, "nenhuma preferência guardada — valendo os padrões "
                           "do config.py.")
        ui.hint(console, "Troque algo com /modelo, /think, /stt, /vad ou /voz "
                         "e eu lembro na próxima vez.")
        return

    ui.heading(console, "Preferências guardadas:")
    for campo in prefs_mod.campos():
        if campo not in guardadas:
            continue
        valor = guardadas[campo]
        if isinstance(valor, bool):
            valor = "ligado" if valor else "desligado"
        console.print(f"    [{config.UI_COLOR_PROMPT}]{campo}[/]"
                      f"  [{config.UI_COLOR_BRIGHT}]{valor}[/]")
    ui.hint(console, f"[{config.UI_COLOR_PROMPT}]/padroes limpar[/] esquece "
                     f"tudo · {config.PREFS_FILE}")
    # A escuta não entra na lista, e é melhor dizer isso do que deixar alguém
    # concluir que ela ficou ligada de ontem.
    ui.notice(console, "  A escuta pela palavra \"Oráculo\" nunca é lembrada: "
                       "microfone aberto é escolha de cada sessão.")


def _handle_think(ctx: dict) -> None:
    console = ctx["console"]
    chain = ctx["chain"]
    want = not ctx.get("thinking", False)

    if want and not llm_mod.supports_thinking(chain.model_name):
        ui.warn(console, f"O modelo {chain.model_name} não suporta raciocínio (thinking).")
        return

    ctx["thinking"] = want
    chain.set_thinking(want)
    prefs_mod.gravar(think=want)
    if want:
        ui.ok(console, "Raciocínio ativado — Ctrl+O mostra/oculta o texto.")
    else:
        ui.ok(console, "Raciocínio (thinking) desativado.")


def _handle_vad(ctx: dict) -> None:
    """Alterna a parada automática da gravação (VAD) e o push-to-talk."""
    console = ctx["console"]
    quer = not config.VAD_ENABLED

    if quer:
        # Só liga se o modelo realmente carregar: melhor recusar aqui, com a
        # causa na tela, do que falhar no meio da próxima gravação.
        from core import vad

        if not vad.disponivel():
            ui.warn(console, "VAD indisponível — o push-to-talk continua ativo.")
            return

    config.VAD_ENABLED = quer
    prefs_mod.gravar(vad=quer)
    if quer:
        ui.ok(console, "VAD ativado — a gravação para sozinha quando você "
                       "parar de falar.")
    else:
        ui.ok(console, "VAD desativado — Enter para gravar, Enter de novo "
                       "para parar.")


def _handle_despertar(ctx: dict) -> None:
    """Alterna a escuta pela palavra de despertar.

    Ao ligar, diz em voz alta o que isso implica: o microfone fica aberto. Essa
    frase não é enfeite — é o que faz do modo uma escolha informada.
    """
    console = ctx["console"]
    quer = not config.WAKE_ENABLED

    if quer:
        from core import wake

        if not wake.disponivel():
            ui.warn(console, "Escuta indisponível — o push-to-talk continua ativo.")
            motivo = wake.motivo_indisponivel()
            for linha in motivo.splitlines():
                if linha.strip():
                    ui.warn(console, f"  {linha.strip()}")
            return

    config.WAKE_ENABLED = quer
    if quer:
        ui.ok(console,
              f'Escuta ativada — o microfone fica aberto e eu respondo '
              f'quando você disser "{config.WAKE_WORD}".')
        ui.notice(console,
                  "  Nada é gravado nem transcrito antes disso. Ctrl+C encerra "
                  "a escuta.")
        if not ctx.get("voice_mode"):
            ui.notice(console, "  Só vale no modo voz — use /voz para entrar.")
    else:
        ui.ok(console, "Escuta desativada — o microfone só abre quando você "
                       "pedir.")


def _handle_dono(ctx: dict) -> None:
    """Alterna a verificação de voz.

    Ao ligar, diz o que ela é e o que ela não é: um filtro de sala, não uma
    autenticação. Prometer mais do que um vetor de timbre entrega seria a mesma
    desonestidade que o system prompt evita.
    """
    console = ctx["console"]
    quer = not config.LOCUTOR_ENABLED

    if quer:
        from core import locutor

        if not locutor.disponivel():
            ui.warn(console, "Verificação de voz indisponível.")
            for linha in locutor.motivo_indisponivel().splitlines():
                if linha.strip():
                    ui.warn(console, f"  {linha.strip()}")
            return

    config.LOCUTOR_ENABLED = quer
    prefs_mod.gravar(dono=quer)
    if quer:
        ui.ok(console, "Verificação de voz ativada — só respondo à sua voz.")
        ui.notice(console,
                  "  Fala de outra pessoa é descartada sem transcrever. Frases "
                  "muito curtas passam: não dá para julgar timbre em meio "
                  "segundo.")
        if not ctx.get("voice_mode"):
            ui.notice(console, "  Só vale no modo voz — use /voz para entrar.")
    else:
        ui.ok(console, "Verificação de voz desativada — respondo a qualquer voz.")


def _handle_comandos(ctx: dict) -> None:
    """Alterna as ações no computador.

    Ao ligar, lista o que ele passa a poder fazer — a lista sai do registro, que
    é a mesma fonte do system prompt. E diz o que está travado e por quê: um
    recurso que existe mas não aparece é indistinguível de um que quebrou.

    NÃO é gravado nas preferências, de propósito. Ver core/prefs.py.
    """
    from core import acoes as acoes_mod

    console = ctx["console"]
    if config.ACOES_ENABLED:
        config.ACOES_ENABLED = False
        ctx["chain"].set_acoes(False)
        ui.ok(console, "Ações desativadas — voltei a só gerar texto.")
        return

    config.ACOES_ENABLED = True
    ctx["chain"].set_acoes(True)
    ui.ok(console, "Ações ativadas — posso mexer no computador.")
    liberado, motivo = acoes_mod.pode_destrutiva(ctx)
    for acao in acoes_mod.ACOES.values():
        if acao.destrutiva and not liberado:
            continue
        args = ", ".join(acao.parametros) or "—"
        console.print(f"    [{config.UI_COLOR_PROMPT}]{acao.nome}[/] "
                      f"[{config.UI_COLOR_FAINT}]{args}[/]")
    if not liberado:
        ui.notice(console, f"  desligar/reiniciar indisponíveis: {motivo}")
    else:
        ui.notice(console, "  desligar/reiniciar pedem confirmação sempre")
    ui.notice(console, "  vale só nesta sessão — /comandos não é lembrado")


def _ligar_notas(ctx: dict) -> bool:
    """Carrega o índice e liga a consulta na chain. True se conseguiu."""
    from core import rag

    console = ctx["console"]
    try:
        consulta = rag.Consulta.abrir()
    except rag.RagError as exc:
        ui.warn(console, f"Notas indisponíveis: {exc}")
        return False
    ctx["chain"].set_notas(consulta)
    ctx["rag"] = consulta
    config.RAG_ENABLED = True
    return True


def _handle_notas(ctx: dict) -> None:
    """Alterna a consulta às notas do Obsidian.

    Ligar troca o system prompt junto (ver OraculoChain.set_notas): o Oráculo
    só anuncia que lê as suas notas enquanto realmente lê.
    """
    console = ctx["console"]
    if config.RAG_ENABLED:
        ctx["chain"].set_notas(None)
        ctx.pop("rag", None)
        config.RAG_ENABLED = False
        prefs_mod.gravar(notas=False)
        ui.ok(console, "Consulta às notas desativada.")
        return

    if not _ligar_notas(ctx):
        ui.hint(console, "  Indexe primeiro com [%s]/indexar[/]." %
                config.UI_COLOR_PROMPT)
        return

    prefs_mod.gravar(notas=True)
    consulta = ctx["rag"]
    meta = consulta.indice.meta
    ui.ok(console, f"Consulta às notas ativada — {meta['trechos']} trechos de "
                   f"{meta['notas']} notas.")
    ui.notice(console, f"  Vault: {meta['vault']}")
    if consulta.indice.desatualizado():
        ui.warn(console, "  O vault mudou desde a indexação — rode /indexar.")


def _handle_indexar(ctx: dict) -> None:
    """(Re)constrói o índice das notas, com barra de progresso.

    O trabalho pesado é o embedding de ~1300 trechos, e ele acontece no Ollama;
    aqui é só I/O e espera. Ao terminar, o índice em uso é substituído na hora —
    reindexar e continuar consultando o índice velho seria o tipo de estado
    desencontrado que ninguém percebe.
    """
    from core import rag

    console = ctx["console"]
    try:
        raiz = rag.vault_dir()
    except rag.RagError as exc:
        ui.error(console, str(exc))
        return

    notas = rag.listar_notas(raiz)
    ui.notice(console, f"Indexando {len(notas)} notas de {raiz}...")
    inicio = time.monotonic()
    try:
        with console.status(f"[{config.UI_COLOR_ACCENT}]lendo as notas...",
                            spinner=ui.IRIS) as status:
            def progresso(feito: int, total: int) -> None:
                status.update(f"[{config.UI_COLOR_ACCENT}]embutindo "
                              f"{feito}/{total} trechos...")

            indice = rag.construir(raiz, progresso=progresso)
        caminho = indice.gravar()
    except rag.RagError as exc:
        ui.error(console, f"Falhou: {exc}")
        return
    except KeyboardInterrupt:
        # Interromper aqui não deixa índice pela metade: o arquivo só é escrito
        # depois que todos os vetores existem.
        ui.warn(console, "Indexação interrompida — o índice anterior continua valendo.")
        return

    ui.ok(console, f"{indice.meta['notas']} notas em "
                   f"{indice.meta['trechos']} trechos "
                   f"({time.monotonic() - inicio:.1f}s).")
    ui.notice(console, f"  {caminho} · "
                       f"{caminho.stat().st_size / 1024:.0f} KB")
    if config.RAG_ENABLED:
        ctx["chain"].set_notas(rag.Consulta(indice))
        ctx["rag"] = ctx["chain"].recuperar
    else:
        ui.hint(console, "  Ligue com [%s]/notas[/]." % config.UI_COLOR_PROMPT)


def _handle_buscar(arg: str, ctx: dict) -> None:
    """Mostra o que a busca traria, sem gastar o LLM.

    É o instrumento antes da dedução, o mesmo papel do `--testar-microfone` do
    wake word: quando a resposta com notas sai ruim, os dois suspeitos são
    "veio o trecho errado" e "veio o certo e o modelo ignorou", e a resposta
    sozinha não distingue. Aqui aparece exatamente o que iria para o contexto —
    e, marcado com um x, o que o limiar barrou e por qual score.
    """
    from core import rag

    console = ctx["console"]
    if not arg:
        ui.warn(console, "Uso: /buscar <pergunta>")
        return

    consulta = ctx.get("rag")
    if consulta is None:
        try:
            consulta = rag.Consulta.abrir()
        except rag.RagError as exc:
            ui.warn(console, f"Notas indisponíveis: {exc}")
            return

    inicio = time.monotonic()
    achados = consulta.indice.buscar(arg, k=max(config.RAG_TOP_K, 6), minimo=-1.0)
    ms = (time.monotonic() - inicio) * 1000
    ui.heading(console, f"Busca nas notas  ({ms:.0f} ms, "
                        f"limiar {config.RAG_MIN_SCORE})")
    if not achados:
        ui.notice(console, "  nada no índice.")
        return
    for posicao, achado in enumerate(achados, 1):
        passou = achado.score >= config.RAG_MIN_SCORE
        marca = " " if passou else "x"
        cor = config.UI_COLOR_ACCENT if passou else config.UI_COLOR_FAINT
        fonte = achado.trecho.fonte
        if len(fonte) > 64:
            fonte = fonte[:61] + "..."
        console.print(f"    [{config.UI_COLOR_ALERT}]{marca}[/] "
                      f"[{config.UI_COLOR_FAINT}]{posicao}.[/] "
                      f"[{cor}]{achado.score:.3f}[/]  "
                      f"[{config.UI_COLOR_SOFT}]{fonte}[/]")
        console.print(f"          [{config.UI_COLOR_FAINT}]"
                      f"{achado.trecho.caminho}[/]")
    barrados = sum(1 for a in achados if a.score < config.RAG_MIN_SCORE)
    if barrados:
        ui.notice(console, f"  {barrados} barrado(s) pelo limiar (x).")
    # Sem esta linha o resultado parece um bug de ordenação: a ordem é a da
    # fusão vetor+termos e o número é só o cosseno, então ele SOBE e DESCE pela
    # lista. Ver o porquê dos dois papéis em core/rag.py (Indice.buscar).
    ui.notice(console, "  ordem: busca híbrida · número: cosseno, que é o que "
                       "o limiar corta")


def _stt_detalhes() -> dict[str, str]:
    """Uma linha por motor, lida da configuração de verdade.

    Escolher entre whisper e parakeet é escolher entre GPU e limite de duração;
    sem esses números a lista é só dois nomes, e a decisão fica adivinhada.
    """
    return {
        "whisper": f"{config.WHISPER_MODEL} · "
                   f"{config.WHISPER_DEVICE.upper()} · "
                   f"{config.WHISPER_COMPUTE_TYPE}",
        "parakeet": f"CPU · limite ~{config.TRANSCRIBE_PARAKEET_LIMIT:.0f}s "
                    f"por clipe",
    }


def _handle_stt(arg: str, ctx: dict) -> None:
    console = ctx["console"]
    arg = arg.lower()

    if not arg:
        ui.heading(console, "Motores de transcrição (STT):")
        detalhes = _stt_detalhes()
        for engine in STT_ENGINES:
            mark = (f"  [{config.UI_COLOR_ACCENT}](atual)[/]"
                    if engine == config.STT_ENGINE else "")
            info = detalhes.get(engine, "")
            cauda = f"  [{config.UI_COLOR_FAINT}]{info}[/]" if info else ""
            console.print(f"    • [{config.UI_COLOR_BRIGHT}]{engine}[/]"
                          f"{mark}{cauda}")
        ui.hint(console, f"Use [{config.UI_COLOR_PROMPT}]/stt <motor>[/] "
                         f"para trocar.")
        return

    if arg not in STT_ENGINES:
        ui.warn(console, f"Motor desconhecido: {arg} "
                f"(opções: {', '.join(STT_ENGINES)})")
        return

    # transcribe() lê config.STT_ENGINE a cada chamada, então sobrescrever aqui
    # já troca o motor da próxima transcrição — sem reiniciar.
    from core import stt

    config.STT_ENGINE = arg
    prefs_mod.gravar(stt=arg)
    ui.ok(console, f"Motor de STT trocado para {arg}.")
    if not stt.available():
        ui.warn(console, f"Dependências de '{arg}' não instaladas — a transcrição "
                f"vai falhar até instalá-las.")


_SAVE_FLAGS = {"--salvar", "-s"}

TRANSCREVER_USO = (
    f"  [{config.UI_COLOR_FAINT}]Uso:[/] "
    f"[{config.UI_COLOR_PROMPT}]/transcrever <arquivo>[/] "
    f"[{config.UI_COLOR_FAINT}][--salvar][/]\n"
    f"  [{config.UI_COLOR_FAINT}]--salvar grava a transcrição em Markdown "
    f"ao lado do áudio.[/]"
)


def _parse_alvo(arg: str) -> tuple[str, bool]:
    """Separa o caminho das flags.

    O caminho vem sem aspas na maioria das vezes e pode ter espaços (áudios do
    WhatsApp têm), então as flags só são reconhecidas no fim da linha e o resto
    inteiro é tratado como um caminho só."""
    salvar = False
    tokens = arg.split()
    while tokens and tokens[-1].lower() in _SAVE_FLAGS:
        salvar = True
        tokens.pop()

    caminho = " ".join(tokens)
    if len(caminho) > 1 and caminho[0] == caminho[-1] and caminho[0] in "\"'":
        caminho = caminho[1:-1]
    return caminho, salvar


def _com_progresso(segments, status, secs: float | None):
    """Repassa os segmentos atualizando o spinner com a posição no áudio."""
    from core.transcript import hms

    for seg in segments:
        total = f"/{hms(secs)}" if secs else ""
        status.update(f"[{config.UI_COLOR_SOFT}]Transcrevendo...[/] "
                      f"[{config.UI_COLOR_FAINT}]{hms(seg[1])}{total}[/]")
        yield seg


def _handle_transcrever(arg: str, ctx: dict) -> None:
    console = ctx["console"]

    if not arg:
        console.print(TRANSCREVER_USO)
        return

    caminho, salvar = _parse_alvo(arg)
    path = Path(caminho).expanduser()
    if not path.is_file():
        ui.warn(console, f"Arquivo não encontrado: {path}")
        return

    from core import stt, transcript

    if not stt.available():
        ui.warn(console, f"O motor '{config.STT_ENGINE}' não está instalado — "
                         f"veja /stt para trocar de motor.")
        return

    if path.suffix.lower() not in config.TRANSCRIBE_EXTENSIONS:
        ui.warn(console, f"'{path.suffix}' não parece um formato de áudio; "
                         f"vou tentar mesmo assim.")

    secs = stt.duration(str(path))
    limite = config.TRANSCRIBE_PARAKEET_LIMIT
    if config.STT_ENGINE == "parakeet" and secs and secs > limite:
        ui.warn(console, f"O parakeet trunca clipes acima de {limite:.0f}s. "
                         f"Para este áudio, use /stt whisper.")

    dur = f"  [{config.UI_COLOR_FAINT}]({transcript.hms(secs)})[/]" if secs else ""
    console.print(f"  [bold {config.UI_COLOR_ACCENT}]Transcrevendo[/] "
                  f"[{config.UI_COLOR_BRIGHT}]{path.name}[/]{dur}"
                  f"  [{config.UI_COLOR_FAINT}]· {transcript.engine_label()}[/]")

    paras: list[tuple[float, str]] = []
    interrompido = False
    try:
        with console.status(f"[{config.UI_COLOR_SOFT}]Carregando o motor de "
                            f"transcrição...[/]", spinner=ui.IRIS) as status:
            segments = _com_progresso(stt.transcribe_segments(str(path)),
                                      status, secs)
            for start, texto in transcript.paragraphs(segments):
                paras.append((start, texto))
                # Montado como Text, não como marcação: o parágrafo vem da
                # transcrição e um "[" solto no meio da fala viraria uma tag.
                linha = Text("  ")
                if config.TRANSCRIBE_TIMESTAMPS:
                    linha.append(f"[{transcript.hms(start)}]  ",
                                 style=config.UI_COLOR_FAINT)
                linha.append(texto, style=config.UI_COLOR_BODY)
                console.print(linha)
    except KeyboardInterrupt:
        interrompido = True
        ui.warn(console, "transcrição interrompida")
    except RuntimeError as exc:      # dependência faltando
        ui.warn(console, str(exc))
        return
    except Exception as exc:         # noqa: BLE001 — áudio ilegível, disco, etc.
        ui.error(console, f"Erro ao transcrever: {exc}")
        return

    if not paras:
        ui.notice(console, "nada foi transcrito — o áudio tem fala?")
        return

    if salvar:
        try:
            destino = transcript.save(paras, path, secs)
            parcial = f" [{config.UI_COLOR_FAINT}](parcial)[/]" if interrompido else ""
            ui.hint(console, f"[{config.UI_COLOR_ACCENT}]Transcrição salva em[/] "
                             f"[{config.UI_COLOR_BRIGHT}]{destino}[/]{parcial}")
        except OSError as exc:
            ui.error(console, f"Não consegui salvar: {exc}")
    elif not interrompido:
        ui.hint(console, f"use [{config.UI_COLOR_PROMPT}]--salvar[/] para "
                         f"gravar em Markdown")


def handle(raw: str, ctx: dict) -> bool:
    raw = raw.strip()
    if not raw.startswith("/"):
        return False

    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    console = ctx["console"]

    if cmd in config.EXIT_COMMANDS:
        ctx["running"] = False
        ui.ok(console, "Encerrando...")
        return True

    if cmd in {"/ajuda", "/help", "/?"}:
        console.print(AJUDA_TEXT)
        return True

    if cmd == "/limpar":
        ctx["chain"].memory.clear()
        ui.ok(console, "Memória da sessão limpa.")
        return True

    if cmd == "/voz":
        ctx["voice_mode"] = not ctx["voice_mode"]
        prefs_mod.gravar(voz=ctx["voice_mode"])
        estado = "ativado" if ctx["voice_mode"] else "desativado"
        ui.ok(console, f"Modo voz {estado}.")
        return True

    if cmd == "/retomar":
        _handle_retomar(arg, ctx)
        return True

    if cmd in {"/padroes", "/padrões"}:
        _handle_padroes(arg, ctx)
        return True

    if cmd == "/vad":
        _handle_vad(ctx)
        return True

    if cmd == "/despertar":
        _handle_despertar(ctx)
        return True

    if cmd == "/dono":
        _handle_dono(ctx)
        return True

    if cmd in {"/comandos", "/acoes", "/ações"}:
        _handle_comandos(ctx)
        return True

    if cmd == "/notas":
        _handle_notas(ctx)
        return True

    if cmd == "/indexar":
        _handle_indexar(ctx)
        return True

    if cmd == "/buscar":
        _handle_buscar(arg, ctx)
        return True

    if cmd == "/think":
        _handle_think(ctx)
        return True

    if cmd == "/stt":
        _handle_stt(arg, ctx)
        return True

    if cmd in {"/transcrever", "/transcricao"}:
        _handle_transcrever(arg, ctx)
        return True

    if cmd == "/modelo":
        _handle_modelo(arg, ctx)
        return True

    ui.warn(console, f"Comando desconhecido: {cmd}  (veja /ajuda)")
    return True
