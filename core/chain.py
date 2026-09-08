"""Pipeline principal: system prompt + memória + LLM."""

from collections.abc import Callable, Iterator

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from core import text as textproc
from core.llm import build_llm
from core.memory import ConversationMemory


class OraculoChain:
    """Orquestra prompt, memória e modelo para uma sessão de conversa."""

    def __init__(self, model_name: str = config.OLLAMA_MODEL):
        self.model_name = model_name
        self.memory = ConversationMemory()
        # Metadata do último stream (uso de tokens/durações do Ollama) para a
        # telemetria ler. Preenchido ao final de cada stream bem-sucedido.
        self.last_usage: dict = {"usage": None, "meta": None}
        # Raciocínio (thinking) SEMPRE explícito, nunca None.
        #
        # Deixar o padrão do modelo decidir faz a interface mentir: o gemma4
        # raciocina por padrão, então com `reasoning=None` o Oráculo pensava em
        # todo turno enquanto o /think e a barra de status diziam "desligado" —
        # e só passava a obedecer depois de alternar o /think uma vez, que é o
        # que finalmente escrevia o valor explícito.
        #
        # Começa desligado (custo zero); main.py liga no arranque se
        # THINKING_DEFAULT estiver ligado E o modelo suportar. `reasoning=False`
        # é aceito por qualquer modelo — só `True` dá 400 em quem não suporta.
        self.reasoning: bool = False
        # Consulta às notas (Fase 4). É uma função `pergunta -> (bloco, fontes)`
        # injetada de fora — a chain não conhece o Obsidian, não carrega índice
        # nenhum e, com ela em None, o turno é byte a byte o que era antes
        # (invariante 5: custo zero quando desligado).
        self.recuperar: Callable[[str], tuple[str, list[str]]] | None = None
        # Fontes usadas no último turno, para o rodapé mostrar.
        self.last_sources: list[str] = []
        self._montar_prompt(notas=False)
        self.llm = build_llm(self.model_name, reasoning=self.reasoning)
        self.pipeline = self.prompt | self.llm

    def _montar_prompt(self, notas: bool) -> None:
        """(Re)monta o template. `notas` decide qual system prompt entra.

        O bloco recuperado é um placeholder PRÓPRIO, e não um pedaço colado no
        `{input}`: o que vai para a memória é a pergunta que o usuário fez, e
        colar as notas nela guardaria quatro trechos do vault dentro do
        histórico da conversa — a cada turno, empilhando, até a janela estourar
        com contexto que já cumpriu sua função.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.build_system_prompt(notas)),
                MessagesPlaceholder(variable_name="history"),
                MessagesPlaceholder(variable_name="notas"),
                ("human", "{input}"),
            ]
        )

    def set_model(self, model_name: str) -> None:
        """Troca o modelo ativo em tempo de execução, preservando memória e reasoning."""
        self.model_name = model_name
        self.llm = build_llm(model_name, reasoning=self.reasoning)
        self.pipeline = self.prompt | self.llm

    def set_thinking(self, enabled: bool) -> None:
        """Liga/desliga o raciocínio (thinking), recriando o LLM."""
        self.reasoning = bool(enabled)
        self.llm = build_llm(self.model_name, reasoning=self.reasoning)
        self.pipeline = self.prompt | self.llm

    def set_notas(self, recuperar: "Callable[[str], tuple[str, list[str]]] | None") -> None:
        """Liga (função) ou desliga (None) a consulta às notas.

        Trocar o recuperador troca junto o system prompt, e isso é o invariante
        1 em código: o Oráculo declara que lê as suas notas exatamente enquanto
        as lê, e volta a declarar que não acessa arquivos no instante em que
        para. As duas metades da lista de limitações precisam continuar
        verdadeiras, e a única forma de garantir isso é elas mudarem juntas.
        """
        self.recuperar = recuperar
        self._montar_prompt(notas=recuperar is not None)
        self.pipeline = self.prompt | self.llm

    def stream(self, user_input: str) -> Iterator[tuple[str, str]]:
        """Gera a resposta em streaming como eventos `(tipo, texto)`:
          - ("think", ...):  tokens de raciocínio (só se o thinking estiver ligado);
          - ("answer", ...): tokens da resposta final.
        Só a resposta final entra na memória; o raciocínio é efêmero.
        """
        chunks: list[str] = []
        full = None
        notas: list[BaseMessage] = []
        self.last_sources = []
        if self.recuperar is not None:
            # Recuperação nunca derruba o turno: sem o Ollama de embeddings, ou
            # com o índice ilegível, o Oráculo responde sem as notas em vez de
            # falhar. É a mesma regra da verificação de voz — o modo de falha
            # inaceitável é o que emudece o assistente sem dizer por quê.
            try:
                bloco, fontes = self.recuperar(user_input)
            except Exception:  # noqa: BLE001
                bloco, fontes = "", []
            if bloco:
                notas = [SystemMessage(content=bloco)]
                self.last_sources = fontes
        for chunk in self.pipeline.stream(
            {"input": user_input, "history": self.memory.messages,
             "notas": notas}
        ):
            # Agrega TODOS os chunks (mesmo sem conteúdo) para preservar o
            # metadata de uso/duração que o Ollama anexa ao chunk final.
            full = chunk if full is None else full + chunk
            # Raciocínio vem num canal separado (additional_kwargs), nunca em content.
            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            if reasoning:
                yield ("think", reasoning)
            # Remove caracteres CJK que o modelo às vezes vaza no meio do texto.
            text = textproc.strip_cjk(chunk.content) if chunk.content else ""
            if text:
                chunks.append(text)
                yield ("answer", text)

        # Só roda se o stream foi consumido até o fim (numa interrupção via
        # KeyboardInterrupt o gerador é fechado antes daqui — last_usage e a
        # memória não são atualizados, exatamente como antes).
        self.last_usage = {
            "usage": getattr(full, "usage_metadata", None),
            "meta": getattr(full, "response_metadata", None),
        }
        response = "".join(chunks)
        self.memory.add_user(user_input)
        self.memory.add_assistant(response)
