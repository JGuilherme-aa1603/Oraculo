"""Pipeline principal: system prompt + memória + LLM."""

from collections.abc import Iterator

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
        # Raciocínio (thinking): None = padrão do modelo; True/False = explícito.
        self.reasoning: bool | None = None
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.llm = build_llm(self.model_name, reasoning=self.reasoning)
        self.pipeline = self.prompt | self.llm

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

    def stream(self, user_input: str) -> Iterator[tuple[str, str]]:
        """Gera a resposta em streaming como eventos `(tipo, texto)`:
          - ("think", ...):  tokens de raciocínio (só se o thinking estiver ligado);
          - ("answer", ...): tokens da resposta final.
        Só a resposta final entra na memória; o raciocínio é efêmero.
        """
        chunks: list[str] = []
        full = None
        for chunk in self.pipeline.stream(
            {"input": user_input, "history": self.memory.messages}
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
