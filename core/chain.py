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
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.llm = build_llm(self.model_name)
        self.pipeline = self.prompt | self.llm

    def set_model(self, model_name: str) -> None:
        """Troca o modelo ativo em tempo de execução, preservando a memória."""
        self.model_name = model_name
        self.llm = build_llm(model_name)
        self.pipeline = self.prompt | self.llm

    def stream(self, user_input: str) -> Iterator[str]:
        """Gera a resposta em streaming, acumulando na memória ao final."""
        chunks: list[str] = []
        full = None
        for chunk in self.pipeline.stream(
            {"input": user_input, "history": self.memory.messages}
        ):
            # Agrega TODOS os chunks (mesmo sem conteúdo) para preservar o
            # metadata de uso/duração que o Ollama anexa ao chunk final.
            full = chunk if full is None else full + chunk
            # Remove caracteres CJK que o modelo às vezes vaza no meio do texto.
            text = textproc.strip_cjk(chunk.content) if chunk.content else ""
            if text:
                chunks.append(text)
                yield text

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
