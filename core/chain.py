"""Pipeline principal: system prompt + memória + LLM."""

from collections.abc import Iterator

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from core.llm import build_llm
from core.memory import ConversationMemory


class OraculoChain:
    """Orquestra prompt, memória e modelo para uma sessão de conversa."""

    def __init__(self):
        self.memory = ConversationMemory()
        self.llm = build_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.pipeline = self.prompt | self.llm

    def stream(self, user_input: str) -> Iterator[str]:
        """Gera a resposta em streaming, acumulando na memória ao final."""
        chunks: list[str] = []
        for chunk in self.pipeline.stream(
            {"input": user_input, "history": self.memory.messages}
        ):
            text = chunk.content
            if text:
                chunks.append(text)
                yield text

        response = "".join(chunks)
        self.memory.add_user(user_input)
        self.memory.add_assistant(response)
