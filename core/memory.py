"""Memória de conversação da sessão atual.

Mantém um histórico em memória (volátil) limitado às últimas N mensagens,
para preservar contexto sem estourar o num_ctx do modelo.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

import config


class ConversationMemory:
    """Histórico de mensagens da sessão, com janela deslizante."""

    def __init__(self, max_messages: int = config.MAX_HISTORY_MESSAGES):
        self.max_messages = max_messages
        self._history = InMemoryChatMessageHistory()

    def add_user(self, content: str) -> None:
        self._history.add_user_message(content)
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._history.add_ai_message(content)
        self._trim()

    @property
    def messages(self) -> list[BaseMessage]:
        return self._history.messages

    def _trim(self) -> None:
        """Mantém no máximo `max_messages`, removendo mensagens antigas em PARES.

        Como o histórico alterna user/assistant, remover sempre um número par pela
        frente preserva pares completos e evita deixar uma resposta órfã (uma
        mensagem da IA sem a pergunta correspondente) no início da janela.
        """
        msgs = self._history.messages
        excess = len(msgs) - self.max_messages
        if excess > 0:
            remove = excess + (excess % 2)  # arredonda para cima até um número par
            self._history.messages = msgs[remove:]

    def clear(self) -> None:
        self._history.clear()
