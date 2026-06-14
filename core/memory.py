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
        """Mantém apenas as últimas `max_messages` mensagens."""
        msgs = self._history.messages
        if len(msgs) > self.max_messages:
            self._history.messages = msgs[-self.max_messages:]

    def clear(self) -> None:
        self._history.clear()
