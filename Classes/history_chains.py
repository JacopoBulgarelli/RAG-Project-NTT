from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Union
from pydantic import BaseModel

class BaseMessage(BaseModel):
    """Base class for messages"""
    type: str
    content: Union[str, Dict]

class InMemoryHistory(BaseModel):
    """In-memory implementation of chat message history"""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the history"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all messages from the history"""
        self.messages = []

    def get_conversation(self):
        """Return the questions and answers in the conversation"""
        conversation = []
        for message in self.messages:
            conversation.append({"type": message.type, "content": message.content})
        return conversation

class MessageHistoryStore:
    """Store for managing message histories by session ID"""
    def __init__(self):
        self.store = {}

    def get_by_session_id(self, session_id: str) -> InMemoryHistory:
        """Retrieve the message history for a given session ID, create a new history if needed"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryHistory()
        return self.store[session_id]

class ChainWithHistory:
    def __init__(self, qa_chain, message_history_store: MessageHistoryStore, input_messages_key="question", history_messages_key="history"):
        self.qa_chain = qa_chain
        self.message_history_store = message_history_store
        self.input_messages_key = input_messages_key
        self.history_messages_key = history_messages_key

    def invoke(self, input_data: Dict, config: Dict = None):
        """Invoke the Q&A chain with the message history"""
        session_id = config.get("configurable", {}).get("session_id")
        if not session_id:
            raise ValueError("Session ID is required to fetch history.")

        # Fetch the history using the session_id
        history = self.message_history_store.get_by_session_id(session_id)

        # Add the history messages to the input_data
        input_data[self.input_messages_key] = history.messages

        # Process the input data through the Q&A chain
        response = self.qa_chain.invoke(input_data)

        # Add user and bot messages to history
        user_message = BaseMessage(type="user", content=input_data[self.input_messages_key])
        history.add_messages([user_message])

        bot_message = BaseMessage(type="bot", content=response)
        history.add_messages([bot_message])

        return response
