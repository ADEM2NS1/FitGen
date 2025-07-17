# my_chatbot_backend/models.py

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    # Relationship to conversations: one user can have many conversations
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    owner_id = Column(Integer, ForeignKey("users.id"))

    # Relationship to user: many conversations belong to one user
    owner = relationship("User", back_populates="conversations")
    # Relationship to messages: one conversation can have many messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # e.g., "user", "assistant"
    content = Column(Text) # Use Text for potentially long messages
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to conversation: many messages belong to one conversation
    conversation = relationship("Conversation", back_populates="messages")