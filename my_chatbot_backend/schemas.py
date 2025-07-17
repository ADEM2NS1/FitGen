# my_chatbot_backend/schemas.py

from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# --- User Schemas ---
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    # conversations: List[Conversation] = [] # Optional: if you want to embed conversations directly
    class Config:
        orm_mode = True # Enables Pydantic to read ORM models

# --- Token Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- Conversation Schemas ---
class ConversationBase(BaseModel):
    title: str

class ConversationCreate(ConversationBase):
    pass # No extra fields needed for creation beyond base

class Conversation(ConversationBase):
    id: int
    owner_id: int
    created_at: datetime

    class Config:
        orm_mode = True

# --- Message Schemas ---
class MessageBase(BaseModel):
    role: str
    content: str

class MessageCreate(MessageBase):
    pass

class Message(MessageBase):
    id: int
    conversation_id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# --- Response Schemas ---
class MessageResponse(BaseModel):
    message: str