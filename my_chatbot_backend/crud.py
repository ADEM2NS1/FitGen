# my_chatbot_backend/crud.py

from sqlalchemy.orm import Session
from . import models, schemas
from .auth import get_password_hash # Import for hashing passwords

# --- User CRUD Operations ---
def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Conversation CRUD Operations ---
def get_conversations(db: Session, user_id: int):
    return db.query(models.Conversation).filter(models.Conversation.owner_id == user_id).all()

def create_conversation(db: Session, conversation: schemas.ConversationCreate, user_id: int):
    db_conversation = models.Conversation(**conversation.dict(), owner_id=user_id)
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation

def delete_conversation(db: Session, conversation_id: int, user_id: int):
    """
    Deletes a conversation and all its associated messages from the database,
    but only if it belongs to the specified user.
    """
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.owner_id == user_id
    ).first()

    if conversation:
        # Delete associated messages first
        db.query(models.Message).filter(models.Message.conversation_id == conversation_id).delete(synchronize_session=False)
        db.delete(conversation)
        db.commit()
        return True # Indicate successful deletion
    return False # Indicate conversation not found or not owned by user

# --- Message CRUD Operations ---
def get_messages(db: Session, conversation_id: int):
    return db.query(models.Message).filter(models.Message.conversation_id == conversation_id).order_by(models.Message.timestamp).all()

def create_message(db: Session, message: schemas.MessageCreate, conversation_id: int):
    db_message = models.Message(**message.dict(), conversation_id=conversation_id)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message