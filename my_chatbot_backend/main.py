# my_chatbot_backend/main.py

import os
from datetime import timedelta
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware # For CORS

from . import crud, models, schemas, auth
from .database import SessionLocal, engine # Import SessionLocal and engine

# Create all database tables (if they don't exist)
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- CORS Configuration ---
# Adjust origins to match your frontend's URL
origins = [
    "http://localhost",
    "http://localhost:8501", # Your Streamlit app runs on this by default
    "http://127.0.0.1:8501",
    # Add any other origins where your frontend might be hosted
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- User Authentication and Registration ---

@app.post("/register", response_model=schemas.MessageResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    crud.create_user(db=db, user=user)
    return {"message": "User registered successfully"}

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Conversation Endpoints ---

@app.get("/conversations", response_model=List[schemas.Conversation])
def get_user_conversations(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    Retrieve all conversations for the logged-in user.
    """
    conversations = crud.get_conversations(db=db, user_id=current_user.id)
    return conversations

@app.post("/conversations", response_model=schemas.Conversation)
def create_new_conversation(
    conversation: schemas.ConversationCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    Create a new conversation for the logged-in user.
    """
    return crud.create_conversation(db=db, conversation=conversation, user_id=current_user.id)

@app.delete("/conversations/{conversation_id}", response_model=schemas.MessageResponse)
def delete_user_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    Delete a specific conversation by ID.
    Only the owner of the conversation can delete it.
    """
    success = crud.delete_conversation(db=db, conversation_id=conversation_id, user_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or not authorized")
    return {"message": "Conversation deleted successfully"}


# --- Message Endpoints ---

@app.get("/conversations/{conversation_id}/messages", response_model=List[schemas.Message])
def get_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    Retrieve all messages for a specific conversation.
    Ensures the conversation belongs to the current user.
    """
    # First, verify that the conversation belongs to the current user
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.owner_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or not authorized")

    messages = crud.get_messages(db=db, conversation_id=conversation_id)
    return messages

@app.post("/conversations/{conversation_id}/messages", response_model=schemas.Message)
def create_conversation_message(
    conversation_id: int,
    message: schemas.MessageCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    """
    Create a new message within a specific conversation.
    Ensures the conversation belongs to the current user.
    """
    # First, verify that the conversation belongs to the current user
    conversation = db.query(models.Conversation).filter(
        models.Conversation.id == conversation_id,
        models.Conversation.owner_id == current_user.id
    ).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found or not authorized")

    return crud.create_message(db=db, message=message, conversation_id=conversation_id)