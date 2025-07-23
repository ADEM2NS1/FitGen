# my_chatbot_backend/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- CHANGE THIS LINE ---
# This will create a file named 'sql_app.db' in the same directory as your main.py
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db" 
# For PostgreSQL or MySQL, it would look like:
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@host:port/dbname"
# SQLALCHEMY_DATABASE_URL = "mysql://user:password@host:port/dbname"
# ------------------------

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} # check_same_thread is needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# This function might be in main.py, but ensure it's called to create tables
def create_db_tables():
    Base.metadata.create_all(bind=engine)