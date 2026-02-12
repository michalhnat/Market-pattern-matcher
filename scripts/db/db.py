import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "mysecretpassword")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "market_data")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_engine(echo: bool = False) -> Engine:
    database_url = os.getenv("DATABASE_URL") or get_database_url()
    return create_engine(database_url, echo=echo)


def get_session() -> Session:
    engine = get_engine()
    session_factory = sessionmaker(bind=engine)
    return session_factory()
