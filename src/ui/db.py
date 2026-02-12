import os
from datetime import date, datetime

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    Index,
    Numeric,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class MarketData(Base):
    __tablename__ = "market_data"

    ticker = Column(String(10), primary_key=True, nullable=False)
    interval = Column(String(10), primary_key=True, nullable=False)
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    open = Column(Numeric(12, 4))
    high = Column(Numeric(12, 4))
    low = Column(Numeric(12, 4))
    close = Column(Numeric(12, 4))
    volume = Column(BigInteger, nullable=True)

    __table_args__ = (
        Index("idx_ticker_interval_ts", "ticker", "interval", "timestamp"),
    )


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


def get_market_data(
    ticker: str,
    interval: str,
    start_date: str | datetime | date,
    end_date: str | datetime | date,
) -> pd.DataFrame:
    session = get_session()

    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    results = (
        session.query(MarketData)
        .filter(
            MarketData.ticker == ticker.upper(),
            MarketData.interval == interval,
            MarketData.timestamp >= start_date,
            MarketData.timestamp <= end_date,
        )
        .order_by(MarketData.timestamp)
        .all()
    )

    session.close()

    if not results:
        return pd.DataFrame()

    data = {
        "Date": [r.timestamp for r in results],
        "Open": [
            float(r.open) if r.open is not None else 0.0 for r in results
        ],
        "High": [
            float(r.high) if r.high is not None else 0.0 for r in results
        ],
        "Low": [
            float(r.low) if r.low is not None else 0.0 for r in results
        ],
        "Close": [
            float(r.close) if r.close is not None else 0.0 for r in results
        ],
        "Volume": [
            int(r.volume) if r.volume is not None else 0 for r in results
        ],
    }

    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    return df
