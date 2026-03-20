from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Numeric,
    String,
    text,
)
from sqlalchemy.orm import declarative_base

from scripts.db.db import get_engine

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


def init_database() -> None:
    engine = get_engine(echo=True)
    Base.metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        connection.execute(
            text(
                """
                SELECT create_hypertable(
                    'market_data',
                    'timestamp',
                    partitioning_column => 'ticker',
                    number_partitions => 4,
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                )
                """
            )
        )


if __name__ == "__main__":
    init_database()
