import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from scripts.db.db import get_engine, get_session
from scripts.db.init_db import Base, MarketData

logger = logging.getLogger(__name__)


def sync_csv(csv_path: Path, ticker: str, interval: str) -> None:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    df = df.reset_index()
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["volume"] = df["volume"].fillna(0).astype(int)

    session = get_session()
    batch_size = 500
    rows = df.to_dict(orient="records")

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        values = [
            {
                "ticker": ticker,
                "interval": interval,
                "timestamp": row["timestamp"],
                "open": float(row["open"]) if pd.notna(row["open"]) else None,
                "high": float(row["high"]) if pd.notna(row["high"]) else None,
                "low": float(row["low"]) if pd.notna(row["low"]) else None,
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "volume": int(row["volume"]),
            }
            for row in batch
        ]
        stmt = insert(MarketData).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "interval", "timestamp"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        session.execute(stmt)

    session.commit()
    session.close()
    logger.info(f"Synced {len(rows)} rows for {ticker} @ {interval}")


if __name__ == "__main__":
    engine = get_engine(echo=True)
    Base.metadata.create_all(engine)

    data_dir = Path("data/raw")
    for csv_file in data_dir.glob("*.csv"):
        parts = csv_file.stem.split("_")
        if len(parts) >= 2:
            ticker = parts[0].upper()
            interval = parts[1]
        else:
            ticker = parts[0].upper()
            interval = "1d"
        sync_csv(csv_file, ticker, interval)
