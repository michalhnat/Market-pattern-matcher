import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from config import INTERVALS, Config
from scripts.db.db import get_engine, get_session
from scripts.db.init_db import Base, MarketData
from scripts.db.sync_db import sync_csv
from src.core.index import IndexBuilder
from src.core.train import Trainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tickers", tags=["tickers"])

_jobs: dict[str, dict[str, Any]] = {}


class AddTickerRequest(BaseModel):
    ticker: str
    intervals: list[str] | None = None
    window_size: int = 30
    epochs: int = 20
    batch_size: int = 64
    embedding_dim: int = 32


class AddTickerResponse(BaseModel):
    job_id: str
    ticker: str
    status: str
    message: str


def download_ticker_data(ticker: str, interval: str, output_dir: Path) -> Path:
    period = INTERVALS[interval]
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if "Date" in df.columns and hasattr(df["Date"].dtype, "tz"):
        df["Date"] = df["Date"].dt.tz_localize(None)

    csv_path = output_dir / f"{ticker}_{interval}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def preprocess_ticker_data(
    csv_path: Path, ticker: str, interval: str, window_size: int, output_dir: Path
) -> tuple[Path, Path]:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=str.capitalize)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    prices = df[["Open", "High", "Low", "Close"]].values
    volumes = df["Volume"].values.reshape(-1, 1)

    all_windows = []
    all_metadata = []

    for i in range(len(df) - window_size + 1):
        window_prices = prices[i:i + window_size]
        window_volumes = volumes[i:i + window_size]

        p_min, p_max = window_prices.min(), window_prices.max()
        if p_max - p_min < 1e-8:
            continue

        normed_prices = (window_prices - p_min) / (p_max - p_min)
        v_min, v_max = window_volumes.min(), window_volumes.max()
        normed_volumes = (
            (window_volumes - v_min) / (v_max - v_min) if v_max > v_min else window_volumes
        )

        window_features = np.hstack([normed_prices, normed_volumes])
        all_windows.append(window_features)

        window_dates = df["Date"].iloc[i:i + window_size]
        all_metadata.append({
            "ticker": ticker,
            "start_date": str(window_dates.iloc[0]),
            "end_date": str(window_dates.iloc[-1])
        })

    if not all_windows:
        raise ValueError(f"No valid windows for {ticker}")

    folder_name = f"{ticker}_{interval}_minmax_{window_size}"
    preprocessed_dir = output_dir / folder_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    data_path = preprocessed_dir / "data.npy"
    np.save(data_path, np.array(all_windows, dtype=np.float32))
    pd.DataFrame(all_metadata).to_csv(preprocessed_dir / "metadata.csv", index=False)

    return preprocessed_dir, data_path


def process_single_interval(
    ticker: str,
    interval: str,
    request: AddTickerRequest,
    project_root: Path
) -> str:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_ticker_data(ticker, interval, raw_dir)

    preprocessed_dir = project_root / "data" / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    processed_dir, _ = preprocess_ticker_data(
        csv_path, ticker, interval, request.window_size, preprocessed_dir
    )

    config = Config(
        data_dir=processed_dir,
        interval=interval,
        epochs=request.epochs,
        batch_size=request.batch_size,
        embedding_dim=request.embedding_dim,
        window_size=request.window_size
    )

    trainer = Trainer(config)
    trainer.run_training()

    builder = IndexBuilder(config)
    builder.build_index(model_path=config.model_path)

    engine = get_engine()
    Base.metadata.create_all(engine)
    sync_csv(csv_path, ticker, interval)

    return f"{interval}"


def process_ticker(job_id: str, ticker: str, request: AddTickerRequest) -> None:
    intervals = request.intervals or ["1d"]
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    _jobs[job_id]["status"] = "processing"

    for interval in intervals:
        if interval not in INTERVALS:
            _jobs[job_id]["progress"].append(f"{interval}: invalid")
            continue

        try:
            result = process_single_interval(ticker, interval, request, project_root)
            _jobs[job_id]["progress"].append(result)
            logger.info(f"{ticker} @ {interval} done")
        except Exception as e:
            _jobs[job_id]["progress"].append(f"{interval}: ✗ {str(e)}")
            logger.error(f"{ticker} @ {interval} failed: {e}")

    _jobs[job_id]["status"] = "completed"
    _jobs[job_id]["completed_at"] = datetime.now().isoformat()


@router.post("/add", response_model=AddTickerResponse)
async def add_ticker(
    request: AddTickerRequest, background_tasks: BackgroundTasks
) -> AddTickerResponse:
    ticker = request.ticker.upper()
    job_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _jobs[job_id] = {
        "ticker": ticker,
        "status": "queued",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "progress": []
    }

    background_tasks.add_task(process_ticker, job_id, ticker, request)

    return AddTickerResponse(
        job_id=job_id,
        ticker=ticker,
        status="queued",
        message=f"Processing {ticker}"
    )


@router.get("/status/{job_id}")
async def get_status(job_id: str) -> dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    return [
        {"job_id": jid, **job}
        for jid, job in sorted(
            _jobs.items(),
            key=lambda x: x[1]["started_at"],
            reverse=True
        )[:10]
    ]


@router.get("/available")
async def get_available_tickers() -> list[dict[str, Any]]:
    """Get available tickers from database."""
    session = get_session()
    try:
        stmt = (
            select(MarketData.ticker, MarketData.interval)
            .distinct()
            .order_by(MarketData.ticker, MarketData.interval)
        )
        result = session.execute(stmt).all()
        return [{"ticker": row[0], "interval": row[1]} for row in result]
    finally:
        session.close()
