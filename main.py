import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yfinance as yf

from config import INTERVALS, Config
from scripts.db.db import get_engine
from scripts.db.init_db import Base
from scripts.db.sync_db import sync_csv
from src.core.index import IndexBuilder
from src.core.search import PatternSearcher
from src.core.train import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_command(args: argparse.Namespace) -> None:
    config = Config(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        device=args.device,
        window_size=args.window_size,
        interval=args.interval,
    )

    logger.info(f"Starting training for {config.ticker}...")
    trainer = Trainer(config)
    trainer.run_training()


def index_command(args: argparse.Namespace) -> None:
    config = Config(
        data_dir=args.data,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        device=args.device,
        interval=args.interval,
    )

    logger.info(f"Building index for {config.ticker}...")
    builder = IndexBuilder(config)
    builder.build_index(model_path=args.model)


def search_command(args: argparse.Namespace) -> None:
    config = Config(
        device=args.device,
        top_k=args.top_k
    )

    searcher = PatternSearcher(config)

    index_path = args.index
    if not index_path and args.data:
        config.data_dir = args.data
        index_path = config.index_path

    if not index_path:
        logger.error("Must provide --index path or --data path to resolve default index.")
        return

    searcher.load_resources(index_path=index_path)

    results = searcher.search(
        query_index=args.query_index,
        query_date=args.date,
        top_k=args.top_k,
        include_self=args.include_self
    )

    print(
        f"\nSearch Results for {config.ticker} "
        f"(Query: {'Index ' + str(args.query_index)}"
        f" {args.date if args.query_index is None else ''})"
    )
    print("-" * 85)
    header = (
        f"{'Rank':<5} | {'Distance':<10} | {'Start Date':<12} | "
        f"{'End Date':<12} | {'Ticker':<8} | {'Window Idx':<10}"
    )
    print(header)
    print("-" * 85)

    for r in results:
        print(
            f"{r.rank:<5} | {r.distance:<10.4f} | "
            f"{r.start_date:<12} | {r.end_date:<12} | "
            f"{r.ticker:<8} | {r.window_index:<10}"
        )
    print("-" * 85)


def download_ticker_data(ticker: str, interval: str, output_dir: Path) -> Path:
    period = INTERVALS[interval]
    logger.info(f"Downloading {ticker} @ {interval} (period={period})...")
    csv_path = output_dir / f"{ticker}_{interval}.csv"

    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker} @ {interval}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel('Ticker')

    # Strip timezone for consistent storage
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index.name = "Date"
    data.to_csv(csv_path)
    logger.info(f"Downloaded {len(data)} rows to {csv_path}")
    return csv_path


def preprocess_ticker_data(
    csv_path: Path, ticker: str, interval: str, window_size: int, output_dir: Path
) -> tuple[Path, Path]:
    logger.info(f"Preprocessing {ticker} @ {interval}...")

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    if len(df) < window_size:
        raise ValueError(
            f"Not enough data for {ticker} @ {interval} "
            f"(need {window_size}, have {len(df)})"
        )

    prices = df[["Open", "High", "Low", "Close"]]
    volume = df["Volume"]
    dates = df.index

    all_windows = []
    all_metadata = []

    for i in range(len(prices) - window_size + 1):
        window = prices[i : i + window_size]
        volume_window = volume[i : i + window_size]
        window_dates = dates[i : i + window_size]

        pmin, pmax = window.values.min(), window.values.max()
        if pmax - pmin == 0:
            continue

        normalized_window = (window - pmin) / (pmax - pmin)
        vol = np.log1p(volume_window.values.astype(np.float64))
        vmin, vmax = vol.min(), vol.max()
        normalized_vol = (vol - vmin) / (vmax - vmin) if vmax > vmin else np.zeros(window_size)

        result = np.stack([
            normalized_window["Open"].values,
            normalized_window["High"].values,
            normalized_window["Low"].values,
            normalized_window["Close"].values,
            normalized_vol,
        ])

        all_windows.append(result)

        start_date = window_dates[0]
        end_date = window_dates[-1]
        start_date_str = str(start_date)
        end_date_str = str(end_date)

        all_metadata.append({
            "ticker": ticker,
            "start_date": start_date_str,
            "end_date": end_date_str
        })

    if not all_windows:
        raise ValueError(f"No valid windows found for {ticker}")

    folder_name = f"{ticker}_{interval}_minmax_{window_size}"
    preprocessed_dir = output_dir / folder_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    data_path = preprocessed_dir / "data.npy"
    meta_path = preprocessed_dir / "metadata.csv"

    np.save(data_path, np.array(all_windows, dtype=np.float32))
    pd.DataFrame(all_metadata).to_csv(meta_path, index=False)

    logger.info(f"Preprocessed {len(all_windows)} windows to {preprocessed_dir}")
    return preprocessed_dir, data_path


def train_ticker_model(
    preprocessed_dir: Path,
    ticker: str,
    interval: str,
    window_size: int,
    args: argparse.Namespace,
) -> Path:
    logger.info(f"Training autoencoder for {ticker} @ {interval}...")

    config = Config(
        data_dir=preprocessed_dir,
        interval=interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        device=args.device,
        window_size=window_size
    )

    trainer = Trainer(config)
    trainer.run_training()

    project_root = Path(__file__).resolve().parent
    model_path = (
        project_root / "models"
        / f"{ticker}_{interval}_w{window_size}_e{args.embedding_dim}.pt"
    )

    logger.info(f"Training complete: {model_path}")
    return model_path


def build_ticker_index(
    preprocessed_dir: Path,
    model_path: Path,
    ticker: str,
    interval: str,
    window_size: int,
    args: argparse.Namespace,
) -> Path:
    logger.info(f"Building FAISS index for {ticker} @ {interval}...")

    config = Config(
        data_dir=preprocessed_dir,
        interval=interval,
        embedding_dim=args.embedding_dim,
        window_size=window_size,
        device=args.device
    )

    builder = IndexBuilder(config)
    builder.build_index(model_path=model_path)

    project_root = Path(__file__).resolve().parent
    index_path = (
        project_root / "models"
        / f"{ticker}_{interval}_w{window_size}.faiss"
    )

    logger.info(f"Index built: {index_path}")
    return index_path


def sync_ticker_to_db(csv_path: Path, ticker: str, interval: str) -> None:
    logger.info(f"Syncing {ticker} @ {interval} to database...")

    try:
        engine = get_engine()
        Base.metadata.create_all(engine)
        sync_csv(csv_path, ticker, interval)
    except Exception as e:
        logger.warning(f"Database sync failed: {e}")


def add_ticker_command(args: argparse.Namespace) -> None:
    ticker = args.ticker.upper()
    window_size = args.window_size

    if args.intervals:
        intervals = [i.strip() for i in args.intervals.split(",")]
    else:
        intervals = list(INTERVALS.keys())

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    preprocessed_base = project_root / "data" / "preprocessed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_base.mkdir(parents=True, exist_ok=True)

    succeeded = []
    failed = []

    for interval in intervals:
        if interval not in INTERVALS:
            logger.warning(f"Unknown interval '{interval}', skipping")
            failed.append(interval)
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {ticker} @ {interval}")
        logger.info(f"{'=' * 60}")

        try:
            csv_path = download_ticker_data(ticker, interval, raw_dir)

            processed_dir, data_path = preprocess_ticker_data(
                csv_path, ticker, interval, window_size, preprocessed_base
            )

            model_path = train_ticker_model(
                processed_dir, ticker, interval, window_size, args
            )

            build_ticker_index(
                processed_dir, model_path, ticker, interval, window_size, args
            )

            sync_ticker_to_db(csv_path, ticker, interval)

            succeeded.append(interval)
            logger.info(f"{ticker} @ {interval} complete!")

        except Exception as e:
            logger.error(f"Failed for {ticker} @ {interval}: {e}")
            failed.append(interval)
            continue

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{ticker} - Summary")
    logger.info(f"{'=' * 60}")
    if succeeded:
        logger.info(f"  Succeeded: {', '.join(succeeded)}")
    if failed:
        logger.info(f"  Failed: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Market Pattern Matcher CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_ticker_parser = subparsers.add_parser(
        "add-ticker",
        help="Add a new ticker (download, preprocess, train, index, sync)"
    )
    add_ticker_parser.add_argument("ticker", type=str, help="Ticker symbol (e.g., SPY, AAPL)")
    add_ticker_parser.add_argument(
        "--intervals", type=str, default=None,
        help=f"Comma-separated intervals to process (default: all = {','.join(INTERVALS.keys())})"
    )
    add_ticker_parser.add_argument("--window-size", type=int, default=30, help="Window size")
    add_ticker_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    add_ticker_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    add_ticker_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    add_ticker_parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dim")
    add_ticker_parser.add_argument("--device", type=str, default="auto", help="Device")
    add_ticker_parser.set_defaults(func=add_ticker_command)

    train_parser = subparsers.add_parser("train", help="Train autoencoder")
    train_parser.add_argument("--data", type=Path, required=True, help="Preprocessed data dir")
    train_parser.add_argument("--interval", type=str, default="1d", help="Interval")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--embedding-dim", type=int, default=32)
    train_parser.add_argument("--window-size", type=int, default=30)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.set_defaults(func=train_command)

    index_parser = subparsers.add_parser("index", help="Build FAISS index")
    index_parser.add_argument("--data", type=Path, required=True, help="Preprocessed data dir")
    index_parser.add_argument("--model", type=Path, help="Path to model checkpoint")
    index_parser.add_argument("--interval", type=str, default="1d", help="Interval")
    index_parser.add_argument("--embedding-dim", type=int, default=32)
    index_parser.add_argument("--window-size", type=int, default=30)
    index_parser.add_argument("--device", type=str, default="auto")
    index_parser.set_defaults(func=index_command)

    search_parser = subparsers.add_parser("search", help="Search for patterns")
    search_parser.add_argument("--index", type=Path, help="Path to .faiss index file")
    search_parser.add_argument("--data", type=Path, help="Path to data dir")
    search_parser.add_argument("--query-index", type=int, help="Window index to query")
    search_parser.add_argument("--date", type=str, help="Date to query (YYYY-MM-DD)")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--include-self", action="store_true")
    search_parser.add_argument("--device", type=str, default="auto")
    search_parser.set_defaults(func=search_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
