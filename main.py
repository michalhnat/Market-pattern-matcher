import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from src.core.train import Trainer
from src.core.index import IndexBuilder
from src.core.search import PatternSearcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_command(args):
    config = Config(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        device=args.device,
        window_size=args.window_size
    )
    
    logger.info(f"Starting training for {config.ticker}...")
    trainer = Trainer(config)
    trainer.run_training()


def index_command(args):
    config = Config(
        data_dir=args.data,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        device=args.device
    )
    
    logger.info(f"Building index for {config.ticker}...")
    builder = IndexBuilder(config)
    builder.build_index(model_path=args.model)


def search_command(args):
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
    
    print(f"\nSearch Results for {config.ticker} (Query: {'Index ' + str(args.query_index) if args.query_index is not None else args.date})")
    print("-" * 85)
    print(f"{'Rank':<5} | {'Distance':<10} | {'Start Date':<12} | {'End Date':<12} | {'Ticker':<8} | {'Window Idx':<10}")
    print("-" * 85)
    
    for r in results:
        print(f"{r.rank:<5} | {r.distance:<10.4f} | {r.start_date:<12} | {r.end_date:<12} | {r.ticker:<8} | {r.window_index:<10}")
    print("-" * 85)


def main():
    parser = argparse.ArgumentParser(description="Market Pattern Matcher CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train autoencoder")
    train_parser.add_argument("--data", type=Path, required=True, help="Path to preprocessed data dir")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--embedding-dim", type=int, default=32)
    train_parser.add_argument("--window-size", type=int, default=30)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.set_defaults(func=train_command)

    index_parser = subparsers.add_parser("index", help="Build FAISS index")
    index_parser.add_argument("--data", type=Path, required=True, help="Path to preprocessed data dir")
    index_parser.add_argument("--model", type=Path, help="Path to model checkpoint")
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
