import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import datetime

import pandas as pd

from config import Config
from src.api.catalog import IndexCatalog
from src.core.search import MatchResult, PatternSearcher


def find_query_index(start_date: datetime.date, metadata_df: pd.DataFrame) -> int:
    target_date = pd.to_datetime(start_date)
    dates = pd.to_datetime(metadata_df["start_date"])
    return int((dates - target_date).abs().idxmin())


def search_patterns(
    ticker: str, window_size: int, date: datetime.date, top_k: int = 5
) -> list[MatchResult]:
    config = Config()
    catalog = IndexCatalog()
    metadata = catalog.get(ticker=ticker, window_size=window_size)
    searcher = PatternSearcher(config=config)
    searcher.load_resources(index_path=metadata.index_path)

    query_idx = find_query_index(date, searcher.metadata_df)

    return searcher.search(query_index=query_idx, top_k=top_k, include_self=False)


def list_indexes() -> dict[str, list[int]]:
    catalog = IndexCatalog()
    return catalog.list_indexes()
