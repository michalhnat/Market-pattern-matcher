import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import datetime

import pandas as pd

from config import Config
from src.api.catalog import IndexCatalog
from src.api.schemas import MatchResult, SearchResponse, WindowDates
from src.core.search import PatternSearcher


def find_query_index(start_date: datetime.date, metadata_df: pd.DataFrame) -> int:
    target_date = pd.to_datetime(start_date)
    dates = pd.to_datetime(metadata_df["start_date"])
    return int((dates - target_date).abs().idxmin())


def search_patterns(
    ticker: str, window_size: int, date: datetime.date, top_k: int = 5
) -> SearchResponse:
    config = Config()
    catalog = IndexCatalog()
    metadata = catalog.get(ticker=ticker, window_size=window_size)
    searcher = PatternSearcher(config=config)
    searcher.load_resources(index_path=metadata.index_path)

    query_idx = find_query_index(date, searcher.metadata_df)

    # Get query window dates
    query_row = searcher.metadata_df.iloc[query_idx]
    query_dates = WindowDates(
        start_date=str(query_row["start_date"]),
        end_date=str(query_row["end_date"])
    )

    # Get next window dates (what happens after the query)
    next_idx = query_idx + 1
    if next_idx < len(searcher.metadata_df):
        next_row = searcher.metadata_df.iloc[next_idx]
        next_dates = WindowDates(
            start_date=str(next_row["start_date"]),
            end_date=str(next_row["end_date"])
        )
    else:
        next_dates = WindowDates(
            start_date=str(query_row["end_date"]),
            end_date=str(query_row["end_date"])
        )

    # Get similar patterns
    core_results = searcher.search(query_index=query_idx, top_k=top_k, include_self=False)

    # Convert to API MatchResult with ticker
    matches = [
        MatchResult(
            rank=r.rank,
            distance=r.distance,
            ticker=ticker,
            pattern_dates=WindowDates(
                start_date=r.pattern_dates.start_date,
                end_date=r.pattern_dates.end_date
            ),
            next_dates=WindowDates(
                start_date=r.next_dates.start_date,
                end_date=r.next_dates.end_date
            )
        )
        for r in core_results
    ]

    return SearchResponse(
        query_dates=query_dates,
        next_dates=next_dates,
        raw_data_path=f"data/raw/{ticker}.csv",
        matches=matches
    )


def list_indexes() -> dict[str, list[int]]:
    catalog = IndexCatalog()
    return catalog.list_indexes()
