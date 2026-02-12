from pydantic import BaseModel, Field


class WindowDates(BaseModel):
    start_date: str = Field(..., description="Start date of the window")
    end_date: str = Field(..., description="End date of the window")


class MatchResult(BaseModel):
    rank: int = Field(..., description="Rank of the match")
    distance: float = Field(..., description="Distance score of the match")
    ticker: str = Field(..., description="Ticker symbol")
    pattern_dates: WindowDates = Field(
        ..., description="Dates of the matched pattern window"
    )
    next_dates: WindowDates = Field(
        ..., description="Dates of the window after the match (what happened next)"
    )


class SearchResponse(BaseModel):
    query_dates: WindowDates = Field(..., description="Dates of the query window")
    next_dates: WindowDates = Field(..., description="Dates of the window after the query")
    raw_data_path: str = Field(..., description="Path to the raw data CSV")
    matches: list[MatchResult] = Field(..., description="List of similar patterns")


class IndexMetadata(BaseModel):
    ticker: str = Field(..., description="Ticker symbol for the index")
    window_size: int = Field(..., description="Window size used for the index")
    embedding_dim: int = Field(..., description="Dimension of the embeddings in the index")
    num_windows: int = Field(..., description="Number of windows indexed")
