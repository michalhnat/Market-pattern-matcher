from pydantic import BaseModel, Field


class WindowDates(BaseModel):
    start_date: str = Field(..., description="Start date of the window")
    end_date: str = Field(..., description="End date of the window")


class MatchResult(BaseModel):
    rank: int = Field(..., description="Rank of the match")
    distance: float = Field(..., description="Distance score")
    ticker: str = Field(..., description="Ticker symbol")
    interval: str = Field(..., description="Timeframe interval")
    pattern_dates: WindowDates = Field(
        ..., description="Dates of the matched pattern window"
    )
    next_dates: WindowDates = Field(
        ..., description="Dates of the window after the match"
    )


class SearchResponse(BaseModel):
    query_dates: WindowDates = Field(..., description="Dates of the query window")
    next_dates: WindowDates = Field(
        ..., description="Dates of the window after the query"
    )
    ticker: str = Field(..., description="Ticker symbol")
    interval: str = Field(..., description="Timeframe interval")
    matches: list[MatchResult] = Field(
        ..., description="List of similar patterns"
    )


class IndexMetadata(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    interval: str = Field(..., description="Timeframe interval")
    window_size: int = Field(..., description="Window size")
    embedding_dim: int = Field(..., description="Embedding dimension")
    num_windows: int = Field(..., description="Number of windows indexed")
