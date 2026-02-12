from pydantic import BaseModel, Field


class MatchResult(BaseModel):
    rank: int = Field(..., description="Rank of the match")
    distance: float = Field(..., description="Distance score of the match")
    window_index: int = Field(..., description="Index of the matched window in the dataset")
    start_date: str = Field(..., description="Start date of the matched window")
    end_date: str = Field(..., description="End date of the matched window")
    ticker: str = Field(..., description="Ticker symbol associated with the matched window")
    metadata_path: str = Field(..., description="Path to the metadata CSV for the matched window")
    raw_data_path: str = Field(..., description="Path to the raw data for the matched window")


class IndexMetadata(BaseModel):
    ticker: str = Field(..., description="Ticker symbol for the index")
    window_size: int = Field(..., description="Window size used for the index")
    embedding_dim: int = Field(..., description="Dimension of the embeddings in the index")
    num_windows: int = Field(..., description="Number of windows indexed")
