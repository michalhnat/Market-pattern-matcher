from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from src.api.crud import search_patterns
from src.api.schemas import SearchResponse

router = APIRouter(
    prefix="/search",
    tags=["search"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=SearchResponse)
async def search(
    start_date: datetime = Query(..., description="Start date for pattern search"),  # noqa: B008
    ticker: str = Query("SPY", description="Ticker symbol"),  # noqa: B008
    interval: str = Query("1d", description="Timeframe interval"),  # noqa: B008
    window_size: int = Query(30, description="Window size"),  # noqa: B008
    top_k: int = Query(5, description="Number of results"),  # noqa: B008
) -> SearchResponse:
    try:
        response = search_patterns(
            ticker=ticker,
            interval=interval,
            window_size=window_size,
            date=start_date.date(),
            top_k=top_k,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Search failed: {e!s}"
        ) from e
