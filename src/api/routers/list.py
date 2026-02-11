from fastapi import APIRouter

from src.api.crud import list_indexes
from src.api.schemas import IndexMetadata

router = APIRouter(
    prefix="/list",
    tags=["list"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=list[IndexMetadata])
async def search() -> list[IndexMetadata]:
    try:
        results = list_indexes()
        return results
    except Exception as e:
        raise ValueError(f"Search failed: {e}") from e
