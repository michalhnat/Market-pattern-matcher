from fastapi import APIRouter, HTTPException

from src.api.crud import list_indexes
from src.api.schemas import IndexMetadata

router = APIRouter(
    prefix="/indexes",
    tags=["indexes"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=list[IndexMetadata])
async def get_indexes() -> list[IndexMetadata]:
    try:
        return list_indexes()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list indexes: {e!s}"
        ) from e
