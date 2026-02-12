from contextlib import asynccontextmanager

from fastapi import FastAPI

from scripts.db.db import get_engine
from scripts.db.init_db import Base
from src.api.routers.list import router as list_router
from src.api.routers.search import router as search_router
from src.api.routers.ticker import router as ticker_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Market Pattern Matcher API", version="0.2.0", lifespan=lifespan)

app.include_router(search_router)
app.include_router(list_router)
app.include_router(ticker_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
