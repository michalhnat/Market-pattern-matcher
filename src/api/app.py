from fastapi import FastAPI

from src.api.routers.list import router as list_router
from src.api.routers.search import router as search_router
from src.api.routers.ticker import router as ticker_router

app = FastAPI(title="Market Pattern Matcher API", version="0.2.0")

app.include_router(search_router)
app.include_router(list_router)
app.include_router(ticker_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
