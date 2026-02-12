from fastapi import FastAPI

from src.api.routers.search import router as search_router

app = FastAPI(title="Market Pattern Matcher API", version="0.1.0")

app.include_router(search_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
