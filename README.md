# Market Pattern Matcher

This project allows you to choose given time window and search for similar periods that are available in history for given ticker.

Search engine uses an autoencoder for feature extraction and FAISS for vector search. The project also includes an interactive frontend built with Streamlit.

## Features

- Autoendcoder: custom autoencoder is used to create embeddings.
- Vector Search: FAISS index for vector search.
- API: FastAPI backend for handling search requests.
- TimescaleDB: TimescaleDB for storing historical data.
- Containerization: Dockerized application for easy deployment.
- Interactive UI: Streamlit frontend for user interaction.

## Usage

### Environment Variables
Create a `.env` file in the root directory of the project and add the environment variables as shown in the `.env.example` file.

### Docker
1. Clone the repository
2. Build and run using docker-compose:
```bash
docker-compose up --build
```
5. Access fronted on: http://localhost:8501

### Development
1. Sync dependencies:
```bash
uv sync
```

2. Run API:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

3. Run UI:
```bash
uv run streamlit run ui/app.py --server.port 8501
```
## UI
<img width="1512" height="857" alt="image" src="https://github.com/user-attachments/assets/d04647ca-b153-4e33-bb59-178d1e6ef6ab" />

## TODO
   - Better timeframes managment - currently data is gathered using yfinance, therfore for smaller timeframe intervals - shorted periods are avalivable
   - Precise time choosing for smaller intervals
   - CI
