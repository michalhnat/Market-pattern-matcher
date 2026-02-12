import json
from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT


@dataclass
class IndexMetadata:
    ticker: str
    interval: str
    window_size: int
    embedding_dim: int
    num_vectors: int
    model_checkpoint: str
    data_path: str
    metadata_csv_path: str
    index_path: Path


class IndexCatalog:
    def __init__(self, faiss_dir: Path | None = None) -> None:
        self.faiss_dir = faiss_dir or (PROJECT_ROOT / "faiss")
        self._catalog: dict[str, dict[str, dict[int, IndexMetadata]]] = (
            self._discover_indexes()
        )

    def _discover_indexes(
        self,
    ) -> dict[str, dict[str, dict[int, IndexMetadata]]]:
        catalog: dict[str, dict[str, dict[int, IndexMetadata]]] = {}

        for meta_file in self.faiss_dir.glob("*.index_meta.json"):
            with open(meta_file) as f:
                data = json.load(f)

            index_path = meta_file.with_name(
                meta_file.stem.replace(".index_meta", "") + ".faiss"
            )
            if not index_path.exists():
                continue

            metadata = IndexMetadata(
                ticker=data["ticker"],
                interval=data.get("interval", "1d"),
                window_size=data["window_size"],
                embedding_dim=data["embedding_dim"],
                num_vectors=data["num_vectors"],
                model_checkpoint=data["model_checkpoint"],
                data_path=data["data_path"],
                metadata_csv_path=data["metadata_csv_path"],
                index_path=index_path,
            )

            catalog.setdefault(metadata.ticker, {})
            catalog[metadata.ticker].setdefault(metadata.interval, {})
            catalog[metadata.ticker][metadata.interval][
                metadata.window_size
            ] = metadata

        return catalog

    def get(
        self, ticker: str, interval: str = "1d", window_size: int = 30
    ) -> IndexMetadata:
        if ticker not in self._catalog:
            raise ValueError(f"No index for ticker: {ticker}")
        if interval not in self._catalog[ticker]:
            available = list(self._catalog[ticker].keys())
            raise ValueError(
                f"No index for {ticker} @ {interval}. "
                f"Available intervals: {available}"
            )
        if window_size not in self._catalog[ticker][interval]:
            available = list(self._catalog[ticker][interval].keys())
            raise ValueError(
                f"No index for {ticker} @ {interval} with window "
                f"{window_size}. Available: {available}"
            )
        return self._catalog[ticker][interval][window_size]

    def list_indexes(self) -> list[dict]:
        result = []
        for ticker, intervals in self._catalog.items():
            for interval, sizes in intervals.items():
                for window_size, meta in sizes.items():
                    result.append({
                        "ticker": ticker,
                        "interval": interval,
                        "window_size": window_size,
                        "embedding_dim": meta.embedding_dim,
                        "num_windows": meta.num_vectors,
                    })
        return result
