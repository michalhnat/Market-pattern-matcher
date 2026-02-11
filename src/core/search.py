import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import faiss
from config import Config
from src.core.model import MarketAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    rank: int
    distance: float
    window_index: int
    start_date: str
    end_date: str
    ticker: str


class PatternSearcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = self._resolve_device()
        self.index = None
        self.meta_info = None
        self.metadata_df = None
        self.model = None
        self.data_memmap = None

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def load_resources(self, index_path: Path | None = None) -> None:
        idx_path = index_path or self.config.index_path
        meta_path = idx_path.with_suffix(".index_meta.json")

        if not idx_path.exists():
            raise FileNotFoundError(f"Index not found at {idx_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Index metadata not found at {meta_path}")

        with open(meta_path) as f:
            self.meta_info = json.load(f)

        self.index = faiss.read_index(str(idx_path))
        self.metadata_df = pd.read_csv(self.meta_info["metadata_csv_path"])
        self.data_memmap = np.load(self.meta_info["data_path"], mmap_mode="r")

        ckpt_path = self.meta_info["model_checkpoint"]
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model = MarketAutoencoder(
            input_len=self.meta_info["window_size"], embedding_dim=self.meta_info["embedding_dim"]
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def search(
        self,
        query_index: int | None = None,
        query_date: str | None = None,
        top_k: int = 5,
        include_self: bool = False,
    ) -> list[MatchResult]:

        if self.index is None:
            self.load_resources()

        query_vector = None
        target_idx = -1

        if query_index is not None:
            target_idx = query_index
            raw_window = self.data_memmap[target_idx]
            query_vector = self._encode(raw_window)
        elif query_date is not None:
            try:
                target_date = pd.to_datetime(query_date)
                dates = pd.to_datetime(self.metadata_df["start_date"])
                diffs = (dates - target_date).abs()
                target_idx = diffs.idxmin()

                found_date = dates[target_idx]
                if diffs[target_idx].days > 7:
                    logger.warning(
                        f"Requested date {query_date} not found. "
                        f"Closest is {found_date.date()} "
                        f"({diffs[target_idx].days} days away)"
                    )

                raw_window = self.data_memmap[target_idx]
                query_vector = self._encode(raw_window)

            except Exception as e:
                raise ValueError(f"Error resolving date {query_date}: {e}") from e
        else:
            raise ValueError("Must provide either query_index or query_date")

        search_k = top_k + 1 if not include_self else top_k
        distances, indices = self.index.search(query_vector, search_k)

        results = []
        rank = 1

        for i, idx in enumerate(indices[0]):
            dist = distances[0][i]

            if not include_self and idx == target_idx:
                continue

            if rank > top_k:
                break

            meta_row = self.metadata_df.iloc[idx]

            results.append(
                MatchResult(
                    rank=rank,
                    distance=float(dist),
                    window_index=int(idx),
                    start_date=str(meta_row["start_date"]),
                    end_date=str(meta_row["end_date"]),
                    ticker=str(meta_row["ticker"]),
                )
            )
            rank += 1

        return results

    def _encode(self, window_np: np.ndarray) -> np.ndarray:
        window_np = np.array(window_np)  # Copy to make writable
        window_tensor = torch.from_numpy(window_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z, _ = self.model(window_tensor)
        return z.cpu().numpy().astype(np.float32)
