import json
import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from src.core.dataset import MarketDataset
from src.core.model import MarketAutoencoder

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.device = self._resolve_device()

    def _resolve_device(self):
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def build_index(self, model_path: Path = None):
        ckpt_path = model_path or self.config.model_path
        data_path = self.config.data_dir / "data.npy"
        meta_csv_path = self.config.data_dir / "metadata.csv"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")

        logger.info(f"Loading model from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        saved_config = checkpoint.get("config", {})
        window_size = saved_config.get("window_size", self.config.window_size)
        embedding_dim = saved_config.get("embedding_dim", self.config.embedding_dim)
        in_channels = saved_config.get("in_channels", self.config.in_channels)

        model = MarketAutoencoder(input_len=window_size, embedding_dim=embedding_dim).to(
            self.device
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info("Loading dataset...")
        dataset = MarketDataset(data_path, meta_csv_path)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )

        logger.info("Extracting embeddings...")
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z, _ = model(batch)
                embeddings.append(z.cpu().numpy())

        embeddings_matrix = np.concatenate(embeddings, axis=0).astype(np.float32)
        num_vectors, dim = embeddings_matrix.shape

        if dim != embedding_dim:
            logger.warning(f"Extracted embedding dim {dim} != config {embedding_dim}")
        logger.info(f"Building FAISS index for {num_vectors} vectors of dim {dim}...")
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_matrix)

        index_path = self.config.index_path
        index_meta_path = self.config.index_meta_path
        index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving index to {index_path}...")
        faiss.write_index(index, str(index_path))

        meta_info = {
            "ticker": self.config.ticker,
            "window_size": window_size,
            "embedding_dim": embedding_dim,
            "num_vectors": num_vectors,
            "model_checkpoint": str(ckpt_path),
            "data_path": str(data_path),
            "metadata_csv_path": str(meta_csv_path),
        }

        with open(index_meta_path, "w") as f:
            json.dump(meta_info, f, indent=2)

        logger.info(f"Saved index metadata to {index_meta_path}")
        return index_path
