from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # data
    window_size: int = 30
    in_channels: int = 5
    data_dir: Path = None

    # model
    embedding_dim: int = 32

    # training
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    early_stop_patience: int = 7

    # search
    top_k: int = 5

    device: str = "auto"

    @property
    def ticker(self) -> str:
        if self.data_dir is None:
            return "unknown"
        return self.data_dir.name.split("_")[0]

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def faiss_dir(self) -> Path:
        return PROJECT_ROOT / "faiss"

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.ticker}_w{self.window_size}_e{self.embedding_dim}.pt"

    @property
    def index_path(self) -> Path:
        return self.faiss_dir / f"{self.ticker}_w{self.window_size}.faiss"

    @property
    def index_meta_path(self) -> Path:
        return self.faiss_dir / f"{self.ticker}_w{self.window_size}.index_meta.json"
