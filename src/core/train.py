import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from src.core.model import MarketAutoencoder
from src.core.dataset import MarketDataset


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.best_val_loss = float("inf")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _resolve_device(self):
        if self.config.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _build_model(self):
        device = self._resolve_device()
        self.model = MarketAutoencoder(
            self.config.window_size, self.config.embedding_dim
        ).to(device)

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def _build_loss(self):
        self.criterion = nn.MSELoss()

    def _build_dataloaders(self):
        data_path = self.config.data_dir / "data.npy"
        dataset = MarketDataset(str(data_path))

        train_size = int(self.config.train_split * len(dataset))
        val_size = int(self.config.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False
        )

    def _train_one_epoch(self):
        device = self._resolve_device()
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            batch = batch.to(device)
            self.optimizer.zero_grad()
            _z, out = self.model(batch)
            loss = self.criterion(out, batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self):
        device = self._resolve_device()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                _z, out = self.model(batch)
                loss = self.criterion(out, batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def _test_one_epoch(self):
        device = self._resolve_device()
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(device)
                _z, out = self.model(batch)
                loss = self.criterion(out, batch)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)

    def save(self, path=None):
        save_path = path or self.config.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "window_size": self.config.window_size,
                "embedding_dim": self.config.embedding_dim,
                "in_channels": self.config.in_channels,
            },
            "best_val_loss": self.best_val_loss,
        }, save_path)
        print(f"Model saved to {save_path}")

    def load(self, path=None):
        load_path = path or self.config.model_path
        checkpoint = torch.load(load_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def run_training(self):
        self._build_model()
        self._build_optimizer()
        self._build_loss()
        self._build_dataloaders()

        self.best_val_loss = float("inf")
        patience_counter = 0

        print(f"Training on {self._resolve_device()} for {self.config.epochs} epochs")
        print(f"Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}, "
              f"Test: {len(self.test_loader.dataset)}")
        print("-" * 60)

        for epoch in range(self.config.epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save()
            else:
                patience_counter += 1

            marker = " *" if is_best else ""
            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}{marker}")

            if patience_counter >= self.config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (patience={self.config.early_stop_patience})")
                break

        # Final test
        self.load()
        test_loss = self._test_one_epoch()
        print("-" * 60)
        print(f"Test loss: {test_loss:.6f}")

    def run_validation(self):
        self._build_model()
        self._build_loss()
        self._build_dataloaders()
        self.load()
        val_loss = self._validate_one_epoch()
        print(f"Validation loss: {val_loss:.6f}")

    def run_test(self):
        self._build_model()
        self._build_loss()
        self._build_dataloaders()
        self.load()
        test_loss = self._test_one_epoch()
        print(f"Test loss: {test_loss:.6f}")