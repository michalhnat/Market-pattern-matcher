import torch
import torch.nn as nn


class MarketAutoencoder(nn.Module):
    def __init__(self, input_len: int = 30, embedding_dim: int = 32) -> None:
        super().__init__()

        self.input_len = input_len
        downsampled = input_len // 2 // 2
        flattened_size = 64 * downsampled

        self.encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(flattened_size, embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, flattened_size),
            nn.ReLU(),
            nn.Unflatten(1, (64, downsampled)),
            nn.Upsample(size=input_len // 2, mode='nearest'),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size=input_len, mode='nearest'),
            nn.Conv1d(32, 5, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out

    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
