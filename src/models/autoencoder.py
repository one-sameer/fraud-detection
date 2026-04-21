"""
src/models/autoencoder.py
─────────────────────────
Autoencoder-based anomaly detector.

Training strategy:
    Trained ONLY on legitimate transactions. The model learns to compress
    and reconstruct "normal" behaviour. At inference, fraud transactions
    produce a high reconstruction error because they fall outside the
    manifold of normality the encoder has learned.

This gives a complementary, unsupervised signal to the supervised models:
  - Supervised models detect known fraud patterns seen at training time.
  - The autoencoder generalises better to novel/unseen fraud patterns.

Architecture:
    Encoder: 31 → 16 → 8
    Bottleneck: 8
    Decoder: 8 → 16 → 31

The reconstruction error (MSE per sample) is normalised to [0, 1] using
the 99th percentile of legitimate training errors as the upper bound.
This avoids a single outlier inflating the normalisation range.

Interface mirrors the other models exactly:
    .fit(X_train_legit)
    .predict_proba(X) → np.ndarray of shape (n,)  ← normalised recon error
    .save(path) / .load(path)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# ─── Network ──────────────────────────────────────────────────────────────────

class _AENet(nn.Module):

    def __init__(self, input_dim: int, bottleneck: int):
        super().__init__()
        mid = (input_dim + bottleneck) // 2  # 31+8//2 = ~20, use 16

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Linear(mid, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Linear(mid, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction."""
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=1)   # shape: (batch,)


# ─── Wrapper ──────────────────────────────────────────────────────────────────

class AutoencoderModel:

    name = "autoencoder"

    def __init__(
        self,
        input_dim: int = 31,
        bottleneck: int = 8,
        lr: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 150,
        patience: int = 20,
        random_state: int = 42,
    ):
        self.input_dim    = input_dim
        self.bottleneck   = bottleneck
        self.lr           = lr
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.patience     = patience
        self.random_state = random_state

        self.net_          = None
        self.error_p99_    = None   # 99th percentile recon error on legit train set
        self.device_       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── helpers ───────────────────────────────────────────────────

    def _to_tensor(self, X) -> torch.Tensor:
        if hasattr(X, "values"):   # DataFrame → numpy
            X = X.values
        return torch.tensor(X, dtype=torch.float32, device=self.device_)

    def _raw_errors(self, X: np.ndarray) -> np.ndarray:
        """Returns unnormalised per-sample MSE reconstruction errors."""
        self.net_.eval()
        loader = DataLoader(
            TensorDataset(self._to_tensor(X)),
            batch_size=self.batch_size, shuffle=False
        )
        errors = []
        with torch.no_grad():
            for (X_batch,) in loader:
                errors.append(self.net_.reconstruction_error(X_batch).cpu().numpy())
        return np.concatenate(errors)

    # ── public interface ──────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "AutoencoderModel":
        """
        Trains exclusively on legitimate transactions in X_train
        (rows where y_train == 0). y_val is used to select val legit rows.
        """
        torch.manual_seed(self.random_state)

        # !! train only on legit !!
        X_legit = X_train[y_train == 0]
        print(f"    [{self.name}] Training on {len(X_legit):,} legitimate transactions")

        self.net_ = _AENet(self.input_dim, self.bottleneck).to(self.device_)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(self._to_tensor(X_legit)),
            batch_size=self.batch_size, shuffle=True
        )

        # val set: legit only for reconstruction loss, but full val for early stopping signal
        use_val = (X_val is not None) and (y_val is not None)
        X_val_legit = X_val[y_val == 0] if use_val else None

        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            train_loss = 0.0
            for (X_batch,) in train_loader:
                optimizer.zero_grad()
                recon = self.net_(X_batch)
                loss  = criterion(recon, X_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X_legit)
            scheduler.step()

            if use_val and X_val_legit is not None:
                self.net_.eval()
                with torch.no_grad():
                    val_recon = self.net_(self._to_tensor(X_val_legit))
                    val_loss  = criterion(val_recon, self._to_tensor(X_val_legit)).item()

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss  = val_loss
                    patience_count = 0
                    best_state     = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
                else:
                    patience_count += 1
                if patience_count >= self.patience:
                    print(f"    [{self.name}] Early stop at epoch {epoch}  "
                          f"(val_loss={best_val_loss:.6f})")
                    break

            if epoch % 10 == 0:
                msg = f"    [{self.name}] Epoch {epoch:3d}/{self.epochs}  train_loss={train_loss:.6f}"
                if use_val:
                    msg += f"  val_loss={val_loss:.6f}"
                print(msg)

        if best_state is not None:
            self.net_.load_state_dict(
                {k: v.to(self.device_) for k, v in best_state.items()}
            )

        # Calibrate normalisation using the 99th percentile of legit training errors.
        # We use p99 rather than max so that one outlier doesn't compress the whole range.
        train_errors = self._raw_errors(X_legit)
        self.error_p99_ = float(np.percentile(train_errors, 99))
        print(f"    [{self.name}] Recon error p99 (legit train) = {self.error_p99_:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns normalised reconstruction error in [0, 1] as fraud probability.

        Errors above the p99 threshold are clipped to 1.0. This is a calibrated
        anomaly score — it will be re-calibrated in Stage 3 like the other models.
        """
        errors = self._raw_errors(X)
        normalised = np.clip(errors / (self.error_p99_ + 1e-8), 0.0, 1.0)
        return normalised.astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "init_params": {
                "input_dim":  self.input_dim,
                "bottleneck": self.bottleneck,
            },
            "state_dict":  self.net_.state_dict(),
            "error_p99":   self.error_p99_,
        }, path)
        print(f"  [{self.name}] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "AutoencoderModel":
        ckpt   = torch.load(path, map_location="cpu")
        params = ckpt["init_params"]
        obj    = cls(**params)
        obj.net_ = _AENet(params["input_dim"], params["bottleneck"])
        obj.net_.load_state_dict(ckpt["state_dict"])
        obj.net_.eval()
        obj.error_p99_ = ckpt["error_p99"]
        return obj