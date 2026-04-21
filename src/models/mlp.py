"""
src/models/mlp.py
─────────────────
PyTorch MLP for binary fraud classification.

Architecture:
    Input(31) → [Linear → BatchNorm → GELU → Dropout] × 3 → Linear(1) → Sigmoid

Design decisions:
  - GELU instead of ReLU: smoother gradient flow, marginally better on tabular data
  - BatchNorm before activation: stabilises training on the extreme class imbalance
  - Dropout(0.3): regularisation — the fraud class is tiny so overfitting is easy
  - pos_weight in BCEWithLogitsLoss: equivalent of scale_pos_weight in XGBoost
  - Output is a raw logit; sigmoid is applied only at inference via predict_proba()
    so that Stage 3 temperature scaling can operate on the logit directly.

The wrapper class mirrors the classical model interface exactly:
    .fit(X, y, X_val, y_val)
    .predict_proba(X) → np.ndarray of shape (n,)
    .save(path) / .load(path)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# ─── Network definition ───────────────────────────────────────────────────────

class _MLPNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))   # raw logit output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)       # shape: (batch,)


# ─── Wrapper ──────────────────────────────────────────────────────────────────

class MLPModel:

    name = "mlp"

    def __init__(
        self,
        input_dim: int = 31,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 100,
        patience: int = 20,
        random_state: int = 42,
    ):
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout     = dropout
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.patience    = patience
        self.random_state = random_state

        self.net_       = None
        self.device_    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_epoch_ = 0

    # ── internal helpers ──────────────────────────────────────────

    def _to_tensor(self, X) -> torch.Tensor:
        if hasattr(X, "values"):   # DataFrame → numpy
            X = X.values
        return torch.tensor(X, dtype=torch.float32, device=self.device_)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(self._to_tensor(X), self._to_tensor(y.astype(np.float32)))
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ── public interface ──────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "MLPModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.net_ = _MLPNet(self.input_dim, self.hidden_dims, self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )

        # pos_weight: upweights fraud gradient during training
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=self.device_)
        criterion      = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Unweighted criterion for val early stopping — val has only ~56 fraud
        # cases so pos_weight amplifies noise and causes premature stopping
        val_criterion  = nn.BCEWithLogitsLoss()

        train_loader = self._make_loader(X, y, shuffle=True)
        use_val = (X_val is not None) and (y_val is not None)

        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        for epoch in range(1, self.epochs + 1):
            # ── train ──
            self.net_.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self.net_(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X)
            scheduler.step()

            # ── val + early stopping ──
            if use_val:
                val_loss = self._eval_loss(X_val, y_val, val_criterion)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss  = val_loss
                    patience_count = 0
                    best_state     = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
                    self.best_epoch_ = epoch
                else:
                    patience_count += 1
                if patience_count >= self.patience:
                    print(f"    [{self.name}] Early stop at epoch {epoch}  "
                          f"(best epoch {self.best_epoch_}, val_loss={best_val_loss:.5f})")
                    break

            if epoch % 10 == 0:
                msg = f"    [{self.name}] Epoch {epoch:3d}/{self.epochs}  train_loss={train_loss:.5f}"
                if use_val:
                    msg += f"  val_loss={val_loss:.5f}"
                print(msg)

        # restore best weights
        if best_state is not None:
            self.net_.load_state_dict(
                {k: v.to(self.device_) for k, v in best_state.items()}
            )

        return self

    def _eval_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion: nn.Module,
    ) -> float:
        self.net_.eval()
        loader = self._make_loader(X, y, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.net_(X_batch)
                total_loss += criterion(logits, y_batch).item() * len(X_batch)
        return total_loss / len(X)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Returns raw logits (pre-sigmoid). Used by temperature scaling in Stage 3."""
        self.net_.eval()
        loader = DataLoader(
            TensorDataset(self._to_tensor(X)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        logits = []
        with torch.no_grad():
            for (X_batch,) in loader:
                logits.append(self.net_(X_batch).cpu().numpy())
        return np.concatenate(logits)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns fraud probability in [0, 1] via sigmoid(logit)."""
        logits = self.predict_logits(X)
        return torch.sigmoid(torch.tensor(logits)).numpy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "init_params": {
                "input_dim":   self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout":     self.dropout,
            },
            "state_dict":  self.net_.state_dict(),
            "best_epoch":  self.best_epoch_,
        }, path)
        print(f"  [{self.name}] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        ckpt   = torch.load(path, map_location="cpu")
        params = ckpt["init_params"]
        obj    = cls(**params)
        obj.net_ = _MLPNet(
            params["input_dim"],
            params["hidden_dims"],
            params["dropout"],
        )
        obj.net_.load_state_dict(ckpt["state_dict"])
        obj.net_.eval()
        obj.best_epoch_ = ckpt.get("best_epoch", 0)
        return obj