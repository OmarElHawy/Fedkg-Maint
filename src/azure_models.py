"""
Stage 8: neural model + metrics for the Azure binary failure-within-24h task.

Kept deliberately small and self-contained. Reuses nothing that would tie it to
the C-MAPSS pipeline beyond generic helpers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class AzureMLP(nn.Module):
    """input_dim -> 64 -> 64 -> 1 (logit), ReLU + dropout."""

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_criterion(y_train: np.ndarray) -> nn.BCEWithLogitsLoss:
    """BCEWithLogitsLoss with pos_weight = neg/pos from the given training labels."""
    y = np.asarray(y_train)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    w = float(neg) / float(pos) if pos > 0 else 1.0
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], dtype=torch.float32))


# --------------------------------------------------------------------------- #
# Manual batching (avoids DataLoader base_seed draws on the global RNG)
# --------------------------------------------------------------------------- #

def manual_batches(n: int, batch_size: int,
                   rng: Optional[np.random.Generator] = None,
                   shuffle: bool = True) -> List[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        rng.shuffle(idx)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def pr_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, proba))


def prevalence(y: np.ndarray) -> float:
    y = np.asarray(y)
    return float(y.mean()) if len(y) else float("nan")


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold maximizing F1 on the given (validation) set."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.5
    cand = np.unique(np.round(proba, 4))
    if len(cand) > 400:
        cand = np.quantile(proba, np.linspace(0.50, 0.999, 400))
    best_t, best_f1 = 0.5, -1.0
    for t in cand:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def evaluate_binary(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": pr_auc(y_true, proba),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "prevalence": prevalence(y_true),
        "n_samples": int(len(y_true)),
        "n_positive_samples": int(y_true.sum()),
    }


# --------------------------------------------------------------------------- #
# Training (local-only / centralized)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device,
                  batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_local_mlp(input_dim: int, Xtr, ytr, Xval, yval,
                    epochs: int = 40, batch_size: int = 1024, lr: float = 1e-3,
                    dropout: float = 0.2, hidden: int = 64, patience: int = 6,
                    seed: int = 0, device: Optional[torch.device] = None,
                    verbose: bool = False) -> Tuple[AzureMLP, Dict]:
    """
    Train an AzureMLP on (Xtr, ytr), early-stop on validation PR-AUC.
    Selection uses only Xval/yval (never the held-out set).
    Returns the model holding the best-validation state and a history dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AzureMLP(input_dim, hidden=hidden, dropout=dropout).to(device)
    criterion = make_criterion(ytr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float32)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_auc, best_epoch, wait = -np.inf, -1, 0
    history = {"epoch": [], "train_loss": [], "val_pr_auc": []}

    for ep in range(epochs):
        model.train()
        losses = []
        for bidx in manual_batches(len(Xtr), batch_size, rng=rng, shuffle=True):
            xb = torch.as_tensor(Xtr[bidx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(ytr[bidx], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        vproba = predict_proba(model, Xval, device)
        auc = pr_auc(yval, vproba)
        history["epoch"].append(ep)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_pr_auc"].append(auc)
        if verbose:
            print(f"    epoch {ep:02d} loss={np.mean(losses):.4f} val_pr_auc={auc:.4f}")
        if not np.isnan(auc) and auc > best_auc + 1e-6:
            best_auc, best_epoch, wait = auc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    history["best_val_pr_auc"] = best_auc
    history["best_epoch"] = best_epoch
    return model, history


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test():
    rng = np.random.default_rng(0)
    n, d = 4000, 26
    X = rng.normal(size=(n, d)).astype(np.float32)
    z = 1.4 * X[:, 0] - 1.1 * X[:, 3] + 0.5 * X[:, 10]
    y = (rng.random(n) < 1 / (1 + np.exp(-(z - 3.0)))).astype(np.int64)
    assert 0.01 < y.mean() < 0.5
    Xtr, ytr = X[:3000], y[:3000]
    Xva, yva = X[3000:], y[3000:]
    dev = torch.device("cpu")
    model, hist = train_local_mlp(d, Xtr, ytr, Xva, yva, epochs=25, patience=5,
                                  seed=0, device=dev)
    p = predict_proba(model, Xva, dev)
    t = pick_threshold(yva, p)
    m = evaluate_binary(yva, p, t)
    print("self-test metrics:", {k: round(v, 4) for k, v in m.items()})
    assert m["pr_auc"] > prevalence(yva), "PR-AUC should beat prevalence on a signal"
    assert 0.0 <= m["f1"] <= 1.0
    print("azure_models self-test OK")


if __name__ == "__main__":
    _self_test()
