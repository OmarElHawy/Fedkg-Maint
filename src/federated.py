import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import time

from src.models import LSTMRegressor
from src.metrics import evaluate_model
from src.utils import set_seed
from src.privacy import (
    flatten_state, unflatten_like, subtract_states, privatize_delta,
    make_generator,
)


def fedavg(client_states: List[Dict[str, torch.Tensor]],
           client_sizes: List[int]) -> Dict[str, torch.Tensor]:
    """Weighted average of client model weights by number of samples."""
    total = sum(client_sizes)
    avg = {}
    for key in client_states[0]:
        avg[key] = sum(
            s[key].float() * (n / total)
            for s, n in zip(client_states, client_sizes)
        )
    return avg


def snapshot_global(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Detached CPU copy of the model weights (taken at the start of each round)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compute_loss(
    model: nn.Module,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    global_state: Optional[Dict[str, torch.Tensor]] = None,
    mu: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute task loss plus (optionally) the FedProx proximal penalty.

    Returns (loss, prox) where prox = sum over parameters of ||w - w_global||^2
    (0.0 when the penalty is inactive). The global weights are always used as a
    detached constant: gradients never flow into them.
    """
    pred = model(X_batch)
    loss = criterion(pred, y_batch)
    prox = torch.zeros((), device=device)
    if mu > 0 and global_state is not None:
        g = {k: v.to(device) for k, v in global_state.items()}
        for name, w in model.named_parameters():
            prox = prox + ((w - g[name]) ** 2).sum()
        loss = loss + (mu / 2) * prox
    return loss, prox


def proximal_penalty(
    state: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor]
) -> float:
    """sum over all parameters of ||w - w_global||^2 between two state dicts."""
    total = 0.0
    for k in global_state:
        total += ((state[k].float() - global_state[k].float()) ** 2).sum().item()
    return float(total)


def state_distance(state_a: Dict[str, torch.Tensor],
                   state_b: Dict[str, torch.Tensor]) -> float:
    """L2 distance between two state dicts: sqrt(sum of squared differences)."""
    return float(np.sqrt(sum(
        ((state_a[k].float() - state_b[k].float()) ** 2).sum().item()
        for k in state_a
    )))


def client_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    global_state: Optional[Dict[str, torch.Tensor]] = None,
    mu: float = 0.0
) -> float:
    """Train one epoch. If mu > 0, add FedProx proximal penalty to the loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        loss, _ = compute_loss(model, X_batch, y_batch, criterion, device,
                               global_state=global_state, mu=mu)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)

    return total_loss / n_samples


def weighted_val_rmse(
    model: nn.Module,
    client_val_data: Dict[str, Dict],
    device: torch.device,
    clip_max: int = 125,
    batch_size: int = 1024
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted validation RMSE across clients, weighted by number of val windows.

    client_val_data: factory -> {"X_val": ..., "y_val": ...} with y_val raw RUL.
    Returns (weighted_rmse, per_client_rmse).
    """
    model.eval()
    per_client = {}
    for f, d in client_val_data.items():
        X, y = d["X_val"], d["y_val"]
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.FloatTensor(X[i:i + batch_size]).to(device)
                preds.append(model(xb).cpu().numpy())
        pred = np.concatenate(preds) * clip_max
        per_client[f] = float(np.sqrt(np.mean((pred - y) ** 2)))

    weights = {f: len(d["X_val"]) for f, d in client_val_data.items()}
    total = sum(weights.values())
    weighted = sum(per_client[f] * weights[f] for f in per_client) / total
    return weighted, per_client


def run_federated(
    method: str,  # "fedavg" or "fedprox"
    client_data: Dict[str, Dict],  # factory -> {X_train, y_train} with y already / CLIP_MAX
    held_out_X: Optional[np.ndarray] = None,
    held_out_y_clipped: Optional[np.ndarray] = None,  # np.minimum(y, 125)
    held_out_y_unclipped: Optional[np.ndarray] = None,
    n_rounds: int = 30,
    local_epochs: int = 2,
    mu: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    device: Optional[torch.device] = None,
    clip_max: int = 125,
    client_val_data: Optional[Dict[str, Dict]] = None,
    eval_every: int = 5,
    log_every: int = 10,
    verbose: bool = True,
    dp: Optional[Dict] = None,
    val_every: int = 1,
    record_delta_norms: bool = False
) -> Tuple[nn.Module, Dict, Dict]:
    """
    Run federated learning simulation.

    - Runs a FIXED number of rounds. The returned model is the final global
      model after round `n_rounds`. No held-out data ever selects the model.
    - held_out_* : optional, for logging/reporting only. Pass None to never
      touch held-out data (used during tuning).
    - client_val_data : optional, factory -> {X_val, y_val}; if given, a
      weighted validation RMSE is logged every `val_every` rounds (and always
      on the final round). Validation never influences training.
    - dp : optional dict {"C": clip_norm, "sigma": noise_multiplier}. When
      given, each client sends clip(delta, C) + N(0, (sigma*C)^2) and the
      server averages the noisy deltas and adds them to the global weights.
      The noise uses a dedicated Generator seeded from (seed, round, client),
      never the global RNG. When None (default) the behaviour is unchanged.
    - record_delta_norms : if True (or when dp is set), the L2 norm of every
      raw client delta is logged per round in history["client_delta_norms"].
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    factories = sorted(client_data.keys())
    if verbose:
        print(f"  Method: {method}, Clients: {factories}, "
              f"Rounds: {n_rounds}, Local epochs: {local_epochs}, mu: {mu}")

    global_model = LSTMRegressor(
        input_dim=14, hidden_size=64, num_layers=2, dropout=0.2
    ).to(device)
    n_params = sum(p.numel() for p in global_model.parameters())

    client_loaders = {}
    client_sizes = {}
    for f in factories:
        d = client_data[f]
        train_ds = TensorDataset(
            torch.FloatTensor(d["X_train"]),
            torch.FloatTensor(d["y_train"])
        )
        client_loaders[f] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        client_sizes[f] = len(train_ds)

    use_held_out = held_out_X is not None
    held_out_loader = None
    if use_held_out:
        held_out_loader = DataLoader(
            TensorDataset(torch.FloatTensor(held_out_X),
                          torch.FloatTensor(held_out_y_clipped / clip_max)),
            batch_size=batch_size, shuffle=False
        )

    criterion = nn.MSELoss()

    history = {
        "round": [],
        "val_rmse": [],
        "val_round": [],
        "held_out_rmse": [],
        "comm_cost_bytes": [],
        "client_train_loss": [],
        "client_delta_norms": []
    }

    comm_cost = 0
    start_time = time.time()

    for rnd in range(1, n_rounds + 1):
        # Snapshot of the global weights taken at the START of the round:
        # a detached copy, reused for every client and every local epoch.
        global_state = snapshot_global(global_model)
        client_states = []
        client_losses = []
        delta_norms = []

        for f in factories:
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(global_state)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

            for _ in range(local_epochs):
                loss = client_train_epoch(
                    local_model, client_loaders[f], optimizer, criterion, device,
                    global_state=global_state,
                    mu=mu if method == "fedprox" else 0.0
                )

            local_state = snapshot_global(local_model)

            if record_delta_norms or dp is not None:
                raw_delta = flatten_state(subtract_states(local_state, global_state))
                delta_norms.append(float(raw_delta.norm()))

            if dp is None:
                client_states.append(local_state)
            else:
                # Client-level DP on the update: clip, noise, send the noisy delta.
                # The noise Generator is seeded deterministically from
                # (seed, round, client) and is completely separate from the
                # global RNG, so evaluation can never change the noise stream.
                gen = make_generator(seed, rnd, f)
                noisy_flat = privatize_delta(
                    local_state, global_state, dp["C"], dp["sigma"], gen
                )
                noisy_delta = unflatten_like(noisy_flat, global_state)
                # Server-side view: global + noisy delta. Averaging these
                # weighted by client size equals global + weighted mean(delta).
                client_states.append({
                    k: global_state[k] + noisy_delta[k] for k in global_state
                })

            client_losses.append(loss)
            comm_cost += n_params * 4 * 2  # download + upload, float32

        global_state = fedavg(client_states, [client_sizes[f] for f in factories])
        global_model.load_state_dict(global_state)

        if record_delta_norms or dp is not None:
            history["client_delta_norms"].append(delta_norms)

        # Validation (clients only) - logged every val_every rounds and on the
        # final round when provided. Validation never changes training.
        if client_val_data is not None and (rnd % val_every == 0 or rnd == n_rounds):
            val_rmse, _ = weighted_val_rmse(global_model, client_val_data, device, clip_max)
            history["val_rmse"].append(val_rmse)
            history["val_round"].append(rnd)
        else:
            val_rmse = None

        # Held-out logging (never used for any choice) - only if provided
        held_out_rmse = None
        if use_held_out and (rnd % eval_every == 0 or rnd == n_rounds):
            pred, true = evaluate_model(global_model, held_out_loader, device, clip_max)
            held_out_rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
            history["held_out_rmse"].append(held_out_rmse)

        history["round"].append(rnd)
        history["comm_cost_bytes"].append(comm_cost)
        history["client_train_loss"].append(float(np.mean(client_losses)))

        if verbose and (rnd % log_every == 0 or rnd == n_rounds):
            parts = [f"    Round {rnd:>3d}/{n_rounds}:"]
            if val_rmse is not None:
                parts.append(f"val RMSE = {val_rmse:.4f},")
            if held_out_rmse is not None:
                parts.append(f"held-out RMSE = {held_out_rmse:.4f},")
            parts.append(f"train loss = {np.mean(client_losses):.4f},")
            parts.append(f"comm = {comm_cost / 1e6:.1f} MB,")
            parts.append(f"time = {time.time() - start_time:.1f}s")
            print(" ".join(parts))

    # Final metrics on held-out, computed ONLY on the final global model.
    final_metrics = {}
    if use_held_out:
        global_model.eval()
        with torch.no_grad():
            pred = global_model(torch.FloatTensor(held_out_X).to(device)).cpu().numpy() * clip_max
        d = pred - held_out_y_clipped
        final_metrics = {
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "nasa_score": float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))),
            "rmse_unclipped": float(np.sqrt(np.mean((pred - held_out_y_unclipped) ** 2)))
        }
        if verbose:
            print(f"  Final held-out RMSE: {final_metrics['rmse']:.4f}, "
                  f"NASA: {final_metrics['nasa_score']:.4f}")

    return global_model, history, final_metrics


def evaluate_on_factory(
    model: nn.Module,
    X_test: np.ndarray,
    y_test_unclipped: np.ndarray,
    device: torch.device,
    clip_max: int = 125
) -> Dict:
    """Evaluate model on a factory test set. y_test_unclipped is raw RUL."""
    y_test_clipped = np.minimum(y_test_unclipped, clip_max)
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy() * clip_max
    d = pred - y_test_clipped
    return {
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "nasa_score": float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))),
        "rmse_unclipped": float(np.sqrt(np.mean((pred - y_test_unclipped) ** 2)))
    }
