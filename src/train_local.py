import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import time
import json

from src.models import LSTMRegressor
from src.metrics import evaluate_model, compute_metrics
from src.utils import set_seed, check_gpu


def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_max=125):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)
    
    return total_loss / n_samples


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)
    
    return total_loss / n_samples


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int = 14,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    max_epochs: int = 60,
    patience: int = 8,
    seed: int = 0,
    device: Optional[torch.device] = None,
    clip_max: int = 125,
    checkpoint_path: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """
    Train LSTM model with early stopping.
    
    Returns:
        model: trained model (best weights)
        history: dict with train_loss, val_loss, val_rmse per epoch
    """
    set_seed(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": []
    }
    
    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, clip_max)
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        pred, true = evaluate_model(model, val_loader, device, clip_max)
        val_rmse = compute_metrics(pred, true)["rmse"]
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"  Epoch {epoch+1}: Early stopping (no improvement for {patience} epochs)")
            break
    
    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save checkpoint if path provided
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "history": history,
            "best_val_rmse": best_val_rmse,
            "seed": seed
        }, checkpoint_path)
    
    return model, history


def create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, y_test_unclipped,
    batch_size: int = 256,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Create PyTorch DataLoaders from numpy arrays."""
    set_seed(seed)
    
    # Targets are already scaled to [0, 1] in preprocessing (clipped at 125)
    # So we just use them as-is
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test_unclipped


def run_experiment(
    setup_name: str,
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    seeds: List[int],
    device: torch.device,
    results_dir: Path,
    clip_max: int = 125
) -> List[Dict]:
    """
    Run training for multiple seeds.
    
    Args:
        setup_name: e.g., "Local FD001", "Centralized", "FD004-local"
        train_data: dict with X, y, unit
        val_data: dict with X, y, unit
        test_data: dict with X, y, y_unclipped, unit
        seeds: list of seeds to run
        device: torch device
        results_dir: directory to save checkpoints
        clip_max: max RUL value for scaling
    
    Returns:
        List of result dicts (one per seed)
    """
    results = []
    
    for seed in seeds:
        print(f"\n  Running {setup_name} with seed {seed}...")
        set_seed(seed)
        
        # Create dataloaders
        train_loader, val_loader, test_loader, y_test_unclipped = create_dataloaders(
            train_data["X"], train_data["y"],
            val_data["X"], val_data["y"],
            test_data["X"], test_data["y"],
            test_data["y_unclipped"],
            batch_size=256,
            seed=seed
        )
        
        # Train
        checkpoint_path = results_dir / "checkpoints" / f"{setup_name.replace(' ', '_')}_seed{seed}.pt"
        model, history = train_model(
            train_loader, val_loader,
            input_dim=14, hidden_size=64, num_layers=2, dropout=0.2,
            lr=1e-3, max_epochs=60, patience=8,
            seed=seed, device=device, clip_max=clip_max,
            checkpoint_path=checkpoint_path
        )
        
        # Evaluate on test set
        pred, true = evaluate_model(model, test_loader, device, clip_max)
        metrics = compute_metrics(pred, true, y_test_unclipped)
        
        result = {
            "setup": setup_name,
            "seed": seed,
            "test_factory": test_data["factory"],
            "rmse": metrics["rmse"],
            "nasa_score": metrics["nasa_score"],
            "rmse_unclipped": metrics.get("rmse_unclipped", np.nan),
            "epochs_trained": len(history["train_loss"]),
            "best_val_rmse": min(history["val_rmse"])
        }
        results.append(result)
        
        print(f"    Test RMSE: {metrics['rmse']:.4f}, NASA: {metrics['nasa_score']:.4f}, "
              f"RMSE_unclipped: {metrics.get('rmse_unclipped', np.nan):.4f}, "
              f"Epochs: {len(history['train_loss'])}")
    
    return results


if __name__ == "__main__":
    # Quick test
    from src.data import load_factory
    train_df, test_df, rul_df = load_factory("FD001")
    print("Data loaded successfully")