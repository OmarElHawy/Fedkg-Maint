import numpy as np
import torch


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """
    NASA score for RUL prediction.
    
    d = predicted - true
    if d < 0: exp(-d/13) - 1 (early prediction penalty)
    if d >= 0: exp(d/10) - 1 (late prediction penalty, heavier)
    
    Lower is better.
    """
    d = pred - true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def evaluate_model(model, dataloader, device, clip_max=125):
    """
    Evaluate model on a dataloader.
    
    Returns:
        pred: predictions (scaled back to original range)
        true: true values
    """
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
    
    pred = np.concatenate(preds) * clip_max
    true = np.concatenate(trues) * clip_max
    
    return pred, true


def compute_metrics(pred: np.ndarray, true: np.ndarray, true_unclipped: np.ndarray = None) -> dict:
    """
    Compute all metrics.
    
    Args:
        pred: predicted RUL (clipped at 125)
        true: true RUL (clipped at 125) - used for main metrics
        true_unclipped: true RUL without clipping - for extra metric
    
    Returns:
        dict with rmse, nasa_score, and optionally rmse_unclipped
    """
    metrics = {
        "rmse": rmse(pred, true),
        "nasa_score": nasa_score(pred, true)
    }
    if true_unclipped is not None:
        metrics["rmse_unclipped"] = rmse(pred, true_unclipped)
    return metrics


if __name__ == "__main__":
    # Quick test
    pred = np.array([10, 20, 30, 40, 50])
    true = np.array([12, 18, 35, 38, 55])
    print(f"RMSE: {rmse(pred, true):.4f}")
    print(f"NASA Score: {nasa_score(pred, true):.4f}")