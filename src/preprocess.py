import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


SENSORS_TO_KEEP = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
SETTINGS = ["set1", "set2", "set3"]


def get_condition_labels(train_df: pd.DataFrame, n_clusters: int, seed: int = 42) -> np.ndarray:
    """Fit KMeans on the 3 operating settings and return cluster labels."""
    settings_data = train_df[SETTINGS].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(settings_data)
    return labels, kmeans


def compute_stats_per_condition(train_df: pd.DataFrame, condition_labels: np.ndarray, 
                                 sensors: List[str], n_clusters: int) -> Dict:
    """Compute mean and std per sensor per condition cluster using training data only."""
    stats = {}
    for c in range(n_clusters):
        mask = condition_labels == c
        cluster_data = train_df.loc[mask, sensors].values
        stats[c] = {
            "mean": cluster_data.mean(axis=0).tolist(),
            "std": cluster_data.std(axis=0).tolist(),
            "n_samples": int(mask.sum())
        }
    return stats


def apply_normalization(df: pd.DataFrame, condition_labels: np.ndarray, 
                        stats: Dict, sensors: List[str]) -> pd.DataFrame:
    """Apply per-condition normalization to a dataframe."""
    df = df.copy()
    # Convert sensor columns to float to avoid dtype issues
    df[sensors] = df[sensors].astype(float)
    for c in stats:
        mask = condition_labels == c
        if mask.sum() == 0:
            continue
        mean = np.array(stats[c]["mean"])
        std = np.array(stats[c]["std"])
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        df.loc[mask, sensors] = (df.loc[mask, sensors].values - mean) / std
    return df


def clip_rul(rul: np.ndarray, clip: int = 125) -> np.ndarray:
    """Clip RUL values at the specified threshold."""
    return np.minimum(rul, clip)


def create_windows(X: np.ndarray, y: np.ndarray, units: np.ndarray, 
                   window_len: int = 30, stride: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data.
    
    Args:
        X: shape (n_samples, n_features)
        y: shape (n_samples,)
        units: shape (n_samples,) engine IDs
        window_len: length of each window
        stride: step size
    
    Returns:
        X_win: (n_windows, window_len, n_features)
        y_win: (n_windows,) RUL at the last step of each window
        unit_win: (n_windows,) engine ID for each window
    """
    X_win_list = []
    y_win_list = []
    unit_win_list = []
    
    for unit_id in np.unique(units):
        mask = units == unit_id
        X_unit = X[mask]
        y_unit = y[mask]
        
        n_cycles = len(X_unit)
        if n_cycles < window_len:
            continue
        
        for start in range(0, n_cycles - window_len + 1, stride):
            end = start + window_len
            X_win_list.append(X_unit[start:end])
            y_win_list.append(y_unit[end - 1])  # RUL at the last cycle of the window
            unit_win_list.append(unit_id)
    
    if len(X_win_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(X_win_list), np.array(y_win_list), np.array(unit_win_list)


def create_test_windows(test_df: pd.DataFrame, rul_df: pd.DataFrame, 
                        sensors: List[str], window_len: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create test windows: last window of each test engine.
    If engine has fewer than window_len cycles, pad at front by repeating first row.
    """
    X_win_list = []
    y_win_list = []
    unit_win_list = []
    
    for unit_id in test_df["unit"].unique():
        unit_data = test_df[test_df["unit"] == unit_id].sort_values("cycle")
        X_unit = unit_data[sensors].values
        n_cycles = len(X_unit)
        
        if n_cycles >= window_len:
            X_win = X_unit[-window_len:]
        else:
            # Pad at front by repeating first row
            pad_len = window_len - n_cycles
            first_row = X_unit[0:1].repeat(pad_len, axis=0)
            X_win = np.vstack([first_row, X_unit])
        
        X_win_list.append(X_win)
        # True RUL from RUL file (not clipped)
        true_rul = rul_df.loc[unit_id - 1, "RUL"]  # RUL file is 0-indexed, unit IDs start at 1
        y_win_list.append(true_rul)
        unit_win_list.append(unit_id)
    
    return np.array(X_win_list), np.array(y_win_list), np.array(unit_win_list)


def split_engines(train_df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Split engines into train and validation sets by engine ID."""
    np.random.seed(seed)
    unique_engines = train_df["unit"].unique()
    np.random.shuffle(unique_engines)
    n_val = int(len(unique_engines) * val_ratio)
    val_engines = unique_engines[:n_val]
    train_engines = unique_engines[n_val:]
    return train_engines, val_engines


def preprocess_factory(name: str, window: int = 30, clip: int = 125, seed: int = 42,
                       raw_dir: Optional[str] = None, processed_dir: Optional[str] = None) -> Dict:
    """
    Full preprocessing pipeline for one factory.
    
    Args:
        name: Factory name (FD001, FD002, FD003, FD004)
        window: Window length
        clip: RUL clip value
        seed: Random seed
        raw_dir: Path to raw data directory
        processed_dir: Path to save processed data
    
    Returns:
        Dictionary with shapes and stats info
    """
    from src.data import load_factory
    
    # Load data
    train_df, test_df, rul_df = load_factory(name, raw_dir)
    
    # Determine number of clusters for KMeans
    n_clusters = 6 if name in ["FD002", "FD004"] else 1
    
    # Split engines into train/val
    train_engines, val_engines = split_engines(train_df, val_ratio=0.2, seed=seed)
    
    train_split = train_df[train_df["unit"].isin(train_engines)].copy()
    val_split = train_df[train_df["unit"].isin(val_engines)].copy()
    
    # Get condition labels for training split only
    train_cond_labels, kmeans = get_condition_labels(train_split, n_clusters, seed)
    
    # Compute normalization stats per condition using training split only
    stats = compute_stats_per_condition(train_split, train_cond_labels, SENSORS_TO_KEEP, n_clusters)
    
    # Apply normalization to train, val, and test
    # Get condition labels for val and test using the same KMeans
    val_settings = val_split[SETTINGS].values
    val_cond_labels = kmeans.predict(val_settings)
    
    test_settings = test_df[SETTINGS].values
    test_cond_labels = kmeans.predict(test_settings)
    
    # Normalize
    train_norm = apply_normalization(train_split, train_cond_labels, stats, SENSORS_TO_KEEP)
    val_norm = apply_normalization(val_split, val_cond_labels, stats, SENSORS_TO_KEEP)
    test_norm = apply_normalization(test_df, test_cond_labels, stats, SENSORS_TO_KEEP)
    
    # Prepare arrays for windowing
    X_train = train_norm[SENSORS_TO_KEEP].values
    y_train = clip_rul(train_norm["RUL"].values, clip)
    units_train = train_norm["unit"].values
    
    X_val = val_norm[SENSORS_TO_KEEP].values
    y_val = clip_rul(val_norm["RUL"].values, clip)
    units_val = val_norm["unit"].values
    
    # Create windows
    X_train_win, y_train_win, unit_train_win = create_windows(X_train, y_train, units_train, window)
    X_val_win, y_val_win, unit_val_win = create_windows(X_val, y_val, units_val, window)
    
    # Test windows (last window per engine, true RUL not clipped)
    X_test_win, y_test_win, unit_test_win = create_test_windows(test_norm, rul_df, SENSORS_TO_KEEP, window)
    
    # Save processed data
    if processed_dir is None:
        processed_dir = Path("data/processed")
    else:
        processed_dir = Path(processed_dir)
    
    factory_dir = processed_dir / name
    factory_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.savez(
        factory_dir / "arrays.npz",
        X_train=X_train_win, y_train=y_train_win, unit_train=unit_train_win,
        X_val=X_val_win, y_val=y_val_win, unit_val=unit_val_win,
        X_test=X_test_win, y_test=y_test_win, unit_test=unit_test_win
    )
    
    # Save stats
    stats_json = {
        "sensors": SENSORS_TO_KEEP,
        "n_clusters": n_clusters,
        "kmeans_centers": kmeans.cluster_centers_.tolist(),
        "stats_per_cluster": stats,
        "window": window,
        "clip": clip,
        "seed": seed,
        "train_engines": train_engines.tolist(),
        "val_engines": val_engines.tolist()
    }
    
    with open(factory_dir / "stats.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    
    return {
        "name": name,
        "train_shape": X_train_win.shape,
        "val_shape": X_val_win.shape,
        "test_shape": X_test_win.shape,
        "n_train_engines": len(train_engines),
        "n_val_engines": len(val_engines),
        "n_test_engines": len(unit_test_win),
        "n_clusters": n_clusters
    }


if __name__ == "__main__":
    for name in ["FD001", "FD002", "FD003", "FD004"]:
        result = preprocess_factory(name)
        print(f"{name}: train={result['train_shape']}, val={result['val_shape']}, test={result['test_shape']}")