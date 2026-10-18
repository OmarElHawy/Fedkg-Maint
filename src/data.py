import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss"

COLUMNS = [
    "unit", "cycle", "set1", "set2", "set3",
    *[f"s{i}" for i in range(1, 22)]
]


def _load_factory_files(name: str, raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and RUL files for a given factory."""
    train_file = raw_dir / f"train_{name}.txt"
    test_file = raw_dir / f"test_{name}.txt"
    rul_file = raw_dir / f"RUL_{name}.txt"

    train = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["RUL"])

    return train, test, rul


def _add_rul_to_train(train: pd.DataFrame) -> pd.DataFrame:
    """Add RUL column to train dataframe: max cycle per engine minus current cycle."""
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train = train.copy()
    train["RUL"] = max_cycle - train["cycle"]
    return train


def load_factory(name: str, raw_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a factory dataset.

    Args:
        name: Factory name, one of "FD001", "FD002", "FD003", "FD004"
        raw_dir: Path to raw data directory (defaults to project_root/data/raw/cmapss)

    Returns:
        Tuple of (train_df, test_df, rul_df)
        - train_df has columns: unit, cycle, set1, set2, set3, s1..s21, RUL
        - test_df has columns: unit, cycle, set1, set2, set3, s1..s21
        - rul_df has one column: RUL (true RUL for each test engine)
    """
    if raw_dir is None:
        raw_path = DEFAULT_RAW_DIR
    else:
        raw_path = Path(raw_dir)
    train, test, rul = _load_factory_files(name, raw_path)
    train = _add_rul_to_train(train)
    return train, test, rul


if __name__ == "__main__":
    for name in ["FD001", "FD002", "FD003", "FD004"]:
        train, test, rul = load_factory(name)
        print(f"{name}: train={train.shape}, test={test.shape}, rul={rul.shape}")
        print(f"  Engines train: {train['unit'].nunique()}, test: {test['unit'].nunique()}")
        print(f"  Avg engine life (train): {train.groupby('unit')['cycle'].max().mean():.1f}")