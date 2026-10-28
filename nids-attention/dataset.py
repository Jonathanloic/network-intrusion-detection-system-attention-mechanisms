"""
CIC-IDS2017 data loading and preprocessing.

Download the dataset from:
  https://www.unb.ca/cic/datasets/ids-2017.html
  or Zenodo: https://zenodo.org/records/5527231

Place all CSV files in a folder and pass the path to load_cicids2017().
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader


# ── Label mapping: 15 raw labels → 8 attack families ─────────────────────────
LABEL_MAP = {
    'BENIGN': 'Benign',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'Heartbleed': 'DoS',
    'Web Attack \x96 Brute Force': 'WebAttack',
    'Web Attack \x96 XSS': 'WebAttack',
    'Web Attack \x96 Sql Injection': 'WebAttack',
    'Infiltration': 'Infiltration',
    'Bot': 'Bot',
    'PortScan': 'PortScan',
    'DDoS': 'DDoS',
}

# Columns to drop (non-numeric, identifiers, or problematic)
DROP_COLS = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp',
             'Source Port', 'Destination Port', 'Protocol']


def load_cicids2017(data_dir, label_col=' Label'):
    """Load and merge all CIC-IDS2017 CSVs from a directory."""
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in csv_files:
        print(f"  Loading {f}...")
        df = pd.read_csv(os.path.join(data_dir, f), low_memory=False)
        # CIC-IDS2017 quirk: column names have leading spaces
        df.columns = df.columns.str.strip()
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(data):,}")
    return data


def preprocess(data, label_col='Label'):
    """Clean, encode, and normalise the dataset."""
    # Strip label whitespace
    data[label_col] = data[label_col].str.strip()

    # Map to attack families
    data['attack_family'] = data[label_col].map(LABEL_MAP)
    data = data.dropna(subset=['attack_family'])

    # Drop identifier columns
    drop = [c for c in DROP_COLS if c in data.columns]
    data = data.drop(columns=drop + [label_col], errors='ignore')

    # Replace inf with NaN, drop NaN rows
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Remove duplicates
    data = data.drop_duplicates()

    # Separate features and labels
    y_raw = data.pop('attack_family')
    X = data.select_dtypes(include=[np.number]).values.astype(np.float32)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)

    print(f"  Clean samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {le.classes_}")
    print(f"  Distribution:")
    for cls in le.classes_:
        count = (y_raw == cls).sum()
        print(f"    {cls}: {count:,} ({100*count/len(y_raw):.1f}%)")

    return X, y, le


def normalise_and_reshape(X_train, X_test, n_groups=None):
    """
    Normalise features and reshape into [batch, channels, seq_len].

    n_groups: number of feature groups to form "channels".
              If None, uses [batch, 1, n_features] (single channel).
              If e.g. 8, groups 78 features into 8 channels of ~10 each.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]

    if n_groups is None or n_groups == 1:
        # Single channel: [batch, 1, features]
        X_train = X_train[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]
    else:
        # Group features into channels: [batch, n_groups, group_size]
        group_size = n_features // n_groups
        # Trim features to fit evenly
        trim = n_groups * group_size
        X_train = X_train[:, :trim].reshape(-1, n_groups, group_size)
        X_test = X_test[:, :trim].reshape(-1, n_groups, group_size)

    return X_train, X_test, scaler


class FlowDataset(Dataset):
    """PyTorch dataset for network flow classification."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_kfold_splits(X, y, n_splits=5, seed=42):
    """Stratified k-fold splits (analogous to LOSO in thesis)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'
    data = load_cicids2017(data_dir)
    X, y, le = preprocess(data)
    print(f"\nReady: X={X.shape}, y={y.shape}, classes={len(le.classes_)}")
