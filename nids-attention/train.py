"""
Training and evaluation loop for one fold.
Protocol mirrors the thesis: fixed epochs, no early stopping,
evaluate the final-epoch model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from dataset import FlowDataset
from model import build_model
from reproducibility import set_seed


def train_one_fold(X_train, y_train, X_test, y_test, config, device='cuda'):
    """
    Train and evaluate one fold.

    Returns:
        dict with accuracy, f1_macro, f1_per_class, confusion_matrix,
        n_params, config
    """
    set_seed(config['seed'])

    # DataLoaders
    train_ds = FlowDataset(X_train, y_train)
    test_ds = FlowDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 256),
                              shuffle=True, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=config.get('batch_size', 256),
                             shuffle=False, num_workers=0)

    # Model
    model = build_model(config).to(device)
    n_params = model.count_parameters()

    # Optimiser (same as thesis: AdamW, lr=1e-3, wd=0.01)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.get('lr', 1e-3),
                                   weight_decay=config.get('weight_decay', 0.01))

    # Loss (with class weights for imbalanced data)
    if config.get('class_weights') is not None:
        weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training
    epochs = config.get('epochs', 100)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        if (epoch + 1) % 25 == 0:
            avg_loss = epoch_loss / len(train_ds)
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}")

    # Evaluation (final epoch, no checkpoint selection)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per.tolist(),
        'confusion_matrix': cm.tolist(),
        'n_params': n_params,
        'config': config,
    }
