# Attention Mechanisms for Network Intrusion Detection

Applying the four attention mechanisms from the thesis
*"Attention Mechanisms for Surface Vibration-based Gesture Recognition"*
to network intrusion detection on CIC-IDS2017.

## Quick Start

### 1. Download the dataset

Download CIC-IDS2017 CSVs from:
- [UNB Official](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Zenodo (preprocessed)](https://zenodo.org/records/5527231)

Place all CSV files in `./data/`.

### 2. Install dependencies

```bash
pip install torch numpy pandas scikit-learn
```

### 3. Verify the model

```bash
python model.py
```

This prints parameter counts for all variants and runs a forward pass test.

### 4. Run all experiments

```bash
# Full run: 11 variants × 5 folds × 5 seeds = 275 training runs
python experiments.py --data-dir ./data --output-dir ./results --epochs 100

# Quick test (1 fold, 1 seed, 10 epochs)
python experiments.py --data-dir ./data --output-dir ./results_test \
    --n-folds 2 --n-runs 1 --epochs 10
```

### 5. Results

Results are saved as JSON in `--output-dir`. The final summary table
prints to stdout with accuracy, F1, parameter count, and Δ accuracy
vs baseline.

## Architecture

Same depthwise-separable 1D-CNN structure as the thesis:

```
Input [B, 1, 78] → Conv1D (1→32) → BN → ReLU
  → 4× DSConvBlock (DW → PW → BN → MaxPool → ReLU → Attention → Dropout)
  → GAP → Linear → 8 classes
```

Attention module inserted after ReLU, before Dropout in every block.

## Variants

| # | Variant | Attention | Hyperparameter |
|---|---------|-----------|----------------|
| 1 | Baseline | None | — |
| 2–5 | SE | Channel (bottleneck) | r ∈ {2, 4, 8, 16} |
| 6 | ECA | Channel (local conv) | k = 3 |
| 7 | Temporal | Temporal | k = 7 |
| 8–11 | CT | Channel + temporal | r ∈ {2, 4, 8, 16} |

## Evaluation Protocol

- **CV:** Stratified 5-fold (balanced class representation)
- **Runs:** 5 independent seeds (42–46)
- **Epochs:** 100 (fixed, no early stopping)
- **Metric:** Accuracy and macro F1 (important for class imbalance)
- **Class weights:** Inverse frequency, applied to cross-entropy loss
- **Reproducibility:** cuDNN deterministic, all seeds fixed

## Dataset: CIC-IDS2017

| Property | Value |
|----------|-------|
| Samples | ~2.8M flows |
| Features | 78 numeric flow statistics |
| Classes | 8 (Benign + 7 attack families) |
| Imbalance | Benign ~83%, Infiltration <0.1% |

Attack families: Benign, BruteForce, DoS, WebAttack,
Infiltration, Bot, PortScan, DDoS.

## File Structure

```
├── dataset.py          # Data loading, preprocessing, CIC-IDS2017 specifics
├── attention.py        # SE, ECA, Temporal, Channel-Temporal modules
├── model.py            # NIDSNet: depthwise-separable 1D-CNN + attention
├── train.py            # Training and evaluation loop (one fold)
├── experiments.py      # Full experiment orchestration (all variants)
├── reproducibility.py  # Deterministic seeding
└── README.md
```
