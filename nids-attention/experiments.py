"""
Full experiment runner.

Mirrors the thesis protocol:
  - Stratified 5-fold CV (analogous to LOSO)
  - 5 independent runs per variant (seeds 42–46)
  - 11 variants: baseline + 4 SE + 1 ECA + 1 temporal + 4 CT
  - Results saved as JSON

Usage:
    python experiments.py --data-dir ./data --output-dir ./results
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

from dataset import (load_cicids2017, preprocess, normalise_and_reshape,
                     get_kfold_splits)
from train import train_one_fold


# ── Variant definitions ──────────────────────────────────────────────────────
VARIANTS = [
    {'attention': 'none',     'reduction': 0,  'label': 'Baseline'},
    {'attention': 'se',       'reduction': 2,  'label': 'SE (r=2)'},
    {'attention': 'se',       'reduction': 4,  'label': 'SE (r=4)'},
    {'attention': 'se',       'reduction': 8,  'label': 'SE (r=8)'},
    {'attention': 'se',       'reduction': 16, 'label': 'SE (r=16)'},
    {'attention': 'eca',      'reduction': 0,  'label': 'ECA'},
    {'attention': 'temporal', 'reduction': 0,  'label': 'Temporal'},
    {'attention': 'ct',       'reduction': 2,  'label': 'CT (r=2)'},
    {'attention': 'ct',       'reduction': 4,  'label': 'CT (r=4)'},
    {'attention': 'ct',       'reduction': 8,  'label': 'CT (r=8)'},
    {'attention': 'ct',       'reduction': 16, 'label': 'CT (r=16)'},
]

# Default model config
BASE_CONFIG = {
    'in_channels': 1,
    'seq_len': 78,
    'n_classes': 8,
    'n_blocks': 4,
    'channels': 32,
    'kernel_size': 9,
    'temporal_kernel': 7,
    'dropout': 0.1,
    'lr': 1e-3,
    'weight_decay': 0.01,
    'batch_size': 256,
    'epochs': 100,
    'class_weights': None,
}


def compute_class_weights(y):
    """Inverse frequency class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return weights.tolist()


def run_all(data_dir, output_dir, n_folds=5, n_runs=5, base_seed=42,
            use_class_weights=True, device='cuda'):
    """Run all variants across folds and seeds."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading CIC-IDS2017...")
    data = load_cicids2017(data_dir)
    X_raw, y, le = preprocess(data)

    # Class weights
    cw = compute_class_weights(y) if use_class_weights else None

    # Update config
    config_base = {**BASE_CONFIG}
    config_base['n_classes'] = len(le.classes_)
    config_base['seq_len'] = X_raw.shape[1]
    config_base['class_weights'] = cw

    # ── Summary storage ──────────────────────────────────────────────────────
    all_results = {}

    for variant in VARIANTS:
        label = variant['label']
        print(f"\n{'='*60}")
        print(f"Variant: {label}")
        print(f"{'='*60}")

        run_accs = []
        run_f1s = []
        run_results = []

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            print(f"\n  Run {run_idx+1}/{n_runs} (seed={seed})")

            # Get folds for this seed
            splits = get_kfold_splits(X_raw, y, n_splits=n_folds, seed=seed)

            fold_accs = []
            fold_f1s = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f"    Fold {fold_idx+1}/{n_folds}")

                # Split and normalise
                X_train, X_test, _ = normalise_and_reshape(
                    X_raw[train_idx], X_raw[test_idx], n_groups=1)
                y_train, y_test = y[train_idx], y[test_idx]

                # Build config for this run
                cfg = {**config_base}
                cfg['attention'] = variant['attention']
                cfg['reduction'] = variant['reduction']
                cfg['seed'] = seed
                cfg['in_channels'] = X_train.shape[1]
                cfg['seq_len'] = X_train.shape[2]

                # Train and evaluate
                result = train_one_fold(
                    X_train, y_train, X_test, y_test, cfg, device)

                fold_accs.append(result['accuracy'])
                fold_f1s.append(result['f1_macro'])

                print(f"      Acc={result['accuracy']:.4f}  "
                      f"F1={result['f1_macro']:.4f}  "
                      f"Params={result['n_params']:,}")

            # Aggregate folds
            run_acc = np.mean(fold_accs)
            run_f1 = np.mean(fold_f1s)
            run_accs.append(run_acc)
            run_f1s.append(run_f1)
            run_results.append({
                'seed': seed,
                'fold_accs': fold_accs,
                'fold_f1s': fold_f1s,
                'mean_acc': run_acc,
                'mean_f1': run_f1,
            })

            print(f"  Run {run_idx+1} mean: Acc={run_acc:.4f}  F1={run_f1:.4f}")

        # ── Aggregate runs ───────────────────────────────────────────────────
        summary = {
            'variant': label,
            'attention': variant['attention'],
            'reduction': variant['reduction'],
            'n_params': result['n_params'],
            'mean_acc': float(np.mean(run_accs)),
            'std_acc': float(np.std(run_accs)),   # population std
            'mean_f1': float(np.mean(run_f1s)),
            'std_f1': float(np.std(run_f1s)),
            'runs': run_results,
        }
        all_results[label] = summary

        print(f"\n  {label}: "
              f"Acc={summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}  "
              f"F1={summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}  "
              f"Params={summary['n_params']:,}")

        # Save per-variant
        with open(os.path.join(output_dir, f"{label.replace(' ', '_')}.json"),
                  'w') as f:
            json.dump(summary, f, indent=2)

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Variant':<20} {'Acc (%)':>12} {'F1 (%)':>12} {'Params':>10} {'Δ Acc':>8}")
    print(f"{'-'*80}")

    baseline_acc = all_results['Baseline']['mean_acc']

    for label in [v['label'] for v in VARIANTS]:
        r = all_results[label]
        delta = r['mean_acc'] - baseline_acc
        delta_str = f"{delta:+.2f}" if label != 'Baseline' else "—"
        print(f"{label:<20} "
              f"{100*r['mean_acc']:>5.2f} ± {100*r['std_acc']:.2f} "
              f"{100*r['mean_f1']:>5.2f} ± {100*r['std_f1']:.2f} "
              f"{r['n_params']:>10,} "
              f"{delta_str:>8}")

    # Save full results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing CIC-IDS2017 CSVs')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    BASE_CONFIG['epochs'] = args.epochs

    run_all(args.data_dir, args.output_dir,
            n_folds=args.n_folds, n_runs=args.n_runs,
            base_seed=args.base_seed, device=args.device)
