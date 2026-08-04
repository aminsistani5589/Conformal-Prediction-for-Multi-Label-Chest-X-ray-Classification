"""
main.py
-------
End‑to‑end evaluation of CheXNet with several conformal prediction methods,
including the new Mondrian (label‑conditional) approach.
After evaluation, saves all conformal parameters for later single‑image inference.

New features:
- Class prevalence (positive count & percentage) in the class‑conditional coverage table.
- Multi‑α evaluation (α ∈ {0.01, 0.05, 0.1, 0.15, 0.2}) to draw Coverage‑Efficiency frontiers.
- Two plots:
    1. Achieved Coverage vs Average Set Size (Coverage‑Efficiency Frontier)
    2. Target Coverage (1‑α) vs Average Set Size
"""

import torch
import glob
import os
import tempfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import config
from xray_model.model_loader import load_chexnet_model
from xray_model.predictor import run_prediction
from utils.data_mapping import format_findings_for_prompt

# ----------------------------------------------------------------------
# Conformal prediction modules
# ----------------------------------------------------------------------
from THR import (
    fit_thr,
    fit_ecot,
    predict_sets,
    evaluate_prediction_sets,
)
from APS import fit_aps as old_fit_aps, predict_sets_aps as old_predict_aps
from RAPS import fit_raps as old_fit_raps, predict_sets_raps as old_predict_raps
from mondrian import fit_mondrian, predict_mondrian

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
DEFAULT_CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

CLASS_THRESHOLDS = {
    'Atelectasis': 0.5, 'Cardiomegaly': 0.5, 'Effusion': 0.6,
    'Infiltration': 0.5, 'Mass': 0.5, 'Nodule': 0.5,
    'Pneumonia': 0.5, 'Pneumothorax': 0.65, 'Consolidation': 0.5,
    'Edema': 0.55, 'Emphysema': 0.55, 'Fibrosis': 0.62,
    'Pleural_Thickening': 0.55, 'Hernia': 0.85
}


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def label_to_vector(label_array, class_names=DEFAULT_CLASS_NAMES):
    vector = np.zeros(len(class_names), dtype=np.int32)
    for name in label_array:
        name = name.strip()
        if name == "No Finding":
            continue
        try:
            vector[class_names.index(name)] = 1
        except ValueError:
            print(f"Warning: disease '{name}' not in class list; ignoring.")
    return vector


def read_parquet_with_fallback(file_path):
    methods = [
        lambda: pd.read_parquet(file_path),
        lambda: pq.read_table(file_path).to_pandas(),
        lambda: _read_parquet_in_batches(file_path),
    ]
    for i, method in enumerate(methods, 1):
        try:
            df = method()
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            print(f"  Method {i} failed: {type(e).__name__}: {str(e)[:100]}")
    raise RuntimeError(f"All methods failed to read {file_path}")


def _read_parquet_in_batches(file_path):
    parquet_file = pq.ParquetFile(file_path)
    dfs = []
    for batch in parquet_file.iter_batches(batch_size=100):
        dfs.append(batch.to_pandas())
    return pd.concat(dfs, ignore_index=True) if dfs else None


def extract_image_bytes(img_cell):
    if isinstance(img_cell, bytes):
        return img_cell
    if isinstance(img_cell, dict):
        return img_cell.get('bytes') or img_cell.get('path')
    if isinstance(img_cell, str):
        return img_cell
    if hasattr(img_cell, 'as_py'):
        try:
            return img_cell.as_py()
        except Exception:
            return None
    return None


def save_bytes_to_temp_png(img_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    with open(tmp.name, "wb") as f:
        f.write(img_bytes)
    return tmp.name




# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
if __name__ == '__main__':
    parquet_files = glob.glob(os.path.join(config.PARQUET_DIR, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")

    all_probs = []
    all_labels = []
    total_processed = 0

    model = load_chexnet_model(config.CKPT_PATH, config.N_CLASSES)

    for pq_file in parquet_files:
        print(f"\n--- Processing {pq_file} (total images so far: {total_processed}) ---")
        try:
            df = read_parquet_with_fallback(pq_file)
            print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")

            if 'image' not in df.columns:
                print("No 'image' column, skipping.")
                continue

            label_col = None
            for col in ['label', 'labels', 'targets', 'finding_labels']:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                for col in df.columns:
                    if 'label' in col.lower():
                        label_col = col
                        break
            if label_col is None:
                print("No label column found, skipping.")
                continue

            print(f"Using label column: '{label_col}'")

            file_processed = 0
            for idx, row in df.iterrows():
                temp_path = None
                try:
                    img_cell = row['image']
                    img_data = extract_image_bytes(img_cell)
                    if img_data is None:
                        continue

                    if isinstance(img_data, bytes):
                        temp_path = save_bytes_to_temp_png(img_data)
                        image_path = temp_path
                    elif isinstance(img_data, str) and os.path.exists(img_data):
                        image_path = img_data
                    else:
                        continue

                    probs = run_prediction(model, image_path)
                    if probs.ndim == 2:
                        probs = probs.squeeze(0)

                    raw_label = row[label_col]
                    if isinstance(raw_label, np.ndarray):
                        raw_label = raw_label.tolist()
                    label_vec = label_to_vector(raw_label)

                    if len(label_vec) != config.N_CLASSES:
                        continue

                    all_probs.append(to_numpy(probs))
                    all_labels.append(label_vec)
                    total_processed += 1
                    file_processed += 1

                    active = [DEFAULT_CLASS_NAMES[i] for i, v in enumerate(label_vec) if v]
                    print(f"   ✅ [{total_processed}] row {idx}: active={active if active else 'No Finding'}")

                except Exception as e:
                    print(f"   ❌ row {idx}: {type(e).__name__} – {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            print(f"--- Finished {pq_file}: {file_processed} images processed ---")

        except Exception as e:
            print(f"Failed to process {pq_file}: {e}")

    if not all_probs:
        print("\nNo data collected. Exiting.")
        exit()

    all_probs = np.stack(all_probs)
    all_labels = np.stack(all_labels)

    cal_probs, test_probs, cal_labels, test_labels = train_test_split(
        all_probs, all_labels, test_size=0.5, random_state=42
    )
    N_test = test_probs.shape[0]
    print(f"\nCalibration: {cal_probs.shape[0]}, Test: {N_test}")

    # ------------------------------------------------------------------
    # Class prevalence in test set
    # ------------------------------------------------------------------
    test_positives = test_labels.sum(axis=0)
    prevalence_pct = (test_positives / N_test) * 100.0

    # ------------------------------------------------------------------
    # Multi‑α evaluation
    # ------------------------------------------------------------------
    alphas = [0.01, 0.05, 0.10, 0.15, 0.20]           # target miscoverage
    target_coverages = [1 - a for a in alphas]

    methods_order = ["ECOT", "THR", "Mondrian", "APS (legacy)", "RAPS (legacy)", "Fixed Threshold"]
    # Store overall metrics per method across alphas
    results = {
        name: {
            'coverage': [],
            'avg_set_size': [],
            'f1': []
        } for name in methods_order
    }

    # For printing the class‑conditional table we keep the metrics at α=0.1
    alpha01_metrics = {}

    for alpha in alphas:
        print(f"\n{'='*80}")
        print(f"⚙️  Evaluating for α = {alpha:.2f} (target coverage {1-alpha:.0%})")
        print('='*80)

        # 1. Fixed thresholds
        fixed_thresh = np.array([CLASS_THRESHOLDS[name] for name in DEFAULT_CLASS_NAMES])
        fixed_pred = predict_sets(test_probs, fixed_thresh)
        fixed_met = evaluate_prediction_sets(fixed_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # 2. ECOT
        ecot_thresh = fit_ecot(cal_probs, cal_labels, alpha)
        ecot_pred = predict_sets(test_probs, ecot_thresh)
        ecot_met = evaluate_prediction_sets(ecot_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # 3. THR
        thr_thresh = fit_thr(cal_probs, cal_labels, alpha)
        thr_pred = predict_sets(test_probs, thr_thresh)
        thr_met = evaluate_prediction_sets(thr_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # 4. Mondrian
        mondrian_thresh = fit_mondrian(cal_probs, cal_labels, alpha)
        mondrian_pred = predict_mondrian(test_probs, mondrian_thresh)
        mondrian_met = evaluate_prediction_sets(mondrian_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # 5. APS
        q_aps = old_fit_aps(cal_probs, cal_labels, alpha)
        aps_pred = old_predict_aps(test_probs, q_aps)
        aps_met = evaluate_prediction_sets(aps_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # 6. RAPS
        lam_raps = 0.01
        k_reg_raps = 1
        q_raps = old_fit_raps(cal_probs, cal_labels, alpha, lam=lam_raps, k_reg=k_reg_raps)
        raps_pred = old_predict_raps(test_probs, q_raps, lam_raps, k_reg_raps)
        raps_met = evaluate_prediction_sets(raps_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # Collect metrics & compute CIs
        current_methods = [
            ("ECOT", ecot_met, ecot_pred),
            ("THR", thr_met, thr_pred),
            ("Mondrian", mondrian_met, mondrian_pred),
            ("APS (legacy)", aps_met, aps_pred),
            ("RAPS (legacy)", raps_met, raps_pred),
            ("Fixed Threshold", fixed_met, fixed_pred),
        ]

        for name, met, pred_set in current_methods:

            results[name]['coverage'].append(met['coverage'])
            results[name]['avg_set_size'].append(met['avg_set_size'])
            results[name]['f1'].append(met['set_f1'])

        # Store full metrics for α=0.1 (the reference table)
        if alpha == 0.1:
            alpha01_metrics = {name: met for name, met, _ in current_methods}
            # Save thresholds for later single‑image inference (original behaviour)
            save_dir = "conformal_params"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "fixed_thresh.npy"), fixed_thresh)
            np.save(os.path.join(save_dir, "ecot_thresh.npy"), ecot_thresh)
            np.save(os.path.join(save_dir, "thr_thresh.npy"), thr_thresh)
            np.save(os.path.join(save_dir, "mondrian_thresh.npy"), mondrian_thresh)
            np.save(os.path.join(save_dir, "q_aps.npy"), np.array(q_aps))
            np.save(os.path.join(save_dir, "q_raps.npy"), np.array(q_raps))
            np.save(os.path.join(save_dir, "lam_raps.npy"), np.array(lam_raps))
            np.save(os.path.join(save_dir, "k_reg_raps.npy"), np.array(k_reg_raps))
            np.save(os.path.join(save_dir, "alpha.npy"), np.array(alpha))
            print("\n💾 Conformal parameters for α=0.1 saved to 'conformal_params/'.")

    # ------------------------------------------------------------------
    # Print detailed class‑conditional coverage table for α = 0.1
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("📊 Class‑conditional coverage table for α = 0.1")
    print("=" * 120)

    # Build header
    header = f"{'Class':<22} {'Test Positives':>14} {'Prevalence (%)':>14}"
    short_names = ["ECOT", "THR", "Mondrian", "APS", "RAPS", "Fixed"]
    for sn in short_names:
        header += f" {sn:>8}"
    print(header)
    print("-" * (22 + 14 + 14 + 8 * len(short_names)))

    for idx, cls in enumerate(DEFAULT_CLASS_NAMES):
        row = f"{cls:<22} {int(test_positives[idx]):14d} {prevalence_pct[idx]:14.2f}"
        for name, sn in zip(methods_order, short_names):
            cc = alpha01_metrics[name]['class_conditional_coverage'].get(cls, 1.0)
            row += f" {cc:8.4f}"
        print(row)

    print("\nWorst class‑conditional coverage at α=0.1:")
    for name in methods_order:
        cls, cov = alpha01_metrics[name]['worst_class_coverage']
        print(f"  {name:<20}: {cls} (coverage={cov:.4f})")

    # ------------------------------------------------------------------
    # Summary table for all α (overall metrics) – now without CIs
    # ------------------------------------------------------------------
    print("📊 Overall metrics across α levels")
    print("=" * 90)
    for name in methods_order:
        print(f"\n--- {name} ---")
        print(f"{'α':>6}  {'Target Cov':>11}  {'Coverage':>10}  {'Avg Set Size':>13}  {'F1':>7}")
        for a, tc, cov, sz, f1 in zip(
            alphas, target_coverages,
            results[name]['coverage'],
            results[name]['avg_set_size'],
            results[name]['f1']
        ):
            print(f"{a:6.2f}  {tc:11.2%}  {cov:10.4f}  {sz:13.2f}  {f1:7.4f}")

    # ------------------------------------------------------------------
    # Plot 1: Coverage‑Efficiency Frontier (Achieved Coverage vs Set Size)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', 'D', '^', 'v', '<']

    for (name, color, marker) in zip(methods_order, colors, markers):
        cov = results[name]['coverage']
        sz = results[name]['avg_set_size']
        plt.plot(sz, cov, marker=marker, color=color, label=name, linewidth=2, markersize=8)

    # Add horizontal lines for typical target coverages
    for tc in [0.80, 0.85, 0.90, 0.95, 0.99]:
        plt.axhline(y=tc, color='gray', linestyle=':', alpha=0.5)
        plt.text(plt.xlim()[1] * 0.95, tc, f'{tc:.0%}', va='center', fontsize=8, color='gray')

    plt.xlabel('Average Set Size')
    plt.ylabel('Achieved Coverage')
    plt.title('Coverage‑Efficiency Frontier')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('coverage_efficiency_frontier.png', dpi=300)
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: Target Coverage vs Average Set Size
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for name, color, marker in zip(methods_order, colors, markers):
        plt.plot(target_coverages, results[name]['avg_set_size'],
                 marker=marker, color=color, label=name, linewidth=2, markersize=8)

    plt.xlabel('Target Coverage (1−α)')
    plt.ylabel('Average Set Size')
    plt.title('Set Size Required to Achieve Target Coverage')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_coverage_vs_setsize.png', dpi=300)
    plt.show()

    print("\n🏁 Multi‑α evaluation and plotting complete.")