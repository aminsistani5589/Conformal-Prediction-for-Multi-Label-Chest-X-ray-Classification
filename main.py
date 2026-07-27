"""
main.py
-------
End‑to‑end evaluation of CheXNet with several conformal prediction methods,
including the new Mondrian (label‑conditional) approach.

Usage:
    python main.py
"""

import torch
import glob
import os
import tempfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
    predict_sets,              # generic threshold‑based prediction
    evaluate_prediction_sets,
)
# Legacy APS / RAPS (kept for completeness, though they are broken for multi‑label)
from APS import fit_aps as old_fit_aps, predict_sets_aps as old_predict_aps
from RAPS import fit_raps as old_fit_raps, predict_sets_raps as old_predict_raps
# Mondrian conformal (label‑conditional)
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
    """Convert a torch Tensor or numpy array to a numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def label_to_vector(label_array, class_names=DEFAULT_CLASS_NAMES):
    """Convert a list of disease strings to a binary label vector."""
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


# ----------------------------------------------------------------------
# Robust parquet reading (unchanged logic, slightly cleaned)
# ----------------------------------------------------------------------
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
    """Extract raw image bytes from a variety of storage formats."""
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

    # Load model once
    model = load_chexnet_model(config.CKPT_PATH, config.N_CLASSES)

    for pq_file in parquet_files:
        print(f"\n--- Processing {pq_file} (total images so far: {total_processed}) ---")
        try:
            df = read_parquet_with_fallback(pq_file)
            print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")

            if 'image' not in df.columns:
                print("No 'image' column, skipping.")
                continue

            # Find label column
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

                    # Run model
                    probs = run_prediction(model, image_path)
                    if probs.ndim == 2:
                        probs = probs.squeeze(0)

                    # Ground truth label
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

    # ------------------------------------------------------------------
    # Evaluate all methods on the collected data
    # ------------------------------------------------------------------
    if not all_probs:
        print("\nNo data collected. Exiting.")
        exit()

    all_probs = np.stack(all_probs)
    all_labels = np.stack(all_labels)

    # Split into calibration and test
    cal_probs, test_probs, cal_labels, test_labels = train_test_split(
        all_probs, all_labels, test_size=0.5, random_state=42
    )
    print(f"\nCalibration: {cal_probs.shape[0]}, Test: {test_probs.shape[0]}")

    alpha = 0.1   # target miscoverage

    # 1. Fixed (expert) thresholds
    fixed_thresh = np.array([CLASS_THRESHOLDS[name] for name in DEFAULT_CLASS_NAMES])
    fixed_pred = predict_sets(test_probs, fixed_thresh)
    fixed_met = evaluate_prediction_sets(fixed_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

    # 2. ECOT
    ecot_thresh = fit_ecot(cal_probs, cal_labels, alpha)
    ecot_pred = predict_sets(test_probs, ecot_thresh)
    ecot_met = evaluate_prediction_sets(ecot_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

    # 3. THR (label‑wise conformal)
    thr_thresh = fit_thr(cal_probs, cal_labels, alpha)
    thr_pred = predict_sets(test_probs, thr_thresh)
    thr_met = evaluate_prediction_sets(thr_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

    # 4. Mondrian (label‑conditional)
    mondrian_thresh = fit_mondrian(cal_probs, cal_labels, alpha)
    mondrian_pred = predict_mondrian(test_probs, mondrian_thresh)
    mondrian_met = evaluate_prediction_sets(mondrian_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

    # 5 & 6. Legacy APS / RAPS
    q_aps = old_fit_aps(cal_probs, cal_labels, alpha)
    aps_pred = old_predict_aps(test_probs, q_aps)
    aps_met = evaluate_prediction_sets(aps_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

    q_raps = old_fit_raps(cal_probs, cal_labels, alpha, lam=0.01, k_reg=1)
    raps_pred = old_predict_raps(test_probs, q_raps, lam=0.01, k_reg=1)
    raps_met = evaluate_prediction_sets(raps_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"📊 Full comparison (α = {alpha:.0%})")
    print("=" * 80)

    methods = [
        ("Fixed Threshold", fixed_met),
        ("ECOT", ecot_met),
        ("THR", thr_met),
        ("Mondrian", mondrian_met),
        ("APS (legacy)", aps_met),
        ("RAPS (legacy)", raps_met),
    ]

    # Summary table
    print(f"{'Method':<20} {'Coverage':>8} {'Avg Size':>10} {'F1':>7}")
    print("-" * 50)
    for name, met in methods:
        print(f"{name:<20} {met['coverage']:8.4f} {met['avg_set_size']:10.2f} {met['set_f1']:7.4f}")

    # Class‑conditional coverage
    print("\nClass‑conditional coverage:")
    header = f"{'Class':<22}"
    for name, _ in methods:
        header += f"{name.split()[0]:>10}"
    print(header)
    print("-" * (22 + 10 * len(methods)))
    for cls in DEFAULT_CLASS_NAMES:
        row = f"{cls:<22}"
        for _, met in methods:
            row += f"{met['class_conditional_coverage'].get(cls, 1.0):10.4f}"
        print(row)

    # Worst class per method
    print("\nWorst class‑conditional coverage:")
    for name, met in methods:
        cls, cov = met['worst_class_coverage']
        print(f"  {name:<20}: {cls} (coverage={cov:.4f})")

    print("\n🏁 Evaluation complete.")