# main.py 
import torch
import config
from xray_model.model_loader import load_chexnet_model
from xray_model.predictor import run_prediction, interpret_probabilities
from utils.data_mapping import format_findings_for_prompt
import glob
import os
import tempfile
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Conformal prediction modules
# ----------------------------------------------------------------------
from THR import (
    fit_thr,
    fit_ecot,
    predict_sets,              # threshold‑based prediction (used for THR/ECOT/Fixed)
    evaluate_prediction_sets,  # common evaluation function
)

# Original (broken) APS / RAPS – included for reference
from APS import fit_aps as old_fit_aps, predict_sets_aps as old_predict_aps
from RAPS import fit_raps as old_fit_raps, predict_sets_raps as old_predict_raps

# ----------------------------------------------------------------------
# Constants & utilities 
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
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")

def label_to_vector(label_array, class_names=DEFAULT_CLASS_NAMES):
    vector = np.zeros(len(class_names), dtype=np.int32)
    for name in label_array:
        name = name.strip()
        if name == "No Finding":
            continue
        try:
            idx = class_names.index(name)
            vector[idx] = 1
        except ValueError:
            print(f"Warning: disease '{name}' not found in class list; ignoring.")
    return vector

def read_parquet_with_fallback(file_path):
    methods = [
        lambda: pd.read_parquet(file_path),
        lambda: pq.read_table(file_path).to_pandas(),
        lambda: _read_parquet_in_batches(file_path),
        lambda: _read_parquet_with_column_conversion(file_path),
    ]
    for i, method in enumerate(methods, 1):
        try:
            print(f"  Trying method {i} to read parquet file...")
            df = method()
            if df is not None and len(df) > 0:
                print(f"  ✓ Successfully read with method {i}")
                return df
        except Exception as e:
            print(f"  ✗ Method {i} failed: {type(e).__name__}: {str(e)[:100]}")
            continue
    raise ValueError(f"All methods failed to read {file_path}")

def _read_parquet_in_batches(file_path):
    parquet_file = pq.ParquetFile(file_path)
    dfs = []
    for batch in parquet_file.iter_batches(batch_size=100):
        batch_df = batch.to_pandas()
        dfs.append(batch_df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def _read_parquet_with_column_conversion(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    if 'image' in df.columns:
        df['image'] = df['image'].apply(lambda x: extract_image_bytes_from_nested(x))
    return df

def extract_image_bytes_from_nested(cell):
    if cell is None:
        return None
    if isinstance(cell, bytes):
        return cell
    if isinstance(cell, dict):
        if 'bytes' in cell:
            return cell['bytes']
        for key in ['image', 'data', 'img', 'binary']:
            if key in cell:
                return extract_image_bytes_from_nested(cell[key])
    if isinstance(cell, (list, tuple)):
        if len(cell) > 0:
            if isinstance(cell[0], (int, np.integer)):
                try:
                    return bytes(cell)
                except:
                    pass
    if hasattr(cell, 'as_py'):
        try:
            return cell.as_py()
        except:
            pass
    if isinstance(cell, str):
        return cell
    return cell

def extract_image_bytes(img_cell):
    if isinstance(img_cell, dict):
        return img_cell.get('bytes') or img_cell.get('path')
    elif isinstance(img_cell, bytes):
        return img_cell
    elif isinstance(img_cell, str):
        return img_cell
    elif hasattr(img_cell, 'as_py'):
        try:
            return img_cell.as_py()
        except:
            return None
    return None

def save_bytes_to_temp_png(img_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_path = temp_file.name
    temp_file.close()
    with open(temp_path, "wb") as f:
        f.write(img_bytes)
    return temp_path


# ======================================================================
# Main evaluation
# ======================================================================
if __name__ == '__main__':
    PARQUET_DIR = config.PARQUET_DIR
    parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")

    all_probs_list = []
    all_labels_list = []
    total_processed = 0

    model = load_chexnet_model(config.CKPT_PATH, config.N_CLASSES)

    # --- Data loading (unchanged) ---
    for pq_file in parquet_files:
        print(f"\n--- Opening: {pq_file} | Total processed so far: {total_processed} images ---")
        try:
            df = read_parquet_with_fallback(pq_file)
            print(f"Successfully loaded {len(df)} rows")
            print(f"Columns: {df.columns.tolist()}")

            if 'image' not in df.columns:
                print(f"Skipping {pq_file}: no 'image' column found.")
                continue

            candidate_label_cols = ['label', 'labels', 'targets', 'finding_labels']
            label_col = None
            for col in candidate_label_cols:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                for col in df.columns:
                    if 'label' in col.lower():
                        label_col = col
                        print(f"Found label column by pattern: '{col}'")
                        break
            if label_col is None:
                print(f"Skipping {pq_file}: no suitable label column found (candidates: {candidate_label_cols}).")
                continue

            print(f"Using label column: '{label_col}'")
            file_processed = 0

            for idx, row in df.iterrows():
                img_cell = row['image']
                temp_image_path = None
                image_path = None
                try:
                    if isinstance(img_cell, dict):
                        if 'bytes' in img_cell:
                            extracted = img_cell['bytes']
                        else:
                            print(f"Row {idx}: dict without 'bytes' key. Keys: {img_cell.keys()}")
                            continue
                    else:
                        extracted = extract_image_bytes(img_cell)

                    if extracted is None:
                        print(f"Row {idx}: could not extract image data. Type: {type(img_cell)}")
                        continue

                    if isinstance(extracted, bytes):
                        temp_image_path = save_bytes_to_temp_png(extracted)
                        image_path = temp_image_path
                    elif isinstance(extracted, str):
                        if os.path.exists(extracted):
                            image_path = extracted
                        else:
                            print(f"Row {idx}: file not found -> {extracted}")
                            continue
                    else:
                        print(f"Row {idx}: unsupported image type -> {type(extracted)}")
                        continue

                    raw_probabilities = run_prediction(model, image_path)
                    if raw_probabilities.ndim == 2:
                        raw_probabilities = raw_probabilities.squeeze(0)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    raw_label = row[label_col]
                    if isinstance(raw_label, np.ndarray):
                        raw_label = raw_label.tolist()
                    label = label_to_vector(raw_label)

                    if len(label) != config.N_CLASSES:
                        print(f"Row {idx}: label length {len(label)} != {config.N_CLASSES}; skipping.")
                        continue

                    all_probs_list.append(to_numpy(raw_probabilities))
                    all_labels_list.append(label)
                    total_processed += 1
                    file_processed += 1

                    active_indices = np.where(label == 1)[0]
                    active_names = [DEFAULT_CLASS_NAMES[i] for i in active_indices]
                    print(f"   ✅ [{total_processed}] row {idx}: OK   active: {active_names if active_names else 'No Finding'}")

                except Exception as e:
                    import traceback
                    print(f"Row {idx}: {type(e).__name__} → {e}")
                    traceback.print_exc()
                finally:
                    if temp_image_path and os.path.exists(temp_image_path):
                        try:
                            os.remove(temp_image_path)
                        except:
                            pass

            print(f"--- Finished {pq_file}: {file_processed} images processed ---")

        except Exception as e:
            print(f"Failed to read parquet file {pq_file}: {type(e).__name__} → {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------
    # Evaluation & Comparison (all methods)
    # ---------------------------------------------------------------
    if all_probs_list and all_labels_list:
        all_probs = np.stack(all_probs_list)
        all_labels = np.stack(all_labels_list)
        print(f"\nTotal processed samples: {len(all_probs_list)}")

        cal_probs, test_probs, cal_labels, test_labels = train_test_split(
            all_probs, all_labels, test_size=0.5, random_state=42
        )
        print(f"Calibration set: {cal_probs.shape[0]} samples")
        print(f"Test set: {test_probs.shape[0]} samples")

        alpha = 0.1   # 90% target coverage

        # Fixed Threshold
        fixed_thresh = np.array([CLASS_THRESHOLDS[name] for name in DEFAULT_CLASS_NAMES])
        fixed_pred = predict_sets(test_probs, fixed_thresh)
        fixed_met = evaluate_prediction_sets(fixed_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # ECOT
        ecot_thresh = fit_ecot(cal_probs, cal_labels, alpha)
        ecot_pred = predict_sets(test_probs, ecot_thresh)
        ecot_met = evaluate_prediction_sets(ecot_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # THR (label‑wise conformal)
        thr_thresh = fit_thr(cal_probs, cal_labels, alpha)
        thr_pred = predict_sets(test_probs, thr_thresh)
        thr_met = evaluate_prediction_sets(thr_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # Original APS (broken for multi‑label)
        q_aps_old = old_fit_aps(cal_probs, cal_labels, alpha)
        aps_old_pred = old_predict_aps(test_probs, q_aps_old)
        aps_old_met = evaluate_prediction_sets(aps_old_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # Original RAPS (broken for multi‑label)
        q_raps_old = old_fit_raps(cal_probs, cal_labels, alpha, lam=0.01, k_reg=1)
        raps_old_pred = old_predict_raps(test_probs, q_raps_old, lam=0.01, k_reg=1)
        raps_old_met = evaluate_prediction_sets(raps_old_pred, test_labels, DEFAULT_CLASS_NAMES, alpha)

        # ---- Display results ----
        print("\n" + "="*80)
        print("📊 Full comparison: Fixed, ECOT, THR, old APS/RAPS (broken), calibrated APS/RAPS")
        print("="*80)
        print(f"Target coverage (1-α): {1-alpha:.0%}")
        print()

        methods = [
            ("Fixed Threshold (Baseline 1)",   fixed_met),
            ("ECOT (Baseline 2)",              ecot_met),
            ("THR (Conformal)",                thr_met),
            ("APS",              aps_old_met),
            ("RAPS",             raps_old_met)
        ]

        # Summary table
        print(f"{'Method':<30} {'Coverage':>8} {'Avg Set Size':>13} {'Set F1':>7}")
        print("-"*60)
        for name, met in methods:
            print(f"{name:<30} {met['coverage']:8.4f} {met['avg_set_size']:13.2f} {met['set_f1']:7.4f}")

        # Class‑conditional coverage
        print("\n" + "-"*80)
        print("Class‑conditional coverage:")
        header = f"{'Class':<25}"
        for name, _ in methods:
            short = name.split('(')[0].strip().replace(' ', '_')
            header += f"{short:>10}"
        print(header)
        print("-"*80)
        for cls in DEFAULT_CLASS_NAMES:
            row_str = f"{cls:<25}"
            for _, met in methods:
                cov_val = met['class_conditional_coverage'].get(cls, 1.0)
                row_str += f"{cov_val:10.4f}"
            print(row_str)

        # Worst class per method
        print("\nWorst class‑conditional coverage:")
        for name, met in methods:
            worst_cls, worst_cov = met['worst_class_coverage']
            print(f"  {name:<30}: {worst_cls} (cov={worst_cov:.4f})")

        print("\nNote: Old APS/RAPS use `1 - q_hat` threshold, leading to set size = 1.")
        print("Calibrated versions use `q_hat` directly and honour the multi‑label structure.")

    else:
        print("\nNo valid samples with labels were processed — cannot compute metrics.")

    print("\n🏁 Done processing all parquet images.")