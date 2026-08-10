"""
test_image_hardcoded.py
----------------------
Loads saved conformal parameters and predicts on a single chest X‑ray.
Just set IMAGE_PATH below and run:
    python test_image_hardcoded.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import config
from xray_model.model_loader import load_chexnet_model
from xray_model.predictor import run_prediction

# Conformal modules
from THR import predict_sets
from APS import predict_sets_aps
from RAPS import predict_sets_raps
from mondrian import predict_mondrian

# ========== SETTINGS (change these as you wish) ==========
IMAGE_PATH = r"D:\\projects\\benchmarking-cp-medical-noise-main\\ChestX-ray14\\images\\00000013_030.png"
PARAMS_DIR = "conformal_params"
# =========================================================

DEFAULT_CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def to_numpy(arr):
    import torch
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def load_params(params_dir="conformal_params"):
    fixed = np.load(os.path.join(params_dir, "fixed_thresh.npy"))
    ecot = np.load(os.path.join(params_dir, "ecot_thresh.npy"))
    thr = np.load(os.path.join(params_dir, "thr_thresh.npy"))
    mondrian = np.load(os.path.join(params_dir, "mondrian_thresh.npy"))
    q_aps = float(np.load(os.path.join(params_dir, "q_aps.npy")))
    q_raps = float(np.load(os.path.join(params_dir, "q_raps.npy")))
    lam = float(np.load(os.path.join(params_dir, "lam_raps.npy")))
    k_reg = float(np.load(os.path.join(params_dir, "k_reg_raps.npy")))
    alpha = float(np.load(os.path.join(params_dir, "alpha.npy")))
    return fixed, ecot, thr, mondrian, q_aps, q_raps, lam, k_reg, alpha

if __name__ == "__main__":
    # Load model and parameters
    model = load_chexnet_model(config.CKPT_PATH, config.N_CLASSES)
    fixed, ecot, thr, mondrian, q_aps, q_raps, lam, k_reg, alpha = load_params(PARAMS_DIR)

    # Run prediction
    probs = run_prediction(model, IMAGE_PATH)
    if probs.ndim == 2:
        probs = probs.squeeze(0)
    probs = to_numpy(probs)

    print(f"Model probabilities: {dict(zip(DEFAULT_CLASS_NAMES, np.round(probs, 4)))}")

    probs_batch = probs[np.newaxis, ...]

    # Predict masks (boolean array per class)
    fixed_mask = predict_sets(probs_batch, fixed)[0]
    ecot_mask = predict_sets(probs_batch, ecot)[0]
    thr_mask = predict_sets(probs_batch, thr)[0]
    mondrian_mask = predict_mondrian(probs_batch, mondrian)[0]
    aps_mask = predict_sets_aps(probs_batch, q_aps)[0]
    raps_mask = predict_sets_raps(probs_batch, q_raps, lam, k_reg)[0]

    # Convert masks to class name lists
    def mask_to_names(mask):
        indices = np.where(mask)[0]
        return [DEFAULT_CLASS_NAMES[i] for i in indices]

    sets = {
        "Fixed Threshold": mask_to_names(fixed_mask),
        "ECOT": mask_to_names(ecot_mask),
        "THR": mask_to_names(thr_mask),
        "Mondrian": mask_to_names(mondrian_mask),
        "APS": mask_to_names(aps_mask),
        "RAPS": mask_to_names(raps_mask),
    }

    print(f"\nPrediction sets (α = {alpha:.0%}):")
    for name, s in sets.items():
        print(f"  {name:<20}: {s if s else 'Empty'}")

    # Show image with prediction sets only
    img = mpimg.imread(IMAGE_PATH)
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    lines = []
    for name, labels in sets.items():
        lines.append(f"{name}: {', '.join(labels) if labels else 'Empty set'}")
    text = "\n".join(lines)
    plt.gcf().text(0.5, 0.02, text, ha='center', fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.show()