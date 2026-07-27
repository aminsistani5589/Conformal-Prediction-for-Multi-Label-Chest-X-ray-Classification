"""
mondrian.py
-----------
Mondrian (label‑conditional) conformal prediction for multi‑label classification.

For each disease class k, we compute a threshold q_k using the calibration
scores of *only those instances where the class is truly positive*. This
guarantees conditional coverage:
    P( Y_k ∈ C(X) | Y_k = 1 ) ≥ 1 - α.
"""

import numpy as np

def fit_mondrian(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.1):
    """
    Compute class‑specific conformal thresholds.

    Parameters
    ----------
    cal_probs : np.ndarray, shape (n_cal, n_classes)
        Softmax probabilities from the model.
    cal_labels : np.ndarray, shape (n_cal, n_classes)
        Binary ground‑truth labels (multi‑label, can have multiple 1's).
    alpha : float
        Desired miscoverage rate (default 0.1 → 90% coverage).

    Returns
    -------
    thresholds : np.ndarray, shape (n_classes,)
        Threshold values for each class. A test class k is included in the
        prediction set if 1 - prob_k ≤ thresholds[k].
    """
    n_cal, n_classes = cal_probs.shape
    thresholds = np.ones(n_classes)  # default: never include

    for k in range(n_classes):
        # calibration scores for this class (only positive instances)
        pos_mask = cal_labels[:, k] == 1
        if pos_mask.sum() == 0:
            # no positive examples – threshold stays 1 (never includes)
            continue
        scores_k = 1.0 - cal_probs[pos_mask, k]
        n_k = len(scores_k)

        # Finite‑sample corrected quantile level
        level = min(1.0, (n_k + 1) * (1 - alpha) / n_k)
        q_k = np.quantile(scores_k, level, method='higher')
        thresholds[k] = q_k

    return thresholds


def predict_mondrian(test_probs: np.ndarray, thresholds: np.ndarray):
    """
    Generate prediction sets using Mondrian thresholds.

    Parameters
    ----------
    test_probs : np.ndarray, shape (n_test, n_classes)
        Model probabilities for test instances.
    thresholds : np.ndarray, shape (n_classes,)
        Class‑wise thresholds from fit_mondrian.

    Returns
    -------
    pred_sets : np.ndarray, shape (n_test, n_classes)
        Binary matrix where 1 indicates the class is in the prediction set.
    """
    scores = 1.0 - test_probs   # nonconformity: low prob → high score
    return (scores <= thresholds).astype(int)