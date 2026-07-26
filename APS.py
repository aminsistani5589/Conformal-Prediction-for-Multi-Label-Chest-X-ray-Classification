import numpy as np

def _aps_nonconformity(probs, labels):
    """
    Nonconformity score for multi-label APS.
    For an image with multiple positive labels, we consider the
    *least probable* positive label (i.e., the one with the smallest
    predicted probability among the true classes).
    The score is the total probability mass of all classes that are
    *strictly more likely* than that hardest positive label.
    """
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]

    true_set = set(np.where(labels == 1)[0])
    if len(true_set) == 0:
        return 0.0

    cumsum = 0.0
    for r, idx in enumerate(sorted_idx):
        cumsum += sorted_probs[r]          # include this class
        if idx in true_set:
            true_set.remove(idx)
            if len(true_set) == 0:         # last positive label found
                # Score = sum of probabilities of all classes ranked
                # *before* this hardest positive label.
                return cumsum - sorted_probs[r]

    return 1.0   # fallback, should not happen


def fit_aps(cal_probs, cal_labels, alpha=0.1):
    """
    Calibrate APS.
    Returns q_hat: the (1-alpha) quantile of the nonconformity scores.
    """
    n = len(cal_probs)
    scores = np.array([_aps_nonconformity(cal_probs[i], cal_labels[i]) for i in range(n)])

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, min(q_level, 1.0))
    return q_hat


def predict_sets_aps(probs, q_hat):
    """
    Generate prediction sets using APS.
    Classes are added in descending order of probability.
    A class is included if the cumulative probability mass of all
    *more likely* classes (i.e., the cumsum before adding it) is < q_hat.
    """
    n_samples, n_classes = probs.shape
    pred_sets = np.zeros_like(probs, dtype=np.int32)

    for i in range(n_samples):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        for idx in sorted_idx:
            if cumsum < q_hat:
                pred_sets[i, idx] = 1
                cumsum += probs[i, idx]
            else:
                break
    return pred_sets