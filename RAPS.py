import numpy as np

def _raps_nonconformity(probs, labels, lam, k_reg):
    """
    Nonconformity score for multi-label RAPS.
    We locate the *last* (hardest) true positive label.
    The score is:
        cumulative probability up to and including that label
      + lam * max(0, rank_of_that_label - k_reg)
    """
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]

    true_set = set(np.where(labels == 1)[0])
    if len(true_set) == 0:
        return 0.0

    cumsum = 0.0
    r_last = -1
    for r, idx in enumerate(sorted_idx):
        cumsum += sorted_probs[r]
        if idx in true_set:
            true_set.remove(idx)
            if len(true_set) == 0:
                r_last = r
                break

    score = cumsum  # total prob up to and including the last true label
    penalty = lam * max(0, (r_last + 1) - k_reg)
    return score + penalty


def fit_raps(cal_probs, cal_labels, alpha=0.1, lam=0.01, k_reg=1):
    """
    Calibrate RAPS.
    Returns q_hat: the (1-alpha) quantile of the regularised scores.
    """
    n = len(cal_probs)
    scores = np.array([
        _raps_nonconformity(cal_probs[i], cal_labels[i], lam, k_reg)
        for i in range(n)
    ])

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(scores, min(q_level, 1.0))
    return q_hat


def predict_sets_raps(probs, q_hat, lam=0.01, k_reg=1):
    """
    Generate prediction sets using RAPS.
    Classes are added in descending order of probability.
    A class is included only if the *regularised cumulative sum*
    (probability sum including this class + penalty for the new size)
    does NOT exceed q_hat.
    """
    n_samples, n_classes = probs.shape
    pred_sets = np.zeros_like(probs, dtype=np.int32)

    for i in range(n_samples):
        sorted_idx = np.argsort(-probs[i])
        cumsum = 0.0
        set_size = 0
        for idx in sorted_idx:
            new_size = set_size + 1
            penalty = lam * max(0, new_size - k_reg)
            # Check if we can add this class without exceeding q_hat
            if cumsum + probs[i, idx] + penalty <= q_hat:
                pred_sets[i, idx] = 1
                cumsum += probs[i, idx]
                set_size = new_size
            else:
                break
    return pred_sets
