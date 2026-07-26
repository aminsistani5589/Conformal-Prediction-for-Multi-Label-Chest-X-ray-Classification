# conformal.py
import numpy as np
from sklearn.metrics import f1_score

def fit_thr(cal_probs, cal_labels, alpha=0.1):
    """
    Fit class-wise thresholds using conformal thresholding.

    Parameters:
        cal_probs: array of shape (n_cal, n_classes)
            Predicted probabilities on calibration set.
        cal_labels: array of shape (n_cal, n_classes)
            Binary true labels on calibration set.
        alpha: float
            Target error level. Desired coverage is about (1 - alpha).

    Returns:
        thresholds: array of shape (n_classes,)
            One threshold per class.
    """
    n_cal, n_classes = cal_labels.shape
    thresholds = np.zeros(n_classes)
    
    for j in range(n_classes):
        y = cal_labels[:, j]
        p = cal_probs[:, j]
        
        # Nonconformity score:
        # if true label is 1 -> score = 1 - p
        # if true label is 0 -> score = p
        scores = np.where(y == 1, 1.0 - p, p)
        
        # Rank-corrected quantile level for conformal prediction
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)  # Prevent quantile level > 1
        q = np.quantile(scores, q_level)
        
        # Inclusion threshold = 1 - quantile
        thresholds[j] = 1.0 - q
        
    return thresholds


def fit_ecot(cal_probs, cal_labels, alpha=0.1):
    """
    Empirical Coverage-Optimized Thresholding (ECOT) - non-conformal baseline.

    For each class, use only positive calibration samples and choose a threshold
    such that about (1 - alpha) of positive samples are included.

    Parameters:
        cal_probs: array of shape (n_cal, n_classes)
        cal_labels: array of shape (n_cal, n_classes)
        alpha: float

    Returns:
        thresholds: array of shape (n_classes,)
    """
    n_cal, n_classes = cal_labels.shape
    thresholds = np.zeros(n_classes)
    
    for j in range(n_classes):
        pos_mask = cal_labels[:, j] == 1
        if pos_mask.sum() == 0:
            # If this class does not appear in calibration, never predict it
            thresholds[j] = 1.0
            continue
        pos_probs = cal_probs[pos_mask, j]
        # Choose the alpha-quantile of positive probabilities
        thresholds[j] = np.quantile(pos_probs, alpha)
    
    return thresholds


def predict_sets(probs, thresholds):
    """
    Convert probabilities into binary prediction sets using class-wise thresholds.

    Parameters:
        probs: array of shape (n_samples, n_classes)
        thresholds: array of shape (n_classes,)

    Returns:
        Binary array of shape (n_samples, n_classes)
    """
    return (probs >= thresholds).astype(int)


def evaluate_prediction_sets(pred_sets, true_labels, class_names=None, alpha=None):
    """
    Evaluate prediction sets.

    Parameters:
        pred_sets: binary array of shape (n_samples, n_classes)
        true_labels: binary array of shape (n_samples, n_classes)
        class_names: optional list of class names
        alpha: optional target error level

    Returns:
        Dictionary containing:
            - coverage
            - avg_set_size
            - set_precision
            - set_recall
            - set_f1
            - class_conditional_coverage
            - worst_class_coverage
            - target_alpha
    """
    n_samples, n_classes = true_labels.shape
    true_sets = [set(np.where(true_labels[i])[0]) for i in range(n_samples)]
    pred_sets_list = [set(np.where(pred_sets[i])[0]) for i in range(n_samples)]
    
    # Coverage: true_labels subset of pred_sets
    coverage = np.mean([true_sets[i].issubset(pred_sets_list[i]) for i in range(n_samples)])
    
    # Average set size
    set_sizes = [len(s) for s in pred_sets_list]
    avg_set_size = np.mean(set_sizes)
    
    # Set-based precision, recall, F1 
    precisions = []
    recalls = []
    for i in range(n_samples):
        if len(pred_sets_list[i]) == 0:
            prec = 0.0 if len(true_sets[i]) > 0 else 1.0
        else:
            prec = len(true_sets[i] & pred_sets_list[i]) / len(pred_sets_list[i])
        if len(true_sets[i]) == 0:
            rec = 1.0 if len(pred_sets_list[i]) == 0 else 0.0
        else:
            rec = len(true_sets[i] & pred_sets_list[i]) / len(true_sets[i])
        precisions.append(prec)
        recalls.append(rec)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    if avg_precision + avg_recall > 0:
        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        avg_f1 = 0.0
    
    # Class-conditional coverage
    class_cond_cov = {}
    for j in range(n_classes):
        mask = true_labels[:, j] == 1
        if mask.sum() > 0:
            class_cond_cov[class_names[j] if class_names else str(j)] = \
                np.mean(pred_sets[mask, j])
        else:
            class_cond_cov[class_names[j] if class_names else str(j)] = 1.0
    
    worst_class = min(class_cond_cov, key=class_cond_cov.get)
    worst_cov = class_cond_cov[worst_class]
    
    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "set_precision": avg_precision,
        "set_recall": avg_recall,
        "set_f1": avg_f1,
        "class_conditional_coverage": class_cond_cov,
        "worst_class_coverage": (worst_class, worst_cov),
        "target_alpha": alpha
    }