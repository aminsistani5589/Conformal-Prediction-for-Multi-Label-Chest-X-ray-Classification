## 📌 Overview

Standard multi-label classifiers output a fixed set of labels using a threshold (e.g., 0.5). This ignores model uncertainty and offers no control over false positives / false negatives.  
This repository explores **conformal prediction** methods that turn any trained classifier into a **set predictor** with a user‑specified marginal coverage guarantee.

We experiment on the **CheXpert** dataset, comparing a fixed‑threshold baseline against several state‑of‑the‑art conformal approaches:

| Method | Description |
|--------|-------------|
| **Fixed Threshold** | Classical baseline – labels with probability > 0.5 |
| **THR** (Threshold) | Conformal calibration of a single global threshold |
| **APS** (Adaptive Prediction Sets) | Adaptive threshold per example using sorted probabilities |
| **RAPS** (Regularized APS) | APS with a penalty on set size for tighter sets |
| **Mondrian** | Class‑conditional conformal prediction for fairness across classes |
| **ECOT** (Entropy‑Constrained Optimal Transport) | Optimal transport‑based conformal predictor with size constraints |

All methods are evaluated with respect to:

- ✅ **Coverage** – empirical marginal / class‑conditional coverage
- 📏 **Set Size** – average number of predicted labels
- 📊 **Class‑wise Behaviour** – per‑class coverage, set composition, and failure modes

---
