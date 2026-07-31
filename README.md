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
> **Note (baseline thresholds):** The Fixed Threshold baseline uses class-specific thresholds chosen **heuristically** via exploratory inspection on the held-out calibration/validation pool (not optimized via a formal objective).  
> These thresholds are included to provide a stronger non-conformal reference point than a single global 0.5 cutoff.


All methods are evaluated with respect to:

- ✅ **Coverage** – empirical marginal / class‑conditional coverage
- 📏 **Set Size** – average number of predicted labels
- 📊 **Class‑wise Behaviour** – per‑class coverage, set composition, and failure modes

---

## 📊 Experimental Results

**Dataset split:**  
Calibration set: `2,803` samples | Test set: `2,804` samples  
**Target error rate:** `α = 10%` → desired marginal coverage ≥ **90%**

---

### 🏆 Overall Performance (α = 0.1)

| Method               | Coverage | Avg. Size | F1↑    | Coverage ≥ 90%? |
|----------------------|----------|-----------|--------|:---------------:|
| Fixed Threshold      | 0.8188   | **2.82**  | 0.4411 | ❌              |
| **ECOT**             | **0.9116** | 4.86    | 0.2755 | ✅              |
| THR                  | 0.9290   | 5.53      | 0.2573 | ✅              |
| Mondrian             | 0.9162   | 5.44      | 0.2431 | ✅              |
| APS (legacy)         | 0.8955   | 9.20      | 0.1626 | ❌              |
| RAPS (legacy)        | 0.8959   | 9.55      | 0.1607 | ❌              |

**Key observations:**
- **Baseline (Fixed Threshold)** fails to meet the 90% coverage guarantee, despite having the smallest set size – it is overconfident and unreliable.
- **THR, ECOT, and Mondrian** all provide valid distribution-free coverage (≥90%) with conformal calibration.
- **ECOT** achieves the best trade-off: it is **the only valid method with an average set size below 5**, producing significantly tighter prediction sets than THR or Mondrian.
- The legacy implementations of **APS and RAPS** both undercover (≈89.6%) and yield extremely large sets (≥9.2), highlighting the need for proper regularization when applying conformal prediction to multi-label problems.
- F1 is reported for completeness, but **set predictors trade point-wise precision for guaranteed coverage** – their primary metrics are coverage and set size.

---

### 🔬 Class‑Conditional Coverage Analysis

Coverage per pathology across all methods.  
**Bold** numbers indicate where the 90% target is met for that specific class.

| Class                  | Fixed   | **ECOT** | THR     | Mondrian | APS     | RAPS    |
|------------------------|---------|----------|---------|----------|---------|---------|
| Atelectasis            | 0.7667  | 0.8667   | 0.8970  | 0.8727   | 0.8030  | 0.8000  |
| Cardiomegaly           | 0.8630  | **0.9178** | 0.8904  | **0.9178**  | 0.8630  | 0.8630  |
| Effusion               | 0.7774  | 0.8902   | **0.9116** | 0.8902   | 0.8476  | 0.8476  |
| Infiltration           | 0.5980  | 0.8735   | 0.8959  | 0.8755   | 0.8265  | 0.8204  |
| Mass                   | 0.7727  | 0.8182   | 0.8831  | 0.8182   | 0.7727  | 0.7597  |
| Nodule                 | 0.7500  | **0.9286** | **0.9071** | **0.9429**  | 0.8071  | 0.7857  |
| Pneumonia              | 0.7419  | **0.9032** | **0.9032** | **1.0000**  | 0.7742  | 0.7419  |
| Pneumothorax           | 0.7101  | 0.7899   | 0.8696  | 0.7899   | 0.8841  | 0.8986  |
| Consolidation          | 0.6929  | 0.8740   | 0.8740  | 0.8740   | 0.7874  | 0.7953  |
| Edema                  | 0.8654  | 0.8654   | **0.9423** | 0.8846   | **0.9038** | **0.9038** |
| Emphysema              | **0.9333** | **0.9600** | **0.9600** | **0.9600** | **0.9600** | **0.9467** |
| Fibrosis               | 0.5946  | 0.7297   | **0.9730** | 0.7838   | 0.8378  | 0.8108  |
| Pleural Thickening     | 0.7079  | **0.9101** | 0.8876  | **0.9213**  | 0.8202  | 0.8090  |
| Hernia                 | 0.1429  | 0.4286   | 0.7143  | 0.5714   | 0.8571  | 0.8571  |

---

### 🚨 Most Challenging Classes (Worst-Case Coverage)

| Method            | Worst Class         | Coverage |
|-------------------|---------------------|----------|
| Fixed Threshold   | **Hernia**          | 0.1429   |
| ECOT              | **Hernia**          | 0.4286   |
| THR               | **Hernia**          | 0.7143   |
| Mondrian          | **Hernia**          | 0.5714   |
| APS (legacy)      | Mass                | 0.7727   |
| RAPS (legacy)     | Pneumonia           | 0.7419   |

**Takeaway:**  
Hernia – an extremely rare finding in CheXpert – remains the Achilles' heel for all methods. While conformal predictors improve its coverage dramatically (from **14% → 71%** with THR), none fully meet the 90% target for this class.  
Mondrian’s per-class guarantee also suffers on Hernia, suggesting that further class‑adaptive strategies or data augmentation are needed for long‑tail pathologies.

---

> ✅ **Conclusion:** Conformal prediction turns a black‑box multi‑label classifier into a **reliable, uncertainty‑aware system**. ECOT offers the most efficient valid sets, while THR provides the highest overall coverage. The choice between them depends on the clinical tolerance for false negatives vs. set size.
