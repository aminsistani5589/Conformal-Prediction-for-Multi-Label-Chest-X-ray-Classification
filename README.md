<<<<<<< HEAD
## 🌟 Model in Action
=======
## 📌 Overview

Standard multi-label classifiers output a fixed set of labels using a threshold (e.g., 0.5). This ignores model uncertainty and offers no control over false positives / false negatives.  
This repository explores **conformal prediction** methods that turn any trained classifier into a **set predictor** with a user‑specified marginal coverage guarantee.

We experiment on the **CheXpert** dataset, comparing a fixed‑threshold baseline against several state‑of‑the‑art conformal approaches:

| Method | Description |
|--------|-------------|
| **Fixed Threshold** | Classical baseline |
| **THR** (Threshold) | Conformal calibration of a single global threshold |
| **APS** (Adaptive Prediction Sets) | Adaptive threshold per example using sorted probabilities |
| **RAPS** (Regularized APS) | APS with a penalty on set size for tighter sets |
| **Mondrian** | Class‑conditional conformal prediction for fairness across classes |
| **ECOT** (Entropy‑Constrained Optimal Transport) | Optimal transport‑based conformal predictor with size constraints |
> **Note (baseline thresholds):** The Fixed Threshold baseline uses class-specific thresholds chosen **heuristically** via exploratory inspection on the held-out calibration/validation pool (not optimized via a formal objective).  
> These thresholds are included to provide a stronger non-conformal reference point than a single global 0.5 cutoff.
>>>>>>> 216a4b8411f5026c3f6631e0b9819bb067b0494a


<<<<<<< HEAD
1.  **CheXNet Input:** A chest X-ray image (e.g., `chest_xray_sample.png`) is fed into CheXNet.
2.  **Automated Analysis:** CheXNet quickly identifies potential findings, such as "Pneumonia detected."
3.  **LLM Integration:** This initial finding, combined with relevant patient data, is passed to the Llama LLM.
4.  **Interactive Querying:** The radiologist can then ask specific questions like:
    - "What are the typical symptoms associated with this finding?"
    - "Are there any other conditions that might present similarly?"
    - "What follow-up procedures are recommended?"
5.  **Refined Reports:** The LLM provides intelligent, context-aware answers, helping the radiologist generate more comprehensive and accurate diagnostic reports.

---
=======
All methods are evaluated with respect to:

- ✅ **Coverage** – empirical marginal / class‑conditional coverage
- 📏 **Set Size** – average number of predicted labels
- 📊 **Class‑wise Behaviour** – per‑class coverage, set composition, and failure modes
>>>>>>> 216a4b8411f5026c3f6631e0b9819bb067b0494a

---

<<<<<<< HEAD
- **Synergistic AI Integration:** Seamlessly combines the cutting-edge **CheXNet** radiology image classification model with the robust **Llama 3.3 Nemotron Super 49B v1 API**.
- **Automated Radiological Image Analysis:** Leverages CheXNet to automatically predict potential findings from chest X-ray images, providing a rapid initial assessment.
- **Intelligent Question-Answering:** Enables users to engage in in-depth, contextual Q\&A sessions with the Llama LLM, utilizing initial CheXNet findings and patient-specific medical data as inputs.
- **Enhanced Understanding of Medical Imagery:** Moves beyond simple classification, allowing for nuanced exploration and explanation of potential conditions identified in radiology scans.
- **Flexible Querying:** Supports a wide range of user queries, from basic clarifications of medical terms to complex inquiries about potential implications, differential diagnoses, or recommended follow-up considerations.
- **(Potentially) Extensible Framework:** The modular architecture serves as a flexible foundation for future integration of other AI models, data sources, or specialized medical knowledge bases.

---
=======
## 📊 Experimental Results

**Dataset split:**  
Calibration set: `2,803` samples | Test set: `2,804` samples  
**Target error rate:** `α = 10%` → desired marginal coverage ≥ **90%**
>>>>>>> 216a4b8411f5026c3f6631e0b9819bb067b0494a

---

### 🏆 Overall Performance (α = 0.1)

<<<<<<< HEAD
- [](https://stanfordmlgroup.github.io/projects/chexnet/) - Utilized for precise radiological image classification and prediction.
- [](https://developer.nvidia.com/nemotron-3-8b) - The large language model API for interactive querying and report generation.

---
=======
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
>>>>>>> 216a4b8411f5026c3f6631e0b9819bb067b0494a

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

<<<<<<< HEAD
- Python 3.4+
- pip install pytorch
- pip install groq (for API interaction)
=======
---
>>>>>>> 216a4b8411f5026c3f6631e0b9819bb067b0494a

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

<<<<<<< HEAD
    - Download the ChestX-ray14 dataset images from the official [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC).
    - Decompress the downloaded files and place them into the `ChestX-ray14/images` directory within your cloned repository.
      - _Expected path structure:_ `your_cloned_repo/ChestX-ray14/images/`

3.  **Obtain Llama 3.3 Nemotron API Key:**

    - Create your API key for Llama 3.3 from [Groq](https://groq.com/).
    - Create a file named `GROQ_API_KEY.env` in the root of your cloned repository and add your API key in the following format:
      ```
      GROQ_API_KEY = your_api_key_here
      ```

4.  **Update `model.py` with Local Directory Paths:**

    - Open `model.py` in your preferred text editor.

    - **Define the path to your repository:**

      ```python
      path_to_repository="provide your path to repository here !!!" # Update this to your actual path
      ```

    - **Choose your selected image:**

      - Within `model.py`, locate the `SINGLE_TEST_IMAGE` variable.
      - Define its address, pointing to an image within your `ChestX-ray14/images` directory:
        ```python
        SINGLE_TEST_IMAGE = path_to_repository + '\\ChestX-ray14\\images\\00000003_001.png' # Update filename as needed
        ```
        _(Make sure the image filename corresponds to an actual image you downloaded.)_

5.  **Run the Model:**

    - Execute the `model.py` script from your terminal:
      ```bash
      python model.py
      ```
    - The script will process the image and provide initial results. Feel free to ask any follow-up questions within the interactive session\! 😀

---

## 📄 Project Paper

For a more in-depth understanding of the methodology and experimental results, please refer to our research paper on arXiv:

- **[Link]** _(it will be added very soon.)_
=======
> ✅ **Conclusion:** Conformal prediction turns a black‑box multi‑label classifier into a **reliable, uncertainty‑aware system**. ECOT offers the most efficient valid sets, while THR provides the highest overall coverage. The choice between them depends on the clinical tolerance for false negatives vs. set size.
>>>>>>> 216a4b8411f5026c3f6631e0b9819bb067b0494a
