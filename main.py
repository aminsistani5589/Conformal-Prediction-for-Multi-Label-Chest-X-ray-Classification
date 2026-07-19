# main.py
import torch
import config
from xray_model.model_loader import load_chexnet_model
from xray_model.predictor import run_prediction, interpret_probabilities
from utils.data_mapping import format_findings_for_prompt

# --- EXAMPLE PATIENT DATA (User can modify this) ---
# In a real application, this would be loaded from a database or user input.
PATIENT_INFO = """
Patient Name: John Doe
Age: 65
Sex: Male
ID: 123456

Chief Complaint:
- Shortness of breath for 3 days
- Dry cough and low-grade fever

History of Present Illness:
- Worsening dyspnea over the last 3 days
- Mild, persistent dry cough with mild chest pain on deep inspiration
- Fever around 38°C

Past Medical History:
- Hypertension, controlled
- Type 2 Diabetes Mellitus
- Former smoker (20 pack-years, quit 7 years ago)

Medications:
- Metformin 1000mg/day
- Lisinopril 10mg/day

Allergies: None known

Physical Examination:
- Temp: 38.2°C, RR: 22/min, Pulse: 90/min, BP: 130/80 mmHg
- O2 Saturation: 91% on room air
- Auscultation: Fine crackles at both lung bases

Laboratory Results:
- WBC: 8,400/μL (slightly elevated)
- C-Reactive Protein (CRP): Elevated
"""
MEDICAL_CONTEXT = "Evaluate for infectious or inflammatory lung disease (e.g., pneumonia, heart failure)."

def run_diagnostic_workflow(image_path):
    """
    Orchestrates the full diagnostic workflow from image to final report.
    """
    # 1. Initialize and load the CheXNet model
    print("--- Step 1: Loading X-Ray Analysis Model ---")
    model = load_chexnet_model(config.CKPT_PATH, config.N_CLASSES)
    
    # 2. Analyze the chest X-ray image
    print(f"\n--- Step 2: Analyzing Image: {image_path} ---")
    raw_probabilities = run_prediction(model, image_path)
    
    # 3. Interpret the model's output
    predictions, uncertainties, pred_probs, uncert_probs = interpret_probabilities(raw_probabilities)
    print("\nAI Model Predictions:")
    print(f"  - Findings: {predictions}")
    print(f"  - Uncertainties: {uncertainties}")
    
    # 4. Format AI findings into a text block for the LLM
    image_findings_text = format_findings_for_prompt(predictions, uncertainties, pred_probs, uncert_probs)
    print("\n--- Step 3: Formatting Findings for LLM ---")
    print(image_findings_text)
    
if __name__ == '__main__':
    # Use the image path from the config file
    image_to_diagnose = config.SINGLE_TEST_IMAGE
    
    # Run the entire workflow
    run_diagnostic_workflow(image_to_diagnose)