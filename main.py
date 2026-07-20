# main.py
import torch
import config
from xray_model.model_loader import load_chexnet_model
from xray_model.predictor import run_prediction, interpret_probabilities
from utils.data_mapping import format_findings_for_prompt
import glob
import io
import pandas as pd
from PIL import Image
import os
import tempfile

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
    print("\n--- Step 3: Formatting Findings ---")
    print(image_findings_text)
    


def extract_image_bytes(img_cell):
    """
    Extract bytes from the 'image' cell.
    Supports dict {'bytes': ...}, raw bytes, or string path.
    """
    if isinstance(img_cell, dict):
        return img_cell.get('bytes') or img_cell.get('path')
    elif isinstance(img_cell, bytes):
        return img_cell
    elif isinstance(img_cell, str):
        return img_cell
    return None

def save_bytes_to_temp_png(img_bytes):
    """
    Save image bytes to a temporary PNG file and return its path.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_path = temp_file.name
    temp_file.close()

    with open(temp_path, "wb") as f:
        f.write(img_bytes)

    return temp_path

if __name__ == '__main__':
    PARQUET_DIR = config.PARQUET_DIR
    parquet_files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))

    print(f"Found {len(parquet_files)} parquet files.")

    for pq_file in parquet_files:
        print(f"\nProcessing: {pq_file}")

        try:
            df = pd.read_parquet(pq_file)

            if 'image' not in df.columns:
                print(f"Skipping {pq_file}: no 'image' column found.")
                continue

            for idx, row in df.iterrows():
                img_cell = row['image']
                temp_image_path = None
                image_path = None

                try:
                    extracted = extract_image_bytes(img_cell)

                    if extracted is None:
                        print(f"Row {idx}: could not extract image data.")
                        continue

                    if isinstance(extracted, bytes):
                        temp_image_path = save_bytes_to_temp_png(extracted)
                        image_path = temp_image_path

                    elif isinstance(extracted, str):
                        # If it's already a path
                        if os.path.exists(extracted):
                            image_path = extracted
                        else:
                            print(f"Row {idx}: file not found -> {extracted}")
                            continue

                    else:
                        print(f"Row {idx}: unsupported image type -> {type(extracted)}")
                        continue

                    print(f"Running model on row {idx} -> {image_path}")

                    # Run your full workflow on each image
                    run_diagnostic_workflow(image_path)

                except Exception as e:
                    print(f"Row {idx}: error during processing -> {e}")

                finally:
                    # Clean up temporary file if we created one
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

        except Exception as e:
            print(f"Failed to read parquet file {pq_file}: {e}")

    print("\nDone processing all parquet images.")
