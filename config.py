# config.py
import os
from dotenv import load_dotenv


# --- PATH CONFIGURATION ---
path_to_repository="D:\\projects\\vqa_chest\\CheXLlama-main"

CKPT_PATH = path_to_repository+'\\model.pth.tar'
DATA_DIR = path_to_repository+'\\ChestX-ray14\\images'
PARQUET_DIR = 'D:\\projects\\vqa_chest\\data-chexnet'
path_to_API = path_to_repository + "\\API_KEY.env"

# --- MODEL & DIAGNOSIS CONFIGURATION ---
N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


# Thresholds for classifying a finding as present
CLASS_THRESHOLDS = {
    'Atelectasis': 0.5,
    'Cardiomegaly': 0.6,
    'Effusion': 0.6,
    'Infiltration': 0.5,
    'Mass': 0.5,
    'Nodule': 0.5,
    'Pneumonia': 0.5,
    'Pneumothorax': 0.65,
    'Consolidation': 0.5,
    'Edema': 0.55,
    'Emphysema': 0.55,
    'Fibrosis': 0.62,
    'Pleural_Thickening': 0.55,
    'Hernia': 0.85
}

# If the highest probability across all classes is below this, it's 'No Finding'.
NO_FINDING_THRESHOLD = 1

# Margin for flagging a result as 'uncertain'
# If abs(probability - threshold) <= UNCERTAINTY_MARGIN, it's uncertain.
UNCERTAINTY_MARGIN = 0.1

# --- TORCH CONFIGURATION ---
DEVICE = 'cpu'  # Can be changed to 'cuda' if a GPU is available
DTYPE = 'torch.float'