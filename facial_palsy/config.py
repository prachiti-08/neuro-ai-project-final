from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# facial_palsy/
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# DATASET PATH
# ============================================================

# Dataset structure:
#
# facial_palsy/
# └── data/
#     └── archive/
#         ├── Eye/
#         ├── Eyebrow/
#         └── Mouth/
#
# Therefore:
#
# PROJECT_ROOT / "data" / "archive"

DATASET_ROOT = PROJECT_ROOT / "data" / "archive"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET REGIONS
# ============================================================

REGIONS = {

    "eye": {
        "folder": DATASET_ROOT / "Eye" / "Eye",
        "suffix": "eye",
    },

    "eyebrow": {
        "folder": DATASET_ROOT / "Eyebrow" / "Eyebrow",
        "suffix": "eyebrow",
    },

    "mouth": {
        "folder": DATASET_ROOT / "Mouth" / "Mouth",
        "suffix": "mouth",
    },

}


# ============================================================
# SEVERITY CLASSES
# ============================================================

CLASSES = [
    "Mild",
    "Moderate",
    "Moderate Severe",
    "Severe",
]


CLASS_TO_INDEX = {
    "Mild": 0,
    "Moderate": 1,
    "Moderate Severe": 2,
    "Severe": 3,
}


INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

# Windows-safe setting
NUM_WORKERS = 0

RANDOM_SEED = 42


# ============================================================
# MODEL SETTINGS
# ============================================================

NUM_CLASSES = len(CLASSES)

PRETRAINED = True


# ============================================================
# MODEL FILE PATHS
# ============================================================

MODEL_PATHS = {

    "eye":
        MODEL_DIR / "eye_model.pth",

    "eyebrow":
        MODEL_DIR / "eyebrow_model.pth",

    "mouth":
        MODEL_DIR / "mouth_model.pth",

}