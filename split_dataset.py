import os
import shutil
import random

# ============================================================
# SETTINGS
# ============================================================

SOURCE_DIR = "dataset/train"
OUTPUT_DIR = "dataset_new"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

CLASSES = [
    "hemorrhagic",
    "ischemic",
    "normal"
]

random.seed(SEED)


# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validate", "test"]:
    for class_name in CLASSES:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )


# ============================================================
# SPLIT EACH CLASS
# ============================================================

for class_name in CLASSES:

    source_class_dir = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(source_class_dir):
        raise FileNotFoundError(
            f"Class folder not found: {source_class_dir}"
        )

    images = [
        f for f in os.listdir(source_class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "validate": val_images,
        "test": test_images
    }

    print(f"\n{class_name.upper()}")
    print(f"Total:      {total}")
    print(f"Train:      {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test:       {len(test_images)}")

    for split_name, split_images in splits.items():

        destination_dir = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        for image_name in split_images:

            source_path = os.path.join(
                source_class_dir,
                image_name
            )

            destination_path = os.path.join(
                destination_dir,
                image_name
            )

            shutil.copy2(source_path, destination_path)


# ============================================================
# FINAL COUNTS
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

for split in ["train", "validate", "test"]:

    print(f"\n{split.upper()}")

    total_split = 0

    for class_name in CLASSES:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            class_name
        )

        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ])

        print(f"{class_name:12s}: {count}")
        total_split += count

    print(f"TOTAL:        {total_split}")

print("\nNew dataset created at:")
print(OUTPUT_DIR)