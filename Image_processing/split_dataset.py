import os
import shutil
import random

# Paths
src_dir = r"/home/phamtiendat/Documents/ComputerVision/Image_processing/Raw_Images"
dst_dir = r"/home/phamtiendat/Documents/ComputerVision/Image_processing/Dataset"

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Seed for reproducibility
random.seed(42)

# Create output directories
for split in ["train", "val", "test"]:
    split_path = os.path.join(dst_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Iterate through each class folder
for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get all files in this class
    files = os.listdir(class_path)
    random.shuffle(files)

    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    # Copy files to respective folders
    for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
        split_class_dir = os.path.join(dst_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for file in split_files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(split_class_dir, file))

print("✅ Dataset successfully split into train, val, and test sets!")
