"""
prepare_binary_dataset.py

Convert existing YOLO dataset to binary tree/no-tree classification dataset.

This script uses the existing YOLO dataset (dataset/images/train, dataset/images/val)
and converts it to a binary classification dataset for the tree/no-tree classifier.

- Images WITH tree labels → tree/ class
- Images WITHOUT tree labels → no_tree/ class (or use empty/background images)

Usage:
======
python3 training/prepare_binary_dataset.py
"""

import os
import sys
import shutil
import random
from pathlib import Path

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source dataset paths
SOURCE_IMAGES_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "images", "train")
SOURCE_IMAGES_VAL = os.path.join(PROJECT_ROOT, "dataset", "images", "val")
SOURCE_LABELS_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "labels", "train")
SOURCE_LABELS_VAL = os.path.join(PROJECT_ROOT, "dataset", "labels", "val")

# Target binary dataset paths
TARGET_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "tree_binary")
TARGET_TRAIN_TREE = os.path.join(TARGET_DATA_DIR, "train", "tree")
TARGET_TRAIN_NO_TREE = os.path.join(TARGET_DATA_DIR, "train", "no_tree")
TARGET_VAL_TREE = os.path.join(TARGET_DATA_DIR, "val", "tree")
TARGET_VAL_NO_TREE = os.path.join(TARGET_DATA_DIR, "val", "no_tree")


def has_labels(label_path):
    """Check if an image has corresponding labels (contains trees)."""
    if not os.path.exists(label_path):
        return False
    
    # Check if label file has content
    try:
        with open(label_path, 'r') as f:
            content = f.read().strip()
            return len(content) > 0
    except:
        return False


def prepare_dataset():
    """Prepare binary classification dataset from YOLO dataset."""
    print("=" * 60)
    print("Preparing Binary Tree/No-Tree Dataset")
    print("=" * 60)
    
    # Create target directories
    for d in [TARGET_TRAIN_TREE, TARGET_TRAIN_NO_TREE, TARGET_VAL_TREE, TARGET_VAL_NO_TREE]:
        os.makedirs(d, exist_ok=True)
    
    # Process training images
    print(f"\nProcessing training images from: {SOURCE_IMAGES_TRAIN}")
    train_tree_count = 0
    train_no_tree_count = 0
    
    if os.path.exists(SOURCE_IMAGES_TRAIN):
        for img_file in os.listdir(SOURCE_IMAGES_TRAIN):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            img_path = os.path.join(SOURCE_IMAGES_TRAIN, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(SOURCE_LABELS_TRAIN, label_file)
            
            if has_labels(label_path):
                # Image contains trees
                shutil.copy2(img_path, os.path.join(TARGET_TRAIN_TREE, img_file))
                train_tree_count += 1
            else:
                # Image has no trees
                shutil.copy2(img_path, os.path.join(TARGET_TRAIN_NO_TREE, img_file))
                train_no_tree_count += 1
    
    # Process validation images
    print(f"Processing validation images from: {SOURCE_IMAGES_VAL}")
    val_tree_count = 0
    val_no_tree_count = 0
    
    if os.path.exists(SOURCE_IMAGES_VAL):
        for img_file in os.listdir(SOURCE_IMAGES_VAL):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            img_path = os.path.join(SOURCE_IMAGES_VAL, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(SOURCE_LABELS_VAL, label_file)
            
            if has_labels(label_path):
                shutil.copy2(img_path, os.path.join(TARGET_VAL_TREE, img_file))
                val_tree_count += 1
            else:
                shutil.copy2(img_path, os.path.join(TARGET_VAL_NO_TREE, img_file))
                val_no_tree_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nTraining set:")
    print(f"  Tree images: {train_tree_count}")
    print(f"  No-tree images: {train_no_tree_count}")
    print(f"\nValidation set:")
    print(f"  Tree images: {val_tree_count}")
    print(f"  No-tree images: {val_no_tree_count}")
    print(f"\nOutput directory: {TARGET_DATA_DIR}")
    
    total_tree = train_tree_count + val_tree_count
    total_no_tree = train_no_tree_count + val_no_tree_count
    
    if total_tree == 0:
        print("\nWARNING: No tree images found! Check your YOLO dataset.")
    if total_no_tree == 0:
        print("\nWARNING: No 'no_tree' images found. The dataset may be unbalanced.")
        print("Consider adding background/non-tree images to improve classifier.")
    
    print("\nNext step: Run training")
    print("  python3 training/train_tree_binary_classifier.py")


if __name__ == "__main__":
    prepare_dataset()