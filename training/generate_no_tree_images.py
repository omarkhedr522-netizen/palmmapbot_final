"""
generate_no_tree_images.py

Generate synthetic "no_tree" images for binary classification training.
Creates images with various natural backgrounds but no trees.
"""

import os
import sys
import random
import numpy as np
import cv2
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NO_TREE_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "tree_binary", "train", "no_tree")
NO_TREE_VAL_DIR = os.path.join(PROJECT_ROOT, "data", "tree_binary", "val", "no_tree")

os.makedirs(NO_TREE_TRAIN_DIR, exist_ok=True)
os.makedirs(NO_TREE_VAL_DIR, exist_ok=True)

def generate_sky_image(width=224, height=224):
    """Generate sky-like gradient image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Blue sky gradient
    for i in range(height):
        r = int(100 + 40 * (i / height))
        g = int(150 + 50 * (i / height))
        b = int(200 + 55 * (i / height))
        img[i] = [b, g, r]  # BGR
    return img

def generate_ground_image(width=224, height=224):
    """Generate ground/dirt texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array([40, 70, 120], dtype=np.uint8)  # Brown-ish
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return img

def generate_grass_image(width=224, height=224):
    """Generate grass-like texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array([30, 80, 30], dtype=np.uint8)  # Green
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return img

def generate_sand_image(width=224, height=224):
    """Generate sand/desert texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array([180, 200, 220], dtype=np.uint8)  # Sandy
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return img

def generate_road_image(width=224, height=224):
    """Generate road/asphalt texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array([60, 60, 60], dtype=np.uint8)  # Gray
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    # Add some lines
    for _ in range(3):
        y = random.randint(50, 174)
        cv2.line(img, (0, y), (width, y), (80, 80, 80), 2)
    return img

def generate_water_image(width=224, height=224):
    """Generate water-like texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array([150, 100, 50], dtype=np.uint8)  # Blue
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return img

def generate_concrete_image(width=224, height=224):
    """Generate concrete texture."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    base_color = np.array([180, 180, 180], dtype=np.uint8)  # Gray
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    return img

def generate_mixed_background(width=224, height=224):
    """Generate mixed background (sky + ground)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = random.randint(80, 144)
    
    # Sky part
    for i in range(horizon):
        r = int(100 + 40 * (i / horizon))
        g = int(150 + 50 * (i / horizon))
        b = int(200 + 55 * (i / horizon))
        img[i] = [b, g, r]
    
    # Ground part
    ground_type = random.choice(['dirt', 'grass', 'sand', 'road'])
    if ground_type == 'dirt':
        ground = generate_ground_image(width, height - horizon)
    elif ground_type == 'grass':
        ground = generate_grass_image(width, height - horizon)
    elif ground_type == 'sand':
        ground = generate_sand_image(width, height - horizon)
    else:
        ground = generate_road_image(width, height - horizon)
    
    img[horizon:] = ground

    return img

generators = [
    generate_sky_image,
    generate_ground_image,
    generate_grass_image,
    generate_sand_image,
    generate_road_image,
    generate_water_image,
    generate_concrete_image,
    generate_mixed_background,
]

print("Generating synthetic 'no_tree' images...")
print("=" * 60)

# Generate training images
num_train = 200
for i in range(num_train):
    gen = random.choice(generators)
    img = gen()
    filename = f"no_tree_synthetic_{i:04d}.jpg"
    cv2.imwrite(os.path.join(NO_TREE_TRAIN_DIR, filename), img)
    
print(f"Generated {num_train} training no_tree images")

# Generate validation images
num_val = 50
for i in range(num_val):
    gen = random.choice(generators)
    img = gen()
    filename = f"no_tree_synthetic_{i:04d}.jpg"
    cv2.imwrite(os.path.join(NO_TREE_VAL_DIR, filename), img)
    
print(f"Generated {num_val} validation no_tree images")
print("=" * 60)
print("Done! Now run: python3 training/train_tree_binary_classifier.py")