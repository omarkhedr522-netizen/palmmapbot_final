#!/usr/bin/env python3
"""
Run YOLO validation on the PalmMapBot dataset to get real detection metrics.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

def run_validation():
    """Run YOLO validation and extract metrics."""
    print("=" * 60)
    print("Running YOLO Validation on PalmMapBot Dataset")
    print("=" * 60)
    
    # Check for model
    model_path = "models/palm_tree_detector.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Run validation
    print("Running validation on val set...")
    metrics = model.val(
        data="dataset/data.yaml",
        split="val",
        verbose=False
    )
    
    # Extract results - use correct ultralytics attribute names
    # metrics.box has: p (precision), r (recall), f1, map50, map
    # metrics.speed has: preprocess, inference, postprocess
    precision = float(metrics.box.mp)  # mean precision
    recall = float(metrics.box.mr)     # mean recall
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        "model": "YOLOv8n-palm",
        "dataset_split": "val",
        "num_images": 20,
        "palm_instances": 20,  # From validation output
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1_score, 4),
        "map_0.5": round(float(metrics.box.map50), 4),
        "map_0.5:0.95": round(float(metrics.box.map), 4),
        "inference_ms": round(float(metrics.speed["inference"]), 1),
        "fps": round(1000.0 / float(metrics.speed["inference"]), 1) if metrics.speed["inference"] > 0 else "N/A",
        "weights_path": model_path,
        "notes": "Real validation results from PalmMapBot dataset (val split)"
    }
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Save to CSV
    output_dir = Path("validation/evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([results])
    df.to_csv(output_dir / "detection_metrics.csv", index=False)
    print(f"\nResults saved to {output_dir / 'detection_metrics.csv'}")
    
    # Also save as JSON for easy parsing
    with open(output_dir / "detection_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir / 'detection_metrics.json'}")
    
    return results

if __name__ == "__main__":
    results = run_validation()
    if results:
        print("\n" + "=" * 60)
        print("Validation complete!")
        print("=" * 60)
    else:
        print("\nValidation failed. Check error messages above.")