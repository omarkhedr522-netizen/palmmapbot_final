"""
test_tree_binary_inference.py

Test script for binary tree/no-tree inference.

This script tests the binary tree classifier:
1. Load the trained model
2. Test with a sample image or camera feed
3. Display tree/no-tree detection results

Usage:
======
python3 robot_control/test_tree_binary_inference.py

Or test with a specific image:
python3 robot_control/test_tree_binary_inference.py --image path/to/image.jpg

Or test with camera:
python3 robot_control/test_tree_binary_inference.py --camera
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from robot_control.tree_binary_inference import BinaryTreeClassifier, DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD


def test_with_image(classifier, image_path):
    """Test classifier with a single image."""
    print(f"Testing with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return False
        
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not load image")
        return False
        
    # Run detection
    tree_present, confidence = classifier.detect_tree_presence(frame)
    
    # Display result
    print()
    print("=" * 60)
    print(f"Result: {'TREE DETECTED' if tree_present else 'NO TREE'}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Threshold: {classifier.threshold}")
    print("=" * 60)
    
    # Show image with result
    color = (0, 0, 255) if tree_present else (0, 255, 0)
    text = f"Tree: {tree_present} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Tree Detection Test', frame)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def test_with_camera(classifier, duration=30):
    """Test classifier with camera feed."""
    print("Testing with camera...")
    print(f"Duration: {duration} seconds (press 'q' to quit early)")
    print()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
        
    start_time = time.time()
    frame_count = 0
    tree_count = 0
    
    print("Reading camera... (press 'q' to quit)")
    print("-" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        frame_count += 1
        
        # Run detection
        tree_present, confidence = classifier.detect_tree_presence(frame)
        
        if tree_present:
            tree_count += 1
            
        # Display result on frame
        color = (0, 0, 255) if tree_present else (0, 255, 0)
        text = f"Tree: {tree_present} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add frame count and tree count
        info_text = f"Frames: {frame_count} | Trees: {tree_count}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show frame
        cv2.imshow('Tree Detection Test', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Check duration
        if time.time() - start_time > duration:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("-" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Frames with tree: {tree_count}")
    print(f"Tree detection rate: {tree_count/frame_count*100:.1f}%")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test binary tree classifier")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--camera", action="store_true", help="Test with camera")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                       help=f"Path to model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help=f"Detection threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--duration", type=int, default=30,
                       help="Camera test duration in seconds (default: 30)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Binary Tree Classifier Test")
    print("=" * 60)
    print()
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print()
        print("Please train a model first:")
        print("  python3 training/train_tree_binary_classifier.py")
        sys.exit(1)
        
    # Load classifier
    print("Loading model...")
    classifier = BinaryTreeClassifier(args.model, args.threshold)
    
    if not classifier.load_model():
        print("Error: Failed to load model")
        print(f"Error: {classifier.get_error()}")
        sys.exit(1)
        
    print("Model loaded successfully!")
    print(f"Device: {classifier.device}")
    print()
    
    # Run test
    if args.image:
        success = test_with_image(classifier, args.image)
    elif args.camera:
        success = test_with_camera(classifier, args.duration)
    else:
        # Default: test with camera
        success = test_with_camera(classifier, args.duration)
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()