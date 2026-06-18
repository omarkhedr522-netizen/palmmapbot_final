"""
tree_binary_inference.py

Binary tree/no-tree classifier for immediate visual stop detection.

This module provides fast yes/no tree detection to trigger immediate
robot stopping. It uses a lightweight binary classifier that answers:
"Is there a tree in the frame?" - YES or NO.

This is NOT a replacement for YOLO. It's a fast trigger that stops
the robot immediately when a tree is detected. After stopping, the
main YOLO model runs for detailed palm tree detection and bounding boxes.

Model Requirements:
- Path: models/tree_binary_classifier.pt
- Classes: 0 = no_tree, 1 = tree
- Architecture: Lightweight model (MobileNetV3, EfficientNet-B0, etc.)

Default Threshold: 0.60 (60% confidence)
"""

import os
import sys
import logging
import time

import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "models", "tree_binary_classifier.pt")
DEFAULT_THRESHOLD = 0.60
DEFAULT_IMAGE_SIZE = 224

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False


class BinaryTreeClassifier:
    """Binary tree/no-tree classifier for fast detection."""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD, 
                 image_size=DEFAULT_IMAGE_SIZE, device=None):
        """
        Initialize binary tree classifier.
        
        Args:
            model_path: Path to trained model file (.pt)
            threshold: Confidence threshold for tree detection (0.0-1.0)
            image_size: Input image size for the model
            device: torch device (cuda/cpu). Auto-detected if None.
        """
        self.model_path = model_path
        self.threshold = threshold
        self.image_size = image_size
        self.model = None
        self.device = device
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self._model_loaded = False
        self._load_error = None
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """
        Load the trained binary classifier model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TORCH_AVAILABLE:
            self._load_error = "PyTorch not installed"
            logger.error(self._load_error)
            return False
            
        if not os.path.exists(self.model_path):
            self._load_error = f"Model file not found: {self.model_path}"
            logger.error(self._load_error)
            return False
            
        try:
            # Load model
            # The model is expected to be a binary classifier with 2 output classes
            # We use MobileNetV3-Small as default architecture
            self.model = models.mobilenet_v3_small(weights=None)
            self.model.classifier[3] = nn.Linear(
                self.model.classifier[3].in_features, 
                2  # Binary classification: tree / no_tree
            )
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                # If checkpoint contains state_dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Direct state dict
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            self._model_loaded = True
            self._load_error = None
            
            logger.info(f"Binary tree classifier loaded: {self.model_path}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Threshold: {self.threshold}")
            
            return True
            
        except Exception as e:
            self._load_error = f"Failed to load model: {str(e)}"
            logger.error(self._load_error)
            return False
            
    def detect_tree_presence(self, frame):
        """
        Detect if a tree is present in the frame.
        
        This is a FAST yes/no detection for immediate stopping.
        
        Args:
            frame: OpenCV BGR image (numpy array)
            
        Returns:
            tuple: (tree_present: bool, confidence: float)
        """
        if not self._model_loaded:
            if self.model is None:
                self.load_model()
            if not self._model_loaded:
                logger.error("Model not loaded, returning negative detection")
                return False, 0.0
                
        try:
            # Convert BGR to RGB for torchvision
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale - convert to 3 channels
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
            # Preprocess
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
            # Get probability for "tree" class (index 1)
            tree_prob = probabilities[0][1].item()
            
            tree_present = tree_prob >= self.threshold
            
            if tree_present:
                logger.info(f"TREE DETECTED - confidence: {tree_prob:.4f}")
            else:
                logger.debug(f"No tree - confidence: {tree_prob:.4f}")
                
            return tree_present, round(tree_prob, 4)
            
        except Exception as e:
            logger.error(f"Tree detection error: {str(e)}")
            return False, 0.0
            
    def is_loaded(self):
        """Check if model is loaded."""
        return self._model_loaded
        
    def get_error(self):
        """Get load error message if any."""
        return self._load_error


# Global classifier instance (optional convenience)
_classifier = None


def get_classifier(model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD):
    """
    Get or create global binary classifier instance.
    
    Args:
        model_path: Path to model
        threshold: Detection threshold
        
    Returns:
        BinaryTreeClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = BinaryTreeClassifier(model_path, threshold)
        _classifier.load_model()
    return _classifier


def detect_tree_presence(frame, model_path=DEFAULT_MODEL_PATH, threshold=DEFAULT_THRESHOLD):
    """
    Convenience function for tree detection.
    
    Args:
        frame: OpenCV BGR image
        model_path: Path to model
        threshold: Detection threshold
        
    Returns:
        tuple: (tree_present: bool, confidence: float)
    """
    classifier = get_classifier(model_path, threshold)
    return classifier.detect_tree_presence(frame)


# Test function
if __name__ == "__main__":
    print("Binary Tree Classifier Test")
    print("=" * 40)
    
    # Test with a sample image or camera
    classifier = BinaryTreeClassifier()
    
    if not classifier.load_model():
        print("Failed to load model!")
        print(f"Error: {classifier.get_error()}")
        print("\nTo train a model, run:")
        print("  python training/train_tree_binary_classifier.py")
        sys.exit(1)
        
    print("Model loaded successfully!")
    print(f"Threshold: {classifier.threshold}")
    print(f"Device: {classifier.device}")
    
    # Test with camera if available
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("\nTesting with camera (press 'q' to quit)...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                tree_present, confidence = classifier.detect_tree_presence(frame)
                
                # Display result
                color = (0, 0, 255) if tree_present else (0, 255, 0)
                text = f"Tree: {tree_present} ({confidence:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
                
                cv2.imshow('Tree Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("Camera not available for testing")
    except Exception as e:
        print(f"Camera test error: {e}")