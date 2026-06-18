"""
train_tree_binary_classifier.py

Training script for binary tree/no-tree classifier.

This script trains a lightweight binary classifier that answers:
"Is there a tree in the frame?" - YES or NO.

The trained model is used for immediate visual stop detection.
After stopping, the main YOLO model runs for detailed palm tree
detection and bounding boxes.

Dataset Structure:
==================
data/tree_binary/
    train/
        tree/           # Images containing trees
        no_tree/        # Images without trees
    val/
        tree/
        no_tree/

Usage:
======
python3 training/train_tree_binary_classifier.py

Or with custom parameters:
python3 training/train_tree_binary_classifier.py --epochs 50 --batch_size 32 --lr 0.001

The trained model will be saved to:
models/tree_binary_classifier.pt
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "tree_binary")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "tree_binary_classifier.pt")
DEFAULT_IMAGE_SIZE = 224

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_WORKERS = 2


class TreeBinaryDataset(Dataset):
    """Dataset for binary tree/no-tree classification."""
    
    def __init__(self, root_dir, image_size=DEFAULT_IMAGE_SIZE, transform=None):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing train/val subdirectories
            image_size: Target image size
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Collect all image paths and labels
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all image paths and labels from directory structure."""
        # Class 0: no_tree, Class 1: tree
        classes = {"no_tree": 0, "tree": 1}
        
        for class_name, label in classes.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    filepath = os.path.join(class_dir, filename)
                    self.samples.append((filepath, label))
                    
        logger.info(f"Loaded {len(self.samples)} samples from {self.root_dir}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        # Load image
        image = Image.open(filepath).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_model(num_classes=2, pretrained=True):
    """
    Create a lightweight binary classifier model.
    
    Uses MobileNetV3-Small for efficient inference on Raspberry Pi.
    
    Args:
        num_classes: Number of output classes (2 for binary)
        pretrained: Whether to use pretrained weights
        
    Returns:
        nn.Module: The model
    """
    # Use MobileNetV3-Small for efficiency
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Replace the classifier head for binary classification
    # MobileNetV3-Small has: classifier -> Linear(576, num_classes)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train(data_dir=DEFAULT_DATA_DIR, model_path=DEFAULT_MODEL_PATH,
          batch_size=DEFAULT_BATCH_SIZE, epochs=DEFAULT_EPOCHS,
          learning_rate=DEFAULT_LEARNING_RATE, image_size=DEFAULT_IMAGE_SIZE):
    """
    Train the binary tree classifier.
    
    Args:
        data_dir: Path to dataset directory
        model_path: Path to save trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        image_size: Input image size
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data directories if they don't exist
    os.makedirs(os.path.join(data_dir, "train", "tree"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "train", "no_tree"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val", "tree"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val", "no_tree"), exist_ok=True)
    
    # Check if dataset has images
    train_tree_count = len([f for f in os.listdir(os.path.join(data_dir, "train", "tree")) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    train_no_tree_count = len([f for f in os.listdir(os.path.join(data_dir, "train", "no_tree")) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if train_tree_count == 0 and train_no_tree_count == 0:
        logger.error(f"No training images found in {data_dir}")
        logger.info("Please add images to:")
        logger.info(f"  {os.path.join(data_dir, 'train', 'tree')}")
        logger.info(f"  {os.path.join(data_dir, 'train', 'no_tree')}")
        sys.exit(1)
        
    logger.info(f"Training data: {train_tree_count} tree, {train_no_tree_count} no_tree")
    
    # Create datasets
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TreeBinaryDataset(
        os.path.join(data_dir, "train"),
        image_size=image_size,
        transform=train_transform
    )
    
    val_dataset = TreeBinaryDataset(
        os.path.join(data_dir, "val"),
        image_size=image_size,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            
    # Save final model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    torch.save({
        'model_state_dict': model.state_dict(),
        'image_size': image_size,
        'classes': ['no_tree', 'tree'],
        'architecture': 'mobilenet_v3_small',
        'training_date': datetime.now().isoformat(),
        'best_val_accuracy': best_val_acc
    }, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved: {model_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Architecture: MobileNetV3-Small")
    print()
    print("Next steps:")
    print("1. Test the model: python robot_control/test_tree_binary_inference.py")
    print("2. Run the mission: python robot_control/robot_tree_stop_mission.py")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Train binary tree/no-tree classifier")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                       help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                       help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE,
                       help="Input image size")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PalmMapBot - Binary Tree Classifier Training")
    print("=" * 60)
    print()
    print(f"Data directory: {args.data_dir}")
    print(f"Model output: {args.model_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.image_size}")
    print()
    
    train(
        data_dir=args.data_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        image_size=args.image_size
    )


if __name__ == "__main__":
    main()