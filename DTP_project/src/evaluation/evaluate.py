import os
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.architectures.fashion_trend_model import FashionTrendPredictor, FashionTrendLoss
from src.data_processing.data_processor import FashionDataProcessor, FashionDataset

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )

def load_model(model_path, num_classes, device):
    """Load the trained model"""
    model = FashionTrendPredictor(num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Loaded model from {model_path}")
            logging.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
            logging.info(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
            logging.info(f"Best validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
        else:
            model.load_state_dict(checkpoint)
            logging.info(f"Loaded model state dict from {model_path}")
    else:
        logging.warning(f"Model file {model_path} not found. Using untrained model.")
    
    return model

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, temporal_features, targets in test_loader:
            images, temporal_features, targets = images.to(device), temporal_features.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(images, temporal_features)
            
            # Get style features for loss computation
            cnn_features = model.cnn(images)
            cnn_features = cnn_features.view(images.size(0), -1)
            style_features = model.feature_processor(cnn_features)
            
            # Compute loss
            loss = criterion(predictions, targets, style_features, temporal_features)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy, all_predictions, all_targets

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Trending', 'Trending'],
                yticklabels=['Not Trending', 'Trending'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"Confusion matrix saved to {save_path}")

def plot_metrics(metrics, save_path):
    """Plot evaluation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    ax1.bar(['Test Accuracy'], [metrics['accuracy']], color='skyblue')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy')
    ax1.set_ylim(0, 100)
    
    # Add accuracy value on top of bar
    ax1.text(0, metrics['accuracy'] + 1, f'{metrics["accuracy"]:.2f}%', 
             ha='center', va='bottom', fontweight='bold')
    
    # Plot loss
    ax2.bar(['Test Loss'], [metrics['loss']], color='lightcoral')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    
    # Add loss value on top of bar
    ax2.text(0, metrics['loss'] + 0.01, f'{metrics["loss"]:.4f}', 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logging.info(f"Metrics plot saved to {save_path}")

def main(args):
    # Setup logging
    setup_logging(args.log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load model
    model = load_model(args.model_path, args.num_classes, device)
    
    # Initialize data processor and create test dataset
    processor = FashionDataProcessor(
        image_size=(224, 224),
        data_dir=args.data_dir
    )
    
    train_paths, val_paths, test_paths = processor.create_dataset(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    test_dataset = FashionDataset(test_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    logging.info(f'Test dataset size: {len(test_dataset)}')
    
    # Initialize loss function
    criterion = FashionTrendLoss()
    
    # Evaluate model
    logging.info("Starting model evaluation...")
    test_loss, test_accuracy, predictions, targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Print results
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Generate classification report
    logging.info("\nClassification Report:")
    logging.info(classification_report(targets, predictions, 
                                     target_names=['Not Trending', 'Trending']))
    
    # Create visualizations
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.log_dir, 'confusion_matrix.png')
    plot_confusion_matrix(targets, predictions, cm_path)
    
    # Plot metrics
    metrics = {'accuracy': test_accuracy, 'loss': test_loss}
    metrics_path = os.path.join(args.log_dir, 'evaluation_metrics.png')
    plot_metrics(metrics, metrics_path)
    
    # Save detailed results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions,
        'targets': targets,
        'num_samples': len(test_dataset)
    }
    
    results_path = os.path.join(args.log_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Fashion Trend Prediction Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Number of Test Samples: {len(test_dataset)}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(targets, predictions, 
                                    target_names=['Not Trending', 'Trending']))
    
    logging.info(f"Detailed results saved to {results_path}")
    logging.info("Evaluation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Fashion Trend Predictor')
    parser.add_argument('--model_path', type=str, default='../logs/best_model.pth',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../data',
                      help='Directory containing the dataset')
    parser.add_argument('--log_dir', type=str, default='../logs',
                      help='Directory for saving evaluation results')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    args = parser.parse_args()
    main(args) 