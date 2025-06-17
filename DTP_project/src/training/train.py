import os
import torch
import logging
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, temporal_features, targets) in enumerate(pbar):
        images, temporal_features, targets = images.to(device), temporal_features.to(device), targets.to(device)
        
        # Forward pass
        predictions = model(images, temporal_features)
        
        # Get style features for loss computation
        with torch.no_grad():
            cnn_features = model.cnn(images)
            cnn_features = cnn_features.view(images.size(0), -1)
            style_features = model.feature_processor(cnn_features)
        
        # Compute loss
        loss = criterion(predictions, targets, style_features, temporal_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = predictions.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, temporal_features, targets in val_loader:
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
    
    return total_loss / len(val_loader), 100. * correct / total

def save_model(model, optimizer, epoch, val_loss, val_acc, log_dir, is_best=False):
    """Save model checkpoint with compression"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    
    # Save full checkpoint with compression
    checkpoint_path = os.path.join(log_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
    
    # Save a lighter version for best model
    if is_best:
        # Save only the model state dict with compression
        best_model_path = os.path.join(log_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=True)
        
        # Also save a minimal version for GitHub
        minimal_path = os.path.join(log_dir, 'best_model_minimal.pth')
        minimal_state = {k: v for k, v in model.state_dict().items() 
                        if not k.startswith('cnn.')}  # Exclude CNN weights
        torch.save(minimal_state, minimal_path, _use_new_zipfile_serialization=True)
        
        logging.info(f'Saved best model checkpoint to {best_model_path}')
        logging.info(f'Saved minimal model checkpoint to {minimal_path}')

def main(args):
    # Setup logging and tensorboard
    setup_logging(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Initialize data processor and create datasets
    processor = FashionDataProcessor(
        image_size=(224, 224),
        data_dir=args.data_dir
    )
    
    train_paths, val_paths, test_paths = processor.create_dataset(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    train_dataset = FashionDataset(train_paths)
    val_dataset = FashionDataset(val_paths)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model and training components
    model = FashionTrendPredictor(num_classes=args.num_classes).to(device)
    criterion = FashionTrendLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            logging.info(f'Learning rate decreased from {old_lr:.6f} to {new_lr:.6f}')
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        logging.info(
            f'Epoch {epoch}: '
            f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%'
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model, optimizer, epoch, val_loss, val_acc,
                args.log_dir, is_best=True
            )
        elif epoch % 10 == 0:  # Save checkpoint every 10 epochs
            save_model(
                model, optimizer, epoch, val_loss, val_acc,
                args.log_dir, is_best=False
            )
    
    writer.close()
    logging.info('Training completed!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fashion Trend Predictor')
    parser.add_argument('--data_dir', type=str, default='../data',
                      help='Directory containing the dataset')
    parser.add_argument('--log_dir', type=str, default='../logs',
                      help='Directory for saving logs and checkpoints')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    main(args) 