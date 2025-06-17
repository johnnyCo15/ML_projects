import os
import torch
import logging
from torch.utils.data import DataLoader
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.architectures.fashion_trend_model import FashionTrendPredictor, FashionTrendLoss
from src.data_processing.data_processor import FashionDataProcessor, FashionDataset

def test_dataset():
    """Test the FashionDataset class"""
    logging.info("Testing FashionDataset...")
    
    # Initialize data processor
    processor = FashionDataProcessor(
        image_size=(224, 224),
        data_dir='../data'
    )
    
    # Get a small sample of image paths
    train_paths, val_paths, test_paths = processor.create_dataset(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Create dataset with a small sample
    sample_size = min(5, len(train_paths))
    sample_paths = train_paths[:sample_size]
    dataset = FashionDataset(sample_paths)
    
    # Test dataset operations
    try:
        # Test __len__
        assert len(dataset) == sample_size, f"Dataset length mismatch. Expected {sample_size}, got {len(dataset)}"
        
        # Test __getitem__
        for i in range(sample_size):
            image, temporal_features, target = dataset[i]
            assert image.shape == (3, 224, 224), f"Image shape mismatch at index {i}"
            assert temporal_features.shape == (2,), f"Temporal features shape mismatch at index {i}"
            assert isinstance(target, torch.Tensor), f"Target type mismatch at index {i}"
        
        logging.info("Dataset tests passed successfully!")
        return True
    except Exception as e:
        logging.error(f"Dataset test failed: {str(e)}")
        return False

def test_model():
    """Test the FashionTrendPredictor model"""
    logging.info("Testing FashionTrendPredictor...")
    
    try:
        # Initialize model
        model = FashionTrendPredictor(num_classes=2)
        
        # Create sample input
        batch_size = 2
        sample_images = torch.randn(batch_size, 3, 224, 224)
        sample_temporal = torch.randn(batch_size, 2)
        
        # Test forward pass
        output = model(sample_images, sample_temporal)
        assert output.shape == (batch_size, 2), f"Output shape mismatch. Expected ({batch_size}, 2), got {output.shape}"
        
        logging.info("Model tests passed successfully!")
        return True
    except Exception as e:
        logging.error(f"Model test failed: {str(e)}")
        return False

def test_training_step():
    """Test a single training step"""
    logging.info("Testing training step...")
    
    try:
        # Initialize components
        model = FashionTrendPredictor(num_classes=2)
        criterion = FashionTrendLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create sample batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        temporal_features = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size,))
        
        # Forward pass
        predictions = model(images, temporal_features)
        
        # Get style features from CNN
        with torch.no_grad():
            cnn_features = model.cnn(images)
            cnn_features = cnn_features.view(batch_size, -1)
            style_features = model.feature_processor(cnn_features)
        
        # Compute loss
        loss = criterion(predictions, targets, style_features, temporal_features)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logging.info("Training step test passed successfully!")
        return True
    except Exception as e:
        logging.error(f"Training step test failed: {str(e)}")
        return False

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Run tests
    tests = [
        ("Dataset", test_dataset),
        ("Model", test_model),
        ("Training Step", test_training_step)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        if not test_func():
            all_passed = False
            logger.error(f"{test_name} test failed!")
    
    if all_passed:
        logger.info("\nAll tests passed successfully! The components are working as intended.")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 