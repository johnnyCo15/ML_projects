import os
import sys
import logging
from data_processor import FashionDataProcessor, FashionDataset
from torch.utils.data import DataLoader

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize data processor
    processor = FashionDataProcessor(
        image_size=(224, 224),
        data_dir='../data'  # This will create data directory in src folder
    )
    
    # Download images from CSV
    csv_path = '../../working.csv'  # Updated path to point to working.csv in DTP_project directory
    logger.info(f"Using CSV file at: {os.path.abspath(csv_path)}")
    logger.info("Downloading images...")
    processor.download_images(csv_path)
    
    # Create dataset splits
    logger.info("Creating dataset splits...")
    train_paths, val_paths, test_paths = processor.create_dataset(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Create datasets
    train_dataset = FashionDataset(train_paths)
    val_dataset = FashionDataset(val_paths)
    test_dataset = FashionDataset(test_paths)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Created datasets with sizes:")
    logger.info(f"Training: {len(train_dataset)}")
    logger.info(f"Validation: {len(val_dataset)}")
    logger.info(f"Testing: {len(test_dataset)}")
    
    # Process a sample image to demonstrate feature extraction
    if len(train_paths) > 0:
        sample_image = processor.preprocess_image(train_paths[0])
        if sample_image is not None:
            features = processor.extract_features(sample_image)
            logger.info("Successfully extracted features from sample image")
            logger.info(f"Feature types: {list(features.keys())}")

if __name__ == "__main__":
    main() 