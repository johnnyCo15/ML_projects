import os
import torch
import logging
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.architectures.fashion_trend_model import FashionTrendPredictor

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'prediction.log')),
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
        else:
            model.load_state_dict(checkpoint)
            logging.info(f"Loaded model state dict from {model_path}")
    else:
        logging.error(f"Model file {model_path} not found!")
        return None
    
    model.eval()
    return model

def preprocess_image(image_path, image_size=(224, 224)):
    """Preprocess a single image for prediction"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def extract_temporal_features(image_path):
    """Extract temporal features from image path or metadata"""
    # For now, we'll use dummy temporal features
    # In a real application, you might extract these from:
    # - Image metadata (EXIF data)
    # - File naming conventions
    # - Database records
    
    # Dummy temporal features: [season, year]
    # season: 0=Spring, 1=Summer, 2=Fall, 3=Winter
    # year: normalized year (e.g., 2020 -> 0, 2021 -> 1, etc.)
    
    # Extract year from filename if possible
    filename = os.path.basename(image_path)
    current_year = datetime.now().year
    
    # Try to extract year from filename
    year = current_year
    for i in range(current_year - 10, current_year + 1):
        if str(i) in filename:
            year = i
            break
    
    # Normalize year (0 = 2020, 1 = 2021, etc.)
    normalized_year = (year - 2020) / 10.0
    
    # Determine season (you could make this more sophisticated)
    # For now, we'll use a simple heuristic based on filename
    season = 0  # Default to Spring
    if any(season_name in filename.lower() for season_name in ['summer', 'ss']):
        season = 1
    elif any(season_name in filename.lower() for season_name in ['fall', 'autumn', 'fw']):
        season = 2
    elif any(season_name in filename.lower() for season_name in ['winter', 'aw']):
        season = 3
    
    # Normalize season to [0, 1]
    normalized_season = season / 3.0
    
    return torch.tensor([[normalized_season, normalized_year]], dtype=torch.float32)

def predict_single_image(model, image_path, device, class_names=['Not Trending', 'Trending']):
    """Make prediction on a single image"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Extract temporal features
        temporal_features = extract_temporal_features(image_path)
        temporal_features = temporal_features.to(device)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(image_tensor, temporal_features)
            probabilities = torch.softmax(predictions, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'image_path': image_path
        }
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {str(e)}")
        return None

def predict_batch(model, image_dir, device, class_names=['Not Trending', 'Trending']):
    """Make predictions on all images in a directory"""
    results = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) 
                          if f.lower().endswith(ext)])
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        logging.info(f"Processing {i+1}/{len(image_files)}: {filename}")
        
        result = predict_single_image(model, image_path, device, class_names)
        if result:
            results.append(result)
    
    return results

def save_predictions(results, output_path):
    """Save prediction results to a file"""
    with open(output_path, 'w') as f:
        f.write("Fashion Trend Prediction Results\n")
        f.write("=" * 40 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"Image {i}: {os.path.basename(result['image_path'])}\n")
            f.write(f"Prediction: {result['class_name']}\n")
            f.write(f"Confidence: {result['confidence']:.2%}\n")
            f.write(f"Probabilities: Not Trending: {result['probabilities'][0]:.2%}, "
                   f"Trending: {result['probabilities'][1]:.2%}\n")
            f.write("-" * 30 + "\n")
        
        # Summary statistics
        trending_count = sum(1 for r in results if r['predicted_class'] == 1)
        total_count = len(results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        f.write(f"\nSummary:\n")
        f.write(f"Total images: {total_count}\n")
        f.write(f"Predicted as trending: {trending_count}\n")
        f.write(f"Predicted as not trending: {total_count - trending_count}\n")
        f.write(f"Average confidence: {avg_confidence:.2%}\n")

def main(args):
    # Setup logging
    setup_logging(args.log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load model
    model = load_model(args.model_path, args.num_classes, device)
    if model is None:
        return
    
    # Class names
    class_names = ['Not Trending', 'Trending']
    
    if args.single_image:
        # Predict single image
        logging.info(f"Making prediction on: {args.single_image}")
        result = predict_single_image(model, args.single_image, device, class_names)
        
        if result:
            print(f"\nPrediction Results:")
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"Prediction: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities:")
            for i, (class_name, prob) in enumerate(zip(class_names, result['probabilities'])):
                print(f"  {class_name}: {prob:.2%}")
    
    elif args.image_dir:
        # Predict batch of images
        logging.info(f"Making predictions on images in: {args.image_dir}")
        results = predict_batch(model, args.image_dir, device, class_names)
        
        if results:
            # Print summary
            trending_count = sum(1 for r in results if r['predicted_class'] == 1)
            total_count = len(results)
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"\nPrediction Summary:")
            print(f"Total images processed: {total_count}")
            print(f"Predicted as trending: {trending_count}")
            print(f"Predicted as not trending: {total_count - trending_count}")
            print(f"Average confidence: {avg_confidence:.2%}")
            
            # Save detailed results
            output_path = os.path.join(args.log_dir, 'predictions.txt')
            save_predictions(results, output_path)
            logging.info(f"Detailed results saved to {output_path}")
    
    else:
        logging.error("Please specify either --single_image or --image_dir")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make predictions with Fashion Trend Predictor')
    parser.add_argument('--model_path', type=str, default='../logs/best_model.pth',
                      help='Path to the trained model checkpoint')
    parser.add_argument('--single_image', type=str,
                      help='Path to a single image for prediction')
    parser.add_argument('--image_dir', type=str,
                      help='Directory containing images for batch prediction')
    parser.add_argument('--log_dir', type=str, default='../logs',
                      help='Directory for saving prediction results')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of output classes')
    
    args = parser.parse_args()
    main(args) 