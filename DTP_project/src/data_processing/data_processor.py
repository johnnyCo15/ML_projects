import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import logging

class FashionDataProcessor:
    def __init__(self, 
                 image_size=(224, 224),
                 data_dir='data',
                 raw_dir='raw',
                 processed_dir='processed',
                 features_dir='features'):
        """
        Initialize the FashionDataProcessor
        
        Args:
            image_size (tuple): Target size for processed images
            data_dir (str): Root directory for data
            raw_dir (str): Directory for raw images
            processed_dir (str): Directory for processed images
            features_dir (str): Directory for extracted features
        """
        self.image_size = image_size
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, raw_dir)
        self.processed_dir = os.path.join(data_dir, processed_dir)
        self.features_dir = os.path.join(data_dir, features_dir)
        
        # Create necessary directories
        for dir_path in [self.raw_dir, self.processed_dir, self.features_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def download_images(self, csv_path, batch_size=100):
        """
        Download images from URLs in the CSV file
        
        Args:
            csv_path (str): Path to CSV file containing image URLs
            batch_size (int): Number of images to download in each batch
        """
        df = pd.read_csv(csv_path)
        total_images = len(df)
        
        for i in tqdm(range(0, total_images, batch_size), desc="Downloading images"):
            batch = df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                try:
                    url = row['URL']
                    season = row['SEASON']
                    year = row['YEAR']
                    
                    # Create season-year directory
                    save_dir = os.path.join(self.raw_dir, f"{season}_{year}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Download image
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        # Generate filename from URL
                        filename = os.path.basename(url)
                        save_path = os.path.join(save_dir, filename)
                        
                        with open(save_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                except Exception as e:
                    self.logger.error(f"Error downloading {url}: {str(e)}")

    def preprocess_image(self, image_path):
        """
        Preprocess a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.image_size)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            self.logger.error(f"Error preprocessing {image_path}: {str(e)}")
            return None

    def extract_features(self, image):
        """
        Extract features from an image
        
        Args:
            image (np.ndarray): Preprocessed image
            
        Returns:
            dict: Dictionary containing extracted features
        """
        features = {}
        
        try:
            # Color histogram
            features['color_hist'] = self._extract_color_histogram(image)
            
            # Texture features
            features['texture'] = self._extract_texture_features(image)
            
            # Edge features
            features['edges'] = self._extract_edge_features(image)
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None

    def _extract_color_histogram(self, image):
        """Extract color histogram features"""
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def _extract_texture_features(self, image):
        """Extract texture features using GLCM"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _extract_edge_features(self, image):
        """Extract edge features"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.mean(edges)

    def create_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train/validation/test splits
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of data for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        # Get all image paths
        image_paths = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        
        # Split data
        train_val_paths, test_paths = train_test_split(
            image_paths, test_size=test_size, random_state=random_state
        )
        
        train_paths, val_paths = train_test_split(
            train_val_paths, 
            test_size=val_size/(1-test_size), 
            random_state=random_state
        )
        
        return train_paths, val_paths, test_paths

class FashionDataset(Dataset):
    """Custom Dataset for fashion images"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image 