import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FashionTrendPredictor(nn.Module):
    def __init__(self, num_classes, feature_dim=512, temporal_dim=64):
        """
        Initialize the Fashion Trend Predictor model
        
        Args:
            num_classes (int): Number of output classes (e.g., number of seasons/years to predict)
            feature_dim (int): Dimension of the CNN features
            temporal_dim (int): Dimension of the temporal features
        """
        super(FashionTrendPredictor, self).__init__()
        
        # Load pre-trained ResNet50 as the base CNN
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Temporal processing layers
        self.temporal_processor = nn.Sequential(
            nn.Linear(2, temporal_dim),  # 2 features: season and year
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Combined feature processing
        self.combined_processor = nn.Sequential(
            nn.Linear(feature_dim + temporal_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images, temporal_features):
        """
        Forward pass
        
        Args:
            images (torch.Tensor): Batch of images [batch_size, channels, height, width]
            temporal_features (torch.Tensor): Temporal features [batch_size, 2] (season, year)
            
        Returns:
            torch.Tensor: Predicted fashion trends
        """
        # Process images through CNN
        batch_size = images.size(0)
        cnn_features = self.cnn(images)
        cnn_features = cnn_features.view(batch_size, -1)  # Flatten to [batch_size, 2048]
        cnn_features = self.feature_processor(cnn_features)  # [batch_size, feature_dim]
        
        # Process temporal features
        temporal_features = self.temporal_processor(temporal_features)  # [batch_size, temporal_dim]
        
        # Combine features
        combined_features = torch.cat([cnn_features, temporal_features], dim=1)  # [batch_size, feature_dim + temporal_dim]
        
        # Final prediction
        predictions = self.combined_processor(combined_features)  # [batch_size, num_classes]
        
        return predictions

class FashionTrendLoss(nn.Module):
    def __init__(self, style_weight=0.4, temporal_weight=0.3, trend_weight=0.3):
        """
        Custom loss function for fashion trend prediction
        
        Args:
            style_weight (float): Weight for style consistency loss
            temporal_weight (float): Weight for temporal consistency loss
            trend_weight (float): Weight for trend prediction loss
        """
        super(FashionTrendLoss, self).__init__()
        self.style_weight = style_weight
        self.temporal_weight = temporal_weight
        self.trend_weight = trend_weight
        
    def forward(self, predictions, targets, style_features, temporal_features):
        """
        Compute the combined loss
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels
            style_features (torch.Tensor): Style features from the model
            temporal_features (torch.Tensor): Temporal features from the model
            
        Returns:
            torch.Tensor: Combined loss value (scalar)
        """
        # Cross entropy loss for trend prediction
        trend_loss = F.cross_entropy(predictions, targets)
        
        # Style consistency loss (cosine similarity)
        style_loss = 1 - F.cosine_similarity(style_features, style_features.mean(dim=0, keepdim=True)).mean()
        
        # Temporal consistency loss (smoothness)
        temporal_loss = F.mse_loss(temporal_features[1:], temporal_features[:-1])
        
        # Combine losses
        total_loss = (self.trend_weight * trend_loss + 
                     self.style_weight * style_loss + 
                     self.temporal_weight * temporal_loss)
        
        return total_loss 