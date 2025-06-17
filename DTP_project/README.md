Design Trend Predictor - The goal of this project is to utilize CNNs to predict future design trends based on a designer's previous works

# run "source venv/bin/activate" in ML_projects folder

Current Designer(s): 
    - Yohji Yamamoto 
    - 

# Project Roadmap

## 1. Data Preprocessing and Organization
- Download and organize images from collected URLs
- Create dataset structure:
  - Training/validation/test splits
  - Consistent image sizes
  - Proper labeling based on seasons/years
- Implement data augmentation techniques specific to fashion images

## 2. Feature Engineering
- Extract relevant features:
  - Color palettes
  - Pattern recognition
  - Silhouette analysis
  - Texture features
- Create temporal features based on season/year information
- Extract style elements specific to Yohji Yamamoto's work

## 3. Model Architecture Development
- Design CNN architecture for:
  - Processing temporal fashion data
  - Capturing style evolution over time
  - Combining CNN, LSTM/Transformer, and attention mechanisms

## 4. Implementation Plan
- Create data processing pipeline
- Develop model architecture
- Implement training framework
- Set up evaluation metrics

## 5. Technical Considerations
- Transfer learning from pre-trained models
- Custom loss functions for:
  - Style similarity
  - Temporal consistency
  - Design element preservation
- Contrastive learning for style evolution

## 6. Evaluation Metrics
- Custom metrics for fashion trend prediction
- Visualization tools for:
  - Style evolution
  - Predicted trends
  - Feature importance

## 7. Project Structure
```
DTP_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   ├── architectures/
│   └── weights/
├── src/
│   ├── data_processing/
│   ├── training/
│   └── evaluation/
├── notebooks/
└── tests/
```
