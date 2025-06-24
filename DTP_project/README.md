Design Trend Predictor - The goal of this project is to utilize CNNs to predict future design trends based on a designer's previous works

# Setup & Installation

1. Create and activate your virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install PyTorch (Apple Silicon/CPU):
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   ```
3. Install other requirements:
   ```bash
   pip install -r requirements.txt
   ```

# Current Designer(s):
- Yohji Yamamoto

# Current Progress
- [x] Image scraping from designer URLs
- [x] Data download and organization
- [x] Dataset splitting (train/val/test)
- [x] Data pipeline and feature extraction
- [x] Model architecture (CNN-based)
- [x] Training framework with TensorBoard logging
- [x] Evaluation and metrics reporting
- [x] Prediction and analysis scripts
- [ ] Advanced feature engineering (e.g., temporal/style features)
- [ ] Model improvements (e.g., LSTM/Transformer integration)
- [ ] Full data augmentation pipeline
- [ ] Additional designers

# Project Roadmap

## 1. Data Preprocessing and Organization
- [x] Download and organize images from collected URLs
- [x] Create dataset structure (train/val/test splits, consistent image sizes, labeling)
- [ ] Implement data augmentation techniques specific to fashion images

## 2. Feature Engineering
- [x] Extract basic features (color, pattern, silhouette, texture)
- [ ] Create temporal features based on season/year information
- [ ] Extract style elements specific to Yohji Yamamoto's work

## 3. Model Architecture Development
- [x] Design CNN architecture for fashion data
- [ ] Integrate temporal modeling (LSTM/Transformer, attention)

## 4. Implementation Plan
- [x] Create data processing pipeline
- [x] Develop model architecture
- [x] Implement training framework
- [x] Set up evaluation metrics

## 5. Technical Considerations
- [ ] Transfer learning from pre-trained models
- [ ] Custom loss functions (style similarity, temporal consistency, design element preservation)
- [ ] Contrastive learning for style evolution

## 6. Evaluation Metrics
- [x] Custom metrics for fashion trend prediction
- [x] Visualization tools for style evolution, predicted trends, feature importance

# Running Training with MPS (Apple Silicon)
The training script will automatically use the MPS device if available. No changes are needed, but you can check the logs to confirm the device in use.

# Project Structure
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
