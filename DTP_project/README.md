Design Trend Predictor
=====================
The goal of this project is to utilize CNNs to predict future design trends based on a designer's previous works.

# Setup & Installation

1. **Create and activate your virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install PyTorch (Apple Silicon/CPU):**
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   ```
3. **Install other requirements:**
   ```bash
   pip install -r requirements.txt
   ```

# Pipeline Overview

## 1. Image Scraping
- **Script:** `ImageScraper.py`
- **Command:**
  ```bash
  cd ML_projects/DTP_project
  python ImageScraper.py
  ```
- **Output:** `yohji_images.csv` or `working.csv` with image URLs and metadata.

## 2. Data Processing & Feature Extraction
- **Script:** `src/data_processing/process_data.py`
- **Command:**
  ```bash
  cd src/data_processing
  python process_data.py
  ```
- **Output:**
  - Images organized into `data/processed/train/`, `val/`, `test/`
  - Features saved in `data/features/train/`, `val/`, `test/`

## 3. Model Training
- **Script:** `src/training/train.py`
- **Command:**
  ```bash
  cd ../training
  python train.py --epochs 100 --batch_size 16 --num_workers 4 --learning_rate 0.001
  ```
- **Output:** Model checkpoints and logs in `logs/`

## 4. Model Evaluation
- **Script:** `src/evaluation/evaluate.py`
- **Command:**
  ```bash
  cd ../evaluation
  python evaluate.py --model_path ../logs/best_model.pth --data_dir ../data --log_dir ../logs
  ```
- **Output:** Evaluation results and visualizations in `logs/`

## 5. Prediction
- **Script:** `src/prediction/predict.py`
- **Command:**
  ```bash
  cd ../prediction
  python predict.py --model_path ../logs/best_model.pth --image_dir ../data/processed/test/ --log_dir ../logs
  ```
- **Output:** Prediction results in `logs/predictions.txt`

## 6. Analysis
- **Script:** `src/analysis/analyze_results.py`
- **Command:**
  ```bash
  cd ../analysis
  python analyze_results.py --results_path ../logs/predictions.txt --log_dir ../logs
  ```
- **Output:** Analysis report and visualizations in `logs/`

# Project Structure
```
DTP_project/
├── data/
│   ├── raw/         # Downloaded images
│   ├── processed/   # Images split into train/val/test
│   └── features/    # Extracted features split into train/val/test
├── models/
│   ├── architectures/
│   └── weights/
├── src/
│   ├── data_processing/
│   ├── training/
│   ├── evaluation/
│   ├── prediction/
│   └── analysis/
├── notebooks/
└── tests/
```

# Notes
- The pipeline is fully functional from data collection to analysis.
- The training script will automatically use the MPS device on Apple Silicon if available.
- For troubleshooting or further improvements (e.g., data augmentation, advanced features, new designers), see the code comments and documentation.
