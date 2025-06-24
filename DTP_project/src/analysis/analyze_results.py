import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import logging
import argparse
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'analysis.log')),
            logging.StreamHandler()
        ]
    )

def load_prediction_results(results_path):
    """Load prediction results from file"""
    results = []
    
    with open(results_path, 'r') as f:
        lines = f.readlines()
    
    current_result = {}
    for line in lines:
        line = line.strip()
        if line.startswith('Image'):
            if current_result:
                results.append(current_result)
            current_result = {}
            # Extract image name
            image_name = line.split(': ')[1]
            current_result['image_name'] = image_name
        elif line.startswith('Prediction:'):
            prediction = line.split(': ')[1]
            current_result['prediction'] = prediction
        elif line.startswith('Confidence:'):
            confidence = float(line.split(': ')[1].rstrip('%')) / 100
            current_result['confidence'] = confidence
        elif line.startswith('Probabilities:'):
            try:
                probs = line.split(': ', 1)[1]
                prob_parts = probs.split(',')
                if len(prob_parts) == 2 and ':' in prob_parts[0] and ':' in prob_parts[1]:
                    not_trending_prob = float(prob_parts[0].split(': ')[1].rstrip('%')) / 100
                    trending_prob = float(prob_parts[1].split(': ')[1].rstrip('%')) / 100
                    current_result['not_trending_prob'] = not_trending_prob
                    current_result['trending_prob'] = trending_prob
                else:
                    logging.warning(f"Malformed probabilities line: {line}")
                    current_result['not_trending_prob'] = None
                    current_result['trending_prob'] = None
            except Exception as e:
                logging.warning(f"Error parsing probabilities line: {line} ({e})")
                current_result['not_trending_prob'] = None
                current_result['trending_prob'] = None
    
    if current_result:
        results.append(current_result)
    
    return results

def analyze_predictions(results):
    """Analyze prediction results"""
    if not results:
        logging.error("No results to analyze")
        return None
    
    df = pd.DataFrame(results)
    
    # Basic statistics
    total_images = len(df)
    trending_count = len(df[df['prediction'] == 'Trending'])
    not_trending_count = len(df[df['prediction'] == 'Not Trending'])
    
    avg_confidence = df['confidence'].mean()
    avg_trending_prob = df['trending_prob'].mean()
    avg_not_trending_prob = df['not_trending_prob'].mean()
    
    # Confidence distribution
    high_confidence = len(df[df['confidence'] >= 0.8])
    medium_confidence = len(df[(df['confidence'] >= 0.6) & (df['confidence'] < 0.8)])
    low_confidence = len(df[df['confidence'] < 0.6])
    
    analysis = {
        'total_images': total_images,
        'trending_count': trending_count,
        'not_trending_count': not_trending_count,
        'trending_percentage': (trending_count / total_images) * 100,
        'avg_confidence': avg_confidence,
        'avg_trending_prob': avg_trending_prob,
        'avg_not_trending_prob': avg_not_trending_prob,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence,
        'dataframe': df
    }
    
    return analysis

def create_visualizations(analysis, save_dir):
    """Create visualizations of the analysis"""
    df = analysis['dataframe']
    
    # 1. Prediction Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    prediction_counts = df['prediction'].value_counts()
    plt.pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%')
    plt.title('Prediction Distribution')
    
    # 2. Confidence Distribution
    plt.subplot(2, 3, 2)
    plt.hist(df['confidence'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Images')
    plt.title('Confidence Distribution')
    
    # 3. Confidence by Prediction
    plt.subplot(2, 3, 3)
    df.boxplot(column='confidence', by='prediction', ax=plt.gca())
    plt.title('Confidence by Prediction')
    plt.suptitle('')
    
    # 4. Probability Distribution
    plt.subplot(2, 3, 4)
    plt.hist(df['trending_prob'], bins=20, alpha=0.7, color='lightcoral', label='Trending Prob')
    plt.hist(df['not_trending_prob'], bins=20, alpha=0.7, color='lightgreen', label='Not Trending Prob')
    plt.xlabel('Probability')
    plt.ylabel('Number of Images')
    plt.title('Probability Distribution')
    plt.legend()
    
    # 5. Confidence vs Trending Probability
    plt.subplot(2, 3, 5)
    plt.scatter(df['confidence'], df['trending_prob'], alpha=0.6)
    plt.xlabel('Confidence')
    plt.ylabel('Trending Probability')
    plt.title('Confidence vs Trending Probability')
    
    # 6. Top/Bottom Confidence Images
    plt.subplot(2, 3, 6)
    top_5 = df.nlargest(5, 'confidence')
    bottom_5 = df.nsmallest(5, 'confidence')
    
    x_pos = np.arange(10)
    confidences = list(top_5['confidence']) + list(bottom_5['confidence'])
    colors = ['green'] * 5 + ['red'] * 5
    
    plt.bar(x_pos, confidences, color=colors, alpha=0.7)
    plt.xlabel('Image Rank')
    plt.ylabel('Confidence')
    plt.title('Top 5 (Green) vs Bottom 5 (Red) Confidence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Visualizations saved to {save_dir}")

def generate_report(analysis, save_path):
    """Generate a comprehensive analysis report"""
    with open(save_path, 'w') as f:
        f.write("FASHION TREND PREDICTION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Images Analyzed: {analysis['total_images']}\n")
        f.write(f"Predicted as Trending: {analysis['trending_count']} ({analysis['trending_percentage']:.1f}%)\n")
        f.write(f"Predicted as Not Trending: {analysis['not_trending_count']} ({100-analysis['trending_percentage']:.1f}%)\n\n")
        
        f.write("CONFIDENCE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Confidence: {analysis['avg_confidence']:.3f} ({analysis['avg_confidence']*100:.1f}%)\n")
        f.write(f"High Confidence (≥80%): {analysis['high_confidence']} images\n")
        f.write(f"Medium Confidence (60-80%): {analysis['medium_confidence']} images\n")
        f.write(f"Low Confidence (<60%): {analysis['low_confidence']} images\n\n")
        
        f.write("PROBABILITY ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Average Trending Probability: {analysis['avg_trending_prob']:.3f}\n")
        f.write(f"Average Not Trending Probability: {analysis['avg_not_trending_prob']:.3f}\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 20 + "\n")
        
        if analysis['avg_confidence'] < 0.7:
            f.write("⚠️  Model confidence is relatively low. Consider:\n")
            f.write("   - Training for more epochs\n")
            f.write("   - Adding more training data\n")
            f.write("   - Adjusting model architecture\n\n")
        
        if analysis['trending_percentage'] > 80 or analysis['trending_percentage'] < 20:
            f.write("⚠️  Model predictions are heavily biased. Consider:\n")
            f.write("   - Checking class balance in training data\n")
            f.write("   - Adjusting loss function weights\n")
            f.write("   - Reviewing data preprocessing\n\n")
        
        if analysis['high_confidence'] < analysis['total_images'] * 0.3:
            f.write("⚠️  Few high-confidence predictions. Consider:\n")
            f.write("   - Model may need more training\n")
            f.write("   - Data may be too diverse for current model\n")
            f.write("   - Feature extraction may need improvement\n\n")
        
        f.write("✅ Model appears to be working well if:\n")
        f.write("   - Confidence is above 70%\n")
        f.write("   - Predictions are reasonably balanced\n")
        f.write("   - High-confidence predictions are present\n\n")
    
    logging.info(f"Analysis report saved to {save_path}")

def main(args):
    # Setup logging
    setup_logging(args.log_dir)
    
    # Load prediction results
    logging.info(f"Loading prediction results from {args.results_path}")
    results = load_prediction_results(args.results_path)
    
    if not results:
        logging.error("No results found. Please run prediction first.")
        return
    
    # Analyze results
    logging.info("Analyzing prediction results...")
    analysis = analyze_predictions(results)
    
    if analysis is None:
        return
    
    # Print summary
    print(f"\nANALYSIS SUMMARY:")
    print(f"Total Images: {analysis['total_images']}")
    print(f"Trending Predictions: {analysis['trending_count']} ({analysis['trending_percentage']:.1f}%)")
    print(f"Average Confidence: {analysis['avg_confidence']:.3f} ({analysis['avg_confidence']*100:.1f}%)")
    print(f"High Confidence Images: {analysis['high_confidence']}")
    
    # Create visualizations
    logging.info("Creating visualizations...")
    create_visualizations(analysis, args.log_dir)
    
    # Generate report
    logging.info("Generating analysis report...")
    report_path = os.path.join(args.log_dir, 'analysis_report.txt')
    generate_report(analysis, report_path)
    
    logging.info("Analysis completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Fashion Trend Prediction Results')
    parser.add_argument('--results_path', type=str, default='../logs/predictions.txt',
                      help='Path to prediction results file')
    parser.add_argument('--log_dir', type=str, default='../logs',
                      help='Directory for saving analysis results')
    
    args = parser.parse_args()
    main(args) 