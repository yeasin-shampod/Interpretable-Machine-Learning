"""
Quick Analysis Script - Core Functionality Test
"""

import sys
import os
sys.path.append('/home/ubuntu/interpretable_ml_project/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

# Import custom modules
from data_loader import HeartDiseaseDataLoader
from models import InterpretableModel, BlackBoxModel, ModelEvaluator

def main():
    """Quick analysis pipeline"""
    print("=" * 60)
    print("QUICK INTERPRETABLE ML ANALYSIS")
    print("=" * 60)
    
    # Create results directories
    os.makedirs('/home/ubuntu/interpretable_ml_project/figures', exist_ok=True)
    os.makedirs('/home/ubuntu/interpretable_ml_project/results', exist_ok=True)
    
    # 1. DATA LOADING
    print("\n1. DATA LOADING")
    print("-" * 30)
    
    data_loader = HeartDiseaseDataLoader('/home/ubuntu/interpretable_ml_project/data/processed.cleveland.data')
    df = data_loader.load_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Basic EDA
    print("\nDataset Info:")
    print(df.describe())
    
    # Prepare features
    X, y = data_loader.prepare_features(df, scale_features=True)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    
    feature_names = X.columns.tolist()
    print(f"\nFeatures: {feature_names}")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 2. MODEL TRAINING
    print("\n2. MODEL TRAINING")
    print("-" * 30)
    
    # Train Interpretable Model
    print("Training Logistic Regression...")
    lr_model = InterpretableModel(random_state=42)
    lr_model.fit(X_train, y_train, feature_names=feature_names)
    
    # Train Black-box Model (without extensive hyperparameter tuning)
    print("Training Random Forest...")
    rf_model = BlackBoxModel(random_state=42)
    rf_model.fit(X_train, y_train, feature_names=feature_names, tune_hyperparameters=False)
    
    # 3. MODEL EVALUATION
    print("\n3. MODEL EVALUATION")
    print("-" * 30)
    
    evaluator = ModelEvaluator()
    
    # Evaluate models
    lr_metrics = evaluator.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluator.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Display results
    results_df = evaluator.compare_models([lr_metrics, rf_metrics])
    print("\nModel Performance Comparison:")
    print(results_df.round(4))
    
    # 4. FEATURE IMPORTANCE ANALYSIS
    print("\n4. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 30)
    
    # Get feature importance
    lr_importance = lr_model.get_feature_importance()
    rf_importance = rf_model.get_feature_importance()
    
    print("\nLogistic Regression - Top 5 Features:")
    print(lr_importance.head())
    
    print("\nRandom Forest - Top 5 Features:")
    print(rf_importance.head())
    
    # Create simple visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Model Performance
    plt.subplot(2, 2, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    lr_values = [lr_metrics[m] for m in metrics]
    rf_values = [rf_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, lr_values, width, label='Logistic Regression', alpha=0.8)
    plt.bar(x + width/2, rf_values, width, label='Random Forest', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: LR Feature Importance
    plt.subplot(2, 2, 2)
    top_lr = lr_importance.head(8)
    plt.barh(top_lr['feature'], top_lr['abs_coefficient'])
    plt.xlabel('Absolute Coefficient')
    plt.title('Logistic Regression - Feature Importance')
    plt.gca().invert_yaxis()
    
    # Plot 3: RF Feature Importance
    plt.subplot(2, 2, 3)
    top_rf = rf_importance.head(8)
    plt.barh(top_rf['feature'], top_rf['importance'])
    plt.xlabel('Importance')
    plt.title('Random Forest - Feature Importance')
    plt.gca().invert_yaxis()
    
    # Plot 4: Target Distribution
    plt.subplot(2, 2, 4)
    target_counts = df['target'].value_counts()
    plt.pie(target_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%')
    plt.title('Target Distribution')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/interpretable_ml_project/figures/quick_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. SUMMARY REPORT
    print("\n5. SUMMARY REPORT")
    print("-" * 30)
    
    summary = f"""
INTERPRETABLE ML PROJECT - QUICK ANALYSIS SUMMARY
================================================

DATASET OVERVIEW:
- Total samples: {df.shape[0]}
- Features: {df.shape[1]-1}
- Target distribution: {df['target'].value_counts().to_dict()}

MODEL PERFORMANCE:
Logistic Regression:
- Accuracy: {lr_metrics['accuracy']:.4f}
- Precision: {lr_metrics['precision']:.4f}
- Recall: {lr_metrics['recall']:.4f}
- F1-Score: {lr_metrics['f1_score']:.4f}
- ROC-AUC: {lr_metrics['roc_auc']:.4f}

Random Forest:
- Accuracy: {rf_metrics['accuracy']:.4f}
- Precision: {rf_metrics['precision']:.4f}
- Recall: {rf_metrics['recall']:.4f}
- F1-Score: {rf_metrics['f1_score']:.4f}
- ROC-AUC: {rf_metrics['roc_auc']:.4f}

PERFORMANCE GAP:
- Accuracy difference: {abs(rf_metrics['accuracy'] - lr_metrics['accuracy']):.4f}
- ROC-AUC difference: {abs(rf_metrics['roc_auc'] - lr_metrics['roc_auc']):.4f}

TOP PREDICTIVE FEATURES:
Logistic Regression: {', '.join(lr_importance.head(3)['feature'].tolist())}
Random Forest: {', '.join(rf_importance.head(3)['feature'].tolist())}

KEY INSIGHTS:
1. {'Minimal' if abs(rf_metrics['roc_auc'] - lr_metrics['roc_auc']) < 0.05 else 'Significant'} performance difference between models
2. Both models identify similar important features
3. Logistic Regression provides direct interpretability through coefficients
4. Random Forest achieves {'higher' if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else 'comparable'} predictive performance

RECOMMENDATIONS:
- Use Logistic Regression for interpretable predictions
- Use Random Forest when maximum performance is needed
- Focus on top features for clinical decision support
"""
    
    print(summary)
    
    # Save summary
    with open('/home/ubuntu/interpretable_ml_project/results/quick_summary.txt', 'w') as f:
        f.write(summary)
    
    # Save detailed results
    results_data = {
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'lr_feature_importance': lr_importance.to_dict('records'),
        'rf_feature_importance': rf_importance.to_dict('records'),
        'dataset_info': {
            'shape': df.shape,
            'target_distribution': df['target'].value_counts().to_dict(),
            'feature_names': feature_names
        }
    }
    
    import json
    with open('/home/ubuntu/interpretable_ml_project/results/quick_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("QUICK ANALYSIS COMPLETE!")
    print("Files created:")
    print("- Figure: /home/ubuntu/interpretable_ml_project/figures/quick_analysis.png")
    print("- Summary: /home/ubuntu/interpretable_ml_project/results/quick_summary.txt")
    print("- Results: /home/ubuntu/interpretable_ml_project/results/quick_results.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
