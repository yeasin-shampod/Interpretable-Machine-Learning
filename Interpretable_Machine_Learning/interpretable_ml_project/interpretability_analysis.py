"""
Comprehensive Interpretability Analysis
SHAP, LIME, and Advanced Interpretability Methods
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Import custom modules
from src.data_loader import HeartDiseaseDataLoader
from src.models import InterpretableModel, BlackBoxModel, ModelEvaluator
from src.interpretability import InterpretabilityAnalyzer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("shap").setLevel(logging.WARNING)


root_path = "/Users/yas/Downloads/Interpretable_Machine_Learning_Guidelines/interpretable_ml_project"
data_path = "/Users/yas/Downloads/Interpretable_Machine_Learning_Guidelines/interpretable_ml_project/data"
dataset_name= "processed.cleveland.data"

def main():
    """Comprehensive interpretability analysis"""
    logging.info("=" * 70)
    logging.info("COMPREHENSIVE INTERPRETABILITY ANALYSIS")
    logging.info("=" * 70)
    
    # Create directories
    os.makedirs(os.path.join(root_path, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'results'), exist_ok=True)

    # 1. SETUP DATA AND MODELS
    logging.info("1. SETUP DATA AND MODELS")
    logging.info("-" * 40)

    # loading my selected data
    data_loader = HeartDiseaseDataLoader( data_path=os.path.join(data_path, dataset_name) )
    df = data_loader.load_data()

    # Now, I am preparing features and splitting data
    X, y = data_loader.prepare_features(df, scale_features=True)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    feature_names = X.columns.tolist()

    logging.info(f"Data loaded with {X.shape[0]} samples and {X.shape[1]} features")
    
    logging.info(f"Training set: {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")

    # Train models
    lr_model = InterpretableModel(random_state=42)
    lr_model.fit(X_train, y_train, feature_names=feature_names)
    
    rf_model = BlackBoxModel(random_state=42)
    rf_model.fit(X_train, y_train, feature_names=feature_names, tune_hyperparameters=False)
    
    logging.info(f"Models trained on {X_train.shape[0]} samples")
    logging.info(f"Test set: {X_test.shape[0]} samples")
    
    # 2. SHAP ANALYSIS
    logging.info("2. SHAP ANALYSIS")
    logging.info("-" * 40)
    
    # Setup SHAP for Random Forest
    logging.info("Setting up SHAP for Random Forest...")
    rf_analyzer = InterpretabilityAnalyzer(rf_model, X_train, X_test, feature_names)
    rf_analyzer.setup_shap_explainer(model_type='tree')
    
    # Get SHAP values
    logging.info("Computing SHAP values...")
    rf_shap_values = rf_analyzer.get_shap_values()
    
    # Handle SHAP values format
    if isinstance(rf_shap_values, list):
        rf_shap_values_pos = rf_shap_values[1]  # Positive class
        expected_value = rf_analyzer.shap_explainer.expected_value[1]
    else:
        # For newer SHAP versions, values might be 3D
        if len(rf_shap_values.shape) == 3:
            rf_shap_values_pos = rf_shap_values[:, :, 1]  # Positive class
        else:
            rf_shap_values_pos = rf_shap_values
        expected_value = rf_analyzer.shap_explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    
    logging.info(f"SHAP values shape: {rf_shap_values_pos.shape}")
    logging.info(f"Expected value: {float(expected_value):.4f}")
    
    # Create SHAP summary plot
    logging.info("Creating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(rf_shap_values_pos, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Random Forest\nFeature Impact on Heart Disease Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'figures/shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(rf_shap_values_pos, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'figures/shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create SHAP waterfall plot for first instance
    logging.info("Creating SHAP waterfall plot...")
    plt.figure(figsize=(10, 6))
    
    # Create explanation object for waterfall plot
    shap_explanation = shap.Explanation(
        values=rf_shap_values_pos[0],
        base_values=expected_value,
        data=X_test.iloc[0].values,
        feature_names=feature_names
    )
    
    shap.waterfall_plot(shap_explanation, show=False)
    plt.title('SHAP Waterfall Plot - Individual Prediction (Instance 0)')
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'figures/shap_waterfall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. LIME ANALYSIS
    logging.info("3. LIME ANALYSIS")
    logging.info("-" * 40)
    
    # Setup LIME
    logging.info("Setting up LIME explainer...")
    rf_analyzer.setup_lime_explainer()
    
    # Get LIME explanations for first few instances
    logging.info("Computing LIME explanations...")
    lime_explanations = []
    for i in range(min(3, len(X_test))):
        explanation = rf_analyzer.get_lime_explanation(instance_idx=i, num_features=8)
        lime_explanations.append(explanation)
    
    # Plot LIME explanation for first instance
    logging.info("Creating LIME visualization...")
    explanation = lime_explanations[0]
    lime_values = explanation.as_list()
    features = [item[0] for item in lime_values]
    values = [item[1] for item in lime_values]
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if v < 0 else 'green' for v in values]
    plt.barh(features, values, color=colors, alpha=0.7)
    plt.xlabel('Feature Contribution to Prediction')
    plt.title('LIME Explanation - Individual Prediction (Instance 0)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'figures/lime_explanation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. FEATURE IMPORTANCE COMPARISON
    logging.info("4. FEATURE IMPORTANCE COMPARISON")
    logging.info("-" * 40)
    
    # Get different importance measures
    rf_importance = rf_model.get_feature_importance()
    lr_importance = lr_model.get_feature_importance()
    
    # SHAP global importance
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(rf_shap_values_pos).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    # Merge all importance measures
    comparison_df = rf_importance.merge(
        lr_importance[['feature', 'abs_coefficient']], on='feature', how='outer'
    ).merge(
        shap_importance, on='feature', how='outer'
    )
    
    # Normalize importance scores
    comparison_df['rf_importance_norm'] = comparison_df['importance'] / comparison_df['importance'].max()
    comparison_df['lr_importance_norm'] = comparison_df['abs_coefficient'] / comparison_df['abs_coefficient'].max()
    comparison_df['shap_importance_norm'] = comparison_df['shap_importance'] / comparison_df['shap_importance'].max()
    
    logging.info("Feature Importance Comparison (Top 8):")
    logging.info(comparison_df.head(8)[['feature', 'rf_importance_norm', 'lr_importance_norm', 'shap_importance_norm']].round(3))
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Random Forest importance
    top_rf = comparison_df.nlargest(8, 'rf_importance_norm')
    axes[0].barh(top_rf['feature'], top_rf['rf_importance_norm'], color='lightblue', alpha=0.8)
    axes[0].set_xlabel('Normalized Importance')
    axes[0].set_title('Random Forest\nFeature Importance')
    axes[0].invert_yaxis()
    
    # Logistic Regression importance
    top_lr = comparison_df.nlargest(8, 'lr_importance_norm')
    axes[1].barh(top_lr['feature'], top_lr['lr_importance_norm'], color='lightcoral', alpha=0.8)
    axes[1].set_xlabel('Normalized Importance')
    axes[1].set_title('Logistic Regression\nFeature Importance')
    axes[1].invert_yaxis()
    
    # SHAP importance
    top_shap = comparison_df.nlargest(8, 'shap_importance_norm')
    axes[2].barh(top_shap['feature'], top_shap['shap_importance_norm'], color='lightgreen', alpha=0.8)
    axes[2].set_xlabel('Normalized Importance')
    axes[2].set_title('SHAP Values\nGlobal Importance')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'figures/importance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. PARTIAL DEPENDENCE PLOTS
    logging.info("5. PARTIAL DEPENDENCE PLOTS")
    logging.info("-" * 40)
    
    # Get top 4 features for PDP
    top_features = comparison_df.nlargest(4, 'rf_importance_norm')['feature'].tolist()
    logging.info(f"Creating partial dependence plots for: {top_features}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        feature_idx = feature_names.index(feature)
        
        # Compute partial dependence
        pd_result = partial_dependence(
            rf_model.model, X_train, [feature_idx], 
            kind='average', grid_resolution=50
        )
        
        axes[i].plot(pd_result['grid_values'][0], pd_result['average'][0], 'b-', linewidth=2)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Partial Dependence')
        axes[i].set_title(f'Partial Dependence - {feature}')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Partial Dependence Plots - Top Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, 'figures/partial_dependence.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. INDIVIDUAL PREDICTION ANALYSIS
    logging.info("6. INDIVIDUAL PREDICTION ANALYSIS")
    logging.info("-" * 40)
    
    # Analyze a few individual predictions
    n_instances = min(3, len(X_test))
    
    for i in range(n_instances):
        instance = X_test.iloc[i]
        true_label = y_test.iloc[i]
        rf_pred = rf_model.predict([instance.values])[0]
        rf_prob = rf_model.predict_proba([instance.values])[0, 1]
        lr_pred = lr_model.predict([instance.values])[0]
        lr_prob = lr_model.predict_proba([instance.values])[0, 1]

        logging.info(f"Instance {i}:")
        logging.info(f"  True label: {true_label}")
        logging.info(f"  RF prediction: {rf_pred} (prob: {rf_prob:.3f})")
        logging.info(f"  LR prediction: {lr_pred} (prob: {lr_prob:.3f})")

        # Top SHAP contributions
        shap_contrib = pd.DataFrame({
            'feature': feature_names,
            'shap_value': rf_shap_values_pos[i]
        }).sort_values('shap_value', key=abs, ascending=False)

        logging.info("  Top SHAP contributions:")
        for _, row in shap_contrib.head(3).iterrows():
            logging.info(f"    {row['feature']}: {row['shap_value']:.3f}")
    
    # 7. CORRELATION ANALYSIS
    logging.info("7. CORRELATION ANALYSIS")
    logging.info("-" * 40)
    
    # Correlation between different importance measures
    rf_shap_corr = comparison_df['rf_importance_norm'].corr(comparison_df['shap_importance_norm'])
    lr_shap_corr = comparison_df['lr_importance_norm'].corr(comparison_df['shap_importance_norm'])
    rf_lr_corr = comparison_df['rf_importance_norm'].corr(comparison_df['lr_importance_norm'])
    
    logging.info(f"Correlation between importance measures:")
    logging.info(f"  RF vs SHAP: {rf_shap_corr:.3f}")
    logging.info(f"  LR vs SHAP: {lr_shap_corr:.3f}")
    logging.info(f"  RF vs LR: {rf_lr_corr:.3f}")
    
    # 8. SAVE COMPREHENSIVE RESULTS
    logging.info("8. SAVING RESULTS")
    logging.info("-" * 40)
    
    # Prepare comprehensive results
    results = {
        'feature_importance_comparison': comparison_df.to_dict('records'),
        'shap_analysis': {
            'expected_value': float(expected_value),
            'feature_importance': shap_importance.to_dict('records'),
            'top_features': shap_importance.head(5)['feature'].tolist()
        },
        'lime_analysis': {
            'explanations': [exp.as_list() for exp in lime_explanations]
        },
        'correlations': {
            'rf_shap': float(rf_shap_corr),
            'lr_shap': float(lr_shap_corr),
            'rf_lr': float(rf_lr_corr)
        },
        'top_features_consensus': comparison_df.nlargest(5, 'shap_importance_norm')['feature'].tolist()
    }
    
    # Save results
    import json
    with open(os.path.join(root_path, 'results/interpretability_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    summary = f"""
COMPREHENSIVE INTERPRETABILITY ANALYSIS SUMMARY
==============================================

DATASET: Heart Disease (UCI) - {X_train.shape[0]} train, {X_test.shape[0]} test samples

SHAP ANALYSIS:
- Expected value (baseline): {expected_value:.4f}
- Top 3 features by SHAP importance:
  1. {shap_importance.iloc[0]['feature']}: {shap_importance.iloc[0]['shap_importance']:.3f}
  2. {shap_importance.iloc[1]['feature']}: {shap_importance.iloc[1]['shap_importance']:.3f}
  3. {shap_importance.iloc[2]['feature']}: {shap_importance.iloc[2]['shap_importance']:.3f}

FEATURE IMPORTANCE CORRELATIONS:
- Random Forest vs SHAP: {rf_shap_corr:.3f}
- Logistic Regression vs SHAP: {lr_shap_corr:.3f}
- Random Forest vs Logistic Regression: {rf_lr_corr:.3f}

CONSENSUS TOP FEATURES:
{', '.join(results['top_features_consensus'])}

INTERPRETABILITY INSIGHTS:
1. {'High' if rf_shap_corr > 0.7 else 'Moderate' if rf_shap_corr > 0.4 else 'Low'} consistency between RF and SHAP importance
2. {'High' if lr_shap_corr > 0.7 else 'Moderate' if lr_shap_corr > 0.4 else 'Low'} consistency between LR and SHAP importance
3. SHAP provides model-agnostic explanations
4. LIME offers local instance-specific explanations
5. Partial dependence plots reveal feature-target relationships

FILES GENERATED:
- SHAP summary plot: shap_summary.png
- SHAP bar plot: shap_bar.png
- SHAP waterfall plot: shap_waterfall.png
- LIME explanation: lime_explanation.png
- Feature importance comparison: importance_comparison.png
- Partial dependence plots: partial_dependence.png
- Comprehensive results: interpretability_results.json
"""
    
    logging.info(summary)

    with open(os.path.join(root_path, 'results/interpretability_summary.txt'), 'w') as f:
        f.write(summary)
    
    logging.info("=" * 70)
    logging.info("INTERPRETABILITY ANALYSIS COMPLETE!")
    logging.info("All visualizations and results saved to figures/ and results/ directories")
    logging.info("=" * 70)

if __name__ == "__main__":
    main()
