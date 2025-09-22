"""
Main Analysis Script for Interpretable Machine Learning Project
Heart Disease Prediction with Comprehensive Interpretability Analysis
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loader import HeartDiseaseDataLoader
from src.models import InterpretableModel, BlackBoxModel, ModelEvaluator
from src.interpretability import InterpretabilityAnalyzer
from src.experimental_design import ExperimentalDesign

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

root_path = "/Users/yas/Downloads/Interpretable_Machine_Learning_Guidelines/interpretable_ml_project"
data_path = "/Users/yas/Downloads/Interpretable_Machine_Learning_Guidelines/interpretable_ml_project/data"
dataset_name= "processed.cleveland.data"


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("INTERPRETABLE MACHINE LEARNING PROJECT")
    print("Heart Disease Prediction with Comprehensive Analysis")
    print("=" * 80)
    
    # Create results directories
    os.makedirs(os.path.join(root_path, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'results'), exist_ok=True)

    # 1. DATA LOADING AND PREPROCESSING
    print("\n1. DATA LOADING AND PREPROCESSING")
    print("-" * 40)

    data_loader = HeartDiseaseDataLoader(os.path.join(data_path, dataset_name))
    df = data_loader.load_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Exploratory Data Analysis
    create_eda_visualizations(df)
    
    # Prepare features
    X, y = data_loader.prepare_features(df, scale_features=True)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    
    feature_names = X.columns.tolist()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # 2. MODEL TRAINING
    print("\n2. MODEL TRAINING")
    print("-" * 40)
    
    # Train Interpretable Model (Logistic Regression)
    print("Training Logistic Regression...")
    lr_model = InterpretableModel(random_state=42)
    lr_model.fit(X_train, y_train, feature_names=feature_names)
    
    # Train Black-box Model (Random Forest)
    print("Training Random Forest with hyperparameter tuning...")
    rf_model = BlackBoxModel(random_state=42)
    rf_model.fit(X_train, y_train, feature_names=feature_names, tune_hyperparameters=True)
    
    print("Best RF parameters:", rf_model.best_params)
    
    # 3. MODEL EVALUATION
    print("\n3. MODEL EVALUATION")
    print("-" * 40)
    
    evaluator = ModelEvaluator()
    
    # Evaluate models
    lr_metrics = evaluator.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluator.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Cross-validation
    lr_cv = evaluator.cross_validate_model(lr_model, X_train, y_train)
    rf_cv = evaluator.cross_validate_model(rf_model, X_train, y_train)
    
    # Display results
    results_df = evaluator.compare_models([lr_metrics, rf_metrics])
    print("\nModel Performance Comparison:")
    print(results_df.round(4))
    
    print(f"\nCross-validation scores:")
    print(f"LR: {lr_cv['mean_cv_score']:.4f} ± {lr_cv['std_cv_score']:.4f}")
    print(f"RF: {rf_cv['mean_cv_score']:.4f} ± {rf_cv['std_cv_score']:.4f}")
    
    # 4. INTERPRETABILITY ANALYSIS
    print("\n4. INTERPRETABILITY ANALYSIS")
    print("-" * 40)
    
    # Setup interpretability analyzer for Random Forest
    rf_analyzer = InterpretabilityAnalyzer(rf_model, X_train, X_test, feature_names)
    rf_analyzer.setup_shap_explainer(model_type='tree')
    rf_analyzer.setup_lime_explainer()
    
    # Setup interpretability analyzer for Logistic Regression
    lr_analyzer = InterpretabilityAnalyzer(lr_model, X_train, X_test, feature_names)
    lr_analyzer.setup_shap_explainer(model_type='linear')
    lr_analyzer.setup_lime_explainer()
    
    # Get SHAP values
    print("Computing SHAP values...")
    rf_shap_values = rf_analyzer.get_shap_values()
    lr_shap_values = lr_analyzer.get_shap_values()
    
    # Create SHAP visualizations
    print("Creating SHAP visualizations...")
    rf_analyzer.plot_shap_summary(
        save_path=os.path.join(root_path, 'figures', 'rf_shap_summary.png')
    )
    
    lr_analyzer.plot_shap_summary(
        save_path=os.path.join(root_path, 'figures', 'lr_shap_summary.png')
    )
    
    # Waterfall plots for individual instances
    rf_analyzer.plot_shap_waterfall(
        instance_idx=0,
        save_path=os.path.join(root_path, 'figures', 'rf_shap_waterfall.png')
    )
    
    # LIME explanations
    print("Creating LIME explanations...")
    lime_explanations = []
    for i in range(min(5, len(X_test))):
        lime_exp = rf_analyzer.get_lime_explanation(instance_idx=i)
        lime_explanations.append(lime_exp)
    
    rf_analyzer.plot_lime_explanation(
        instance_idx=0,
        save_path='/home/ubuntu/interpretable_ml_project/figures/rf_lime_explanation.png'
    )
    
    # Feature importance comparison
    print("Creating feature importance comparison...")
    importance_comparison = rf_analyzer.create_feature_importance_comparison(
        lr_model,
        save_path=os.path.join(root_path, 'figures', 'feature_importance_comparison.html')
    )
    
    # Partial dependence plots
    print("Creating partial dependence plots...")
    top_features = importance_comparison.nlargest(4, 'rf_importance_norm')['feature'].tolist()
    rf_analyzer.plot_partial_dependence(
        top_features,
        save_path=os.path.join(root_path, 'figures', 'partial_dependence.png')
    )
    
    # Model behavior analysis
    print("Analyzing model behavior...")
    behavior_analysis = rf_analyzer.analyze_model_behavior(
        save_path=os.path.join(root_path, 'figures', 'model_behavior.html')
    )
    
    # 5. EXPERIMENTAL DESIGN AND HYPOTHESIS TESTING
    print("\n5. EXPERIMENTAL DESIGN AND HYPOTHESIS TESTING")
    print("-" * 40)
    
    experiment = ExperimentalDesign()
    
    # Test all hypotheses
    h1_results = experiment.test_hypothesis_1(lr_model, rf_model, X_test, y_test)
    h2_results = experiment.test_hypothesis_2(rf_shap_values, lime_explanations, feature_names)
    h3_results = experiment.test_hypothesis_3(rf_model.get_feature_importance(), rf_shap_values, feature_names)
    
    models_data = {
        'LogisticRegression': lr_metrics,
        'RandomForest': rf_metrics
    }
    h4_results = experiment.test_hypothesis_4(models_data)
    
    # Generate comprehensive report
    experimental_report = experiment.generate_comprehensive_report(
        save_path=os.path.join(root_path, 'results', 'experimental_report.json')
    )
    
    # Visualize experimental results
    experiment.visualize_results(
        save_path=os.path.join(root_path, 'figures', 'experimental_results.html')
    )
    
    # 6. COMPREHENSIVE RESULTS SUMMARY
    print("\n6. COMPREHENSIVE RESULTS SUMMARY")
    print("-" * 40)
    
    create_final_summary(
        lr_metrics, rf_metrics, lr_cv, rf_cv,
        importance_comparison, experimental_report,
        behavior_analysis
    )
    
    # Create final dashboard
    create_interactive_dashboard(
        df, lr_metrics, rf_metrics, importance_comparison,
        experimental_report
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("Check the following directories for results:")
    print("- Figures: interpretable_ml_project/figures/")
    print("- Results: interpretable_ml_project/results/")
    print("=" * 80)

def create_eda_visualizations(df):
    """Create exploratory data analysis visualizations"""
    print("Creating EDA visualizations...")
    
    # Create subplots for EDA
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Target Distribution', 'Age Distribution by Target', 
                       'Correlation Heatmap', 'Feature Distributions'],
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "heatmap"}, {"type": "box"}]]
    )
    
    # Target distribution
    target_counts = df['target'].value_counts()
    fig.add_trace(
        go.Bar(x=['No Disease', 'Disease'], y=target_counts.values, 
               marker_color=['lightblue', 'lightcoral']),
        row=1, col=1
    )
    
    # Age distribution by target
    fig.add_trace(
        go.Histogram(x=df[df['target']==0]['age'], name='No Disease', 
                    opacity=0.7, marker_color='lightblue'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df[df['target']==1]['age'], name='Disease', 
                    opacity=0.7, marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Correlation heatmap
    corr_matrix = df.corr()
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                  colorscale='RdBu', zmid=0),
        row=2, col=1
    )
    
    # Box plots for key features
    key_features = ['age', 'trestbps', 'chol', 'thalach']
    for i, feature in enumerate(key_features):
        fig.add_trace(
            go.Box(y=df[feature], name=feature, boxpoints='outliers'),
            row=2, col=2
        )
    
    fig.update_layout(
        title='Exploratory Data Analysis - Heart Disease Dataset',
        height=800,
        showlegend=True
    )

    fig.write_html(os.path.join(root_path, 'figures', 'eda_analysis.html'))
    fig.show()

def create_final_summary(lr_metrics, rf_metrics, lr_cv, rf_cv, 
                        importance_comparison, experimental_report, behavior_analysis):
    """Create final summary of all results"""
    
    summary_report = f"""
INTERPRETABLE MACHINE LEARNING PROJECT - FINAL SUMMARY
=====================================================

DATASET: Heart Disease Prediction (UCI ML Repository)
MODELS: Logistic Regression (Interpretable) vs Random Forest (Black-box)

PERFORMANCE COMPARISON:
----------------------
Logistic Regression:
- Accuracy: {lr_metrics['accuracy']:.4f}
- Precision: {lr_metrics['precision']:.4f}
- Recall: {lr_metrics['recall']:.4f}
- F1-Score: {lr_metrics['f1_score']:.4f}
- ROC-AUC: {lr_metrics['roc_auc']:.4f}
- CV Score: {lr_cv['mean_cv_score']:.4f} ± {lr_cv['std_cv_score']:.4f}

Random Forest:
- Accuracy: {rf_metrics['accuracy']:.4f}
- Precision: {rf_metrics['precision']:.4f}
- Recall: {rf_metrics['recall']:.4f}
- F1-Score: {rf_metrics['f1_score']:.4f}
- ROC-AUC: {rf_metrics['roc_auc']:.4f}
- CV Score: {rf_cv['mean_cv_score']:.4f} ± {rf_cv['std_cv_score']:.4f}

TOP IMPORTANT FEATURES:
----------------------
"""
    
    top_features = importance_comparison.nlargest(5, 'rf_importance_norm')
    for idx, row in top_features.iterrows():
        summary_report += f"- {row['feature']}: RF={row['importance']:.3f}, LR={row['abs_coefficient']:.3f}, SHAP={row['shap_importance']:.3f}\n"
    
    summary_report += f"""
HYPOTHESIS TESTING RESULTS:
---------------------------
"""
    
    for hypothesis, result in experimental_report['summary'].items():
        summary_report += f"{hypothesis}: {result['conclusion']} - {result['key_finding']}\n"
    
    summary_report += f"""
KEY INSIGHTS:
------------
1. Performance Trade-off: {'Minimal' if abs(rf_metrics['roc_auc'] - lr_metrics['roc_auc']) < 0.05 else 'Significant'} difference in predictive performance
2. Interpretability: Logistic Regression provides direct coefficient interpretation
3. Feature Consistency: {'High' if importance_comparison['rf_importance_norm'].corr(importance_comparison['shap_importance_norm']) > 0.7 else 'Moderate'} agreement between methods
4. Model Behavior: Mean prediction confidence = {behavior_analysis['mean_confidence']:.3f}

RECOMMENDATIONS:
---------------
1. Use Logistic Regression for high-stakes decisions requiring interpretability
2. Use Random Forest when maximum predictive performance is critical
3. SHAP values provide consistent feature importance across both models
4. Focus on top 5 features for clinical decision support
"""
    
    # Save summary
    with open('/home/ubuntu/interpretable_ml_project/results/final_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print(summary_report)

def create_interactive_dashboard(df, lr_metrics, rf_metrics, importance_comparison, experimental_report):
    """Create interactive dashboard summarizing all results"""
    
    # Create comprehensive dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Model Performance Comparison',
            'Feature Importance Comparison', 
            'Dataset Overview',
            'Hypothesis Testing Results',
            'Performance vs Interpretability',
            'Key Insights'
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "table"}]
        ]
    )
    
    # Model performance comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    lr_values = [lr_metrics[m] for m in metrics]
    rf_values = [rf_metrics[m] for m in metrics]
    
    fig.add_trace(
        go.Bar(name='Logistic Regression', x=metrics, y=lr_values, marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='Random Forest', x=metrics, y=rf_values, marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Feature importance
    top_features = importance_comparison.nlargest(8, 'rf_importance_norm')
    fig.add_trace(
        go.Bar(x=top_features['feature'], y=top_features['rf_importance_norm'], 
               name='RF Importance', marker_color='green'),
        row=1, col=2
    )
    
    # Dataset overview
    target_counts = df['target'].value_counts()
    fig.add_trace(
        go.Pie(labels=['No Disease', 'Disease'], values=target_counts.values,
               marker_colors=['lightblue', 'lightcoral']),
        row=2, col=1
    )
    
    # Hypothesis results table
    hypothesis_data = []
    for h, result in experimental_report['summary'].items():
        hypothesis_data.append([h, result['conclusion'], result['key_finding']])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Hypothesis', 'Conclusion', 'Key Finding']),
            cells=dict(values=list(zip(*hypothesis_data)))
        ),
        row=2, col=2
    )
    
    # Performance vs Interpretability scatter
    models_data = [
        ['Logistic Regression', 1, 5, lr_metrics['roc_auc']],
        ['Random Forest', 3, 2, rf_metrics['roc_auc']]
    ]
    
    fig.add_trace(
        go.Scatter(
            x=[d[1] for d in models_data],
            y=[d[2] for d in models_data],
            mode='markers+text',
            text=[d[0] for d in models_data],
            textposition='top center',
            marker=dict(
                size=[d[3]*30 for d in models_data],
                color=[d[3] for d in models_data],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="ROC-AUC")
            ),
            name='Models'
        ),
        row=3, col=1
    )
    
    # Key insights table
    insights = [
        ['Performance Gap', f"{abs(rf_metrics['roc_auc'] - lr_metrics['roc_auc']):.3f}"],
        ['Top Feature', importance_comparison.iloc[0]['feature']],
        ['Best Model', 'Random Forest' if rf_metrics['roc_auc'] > lr_metrics['roc_auc'] else 'Logistic Regression'],
        ['Interpretability Winner', 'Logistic Regression'],
        ['Recommendation', 'Context-dependent choice']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=list(zip(*insights)))
        ),
        row=3, col=2
    )
    
    fig.update_layout(
        title='Interpretable ML Project - Comprehensive Dashboard',
        height=1200,
        showlegend=True
    )

    fig.write_html('interpretable_ml_project/results/interactive_dashboard.html', include_plotlyjs='cdn')
    fig.show()

if __name__ == "__main__":
    main()
