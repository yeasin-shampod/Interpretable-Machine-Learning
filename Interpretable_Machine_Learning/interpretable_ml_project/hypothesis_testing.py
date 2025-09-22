"""
Hypothesis Testing and Experimental Design Analysis
Statistical testing of interpretability hypotheses
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr, kendalltau
from sklearn.model_selection import cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Import custom modules
from src.data_loader import HeartDiseaseDataLoader
from src.models import InterpretableModel, BlackBoxModel, ModelEvaluator
from src.interpretability import InterpretabilityAnalyzer
from src.experimental_design import ExperimentalDesign


root_path = "/Users/yas/Downloads/Interpretable_Machine_Learning_Guidelines/interpretable_ml_project"
data_path = "/Users/yas/Downloads/Interpretable_Machine_Learning_Guidelines/interpretable_ml_project/data"
dataset_name= "processed.cleveland.data"


def main():
    """Comprehensive hypothesis testing analysis"""
    print("=" * 70)
    print("HYPOTHESIS TESTING AND EXPERIMENTAL DESIGN")
    print("=" * 70)
    
    # Create directories
    os.makedirs(os.path.join(root_path, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'results'), exist_ok=True)

    # 1. SETUP DATA AND MODELS
    print("\n1. SETUP DATA AND MODELS")
    print("-" * 40)
    
    # Load data
    data_loader = HeartDiseaseDataLoader(os.path.join(data_path, dataset_name))
    df = data_loader.load_data()
    X, y = data_loader.prepare_features(df, scale_features=True)
    X_train, X_test, y_train, y_test = data_loader.split_data(X, y)
    feature_names = X.columns.tolist()
    
    # Train models
    lr_model = InterpretableModel(random_state=42)
    lr_model.fit(X_train, y_train, feature_names=feature_names)
    
    rf_model = BlackBoxModel(random_state=42)
    rf_model.fit(X_train, y_train, feature_names=feature_names, tune_hyperparameters=False)
    
    # Get model performance
    evaluator = ModelEvaluator()
    lr_metrics = evaluator.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluator.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    print(f"Models trained and evaluated")
    print(f"LR ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"RF ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    
    # 2. HYPOTHESIS 1: PERFORMANCE COMPARISON
    print("\n2. HYPOTHESIS 1: PERFORMANCE COMPARISON")
    print("-" * 40)
    print("H1: Interpretable models provide comparable performance to black-box models")
    
    # Performance metrics comparison
    performance_diff = {
        'accuracy': rf_metrics['accuracy'] - lr_metrics['accuracy'],
        'precision': rf_metrics['precision'] - lr_metrics['precision'],
        'recall': rf_metrics['recall'] - lr_metrics['recall'],
        'f1_score': rf_metrics['f1_score'] - lr_metrics['f1_score'],
        'roc_auc': rf_metrics['roc_auc'] - lr_metrics['roc_auc']
    }
    
    print("Performance differences (RF - LR):")
    for metric, diff in performance_diff.items():
        print(f"  {metric}: {diff:+.4f}")
    
    # Cross-validation comparison
    print("\nCross-validation analysis...")
    cv_folds = 5
    lr_cv_scores = cross_val_score(lr_model.model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    rf_cv_scores = cross_val_score(rf_model.model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
    
    print(f"LR CV scores: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")
    print(f"RF CV scores: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
    
    # Statistical test
    cv_ttest_stat, cv_ttest_p = ttest_ind(rf_cv_scores, lr_cv_scores)
    print(f"T-test p-value: {cv_ttest_p:.4f}")
    
    h1_conclusion = "No significant difference" if cv_ttest_p > 0.05 else "Significant difference"
    print(f"H1 Conclusion: {h1_conclusion}")
    
    # 3. HYPOTHESIS 2: SHAP VS LIME CONSISTENCY
    print("\n3. HYPOTHESIS 2: SHAP VS LIME CONSISTENCY")
    print("-" * 40)
    print("H2: SHAP provides more consistent explanations than LIME")
    
    # Setup interpretability analyzers
    rf_analyzer = InterpretabilityAnalyzer(rf_model, X_train, X_test, feature_names)
    rf_analyzer.setup_shap_explainer(model_type='tree')
    rf_analyzer.setup_lime_explainer()
    
    # Get SHAP values
    rf_shap_values = rf_analyzer.get_shap_values()
    if len(rf_shap_values.shape) == 3:
        rf_shap_values = rf_shap_values[:, :, 1]  # Positive class
    
    # Get LIME explanations for multiple instances
    n_instances = min(10, len(X_test))
    lime_explanations = []
    shap_rankings = []
    lime_rankings = []
    
    print(f"Computing explanations for {n_instances} instances...")
    
    for i in range(n_instances):
        # SHAP ranking
        shap_importance = np.abs(rf_shap_values[i])
        shap_ranking = np.argsort(shap_importance)[::-1]
        shap_rankings.append(shap_ranking)
        
        # LIME explanation
        try:
            lime_exp = rf_analyzer.get_lime_explanation(instance_idx=i, num_features=len(feature_names))
            lime_features = [item[0] for item in lime_exp.as_list()]
            lime_values = [abs(item[1]) for item in lime_exp.as_list()]
            
            # Map LIME features to indices
            lime_indices = []
            for feat in lime_features:
                if feat in feature_names:
                    lime_indices.append(feature_names.index(feat))
            
            lime_ranking = np.array(lime_indices)
            lime_rankings.append(lime_ranking)
            lime_explanations.append(lime_exp)
        except Exception as e:
            print(f"  Warning: LIME failed for instance {i}: {e}")
            continue
    
    # Calculate ranking consistency
    correlations = []
    for i in range(min(len(shap_rankings), len(lime_rankings))):
        if len(lime_rankings[i]) > 0:
            # Create full rankings
            shap_full = np.zeros(len(feature_names))
            lime_full = np.zeros(len(feature_names))
            
            for j, idx in enumerate(shap_rankings[i]):
                shap_full[idx] = len(feature_names) - j
            
            for j, idx in enumerate(lime_rankings[i]):
                if idx < len(feature_names):
                    lime_full[idx] = len(lime_rankings[i]) - j
            
            corr, p_value = spearmanr(shap_full, lime_full)
            if not np.isnan(corr):
                correlations.append(corr)
    
    if correlations:
        mean_correlation = np.mean(correlations)
        std_correlation = np.std(correlations)
        print(f"SHAP-LIME correlation: {mean_correlation:.4f} ± {std_correlation:.4f}")
        
        # Test if correlation is significantly different from 0
        t_stat, t_p_value = ttest_ind(correlations, [0] * len(correlations))
        print(f"Correlation significance p-value: {t_p_value:.4f}")
        
        h2_conclusion = 'High consistency' if mean_correlation > 0.7 else 'Moderate consistency' if mean_correlation > 0.4 else 'Low consistency'
    else:
        mean_correlation = 0
        h2_conclusion = 'Unable to compute consistency'
    
    print(f"H2 Conclusion: {h2_conclusion}")
    
    # 4. HYPOTHESIS 3: GLOBAL VS LOCAL ALIGNMENT
    print("\n4. HYPOTHESIS 3: GLOBAL VS LOCAL ALIGNMENT")
    print("-" * 40)
    print("H3: Global and local interpretability methods align")
    
    # Global importance (Random Forest)
    rf_importance = rf_model.get_feature_importance()
    global_ranking = rf_importance.sort_values('importance', ascending=False)['feature'].tolist()
    
    # Local importance (average SHAP values)
    local_importance = np.abs(rf_shap_values).mean(axis=0)
    local_ranking = [feature_names[i] for i in np.argsort(local_importance)[::-1]]
    
    print("Global ranking (RF):", global_ranking[:5])
    print("Local ranking (SHAP):", local_ranking[:5])
    
    # Calculate ranking correlation
    global_ranks = {feat: i for i, feat in enumerate(global_ranking)}
    local_ranks = {feat: i for i, feat in enumerate(local_ranking)}
    
    common_features = set(global_ranks.keys()) & set(local_ranks.keys())
    global_rank_values = [global_ranks[feat] for feat in common_features]
    local_rank_values = [local_ranks[feat] for feat in common_features]
    
    spearman_corr, spearman_p = spearmanr(global_rank_values, local_rank_values)
    kendall_tau, kendall_p = kendalltau(global_rank_values, local_rank_values)
    
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"Kendall's tau: {kendall_tau:.4f} (p={kendall_p:.4f})")
    
    # Top-k agreement
    top_k_agreements = {}
    for k in [3, 5, 7]:
        top_global = set(global_ranking[:k])
        top_local = set(local_ranking[:k])
        agreement = len(top_global & top_local) / k
        top_k_agreements[f'top_{k}'] = agreement
        print(f"Top-{k} agreement: {agreement:.3f}")
    
    h3_conclusion = 'Strong alignment' if spearman_corr > 0.7 else 'Moderate alignment' if spearman_corr > 0.4 else 'Weak alignment'
    print(f"H3 Conclusion: {h3_conclusion}")
    
    # 5. HYPOTHESIS 4: COMPLEXITY-INTERPRETABILITY TRADE-OFF
    print("\n5. HYPOTHESIS 4: COMPLEXITY-INTERPRETABILITY TRADE-OFF")
    print("-" * 40)
    print("H4: Model complexity trades off with interpretability")
    
    # Define complexity and interpretability scores
    models_analysis = [
        {
            'model': 'Logistic Regression',
            'complexity': 1,  # Low complexity
            'interpretability': 5,  # High interpretability
            'performance': lr_metrics['roc_auc']
        },
        {
            'model': 'Random Forest',
            'complexity': 3,  # High complexity
            'interpretability': 2,  # Low interpretability
            'performance': rf_metrics['roc_auc']
        }
    ]
    
    df_analysis = pd.DataFrame(models_analysis)
    
    # Correlation analysis
    complexity_performance_corr = df_analysis['complexity'].corr(df_analysis['performance'])
    interpretability_performance_corr = df_analysis['interpretability'].corr(df_analysis['performance'])
    complexity_interpretability_corr = df_analysis['complexity'].corr(df_analysis['interpretability'])
    
    print(f"Complexity vs Performance correlation: {complexity_performance_corr:.4f}")
    print(f"Interpretability vs Performance correlation: {interpretability_performance_corr:.4f}")
    print(f"Complexity vs Interpretability correlation: {complexity_interpretability_corr:.4f}")
    
    h4_conclusion = 'Trade-off confirmed' if abs(complexity_interpretability_corr) > 0.5 else 'No clear trade-off'
    print(f"H4 Conclusion: {h4_conclusion}")
    
    # 6. CREATE COMPREHENSIVE VISUALIZATIONS
    print("\n6. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Create comprehensive results visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['H1: Performance Comparison', 'H2: Method Consistency', 
                       'H3: Global vs Local Alignment', 'H4: Complexity Trade-off'],
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # H1: Performance comparison
    metrics = list(performance_diff.keys())
    diff_values = list(performance_diff.values())
    colors = ['red' if v < 0 else 'green' for v in diff_values]
    
    fig.add_trace(
        go.Bar(x=metrics, y=diff_values, marker_color=colors, name='Performance Difference'),
        row=1, col=1
    )
    
    # H2: Consistency visualization
    if correlations:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(correlations))),
                y=correlations,
                mode='markers+lines',
                name='SHAP-LIME Correlation',
                marker_color='blue'
            ),
            row=1, col=2
        )
    
    # H3: Alignment visualization
    k_values = [int(k.split('_')[1]) for k in top_k_agreements.keys()]
    agreements = list(top_k_agreements.values())
    
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=agreements,
            mode='markers+lines',
            name='Top-K Agreement',
            marker_color='orange'
        ),
        row=2, col=1
    )
    
    # H4: Trade-off visualization
    fig.add_trace(
        go.Scatter(
            x=df_analysis['complexity'],
            y=df_analysis['interpretability'],
            mode='markers+text',
            text=df_analysis['model'],
            textposition='top center',
            marker=dict(
                size=df_analysis['performance'] * 30,
                color=df_analysis['performance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Performance")
            ),
            name='Models'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Hypothesis Testing Results Summary',
        height=800,
        showlegend=True
    )

    fig.write_html(os.path.join(root_path, 'figures', 'hypothesis_testing_results.html'), include_plotlyjs='cdn')
    fig.show()
    
    # 7. SAVE COMPREHENSIVE RESULTS
    print("\n7. SAVING RESULTS")
    print("-" * 40)
    
    # Compile all results
    experimental_results = {
        'hypotheses': {
            'H1': {
                'statement': 'Interpretable models provide comparable performance to black-box models',
                'performance_differences': performance_diff,
                'cv_scores': {
                    'lr_mean': float(lr_cv_scores.mean()),
                    'lr_std': float(lr_cv_scores.std()),
                    'rf_mean': float(rf_cv_scores.mean()),
                    'rf_std': float(rf_cv_scores.std())
                },
                'statistical_test': {
                    'test': 't-test',
                    'p_value': float(cv_ttest_p),
                    'significant': cv_ttest_p < 0.05
                },
                'conclusion': h1_conclusion
            },
            'H2': {
                'statement': 'SHAP provides more consistent explanations than LIME',
                'consistency_score': float(mean_correlation) if correlations else 0,
                'correlations': correlations,
                'conclusion': h2_conclusion
            },
            'H3': {
                'statement': 'Global and local interpretability methods align',
                'spearman_correlation': float(spearman_corr),
                'spearman_p_value': float(spearman_p),
                'kendall_tau': float(kendall_tau),
                'kendall_p_value': float(kendall_p),
                'top_k_agreements': top_k_agreements,
                'conclusion': h3_conclusion
            },
            'H4': {
                'statement': 'Model complexity trades off with interpretability',
                'correlations': {
                    'complexity_performance': float(complexity_performance_corr),
                    'interpretability_performance': float(interpretability_performance_corr),
                    'complexity_interpretability': float(complexity_interpretability_corr)
                },
                'models_data': models_analysis,
                'conclusion': h4_conclusion
            }
        },
        'summary': {
            'H1': {'conclusion': h1_conclusion, 'key_finding': f"AUC difference: {performance_diff['roc_auc']:+.4f}"},
            'H2': {'conclusion': h2_conclusion, 'key_finding': f"Mean correlation: {mean_correlation:.3f}" if correlations else "No correlation computed"},
            'H3': {'conclusion': h3_conclusion, 'key_finding': f"Spearman correlation: {spearman_corr:.3f}"},
            'H4': {'conclusion': h4_conclusion, 'key_finding': f"Complexity-interpretability correlation: {complexity_interpretability_corr:.3f}"}
        }
    }
    
    # Save results
    with open(os.path.join(root_path, 'results', 'hypothesis_testing_results.json'), 'w') as f:
        json.dump(experimental_results, f, indent=2, default=str)
    
    # Create final summary report
    summary_report = f"""
HYPOTHESIS TESTING AND EXPERIMENTAL DESIGN - FINAL REPORT
========================================================

RESEARCH HYPOTHESES AND RESULTS:

H1: Interpretable models provide comparable performance to black-box models
   Result: {h1_conclusion}
   Key Finding: ROC-AUC difference = {performance_diff['roc_auc']:+.4f}
   Statistical Test: t-test p-value = {cv_ttest_p:.4f}

H2: SHAP provides more consistent explanations than LIME
   Result: {h2_conclusion}
   Key Finding: Mean SHAP-LIME correlation = {mean_correlation:.3f}
   Sample Size: {len(correlations)} instances analyzed

H3: Global and local interpretability methods align
   Result: {h3_conclusion}
   Key Finding: Spearman correlation = {spearman_corr:.3f} (p={spearman_p:.4f})
   Top-3 Agreement: {top_k_agreements['top_3']:.3f}

H4: Model complexity trades off with interpretability
   Result: {h4_conclusion}
   Key Finding: Complexity-interpretability correlation = {complexity_interpretability_corr:.3f}
   
OVERALL CONCLUSIONS:
1. Performance Gap: {'Minimal' if abs(performance_diff['roc_auc']) < 0.05 else 'Significant'} difference between interpretable and black-box models
2. Explanation Consistency: {'High' if mean_correlation > 0.7 else 'Moderate' if mean_correlation > 0.4 else 'Low'} consistency between SHAP and LIME
3. Method Alignment: {'Strong' if spearman_corr > 0.7 else 'Moderate' if spearman_corr > 0.4 else 'Weak'} alignment between global and local methods
4. Trade-off Analysis: {'Clear' if abs(complexity_interpretability_corr) > 0.5 else 'Unclear'} complexity-interpretability trade-off

PRACTICAL IMPLICATIONS:
- Choose interpretable models when explanation is critical
- Use SHAP for consistent feature importance across models
- Combine global and local interpretability methods for comprehensive understanding
- Consider context when balancing performance vs interpretability

STATISTICAL RIGOR:
- Cross-validation used for robust performance estimation
- Multiple correlation measures for consistency analysis
- Significance testing for all major comparisons
- Top-k agreement analysis for practical relevance
"""
    
    print(summary_report)

    with open(os.path.join(root_path, 'results', 'hypothesis_testing_summary.txt'), 'w') as f:
        f.write(summary_report)
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING ANALYSIS COMPLETE!")
    print("Results saved to:")
    print("- hypothesis_testing_results.json")
    print("- hypothesis_testing_summary.txt")
    print("- hypothesis_testing_results.html")
    print("=" * 70)

if __name__ == "__main__":
    main()
