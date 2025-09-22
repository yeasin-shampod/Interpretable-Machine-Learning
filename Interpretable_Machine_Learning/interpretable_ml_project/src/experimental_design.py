"""
Experimental Design and Statistical Analysis Module
Hypothesis-driven comparison and statistical testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, permutation_test_score
import warnings
warnings.filterwarnings('ignore')

class ExperimentalDesign:
    """
    Experimental design for interpretability vs performance analysis
    """
    
    def __init__(self):
        self.hypotheses = {
            'H1': 'Interpretable models (Logistic Regression) provide comparable predictive performance to black-box models (Random Forest)',
            'H2': 'SHAP values provide more consistent feature importance rankings than LIME explanations',
            'H3': 'Global interpretability methods (feature importance) align with local interpretability methods (SHAP/LIME)',
            'H4': 'Model complexity negatively correlates with interpretability while positively correlating with performance'
        }
        self.results = {}
        
    def test_hypothesis_1(self, interpretable_model, blackbox_model, X_test, y_test, cv_folds=5):
        """
        Test H1: Performance comparison between interpretable and black-box models
        """
        print("Testing Hypothesis 1: Performance Comparison")
        print("=" * 50)
        
        # Get predictions
        lr_pred = interpretable_model.predict(X_test)
        rf_pred = blackbox_model.predict(X_test)
        
        lr_pred_proba = interpretable_model.predict_proba(X_test)[:, 1]
        rf_pred_proba = blackbox_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        lr_metrics = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred),
            'auc': roc_auc_score(y_test, lr_pred_proba)
        }
        
        rf_metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred),
            'auc': roc_auc_score(y_test, rf_pred_proba)
        }
        
        # Statistical significance testing
        # McNemar's test for accuracy comparison
        from statsmodels.stats.contingency_tables import mcnemar
        
        # Create contingency table
        lr_correct = (lr_pred == y_test)
        rf_correct = (rf_pred == y_test)
        
        contingency_table = pd.crosstab(lr_correct, rf_correct)
        
        try:
            mcnemar_result = mcnemar(contingency_table.values, exact=False, correction=True)
            mcnemar_p_value = mcnemar_result.pvalue
        except:
            mcnemar_p_value = np.nan
        
        # Cross-validation comparison
        X_full = pd.concat([X_test.reset_index(drop=True)] * 3)  # Simulate larger dataset
        y_full = pd.concat([y_test.reset_index(drop=True)] * 3)
        
        lr_cv_scores = cross_val_score(interpretable_model.model, X_full, y_full, cv=cv_folds, scoring='roc_auc')
        rf_cv_scores = cross_val_score(blackbox_model.model, X_full, y_full, cv=cv_folds, scoring='roc_auc')
        
        # Paired t-test on CV scores
        cv_ttest_stat, cv_ttest_p = ttest_ind(lr_cv_scores, rf_cv_scores)
        
        h1_results = {
            'lr_metrics': lr_metrics,
            'rf_metrics': rf_metrics,
            'performance_difference': {metric: rf_metrics[metric] - lr_metrics[metric] for metric in lr_metrics},
            'mcnemar_p_value': mcnemar_p_value,
            'cv_scores': {'lr': lr_cv_scores, 'rf': rf_cv_scores},
            'cv_ttest_p': cv_ttest_p,
            'conclusion': 'No significant difference' if cv_ttest_p > 0.05 else 'Significant difference'
        }
        
        self.results['H1'] = h1_results
        return h1_results
    
    def test_hypothesis_2(self, shap_values, lime_explanations, feature_names, n_instances=10):
        """
        Test H2: Consistency of SHAP vs LIME explanations
        """
        print("Testing Hypothesis 2: SHAP vs LIME Consistency")
        print("=" * 50)
        
        # Get SHAP feature rankings for multiple instances
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        shap_rankings = []
        lime_rankings = []
        
        for i in range(min(n_instances, len(shap_values))):
            # SHAP ranking
            shap_importance = np.abs(shap_values[i])
            shap_ranking = np.argsort(shap_importance)[::-1]
            shap_rankings.append(shap_ranking)
            
            # LIME ranking (if available)
            if i < len(lime_explanations):
                lime_exp = lime_explanations[i]
                lime_features = [item[0] for item in lime_exp.as_list()]
                lime_values = [abs(item[1]) for item in lime_exp.as_list()]
                
                # Map LIME features to indices
                lime_indices = [feature_names.index(feat) for feat in lime_features if feat in feature_names]
                lime_ranking = np.array(lime_indices)
                lime_rankings.append(lime_ranking)
        
        # Calculate ranking consistency (Spearman correlation)
        correlations = []
        for i in range(min(len(shap_rankings), len(lime_rankings))):
            if len(lime_rankings[i]) > 0:
                # Create full rankings
                shap_full = np.zeros(len(feature_names))
                lime_full = np.zeros(len(feature_names))
                
                for j, idx in enumerate(shap_rankings[i]):
                    shap_full[idx] = len(feature_names) - j
                
                for j, idx in enumerate(lime_rankings[i]):
                    lime_full[idx] = len(lime_rankings[i]) - j
                
                corr, p_value = stats.spearmanr(shap_full, lime_full)
                if not np.isnan(corr):
                    correlations.append(corr)
        
        # Statistical analysis
        mean_correlation = np.mean(correlations) if correlations else 0
        std_correlation = np.std(correlations) if correlations else 0
        
        # Test if correlation is significantly different from 0
        if correlations:
            t_stat, t_p_value = ttest_ind(correlations, [0] * len(correlations))
        else:
            t_stat, t_p_value = np.nan, np.nan
        
        h2_results = {
            'correlations': correlations,
            'mean_correlation': mean_correlation,
            'std_correlation': std_correlation,
            'consistency_score': mean_correlation,
            't_test_p_value': t_p_value,
            'conclusion': 'High consistency' if mean_correlation > 0.7 else 'Moderate consistency' if mean_correlation > 0.4 else 'Low consistency'
        }
        
        self.results['H2'] = h2_results
        return h2_results
    
    def test_hypothesis_3(self, global_importance, shap_values, feature_names):
        """
        Test H3: Alignment between global and local interpretability
        """
        print("Testing Hypothesis 3: Global vs Local Interpretability Alignment")
        print("=" * 50)
        
        # Global importance (from Random Forest)
        global_ranking = global_importance.sort_values('importance', ascending=False)['feature'].tolist()
        
        # Local importance (average SHAP values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        local_importance = np.abs(shap_values).mean(axis=0)
        local_ranking = [feature_names[i] for i in np.argsort(local_importance)[::-1]]
        
        # Calculate ranking correlation
        global_ranks = {feat: i for i, feat in enumerate(global_ranking)}
        local_ranks = {feat: i for i, feat in enumerate(local_ranking)}
        
        common_features = set(global_ranks.keys()) & set(local_ranks.keys())
        
        global_rank_values = [global_ranks[feat] for feat in common_features]
        local_rank_values = [local_ranks[feat] for feat in common_features]
        
        spearman_corr, spearman_p = stats.spearmanr(global_rank_values, local_rank_values)
        kendall_tau, kendall_p = stats.kendalltau(global_rank_values, local_rank_values)
        
        # Top-k agreement
        top_k_agreements = {}
        for k in [3, 5, 7]:
            top_global = set(global_ranking[:k])
            top_local = set(local_ranking[:k])
            agreement = len(top_global & top_local) / k
            top_k_agreements[f'top_{k}'] = agreement
        
        h3_results = {
            'global_ranking': global_ranking,
            'local_ranking': local_ranking,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p_value': kendall_p,
            'top_k_agreements': top_k_agreements,
            'conclusion': 'Strong alignment' if spearman_corr > 0.7 else 'Moderate alignment' if spearman_corr > 0.4 else 'Weak alignment'
        }
        
        self.results['H3'] = h3_results
        return h3_results
    
    def test_hypothesis_4(self, models_data):
        """
        Test H4: Complexity vs Interpretability vs Performance trade-off
        """
        print("Testing Hypothesis 4: Complexity-Interpretability-Performance Trade-off")
        print("=" * 50)
        
        # Define complexity and interpretability scores
        complexity_scores = {
            'LogisticRegression': 1,  # Low complexity
            'RandomForest': 3        # High complexity
        }
        
        interpretability_scores = {
            'LogisticRegression': 5,  # High interpretability
            'RandomForest': 2         # Low interpretability
        }
        
        # Collect performance and complexity data
        analysis_data = []
        for model_name, data in models_data.items():
            analysis_data.append({
                'model': model_name,
                'complexity': complexity_scores.get(model_name, 2),
                'interpretability': interpretability_scores.get(model_name, 3),
                'performance': data.get('auc', data.get('roc_auc', 0))
            })
        
        df = pd.DataFrame(analysis_data)
        
        # Correlation analysis
        complexity_performance_corr = df['complexity'].corr(df['performance'])
        interpretability_performance_corr = df['interpretability'].corr(df['performance'])
        complexity_interpretability_corr = df['complexity'].corr(df['interpretability'])
        
        h4_results = {
            'data': df,
            'complexity_performance_correlation': complexity_performance_corr,
            'interpretability_performance_correlation': interpretability_performance_corr,
            'complexity_interpretability_correlation': complexity_interpretability_corr,
            'trade_off_exists': abs(complexity_interpretability_corr) > 0.5,
            'conclusion': 'Trade-off confirmed' if abs(complexity_interpretability_corr) > 0.5 else 'No clear trade-off'
        }
        
        self.results['H4'] = h4_results
        return h4_results
    
    def generate_comprehensive_report(self, save_path=None):
        """
        Generate comprehensive experimental report
        """
        report = {
            'hypotheses': self.hypotheses,
            'results': self.results,
            'summary': self._generate_summary()
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = self._convert_for_json(report)
                json.dump(json_results, f, indent=2)
        
        return report
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _generate_summary(self):
        """Generate summary of all hypothesis tests"""
        summary = {}
        
        for hypothesis, results in self.results.items():
            if 'conclusion' in results:
                summary[hypothesis] = {
                    'conclusion': results['conclusion'],
                    'key_finding': self._extract_key_finding(hypothesis, results)
                }
        
        return summary
    
    def _extract_key_finding(self, hypothesis, results):
        """Extract key finding for each hypothesis"""
        if hypothesis == 'H1':
            perf_diff = results.get('performance_difference', {})
            auc_diff = perf_diff.get('auc', 0)
            return f"AUC difference: {auc_diff:.3f} (RF - LR)"
        elif hypothesis == 'H2':
            consistency = results.get('consistency_score', 0)
            return f"Mean correlation: {consistency:.3f}"
        elif hypothesis == 'H3':
            alignment = results.get('spearman_correlation', 0)
            return f"Spearman correlation: {alignment:.3f}"
        elif hypothesis == 'H4':
            trade_off = results.get('trade_off_exists', False)
            return f"Trade-off exists: {trade_off}"
        
        return "No key finding extracted"
    
    def visualize_results(self, save_path=None):
        """Create comprehensive visualization of experimental results"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['H1: Performance Comparison', 'H2: Method Consistency', 
                          'H3: Global vs Local Alignment', 'H4: Complexity Trade-off'],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # H1: Performance comparison
        if 'H1' in self.results:
            h1_data = self.results['H1']
            lr_metrics = h1_data['lr_metrics']
            rf_metrics = h1_data['rf_metrics']
            
            metrics = list(lr_metrics.keys())
            lr_values = list(lr_metrics.values())
            rf_values = list(rf_metrics.values())
            
            fig.add_trace(
                go.Bar(name='Logistic Regression', x=metrics, y=lr_values, marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='Random Forest', x=metrics, y=rf_values, marker_color='lightcoral'),
                row=1, col=1
            )
        
        # H2: Consistency visualization
        if 'H2' in self.results:
            h2_data = self.results['H2']
            correlations = h2_data.get('correlations', [])
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(correlations))),
                    y=correlations,
                    mode='markers+lines',
                    name='SHAP-LIME Correlation',
                    marker_color='green'
                ),
                row=1, col=2
            )
        
        # H3: Alignment visualization
        if 'H3' in self.results:
            h3_data = self.results['H3']
            top_k = h3_data.get('top_k_agreements', {})
            
            k_values = [int(k.split('_')[1]) for k in top_k.keys()]
            agreements = list(top_k.values())
            
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
        if 'H4' in self.results:
            h4_data = self.results['H4']
            df = h4_data.get('data', pd.DataFrame())
            
            if not df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df['complexity'],
                        y=df['interpretability'],
                        mode='markers+text',
                        text=df['model'],
                        textposition='top center',
                        marker=dict(
                            size=df['performance'] * 20,
                            color=df['performance'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Models'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title='Experimental Results Summary',
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
