"""
Interpretability Analysis Module
Contains SHAP, LIME, and other interpretability methods
"""

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
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class InterpretabilityAnalyzer:
    """
    Comprehensive interpretability analysis using SHAP, LIME, and other methods
    """
    
    def __init__(self, model, X_train, X_test, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        
    def setup_shap_explainer(self, model_type='tree'):
        """Setup SHAP explainer based on model type"""
        if model_type == 'tree':
            self.shap_explainer = shap.TreeExplainer(self.model.model)
        elif model_type == 'linear':
            self.shap_explainer = shap.LinearExplainer(self.model.model, self.X_train)
        else:
            # Use KernelExplainer as fallback
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(self.X_train, 100)
            )
    
    def setup_lime_explainer(self):
        """Setup LIME explainer"""
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )
    
    def get_shap_values(self, X=None):
        """Get SHAP values for the dataset"""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call setup_shap_explainer() first.")
        
        if X is None:
            X = self.X_test
        
        return self.shap_explainer.shap_values(X)
    
    def plot_shap_summary(self, shap_values=None, save_path=None):
        """Create SHAP summary plot"""
        if shap_values is None:
            shap_values = self.get_shap_values()
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Heart Disease Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_waterfall(self, instance_idx=0, shap_values=None, save_path=None):
        """Create SHAP waterfall plot for a single instance"""
        if shap_values is None:
            shap_values = self.get_shap_values()
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
        
        # Create waterfall plot
        expected_value = self.shap_explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        shap_explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=expected_value,
            data=self.X_test.iloc[instance_idx].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_lime_explanation(self, instance_idx=0, num_features=10):
        """Get LIME explanation for a single instance"""
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call setup_lime_explainer() first.")
        
        instance = self.X_test.iloc[instance_idx].values
        explanation = self.lime_explainer.explain_instance(
            instance, 
            self.model.predict_proba, 
            num_features=num_features
        )
        
        return explanation
    
    def plot_lime_explanation(self, instance_idx=0, save_path=None):
        """Plot LIME explanation"""
        explanation = self.get_lime_explanation(instance_idx)
        
        # Extract feature importance from LIME
        lime_values = explanation.as_list()
        features = [item[0] for item in lime_values]
        values = [item[1] for item in lime_values]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        colors = ['red' if v < 0 else 'green' for v in values]
        plt.barh(features, values, color=colors, alpha=0.7)
        plt.xlabel('Feature Contribution')
        plt.title(f'LIME Explanation - Instance {instance_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return explanation
    
    def plot_partial_dependence(self, features, save_path=None):
        """Plot partial dependence plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(features[:4]):
            if i < len(features):
                feature_idx = self.feature_names.index(feature)
                display = PartialDependenceDisplay.from_estimator(
                    self.model.model, 
                    self.X_train, 
                    [feature_idx],
                    feature_names=self.feature_names,
                    ax=axes[i]
                )
                axes[i].set_title(f'Partial Dependence - {feature}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_comparison(self, interpretable_model, save_path=None):
        """Compare feature importance between different methods"""
        # Get feature importance from different methods
        rf_importance = self.model.get_feature_importance()
        lr_importance = interpretable_model.get_feature_importance()
        
        # Get SHAP feature importance
        shap_values = self.get_shap_values()
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': np.abs(shap_values).mean(axis=0)
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
        
        # Create comparison plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Random Forest',
            x=comparison_df['feature'],
            y=comparison_df['rf_importance_norm'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Logistic Regression',
            x=comparison_df['feature'],
            y=comparison_df['lr_importance_norm'],
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            name='SHAP Values',
            x=comparison_df['feature'],
            y=comparison_df['shap_importance_norm'],
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Feature Importance Comparison Across Methods',
            xaxis_title='Features',
            yaxis_title='Normalized Importance',
            barmode='group',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return comparison_df
    
    def analyze_model_behavior(self, save_path=None):
        """Comprehensive model behavior analysis"""
        # Get predictions and probabilities
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Prediction Distribution', 'Probability Distribution', 
                          'Feature Correlation', 'Prediction Confidence'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Prediction distribution
        fig.add_trace(
            go.Histogram(x=y_pred, name='Predictions', nbinsx=2),
            row=1, col=1
        )
        
        # Probability distribution
        fig.add_trace(
            go.Histogram(x=y_pred_proba, name='Probabilities', nbinsx=20),
            row=1, col=2
        )
        
        # Feature correlation heatmap
        corr_matrix = self.X_test.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                name='Correlation'
            ),
            row=2, col=1
        )
        
        # Prediction confidence
        confidence = np.maximum(y_pred_proba, 1 - y_pred_proba)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(confidence))),
                y=confidence,
                mode='markers',
                name='Confidence',
                marker=dict(color=y_pred, colorscale='Viridis')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Model Behavior Analysis',
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return {
            'prediction_distribution': np.bincount(y_pred),
            'mean_probability': y_pred_proba.mean(),
            'mean_confidence': confidence.mean(),
            'correlation_matrix': corr_matrix
        }
