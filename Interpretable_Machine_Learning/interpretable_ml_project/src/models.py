"""
Model Implementation Module
Contains interpretable and black-box models for heart disease prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class InterpretableModel:
    """
    Interpretable Logistic Regression Model
    """
    
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='liblinear'
        )
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y, feature_names=None):
        """Fit the logistic regression model"""
        self.model.fit(X, y)
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance (coefficients)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        coefficients = self.model.coef_[0]
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def get_model_summary(self):
        """Get model summary statistics"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        return {
            'intercept': self.model.intercept_[0],
            'n_features': len(self.feature_names),
            'solver': self.model.solver,
            'regularization': self.model.penalty
        }

class BlackBoxModel:
    """
    Black-box Random Forest Model with hyperparameter tuning
    """
    
    def __init__(self, random_state=42):
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.is_fitted = False
        self.random_state = random_state
        
    def fit(self, X, y, feature_names=None, tune_hyperparameters=True):
        """Fit the Random Forest model with optional hyperparameter tuning"""
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]
        
        if tune_hyperparameters:
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Grid search with cross-validation
            rf = RandomForestClassifier(random_state=self.random_state)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            # Use default parameters
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
            self.model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_model_summary(self):
        """Get model summary statistics"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
            'best_params': self.best_params
        }

class ModelEvaluator:
    """
    Model evaluation utilities
    """
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5):
        """Perform cross-validation"""
        cv_scores = cross_val_score(model.model, X, y, cv=cv, scoring='roc_auc')
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    @staticmethod
    def compare_models(models_metrics):
        """Compare multiple models"""
        comparison_df = pd.DataFrame(models_metrics)
        return comparison_df.sort_values('roc_auc', ascending=False)
