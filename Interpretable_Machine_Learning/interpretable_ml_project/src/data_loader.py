"""
Data Loading and Preprocessing Module for Heart Disease Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseDataLoader:
    """
    Data loader and preprocessor for the Heart Disease dataset from UCI ML Repository
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (0-3)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'slope': 'Slope of the peak exercise ST segment (0-2)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)',
            'target': 'Heart disease presence (0 = no disease; 1 = disease)'
        }
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the heart disease dataset"""
        # Load data
        df = pd.read_csv(self.data_path, names=self.feature_names)
        
        # Handle missing values (marked as '?' in the dataset)
        df = df.replace('?', np.nan)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # For categorical variables, use mode
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # For numerical variables, use median
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        return df
    
    def prepare_features(self, df, scale_features=True):
        """Prepare features for modeling"""
        X = df.drop('target', axis=1)
        y = df['target']
        
        if scale_features:
            # Scale numerical features
            numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            X_scaled = X.copy()
            X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            return X_scaled, y
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_feature_info(self):
        """Get feature information"""
        return self.feature_descriptions
