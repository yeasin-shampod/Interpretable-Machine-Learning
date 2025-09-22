# Interpretable Machine Learning Project
## Heart Disease Prediction with Comprehensive Interpretability Analysis

### Project Overview
This project implements a comprehensive interpretable machine learning analysis for heart disease prediction, comparing interpretable and black-box models while evaluating various interpretability methods. The project is designed to meet the requirements of an interpretable ML seminar at Friedrich-Alexander University.

### Domain & Dataset
- **Domain**: Healthcare/Medical Diagnosis
- **Dataset**: Heart Disease Dataset from UCI ML Repository
- **Target**: Binary classification (presence/absence of heart disease)
- **Features**: 13 clinical and demographic features including age, sex, chest pain type, blood pressure, cholesterol, etc.

### Models Implemented

#### 1. Interpretable Model
- **Algorithm**: Logistic Regression
- **Interpretability**: Direct coefficient interpretation
- **Features**: Feature importance through coefficients, statistical significance testing

#### 2. Black-box Model
- **Algorithm**: Random Forest
- **Optimization**: Hyperparameter tuning with GridSearchCV
- **Features**: High predictive performance, ensemble learning

### Interpretability Methods

#### Local Interpretability
- **SHAP (SHapley Additive exPlanations)**: 
  - TreeExplainer for Random Forest
  - LinearExplainer for Logistic Regression
  - Individual prediction explanations
  - Waterfall plots for decision breakdown

- **LIME (Local Interpretable Model-agnostic Explanations)**:
  - Tabular explainer for individual instances
  - Local surrogate model explanations
  - Feature contribution analysis

#### Global Interpretability
- **Feature Importance**: 
  - Random Forest built-in importance
  - Logistic Regression coefficients
  - SHAP global importance (mean absolute SHAP values)

- **Partial Dependence Plots**: 
  - Individual feature effects on predictions
  - Model behavior visualization

- **Model Behavior Analysis**:
  - Prediction distributions
  - Confidence analysis
  - Feature correlation analysis

### Experimental Design & Hypotheses

#### Hypothesis 1 (H1)
**Statement**: Interpretable models (Logistic Regression) provide comparable predictive performance to black-box models (Random Forest)

**Testing Method**: 
- Performance metrics comparison (accuracy, precision, recall, F1, ROC-AUC)
- Cross-validation analysis
- McNemar's test for statistical significance
- Paired t-test on CV scores

#### Hypothesis 2 (H2)
**Statement**: SHAP values provide more consistent feature importance rankings than LIME explanations

**Testing Method**:
- Spearman rank correlation between SHAP and LIME rankings
- Consistency analysis across multiple instances
- Statistical significance testing

#### Hypothesis 3 (H3)
**Statement**: Global interpretability methods (feature importance) align with local interpretability methods (SHAP/LIME)

**Testing Method**:
- Ranking correlation between global and local importance
- Top-k agreement analysis
- Kendall's tau correlation

#### Hypothesis 4 (H4)
**Statement**: Model complexity negatively correlates with interpretability while positively correlating with performance

**Testing Method**:
- Complexity scoring system
- Interpretability scoring system
- Correlation analysis between complexity, interpretability, and performance

### Project Structure
```
interpretable_ml_project/
├── data/
│   └── processed.cleveland.data
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── models.py               # Model implementations
│   ├── interpretability.py     # SHAP, LIME, and interpretability methods
│   └── experimental_design.py  # Hypothesis testing and statistical analysis
├── figures/                    # Generated visualizations
├── results/                    # Analysis results and reports
├── main_analysis.py           # Main analysis pipeline
└── README.md                  # This file
```

### Key Features

#### Data Processing
- Comprehensive missing value handling
- Feature scaling for numerical variables
- Stratified train-test split
- Exploratory data analysis with visualizations

#### Model Training
- Automated hyperparameter tuning for Random Forest
- Cross-validation for robust performance estimation
- Feature importance extraction for both models

#### Interpretability Analysis
- Multiple interpretability methods comparison
- Individual instance explanations
- Global model behavior analysis
- Interactive visualizations with Plotly

#### Statistical Analysis
- Hypothesis-driven experimental design
- Statistical significance testing
- Comprehensive reporting with confidence intervals
- Performance vs interpretability trade-off analysis

### Results & Insights

#### Performance Comparison
- Both models achieve comparable performance on the heart disease dataset
- Random Forest shows slight improvement in ROC-AUC
- Cross-validation confirms robust performance for both models

#### Feature Importance
- Consistent top features across all interpretability methods
- Chest pain type (cp) and maximum heart rate (thalach) are most predictive
- Strong correlation between different importance measures

#### Interpretability Trade-offs
- Logistic Regression provides direct coefficient interpretation
- SHAP values offer consistent explanations across models
- LIME provides intuitive local explanations but with higher variance

#### Clinical Relevance
- Top predictive features align with medical knowledge
- Model explanations support clinical decision-making
- Interpretability methods reveal feature interactions

### Usage Instructions

#### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly shap lime scipy statsmodels
```

#### Running the Analysis
```bash
cd /home/ubuntu/interpretable_ml_project
python main_analysis.py
```

#### Output Files
- **Figures**: All visualizations saved as PNG and HTML files
- **Results**: JSON reports and text summaries
- **Interactive Dashboard**: Comprehensive HTML dashboard with all results

### Technical Implementation

#### Data Loader (`data_loader.py`)
- Handles UCI heart disease dataset format
- Implements robust missing value imputation
- Provides feature scaling and train-test splitting
- Includes comprehensive data validation

#### Models (`models.py`)
- Interpretable model with coefficient analysis
- Black-box model with hyperparameter optimization
- Comprehensive evaluation metrics
- Cross-validation utilities

#### Interpretability (`interpretability.py`)
- SHAP integration for both tree and linear models
- LIME implementation for tabular data
- Partial dependence plot generation
- Feature importance comparison framework

#### Experimental Design (`experimental_design.py`)
- Hypothesis-driven testing framework
- Statistical significance testing
- Comprehensive reporting system
- Interactive result visualization

### Academic Contributions

#### Methodological Contributions
- Comprehensive comparison framework for interpretability methods
- Statistical testing approach for interpretability consistency
- Performance vs interpretability trade-off quantification

#### Practical Contributions
- Healthcare domain application with clinical relevance
- Reproducible experimental design
- Interactive visualization for stakeholder communication

#### Educational Value
- Clear demonstration of interpretability concepts
- Hands-on implementation of multiple methods
- Statistical rigor in experimental design

### Future Extensions

#### Model Extensions
- Additional interpretable models (Decision Trees, GAMs)
- Deep learning models with interpretability methods
- Ensemble interpretability techniques

#### Interpretability Methods
- Counterfactual explanations
- Anchors explanations
- Integrated gradients for neural networks

#### Domain Applications
- Multi-class classification problems
- Regression tasks with interpretability
- Time-series interpretability

### References
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD.
- Molnar, C. (2020). Interpretable machine learning. Lulu.com.
- UCI ML Repository: Heart Disease Dataset

### Contact & Support
This project was developed as part of an interpretable machine learning seminar. For questions or contributions, please refer to the project documentation and code comments.

---
**Note**: This project demonstrates best practices in interpretable machine learning, combining rigorous experimental design with practical healthcare applications.
