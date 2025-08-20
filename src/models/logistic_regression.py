"""Logistic Regression model implementation."""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .base_model import BaseChurnModel

class LogisticRegressionModel(BaseChurnModel):
    """Logistic Regression model for churn prediction."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("LogisticRegression", random_state)
    
    def _create_model(self, **params) -> LogisticRegression:
        """Create LogisticRegression model with given parameters."""
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            **params
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for LogisticRegression."""
        return {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'liblinear'
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get lightweight hyperparameter grid for regularization.
        
        Overfitting Prevention Techniques:
        - C: Inverse of regularization strength (lower = more regularization)
        - penalty: L1 (feature selection) vs L2 (weight decay) regularization
        - solver: Optimization algorithm (affects regularization effectiveness)
        - class_weight: Handle class imbalance (prevents bias toward majority)
        
        Lightweight grid: ~24 combinations (72 with 3-fold CV)
        """
        return {
            'C': [0.01, 0.1, 1.0, 10.0],  # Key regularization values
            'penalty': ['l1', 'l2'],  # Regularization types
            'solver': ['liblinear', 'saga'],  # Fast solvers that support both penalties
            'class_weight': [None, 'balanced']  # Handle class imbalance
        }
