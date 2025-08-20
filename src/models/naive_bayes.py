"""Naive Bayes model implementation."""

from typing import Dict, Any
from sklearn.naive_bayes import GaussianNB
from .base_model import BaseChurnModel

class NaiveBayesModel(BaseChurnModel):
    """Naive Bayes model for churn prediction."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("NaiveBayes", random_state)
    
    def _create_model(self, **params) -> GaussianNB:
        """Create GaussianNB model with given parameters."""
        return GaussianNB(**params)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for Naive Bayes."""
        return {
            'var_smoothing': 1e-9
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """Get hyperparameter grid for tuning."""
        return {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }
