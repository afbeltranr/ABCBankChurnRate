"""Random Forest model implementation."""

from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseChurnModel

class RandomForestModel(BaseChurnModel):
    """Random Forest model for churn prediction."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("RandomForest", random_state)
    
    def _create_model(self, **params) -> RandomForestClassifier:
        """Create RandomForestClassifier model with given parameters."""
        return RandomForestClassifier(
            random_state=self.random_state,
            **params
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for Random Forest."""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
