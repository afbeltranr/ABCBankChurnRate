"""AdaBoost model implementation."""

from typing import Dict, Any
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseChurnModel

class AdaBoostModel(BaseChurnModel):
    """AdaBoost model for churn prediction."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("AdaBoost", random_state)
    
    def _create_model(self, **params) -> AdaBoostClassifier:
        """Create AdaBoostClassifier model with given parameters."""
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=self.random_state),
            random_state=self.random_state,
            **params
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for AdaBoost."""
        return {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME'
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [25, 50, 100],
            'learning_rate': [0.5, 1.0, 1.5],
            'algorithm': ['SAMME', 'SAMME.R']
        }
