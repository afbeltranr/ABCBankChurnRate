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
        # Extract base estimator parameters
        base_estimator_params = {}
        ada_params = {}
        
        for key, value in params.items():
            if key.startswith('estimator__'):
                base_param = key.replace('estimator__', '')
                base_estimator_params[base_param] = value
            else:
                ada_params[key] = value
        
        # Create base estimator with parameters
        if base_estimator_params:
            base_estimator = DecisionTreeClassifier(
                random_state=self.random_state,
                **base_estimator_params
            )
        else:
            base_estimator = DecisionTreeClassifier(
                max_depth=1, 
                random_state=self.random_state
            )
        
        return AdaBoostClassifier(
            estimator=base_estimator,
            random_state=self.random_state,
            **ada_params
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for AdaBoost."""
        return {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME'
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get comprehensive hyperparameter grid for overfitting prevention.
        
        Overfitting Prevention Techniques:
        - n_estimators: Number of weak learners (more = better but risk overfitting)
        - learning_rate: Shrinkage factor (lower = more conservative)
        - estimator__max_depth: Depth of base decision trees (usually kept shallow)
        - algorithm: SAMME vs SAMME.R (different boosting strategies)
        """
        return {
            'n_estimators': [50, 100, 150, 200],  # Number of boosting stages
            'learning_rate': [0.5, 0.8, 1.0, 1.2],  # Shrinkage factor
            'estimator__max_depth': [1, 2, 3],  # Base estimator depth (stumps to shallow trees)
            'estimator__min_samples_split': [2, 5, 10],  # Split threshold for base estimators
            'estimator__min_samples_leaf': [1, 2, 4]  # Leaf size for base estimators
        }
