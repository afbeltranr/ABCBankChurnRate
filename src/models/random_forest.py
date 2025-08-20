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
        """
        Get lightweight hyperparameter grid for overfitting prevention.
        
        Overfitting Prevention Techniques:
        - max_depth: Controls tree depth (prevents deep, overcomplex trees)
        - min_samples_split: Min samples required to split (prevents splitting on few samples)
        - min_samples_leaf: Min samples in leaf nodes (prevents tiny leaves)
        - max_features: Feature subsampling per tree (bootstrap aggregating)
        - n_estimators: Number of trees (more trees = better generalization)
        
        Lightweight grid: ~48 combinations (144 with 3-fold CV)
        """
        return {
            'n_estimators': [50, 100, 200],  # Fewer estimators for speed
            'max_depth': [5, 10, None],  # Key depth values
            'min_samples_split': [2, 10],  # Split threshold (most impactful values)
            'min_samples_leaf': [1, 4],  # Leaf size threshold
            'max_features': ['sqrt', 0.8],  # Feature subsampling (most effective)
            'bootstrap': [True]  # Always use bootstrap for ensemble diversity
        }
