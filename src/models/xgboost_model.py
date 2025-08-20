"""XGBoost model implementation."""

from typing import Dict, Any
import xgboost as xgb
from .base_model import BaseChurnModel

class XGBoostModel(BaseChurnModel):
    """XGBoost model for churn prediction."""
    
    def __init__(self, random_state: int = 42):
        super().__init__("XGBoost", random_state)
    
    def _create_model(self, **params) -> xgb.XGBClassifier:
        """Create XGBClassifier model with given parameters."""
        return xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            **params
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for XGBoost."""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get lightweight hyperparameter grid for overfitting prevention.
        
        Overfitting Prevention Techniques:
        - max_depth: Controls tree depth (prevents overcomplex trees)
        - learning_rate: Step size shrinkage (prevents overfitting)
        - subsample: Row subsampling ratio (bootstrap sampling)
        - colsample_bytree: Feature subsampling per tree
        - min_child_weight: Min sum of weights in child nodes (regularization)
        - reg_alpha: L1 regularization (feature selection)
        
        Lightweight grid: ~144 combinations (432 with 3-fold CV)
        """
        return {
            'n_estimators': [50, 100, 200],  # Fewer estimators for speed
            'max_depth': [3, 5, 7],  # Tree depth (XGBoost works well with shallow trees)
            'learning_rate': [0.05, 0.1, 0.2],  # Step size shrinkage
            'subsample': [0.8, 0.9],  # Row subsampling (most effective values)
            'colsample_bytree': [0.8, 0.9],  # Feature subsampling per tree
            'min_child_weight': [1, 5],  # Min child weight regularization
            'reg_alpha': [0, 0.1]  # L1 regularization (0 = no reg, 0.1 = moderate reg)
        }
