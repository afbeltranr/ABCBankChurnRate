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
        """Get hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
