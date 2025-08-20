"""Base model interface for churn prediction models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os

class BaseChurnModel(ABC):
    """Abstract base class for churn prediction models."""
    
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def _create_model(self, **params) -> Any:
        """Create the underlying model with given parameters."""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for the model."""
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, list]:
        """Get hyperparameter grid for tuning."""
        pass
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **params) -> 'BaseChurnModel':
        """
        Fit the model to training data.
        
        Args:
            X: Training features
            y: Training target
            **params: Model parameters
        
        Returns:
            self: Fitted model
        """
        # Use default params if none provided
        if not params:
            params = self.get_default_params()
            
        self.model = self._create_model(**params)
        self.feature_names = list(X.columns)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return decision function or dummy probabilities
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 1
            proba[predictions == 1, 1] = 1
            return proba
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Probability of positive class
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> 'BaseChurnModel':
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance if available.
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficient values
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return pd.DataFrame(columns=['feature', 'importance'])
