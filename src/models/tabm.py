"""TabM (Tabular Model) implementation for churn prediction."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .base_model import BaseChurnModel

class TabMModel(BaseChurnModel):
    """
    TabM-inspired model for churn prediction.
    
    This is a simplified implementation that captures the essence of TabM
    using scikit-learn's MLPClassifier with architecture inspired by
    tabular deep learning best practices.
    """
    
    def __init__(self, random_state: int = 42):
        super().__init__("TabM", random_state)
        self.scaler = StandardScaler()
        self.is_scaled = False
    
    def _create_model(self, **params) -> MLPClassifier:
        """Create MLPClassifier model with TabM-inspired architecture."""
        return MLPClassifier(
            random_state=self.random_state,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            **params
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for TabM."""
        return {
            'hidden_layer_sizes': (128, 64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'batch_size': 'auto'
        }
    
    def get_param_grid(self) -> Dict[str, list]:
        """
        Get comprehensive hyperparameter grid for neural network overfitting prevention.
        
        Overfitting Prevention Techniques:
        - hidden_layer_sizes: Network architecture (capacity control)
        - alpha: L2 regularization strength (weight decay)
        - learning_rate_init: Initial learning rate (convergence control)
        - batch_size: Mini-batch size (regularization through noise)
        - early_stopping: Built-in (stops when validation doesn't improve)
        - dropout: Would be ideal but not available in MLPClassifier
        """
        return {
            'hidden_layer_sizes': [
                (64, 32),           # Small network
                (128, 64),          # Medium network  
                (128, 64, 32),      # Deep network
                (256, 128, 64),     # Large network
                (128, 128, 64),     # Wide network
                (64, 64, 32, 16)    # Very deep network
            ],
            'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
            'alpha': [0.0001, 0.001, 0.01, 0.1],  # L2 regularization strength
            'learning_rate_init': [0.0001, 0.001, 0.01],  # Initial learning rate
            'solver': ['adam', 'lbfgs'],  # Optimization algorithms
            'batch_size': [32, 64, 128, 'auto'],  # Mini-batch sizes
            'learning_rate': ['constant', 'adaptive']  # Learning rate schedule
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **params) -> 'TabMModel':
        """
        Fit the TabM model with proper scaling for neural networks.
        
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
        
        # Scale features for neural network
        X_scaled = self.scaler.fit_transform(X)
        self.is_scaled = True
        
        self.model = self._create_model(**params)
        self.feature_names = list(X.columns)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with proper scaling."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.is_scaled:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities with proper scaling."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.is_scaled:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using connection weights.
        
        Note: This is a simplified approach using the magnitude of
        input layer weights as a proxy for feature importance.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        # Get the weights from input layer to first hidden layer
        if hasattr(self.model, 'coefs_') and len(self.model.coefs_) > 0:
            input_weights = self.model.coefs_[0]  # Shape: (n_features, n_hidden)
            
            # Calculate feature importance as mean absolute weight
            feature_importance = np.mean(np.abs(input_weights), axis=1)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return pd.DataFrame(columns=['feature', 'importance'])
