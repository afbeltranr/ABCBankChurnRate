"""Feature preprocessing and engineering for the Bank Churn Prediction project."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ChurnFeatureProcessor(BaseEstimator, TransformerMixin):
    """Custom transformer for preprocessing churn prediction features."""
    
    def __init__(
        self,
        continuous_features,
        categorical_features,
        target_column,
        id_column,
        age_transform="log",
        salary_transform="bin",
        balance_transform="zero_flag"
    ):
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.id_column = id_column
        self.age_transform = age_transform
        self.salary_transform = salary_transform
        self.balance_transform = balance_transform
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Will be set during fit
        self.salary_bins = None
        
    def fit(self, X, y=None):
        """
        Fit the feature processor to the data.
        
        Args:
            X: Input DataFrame
            y: Target variable (not used, included for scikit-learn compatibility)
        """
        # Fit continuous feature transformations
        continuous_data = X[self.continuous_features].copy()
        
        # Age transformation
        if self.age_transform == "log":
            continuous_data["age"] = np.log(continuous_data["age"])
        
        # Salary binning
        if self.salary_transform == "bin":
            self.salary_bins = pd.qcut(
                continuous_data["estimated_salary"],
                q=5,
                labels=False
            )
        
        # Fit scaler on continuous features
        self.scaler.fit(continuous_data)
        
        # Fit label encoders for categorical features
        for feature in self.categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.label_encoders[feature].fit(X[feature])
        
        return self
    
    def transform(self, X):
        """
        Transform the features.
        
        Args:
            X: Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed features
        """
        # Create output DataFrame
        X_transformed = pd.DataFrame()
        
        # Transform continuous features
        continuous_data = X[self.continuous_features].copy()
        
        # Age transformation
        if self.age_transform == "log":
            continuous_data["age"] = np.log(continuous_data["age"])
        
        # Salary transformation
        if self.salary_transform == "bin":
            continuous_data["estimated_salary"] = pd.qcut(
                continuous_data["estimated_salary"],
                q=5,
                labels=False
            )
        
        # Balance transformation
        if self.balance_transform == "zero_flag":
            X_transformed["balance_is_zero"] = (continuous_data["balance"] == 0).astype(int)
            continuous_data["balance"] = np.where(
                continuous_data["balance"] == 0,
                continuous_data["balance"].mean(),
                continuous_data["balance"]
            )
        
        # Scale continuous features
        scaled_features = self.scaler.transform(continuous_data)
        X_transformed = pd.concat([
            X_transformed,
            pd.DataFrame(
                scaled_features,
                columns=self.continuous_features,
                index=X.index
            )
        ], axis=1)
        
        # Transform categorical features
        for feature in self.categorical_features:
            X_transformed[feature] = self.label_encoders[feature].transform(X[feature])
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in one step.
        
        Args:
            X: Input DataFrame
            y: Target variable (not used, included for scikit-learn compatibility)
            
        Returns:
            pd.DataFrame: Transformed features
        """
        return self.fit(X).transform(X)
