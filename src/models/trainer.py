"""Model training and evaluation pipeline."""

import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .logistic_regression import LogisticRegressionModel
from .naive_bayes import NaiveBayesModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .adaboost import AdaBoostModel
from .tabm import TabMModel

class ModelTrainer:
    """Pipeline for training and evaluating multiple churn prediction models."""
    
    def __init__(
        self,
        models_dir: str = "models",
        results_dir: str = "results",
        random_state: int = 42
    ):
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.random_state = random_state
        self.results = {}
        self.trained_models = {}
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'LogisticRegression': LogisticRegressionModel(random_state),
            'NaiveBayes': NaiveBayesModel(random_state),
            'RandomForest': RandomForestModel(random_state),
            'XGBoost': XGBoostModel(random_state),
            'AdaBoost': AdaBoostModel(random_state),
            'TabM': TabMModel(random_state)
        }
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_hyperparameters(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_folds: int = 3,
        scoring: str = 'f1_weighted'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            cv_folds: Number of CV folds
            scoring: Scoring metric
        
        Returns:
            Best parameters found
        """
        print(f"Tuning hyperparameters for {model_name}...")
        
        model = self.models[model_name]
        param_grid = model.get_param_grid()
        
        # Use stratified k-fold for imbalanced datasets
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Create a temporary model for grid search
        temp_model = model._create_model()
        
        # Special handling for TabM (neural network needs scaled data)
        if model_name == 'TabM':
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', temp_model)
            ])
            
            # Adjust parameter names for pipeline
            adjusted_param_grid = {}
            for key, value in param_grid.items():
                adjusted_param_grid[f'model__{key}'] = value
            
            grid_search = GridSearchCV(
                pipeline, adjusted_param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
        else:
            grid_search = GridSearchCV(
                temp_model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
        
        grid_search.fit(X_train, y_train)
        
        # Extract best params (remove pipeline prefix if needed)
        best_params = grid_search.best_params_
        if model_name == 'TabM':
            best_params = {k.replace('model__', ''): v for k, v in best_params.items()}
        
        print(f"Best parameters for {model_name}: {best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_params
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            params: Model parameters (if None, uses defaults)
        """
        print(f"Training {model_name}...")
        start_time = time.time()
        
        model = self.models[model_name]
        
        if params is None:
            params = model.get_default_params()
        
        model.fit(X_train, y_train, **params)
        
        training_time = time.time() - start_time
        print(f"{model_name} training completed in {training_time:.2f} seconds")
        
        # Save the model
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        model.save_model(model_path)
        
        self.trained_models[model_name] = {
            'model': model,
            'params': params,
            'training_time': training_time
        }
    
    def evaluate_model(
        self,
        model_name: str,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on validation and test sets.
        
        Args:
            model_name: Name of the model to evaluate
            X_val: Validation features
            y_val: Validation target
            X_test: Test features
            y_test: Test target
        
        Returns:
            Evaluation results
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        model = self.trained_models[model_name]['model']
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        results = {
            'model_name': model_name,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance.to_dict('records') if not feature_importance.empty else [],
            'training_time': self.trained_models[model_name]['training_time'],
            'parameters': self.trained_models[model_name]['params']
        }
        
        self.results[model_name] = results
        return results
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict[str, Any]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            X_test: Test features
            y_test: Test target
            tune_hyperparameters: Whether to tune hyperparameters
        
        Returns:
            Results for all models
        """
        print("Training all models...")
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Class distribution in training: {y_train.value_counts().to_dict()}")
        
        all_results = {}
        
        for model_name in self.models.keys():
            try:
                # Tune hyperparameters if requested
                if tune_hyperparameters:
                    best_params = self.tune_hyperparameters(model_name, X_train, y_train)
                else:
                    best_params = None
                
                # Train the model
                self.train_model(model_name, X_train, y_train, best_params)
                
                # Evaluate the model
                results = self.evaluate_model(model_name, X_val, y_val, X_test, y_test)
                all_results[model_name] = results
                
                print(f"\n{model_name} Results:")
                print(f"Validation F1: {results['validation_metrics']['f1_score']:.4f}")
                print(f"Test F1: {results['test_metrics']['f1_score']:.4f}")
                print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        results_path = os.path.join(self.results_dir, "model_comparison.json")
        
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'model_name': result['model_name'],
                'validation_metrics': result['validation_metrics'],
                'test_metrics': result['test_metrics'],
                'training_time': result['training_time'],
                'parameters': result['parameters']
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate a comparison report of all models."""
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            test_metrics = result['test_metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1-Score': test_metrics['f1_score'],
                'ROC-AUC': test_metrics['roc_auc'],
                'Training Time (s)': result['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        # Save comparison report
        report_path = os.path.join(self.results_dir, "model_comparison_report.csv")
        comparison_df.to_csv(report_path, index=False)
        
        return comparison_df
    
    def plot_model_comparison(self) -> None:
        """Create visualization comparing model performance."""
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        comparison_df = self.generate_comparison_report()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # F1-Score comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['F1-Score'])
        axes[0, 0].set_title('F1-Score by Model')
        axes[0, 0].set_ylabel('F1-Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # ROC-AUC comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['ROC-AUC'])
        axes[0, 1].set_title('ROC-AUC by Model')
        axes[0, 1].set_ylabel('ROC-AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['Training Time (s)'])
        axes[1, 0].set_title('Training Time by Model')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall metrics heatmap
        metrics_for_heatmap = comparison_df.set_index('Model')[
            ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        ]
        sns.heatmap(metrics_for_heatmap.T, annot=True, cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plot saved to {plot_path}")
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.results:
            raise ValueError("No results available. Train models first.")
        
        best_score = 0
        best_model_name = None
        
        for model_name, result in self.results.items():
            test_score = result['test_metrics'][metric]
            if test_score > best_score:
                best_score = test_score
                best_model_name = model_name
        
        best_model = self.trained_models[best_model_name]['model']
        
        return best_model_name, best_model
