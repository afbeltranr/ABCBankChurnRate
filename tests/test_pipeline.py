"""Integration tests for the complete ML pipeline."""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from src.data.preprocessor import ChurnFeatureProcessor
from src.models.trainer import ModelTrainer
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel

class TestMLPipeline(unittest.TestCase):
    """Test the complete ML pipeline."""
    
    def setUp(self):
        """Set up test data and temporary directories."""
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        self.sample_data = pd.DataFrame({
            'customer_id': range(n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'estimated_salary': np.random.uniform(30000, 150000, n_samples),
            'balance': np.random.exponential(50000, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'country': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
            'credit_card': np.random.choice([0, 1], n_samples),
            'active_member': np.random.choice([0, 1], n_samples),
            'products_number': np.random.choice([1, 2, 3, 4], n_samples),
            'tenure': np.random.randint(0, 11, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        # Set up temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.results_dir = Path(self.temp_dir) / "results"
    
    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_dir)
    
    def test_feature_preprocessing(self):
        """Test the feature preprocessing pipeline."""
        continuous_features = ['credit_score', 'age', 'estimated_salary', 'balance']
        categorical_features = ['gender', 'country', 'credit_card', 'active_member', 'products_number', 'tenure']
        
        preprocessor = ChurnFeatureProcessor(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            target_column='churn',
            id_column='customer_id'
        )
        
        # Separate features and target
        X = self.sample_data.drop(columns=['churn', 'customer_id'])
        y = self.sample_data['churn']
        
        # Test fit_transform
        X_processed = preprocessor.fit_transform(X)
        
        # Assertions
        self.assertIsInstance(X_processed, pd.DataFrame)
        self.assertEqual(len(X_processed), len(X))
        self.assertTrue(len(X_processed.columns) >= len(continuous_features) + len(categorical_features))
        
        # Test transform on new data
        X_new = X.sample(100)
        X_new_processed = preprocessor.transform(X_new)
        self.assertEqual(len(X_new_processed), 100)
    
    def test_individual_models(self):
        """Test individual model training and prediction."""
        # Prepare data
        X = self.sample_data.drop(columns=['churn', 'customer_id'])
        y = self.sample_data['churn']
        
        continuous_features = ['credit_score', 'age', 'estimated_salary', 'balance']
        categorical_features = ['gender', 'country', 'credit_card', 'active_member', 'products_number', 'tenure']
        
        preprocessor = ChurnFeatureProcessor(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            target_column='churn',
            id_column='customer_id'
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        # Test Logistic Regression
        lr_model = LogisticRegressionModel()
        lr_model.fit(X_processed, y)
        
        self.assertTrue(lr_model.is_fitted)
        
        # Test predictions
        predictions = lr_model.predict(X_processed[:100])
        probabilities = lr_model.predict_proba(X_processed[:100])
        
        self.assertEqual(len(predictions), 100)
        self.assertEqual(probabilities.shape, (100, 2))
        
        # Test evaluation
        metrics = lr_model.evaluate(X_processed[:100], y[:100])
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
    
    def test_model_trainer(self):
        """Test the model trainer pipeline."""
        # Prepare data
        X = self.sample_data.drop(columns=['churn', 'customer_id'])
        y = self.sample_data['churn']
        
        continuous_features = ['credit_score', 'age', 'estimated_salary', 'balance']
        categorical_features = ['gender', 'country', 'credit_card', 'active_member', 'products_number', 'tenure']
        
        preprocessor = ChurnFeatureProcessor(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            target_column='churn',
            id_column='customer_id'
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        # Initialize trainer
        trainer = ModelTrainer(
            models_dir=str(self.models_dir),
            results_dir=str(self.results_dir),
            random_state=42
        )
        
        # Test data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_processed, y)
        
        # Check split sizes
        total_size = len(X_processed)
        self.assertLess(len(X_train), total_size)
        self.assertLess(len(X_val), total_size)
        self.assertLess(len(X_test), total_size)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total_size)
        
        # Test training a single model (without hyperparameter tuning for speed)
        trainer.train_model('LogisticRegression', X_train, y_train)
        
        # Check that model was trained
        self.assertIn('LogisticRegression', trainer.trained_models)
        
        # Test evaluation
        results = trainer.evaluate_model('LogisticRegression', X_val, y_val, X_test, y_test)
        
        # Check results structure
        self.assertIn('validation_metrics', results)
        self.assertIn('test_metrics', results)
        self.assertIn('training_time', results)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Prepare data
        X = self.sample_data.drop(columns=['churn', 'customer_id'])
        y = self.sample_data['churn']
        
        continuous_features = ['credit_score', 'age', 'estimated_salary', 'balance']
        categorical_features = ['gender', 'country', 'credit_card', 'active_member', 'products_number', 'tenure']
        
        preprocessor = ChurnFeatureProcessor(
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            target_column='churn',
            id_column='customer_id'
        )
        
        X_processed = preprocessor.fit_transform(X)
        
        # Train model
        model = LogisticRegressionModel()
        model.fit(X_processed, y)
        
        # Save model
        model_path = Path(self.temp_dir) / "test_model.pkl"
        model.save_model(str(model_path))
        
        self.assertTrue(model_path.exists())
        
        # Load model
        new_model = LogisticRegressionModel()
        new_model.load_model(str(model_path))
        
        # Test that loaded model works
        self.assertTrue(new_model.is_fitted)
        predictions_original = model.predict(X_processed[:10])
        predictions_loaded = new_model.predict(X_processed[:10])
        
        np.testing.assert_array_equal(predictions_original, predictions_loaded)

if __name__ == '__main__':
    unittest.main()
