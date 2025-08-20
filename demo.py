"""Quick demo script to test the ML pipeline with sample data."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config.config import config
from src.data.preprocessor import ChurnFeatureProcessor
from src.models.trainer import ModelTrainer
from src.utils.reporting import StakeholderReporter

def create_sample_data(n_samples=2000):
    """Create sample churn data for testing."""
    np.random.seed(42)
    
    # Create realistic churn data
    data = {
        'customer_id': range(n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'age': np.random.randint(18, 80, n_samples),
        'estimated_salary': np.random.uniform(30000, 150000, n_samples),
        'balance': np.random.exponential(50000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'country': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'credit_card': np.random.choice([0, 1], n_samples),
        'active_member': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),  # More active members
        'products_number': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'tenure': np.random.randint(0, 11, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn based on features
    churn_prob = (
        0.1 +  # Base churn rate
        0.2 * (df['active_member'] == 0) +  # Inactive members more likely to churn
        0.1 * (df['products_number'] >= 3) +  # High product users more likely to churn
        0.05 * (df['age'] < 25) +  # Young customers more likely to churn
        0.05 * (df['balance'] == 0)  # Zero balance more likely to churn
    )
    
    df['churn'] = np.random.binomial(1, churn_prob, n_samples)
    
    return df

def main():
    """Run the demo pipeline."""
    print("="*60)
    print("CHURN PREDICTION PIPELINE - DEMO MODE")
    print("="*60)
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data(2000)
    print(f"Sample data created: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    
    # Preprocess features
    print("\nPreprocessing features...")
    X = df.drop(columns=[config.data.target_column, config.data.id_column])
    y = df[config.data.target_column]
    
    preprocessor = ChurnFeatureProcessor(
        continuous_features=config.data.continuous_features,
        categorical_features=config.data.categorical_features,
        target_column=config.data.target_column,
        id_column=config.data.id_column
    )
    
    X_processed = preprocessor.fit_transform(X)
    print(f"Processed features shape: {X_processed.shape}")
    
    # Train models
    print("\nInitializing model trainer...")
    trainer = ModelTrainer(
        models_dir="demo_models",
        results_dir="demo_results",
        random_state=42
    )
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_processed, y)
    
    print("\nTraining models (demo mode - no hyperparameter tuning)...")
    
    # Train all 6 models for comprehensive comparison
    demo_models = ['LogisticRegression', 'NaiveBayes', 'RandomForest', 'XGBoost', 'AdaBoost', 'TabM']
    results = {}
    
    for model_name in demo_models:
        try:
            print(f"\nTraining {model_name}...")
            trainer.train_model(model_name, X_train, y_train)
            result = trainer.evaluate_model(model_name, X_val, y_val, X_test, y_test)
            results[model_name] = result
            
            print(f"✅ {model_name} - F1: {result['test_metrics']['f1_score']:.3f}, "
                  f"ROC-AUC: {result['test_metrics']['roc_auc']:.3f}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
    
    if results:
        print("\n" + "="*50)
        print("DEMO RESULTS SUMMARY")
        print("="*50)
        
        # Generate comparison report
        comparison_df = trainer.generate_comparison_report()
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model('f1_score')
        print(f"\n🏆 Best model: {best_model_name}")
        
        # Generate stakeholder reports
        print("\nGenerating stakeholder reports...")
        reporter = StakeholderReporter("demo_results")
        
        dataset_info = {
            'total_customers': len(df),
            'churn_rate': y.mean(),
            'num_features': X_processed.shape[1],
            'training_samples': len(X_train)
        }
        
        package_paths = reporter.create_comprehensive_stakeholder_package(
            results, best_model_name, dataset_info
        )
        
        print("\n📊 Stakeholder package created in 'demo_results/' folder:")
        for package_type, path in package_paths.items():
            if isinstance(path, list):
                print(f"  • {package_type}: {len(path)} files")
            else:
                print(f"  • {package_type}: {Path(path).name}")
        
        print("\n✅ Demo completed successfully!")
        print("Check the 'demo_results' folder for all outputs.")
        
    else:
        print("❌ No models were successfully trained.")

if __name__ == "__main__":
    main()
