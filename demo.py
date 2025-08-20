"""Quick demo script to test the ML pipeline with real data from Kaggle."""

import sys
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config.config import config
from src.data.preprocessor import ChurnFeatureProcessor
from src.models.trainer import ModelTrainer
from src.utils.reporting import StakeholderReporter

def download_real_churn_data():
    """Download real churn data from Kaggle using the method from EDA notebook."""
    try:
        print("Setting up Kaggle credentials...")
        
        # Get Kaggle token from environment (if available)
        kaggle_token = os.environ.get("KAGGLE_JSON")
        
        if not kaggle_token:
            print("❌ KAGGLE_JSON environment variable not found.")
            print("Using existing real_churn.csv if available...")
            
            # Check if we have the real data file already
            real_data_path = Path("data/real_churn.csv")
            if real_data_path.exists():
                print(f"✅ Using existing real data: {real_data_path}")
                return pd.read_csv(real_data_path)
            else:
                print("❌ No real data file found. Using synthetic data instead.")
                return None
        
        # Set up Kaggle API credentials
        kaggle_config_dir = os.path.expanduser("~/.config/kaggle")
        os.makedirs(kaggle_config_dir, exist_ok=True)
        
        kaggle_json_path = os.path.join(kaggle_config_dir, "kaggle.json")
        with open(kaggle_json_path, "w") as f:
            f.write(kaggle_token)
        
        os.chmod(kaggle_json_path, 0o600)
        
        print("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()
        
        # Download the same dataset from the EDA notebook
        dataset_name = "gauravtopre/bank-customer-churn-dataset"
        download_path = "/tmp"
        
        print(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        # Find the downloaded CSV file
        csv_files = glob.glob(f"{download_path}/*.csv")
        if len(csv_files) == 0:
            print("❌ No CSV files found after download.")
            return None
            
        csv_file_path = csv_files[0]
        print(f"✅ Dataset downloaded: {csv_file_path}")
        
        # Load and return the real data
        df = pd.read_csv(csv_file_path)
        
        # Save a copy to data/ folder for future use
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        local_path = data_dir / "real_churn.csv"
        df.to_csv(local_path, index=False)
        print(f"✅ Real data saved to: {local_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading real data: {str(e)}")
        print("Using synthetic data instead...")
        return None

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
    
    # Try to get real data first
    print("Attempting to download real churn data from Kaggle...")
    df = download_real_churn_data()
    
    if df is not None:
        print(f"✅ Using REAL data: {df.shape}")
        print(f"Real churn rate: {df['churn'].mean():.2%}")
        data_source = "REAL"
    else:
        print("Falling back to synthetic data...")
        df = create_sample_data(2000)
        print(f"Using SYNTHETIC data: {df.shape}")
        print(f"Synthetic churn rate: {df['churn'].mean():.2%}")
        data_source = "SYNTHETIC"
    
    # Preprocess features
    print(f"\nPreprocessing {data_source.lower()} features...")
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
    
    print(f"\nTraining models with {data_source} data (no hyperparameter tuning)...")
    
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
        print(f"{data_source} DATA RESULTS SUMMARY")
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
            'training_samples': len(X_train),
            'data_source': data_source
        }
        
        package_paths = reporter.create_comprehensive_stakeholder_package(
            results, best_model_name, dataset_info
        )
        
        print(f"\n📊 Stakeholder package created with {data_source} data in 'demo_results/' folder:")
        for package_type, path in package_paths.items():
            if isinstance(path, list):
                print(f"  • {package_type}: {len(path)} files")
            else:
                print(f"  • {package_type}: {Path(path).name}")
        
        print(f"\n✅ Demo completed successfully with {data_source} data!")
        print("Check the 'demo_results' folder for all outputs.")
        
    else:
        print("❌ No models were successfully trained.")

if __name__ == "__main__":
    main()
