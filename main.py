"""Main script for running the complete churn prediction pipeline."""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config.config import config
from src.data.data_loader import ChurnDataLoader
from src.data.preprocessor import ChurnFeatureProcessor
from src.models.trainer import ModelTrainer
from src.utils.reporting import StakeholderReporter

def setup_environment():
    """Set up the environment and credentials."""
    # Set up directories
    config.setup_directories()
    
    # Set up credentials from environment variables
    kaggle_json = os.getenv("KAGGLE_JSON")
    gcp_key = os.getenv("GCP_SA_KEY")
    
    if not kaggle_json:
        raise ValueError("KAGGLE_JSON environment variable not set")
    if not gcp_key:
        raise ValueError("GCP_SA_KEY environment variable not set")
    
    # Set up GCP credentials
    gcp_path = "/tmp/sa_credentials.json"
    with open(gcp_path, "w") as f:
        f.write(gcp_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_path
    
    return kaggle_json

def load_and_preprocess_data(download_new: bool = False):
    """Load and preprocess the data."""
    print("Setting up data loader...")
    
    # Initialize data loader
    data_loader = ChurnDataLoader(
        project_id=config.data.project_id,
        dataset_id=config.data.dataset_id,
        table_id=config.data.table_id,
        kaggle_dataset=config.data.kaggle_dataset
    )
    
    if download_new:
        print("Setting up Kaggle credentials...")
        kaggle_json = setup_environment()
        data_loader.setup_kaggle_credentials(kaggle_json)
        
        print("Downloading dataset from Kaggle...")
        csv_path = data_loader.download_dataset()
        
        print("Uploading dataset to BigQuery...")
        data_loader.upload_to_bigquery(csv_path)
    
    print("Fetching data from BigQuery...")
    df = data_loader.fetch_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Check for duplicates
    print("Checking for duplicates...")
    duplicates, complete_duplicates = data_loader.validate_duplicates()
    print(f"Customer ID duplicates: {len(duplicates)}")
    print(f"Complete row duplicates: {len(complete_duplicates)}")
    
    # Remove duplicates if they exist
    if len(complete_duplicates) > 0:
        print("Removing duplicate rows...")
        df = df.drop_duplicates()
        print(f"Dataset shape after deduplication: {df.shape}")
    
    return df

def preprocess_features(df):
    """Preprocess features for model training."""
    print("Preprocessing features...")
    
    # Separate features and target
    X = df.drop(columns=[config.data.target_column, config.data.id_column])
    y = df[config.data.target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Initialize preprocessor
    preprocessor = ChurnFeatureProcessor(
        continuous_features=config.data.continuous_features,
        categorical_features=config.data.categorical_features,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
        age_transform=config.data.age_transform,
        salary_transform=config.data.salary_transform,
        balance_transform=config.data.balance_transform
    )
    
    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Feature columns: {list(X_processed.columns)}")
    
    return X_processed, y, preprocessor

def train_and_evaluate_models(X, y, df, tune_hyperparameters=True):
    """Train and evaluate all models."""
    print("Initializing model trainer...")
    
    trainer = ModelTrainer(
        models_dir=config.models_dir,
        results_dir=config.results_dir,
        random_state=config.model.random_state
    )
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
        X, y,
        test_size=config.model.test_size,
        val_size=config.model.validation_size
    )
    
    print("Training and evaluating models...")
    results = trainer.train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        tune_hyperparameters=tune_hyperparameters
    )
    
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    
    # Generate comparison report
    comparison_df = trainer.generate_comparison_report()
    print(comparison_df.to_string(index=False))
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model('f1_score')
    print(f"\nBest model: {best_model_name}")
    print(f"Best F1-Score: {results[best_model_name]['test_metrics']['f1_score']:.4f}")
    
    # Generate visualizations
    print("\nGenerating comparison plots...")
    trainer.plot_model_comparison()
    
    # Generate stakeholder reports
    print("\nGenerating stakeholder reports...")
    reporter = StakeholderReporter(config.results_dir)
    
    # Create dataset info for reporting
    dataset_info = {
        'total_customers': len(df),
        'churn_rate': y.mean(),
        'num_features': X.shape[1],
        'training_samples': len(X_train)
    }
    
    # Generate comprehensive stakeholder package
    package_paths = reporter.create_comprehensive_stakeholder_package(
        results, best_model_name, dataset_info
    )
    
    print("\nStakeholder package created:")
    for package_type, path in package_paths.items():
        if isinstance(path, list):
            print(f"  {package_type}: {len(path)} files created")
        else:
            print(f"  {package_type}: {path}")
    
    return results, trainer

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run churn prediction pipeline')
    parser.add_argument('--download-new', action='store_true', 
                       help='Download new data from Kaggle')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("BANK CUSTOMER CHURN PREDICTION PIPELINE")
        print("="*60)
        
        # Load and preprocess data
        df = load_and_preprocess_data(download_new=args.download_new)
        
        # Preprocess features
        X, y, preprocessor = preprocess_features(df)
        
        # Train and evaluate models
        results, trainer = train_and_evaluate_models(
            X, y, df,
            tune_hyperparameters=not args.no_tuning
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved in: {config.results_dir}")
        print(f"Models saved in: {config.models_dir}")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
