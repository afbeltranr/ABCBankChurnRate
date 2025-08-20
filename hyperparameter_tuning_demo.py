"""Hyperparameter tuning demo to showcase overfitting prevention techniques."""

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
    """Download real churn data from Kaggle."""
    try:
        # Check if we have the real data file already
        real_data_path = Path("data/real_churn.csv")
        if real_data_path.exists():
            print(f"✅ Using existing real data: {real_data_path}")
            return pd.read_csv(real_data_path)
            
        print("❌ No real data file found. Please run demo.py first to download data.")
        return None
        
    except Exception as e:
        print(f"❌ Error loading real data: {str(e)}")
        return None

def main():
    """Run hyperparameter tuning demo."""
    print("="*80)
    print("HYPERPARAMETER TUNING DEMO - OVERFITTING PREVENTION")
    print("="*80)
    
    # Load real data
    print("Loading real churn data...")
    df = download_real_churn_data()
    
    if df is None:
        print("❌ Could not load real data. Exiting.")
        return
    
    print(f"✅ Loaded real data: {df.shape}")
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
    
    # Initialize model trainer
    print("\nInitializing model trainer...")
    trainer = ModelTrainer(
        models_dir="tuned_models",
        results_dir="tuning_results",
        random_state=42
    )
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_processed, y)
    
    # Models to tune (starting with fastest ones for demo)
    models_to_tune = ['LogisticRegression', 'RandomForest', 'XGBoost']
    
    print(f"\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH OVERFITTING PREVENTION")
    print("="*60)
    
    tuning_results = {}
    
    for model_name in models_to_tune:
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        try:
            # Show parameter grid and estimate processing time
            model = trainer.models[model_name]
            param_grid = model.get_param_grid()
            print(f"\nParameter grid for {model_name}:")
            for param, values in param_grid.items():
                print(f"  • {param}: {values}")
            
            total_combinations = np.prod([len(v) for v in param_grid.values()])
            cv_folds = 3
            total_fits = total_combinations * cv_folds
            
            # Estimate processing time based on model type
            time_estimates = {
                'LogisticRegression': 0.01,  # seconds per fit
                'RandomForest': 0.3,         # seconds per fit
                'XGBoost': 0.2               # seconds per fit
            }
            
            estimated_time_per_fit = time_estimates.get(model_name, 0.5)
            estimated_total_time = total_fits * estimated_time_per_fit
            
            print(f"\n📊 Processing Requirements:")
            print(f"  • Parameter combinations: {total_combinations}")
            print(f"  • With {cv_folds}-fold CV: {total_fits} total fits")
            print(f"  • Estimated time per fit: {estimated_time_per_fit}s")
            print(f"  • Estimated total time: {estimated_total_time/60:.1f} minutes")
            
            if estimated_total_time > 300:  # 5 minutes
                print(f"  ⚠️  WARNING: Estimated time > 5 minutes!")
            else:
                print(f"  ✅ Reasonable processing time for 4-core 16GB setup")
            
            # Tune hyperparameters
            print(f"\n🔧 Tuning hyperparameters for {model_name}...")
            best_params = trainer.tune_hyperparameters(
                model_name, X_train, y_train, cv_folds=3, scoring='f1_weighted'
            )
            
            # Train with best parameters
            print(f"\n🎯 Training {model_name} with best parameters...")
            trainer.train_model(model_name, X_train, y_train, best_params)
            
            # Evaluate
            result = trainer.evaluate_model(model_name, X_val, y_val, X_test, y_test)
            tuning_results[model_name] = {
                'best_params': best_params,
                'result': result
            }
            
            print(f"✅ {model_name} - F1: {result['test_metrics']['f1_score']:.3f}, "
                  f"ROC-AUC: {result['test_metrics']['roc_auc']:.3f}")
            
        except Exception as e:
            print(f"❌ Error tuning {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Results summary
    if tuning_results:
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING RESULTS SUMMARY")
        print("="*60)
        
        for model_name, data in tuning_results.items():
            result = data['result']
            params = data['best_params']
            
            print(f"\n🏆 {model_name}:")
            print(f"  • F1-Score: {result['test_metrics']['f1_score']:.4f}")
            print(f"  • ROC-AUC: {result['test_metrics']['roc_auc']:.4f}")
            print(f"  • Accuracy: {result['test_metrics']['accuracy']:.4f}")
            print(f"  • Best Parameters:")
            for param, value in params.items():
                print(f"    - {param}: {value}")
        
        # Generate comparison report
        comparison_df = trainer.generate_comparison_report()
        print(f"\n📊 Detailed Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model('f1_score')
        print(f"\n🥇 Best tuned model: {best_model_name}")
        
        print("\n✅ Hyperparameter tuning completed!")
        print("📁 Check 'tuned_models/' for saved models")
        print("📊 Check 'tuning_results/' for detailed reports")
        
    else:
        print("❌ No models were successfully tuned.")

if __name__ == "__main__":
    main()
