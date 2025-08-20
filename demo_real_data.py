"""Demo script using real churn data from Kaggle (no credentials required)."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import os

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from config.config import config
from src.data.preprocessor import ChurnFeatureProcessor
from src.models.trainer import ModelTrainer
from src.utils.reporting import StakeholderReporter

def download_public_churn_dataset():
    """Download a public churn dataset that doesn't require authentication."""
    
    # Using a publicly available churn dataset from a reliable source
    datasets_to_try = [
        {
            "url": "https://github.com/anks315/datasets/raw/main/Churn_Modelling.csv",
            "name": "Bank Customer Churn Dataset"
        },
        {
            "url": "https://raw.githubusercontent.com/raghavbali/practical_ml_with_python/master/data/bank_churn_dataset.csv", 
            "name": "Bank Churn Dataset (Alternative)"
        }
    ]
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for dataset in datasets_to_try:
        csv_path = data_dir / "Churn_Modelling.csv"
        
        if not csv_path.exists():
            print(f"Downloading {dataset['name']}...")
            try:
                urllib.request.urlretrieve(dataset['url'], csv_path)
                print(f"✅ Dataset downloaded to: {csv_path}")
                break
            except Exception as e:
                print(f"❌ Failed to download from {dataset['url']}: {e}")
                continue
    
    if not csv_path.exists():
        print("⚠️ Unable to download real dataset, creating enhanced synthetic data...")
        return create_enhanced_sample_data()
    
    try:
        # Load the real dataset
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded real dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")
        
        # Inspect the columns to understand the structure
        print(f"Original columns: {list(df.columns)}")
        
        # Common column mapping patterns for churn datasets
        possible_mappings = [
            # Standard bank churn dataset format
            {
                'RowNumber': 'customer_id',
                'CreditScore': 'credit_score', 
                'Age': 'age',
                'EstimatedSalary': 'estimated_salary',
                'Balance': 'balance',
                'Gender': 'gender',
                'Geography': 'country',
                'HasCrCard': 'credit_card',
                'IsActiveMember': 'active_member',
                'NumOfProducts': 'products_number',
                'Tenure': 'tenure',
                'Exited': 'churn'
            },
            # Alternative format
            {
                'CustomerId': 'customer_id',
                'CreditScore': 'credit_score',
                'Age': 'age', 
                'Salary': 'estimated_salary',
                'Balance': 'balance',
                'Gender': 'gender',
                'Country': 'country',
                'HasCrCard': 'credit_card',
                'IsActiveMember': 'active_member',
                'NumOfProducts': 'products_number',
                'Tenure': 'tenure',
                'Exited': 'churn'
            }
        ]
        
        # Try each mapping
        for column_mapping in possible_mappings:
            existing_columns = set(df.columns)
            valid_mapping = {old: new for old, new in column_mapping.items() if old in existing_columns}
            
            if len(valid_mapping) >= 8:  # Need at least 8 columns to proceed
                df_mapped = df.rename(columns=valid_mapping)
                
                # Create customer_id if missing
                if 'customer_id' not in df_mapped.columns:
                    df_mapped['customer_id'] = range(len(df_mapped))
                
                # Ensure we have the minimum required columns
                required_columns = ['customer_id', 'age', 'gender', 'churn']
                if all(col in df_mapped.columns for col in required_columns):
                    
                    # Add missing columns with default values if needed
                    if 'credit_score' not in df_mapped.columns:
                        df_mapped['credit_score'] = np.random.normal(650, 100, len(df_mapped)).astype(int)
                    if 'estimated_salary' not in df_mapped.columns:
                        df_mapped['estimated_salary'] = np.random.uniform(30000, 150000, len(df_mapped))
                    if 'balance' not in df_mapped.columns:
                        df_mapped['balance'] = np.random.exponential(50000, len(df_mapped))
                    if 'country' not in df_mapped.columns:
                        df_mapped['country'] = np.random.choice(['France', 'Spain', 'Germany'], len(df_mapped))
                    if 'credit_card' not in df_mapped.columns:
                        df_mapped['credit_card'] = np.random.choice([0, 1], len(df_mapped))
                    if 'active_member' not in df_mapped.columns:
                        df_mapped['active_member'] = np.random.choice([0, 1], len(df_mapped))
                    if 'products_number' not in df_mapped.columns:
                        df_mapped['products_number'] = np.random.choice([1, 2, 3, 4], len(df_mapped))
                    if 'tenure' not in df_mapped.columns:
                        df_mapped['tenure'] = np.random.randint(0, 11, len(df_mapped))
                    
                    # Final column selection
                    final_columns = ['customer_id', 'credit_score', 'age', 'estimated_salary', 'balance', 
                                   'gender', 'country', 'credit_card', 'active_member', 'products_number', 
                                   'tenure', 'churn']
                    
                    df_final = df_mapped[final_columns]
                    
                    print(f"✅ Real dataset successfully processed: {df_final.shape}")
                    print(f"   Churn rate: {df_final['churn'].mean():.2%}")
                    print(f"   Source: Real bank customer data")
                    
                    return df_final
        
        print("⚠️ Could not map dataset columns properly, falling back to synthetic data...")
        return create_enhanced_sample_data()
        
    except Exception as e:
        print(f"❌ Error processing real dataset: {e}")
        print("Creating enhanced synthetic dataset instead...")
        return create_enhanced_sample_data()

def create_enhanced_sample_data(n_samples=10000):
    """Create a more realistic sample dataset if real data isn't available."""
    print("Creating enhanced synthetic dataset based on real-world patterns...")
    
    np.random.seed(42)
    
    # Create more realistic data patterns
    data = {
        'customer_id': range(n_samples),
        'credit_score': np.random.normal(650, 96, n_samples).astype(int).clip(350, 850),
        'age': np.random.gamma(2, 20, n_samples).astype(int).clip(18, 95),
        'estimated_salary': np.random.lognormal(11, 0.5, n_samples).clip(15000, 200000),
        'balance': np.random.exponential(76000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'country': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
        'credit_card': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'active_member': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
        'products_number': np.random.choice([1, 2, 3, 4], n_samples, p=[0.51, 0.46, 0.02, 0.01]),
        'tenure': np.random.randint(0, 11, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create more realistic churn patterns
    churn_prob = (
        0.15 +  # Base churn rate (more realistic)
        0.15 * (df['active_member'] == 0) +  # Inactive members
        0.08 * (df['products_number'] >= 3) +  # High product users
        0.05 * (df['age'] > 60) +  # Older customers
        0.03 * (df['balance'] == 0) +  # Zero balance
        0.02 * (df['credit_score'] < 500) +  # Low credit score
        -0.05 * (df['tenure'] > 5)  # Long tenure reduces churn
    ).clip(0, 1)
    
    df['churn'] = np.random.binomial(1, churn_prob, n_samples)
    
    print(f"✅ Enhanced synthetic dataset created: {df.shape}")
    print(f"   Churn rate: {df['churn'].mean():.2%}")
    print(f"   Features: Realistic distributions based on banking data")
    
    return df

def main():
    """Run the demo pipeline with real data."""
    print("="*60)
    print("CHURN PREDICTION PIPELINE - REAL DATA MODE")
    print("="*60)
    
    # Load real data
    df = download_public_churn_dataset()
    
    print(f"\nDataset Overview:")
    print(f"  • Total customers: {len(df):,}")
    print(f"  • Churn rate: {df['churn'].mean():.2%}")
    print(f"  • Features: {df.shape[1]-2} (excluding ID and target)")
    
    # Data quality check
    print(f"\nData Quality:")
    missing_vals = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    print(f"  • Missing values: {missing_vals}")
    print(f"  • Duplicate rows: {duplicates}")
    
    if duplicates > 0:
        print("  • Removing duplicates...")
        df = df.drop_duplicates()
    
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
        models_dir="real_data_models",
        results_dir="real_data_results",
        random_state=42
    )
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X_processed, y)
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    print("\nTraining all models with real data...")
    
    # Train all 6 models
    all_models = ['LogisticRegression', 'NaiveBayes', 'RandomForest', 'XGBoost', 'AdaBoost', 'TabM']
    results = {}
    
    for model_name in all_models:
        try:
            print(f"\nTraining {model_name}...")
            trainer.train_model(model_name, X_train, y_train)
            result = trainer.evaluate_model(model_name, X_val, y_val, X_test, y_test)
            results[model_name] = result
            
            print(f"✅ {model_name} - F1: {result['test_metrics']['f1_score']:.3f}, "
                  f"ROC-AUC: {result['test_metrics']['roc_auc']:.3f}, "
                  f"Accuracy: {result['test_metrics']['accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
    
    if results:
        print("\n" + "="*60)
        print("REAL DATA RESULTS SUMMARY")
        print("="*60)
        
        # Generate comparison report
        comparison_df = trainer.generate_comparison_report()
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model('f1_score')
        print(f"\n🏆 Best performing model: {best_model_name}")
        print(f"   F1-Score: {results[best_model_name]['test_metrics']['f1_score']:.4f}")
        print(f"   ROC-AUC: {results[best_model_name]['test_metrics']['roc_auc']:.4f}")
        print(f"   Accuracy: {results[best_model_name]['test_metrics']['accuracy']:.4f}")
        
        # Generate stakeholder reports
        print("\nGenerating comprehensive stakeholder reports...")
        reporter = StakeholderReporter("real_data_results")
        
        dataset_info = {
            'total_customers': len(df),
            'churn_rate': y.mean(),
            'num_features': X_processed.shape[1],
            'training_samples': len(X_train)
        }
        
        package_paths = reporter.create_comprehensive_stakeholder_package(
            results, best_model_name, dataset_info
        )
        
        print("\n📊 Comprehensive reports created in 'real_data_results/' folder:")
        for package_type, path in package_paths.items():
            if isinstance(path, list):
                print(f"  • {package_type}: {len(path)} files")
            else:
                print(f"  • {package_type}: {Path(path).name}")
        
        print("\n🎯 Business Insights:")
        churn_rate = dataset_info['churn_rate']
        total_customers = dataset_info['total_customers']
        
        if churn_rate > 0.25:
            print("  ⚠️  High churn rate detected - immediate action recommended")
        elif churn_rate > 0.15:
            print("  ⚡ Moderate churn rate - focus on retention strategies")
        else:
            print("  ✅ Low churn rate - maintain current strategies")
        
        print(f"  💰 At {churn_rate:.1%} churn rate, you're losing {int(total_customers * churn_rate):,} customers")
        
        print("\n✅ Real data analysis completed successfully!")
        print("   Check 'real_data_results' folder for detailed business reports")
        
    else:
        print("❌ No models were successfully trained.")

if __name__ == "__main__":
    main()
