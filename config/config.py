"""Configuration settings for the Bank Churn Prediction project."""

from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class DataConfig:
    # BigQuery settings
    project_id: str = os.getenv("GCP_PROJECT_ID", "kagglebigquerybankchurn")
    dataset_id: str = "churn_analysis"
    table_id: str = "Kaggle_churn"
    
    # Kaggle settings
    kaggle_dataset: str = "gauravtopre/bank-customer-churn-dataset"
    
    # Data processing
    target_column: str = "churn"
    id_column: str = "customer_id"
    
    # Feature groups
    continuous_features: List[str] = field(default_factory=lambda: [
        "credit_score",
        "age",
        "estimated_salary",
        "balance"
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        "gender",
        "country",
        "credit_card",
        "active_member",
        "products_number",
        "tenure"
    ])
    
    # Feature engineering settings
    age_transform: str = "log"  # Options: 'log', 'none'
    salary_transform: str = "bin"  # Options: 'bin', 'none'
    balance_transform: str = "zero_flag"  # Options: 'zero_flag', 'none'

@dataclass
class ModelConfig:
    # General settings
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    
    # Cross-validation
    n_splits: int = 5
    
    # Model specific parameters (to be updated with best parameters after tuning)
    tabm_params: Optional[dict] = None
    logistic_params: Optional[dict] = None
    rf_params: Optional[dict] = None
    xgb_params: Optional[dict] = None

@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    models_dir: str = os.path.join(base_dir, "models")
    results_dir: str = os.path.join(base_dir, "results")
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)

# Create global config instance
config = ProjectConfig()
