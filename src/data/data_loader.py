"""
Data loading and preprocessing module for the Bank Churn Prediction project.
Handles Kaggle data download and BigQuery interactions.
"""

import os
from typing import Optional, Tuple
import pandas as pd
from google.cloud import bigquery
from kaggle.api.kaggle_api_extended import KaggleApi

class ChurnDataLoader:
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        kaggle_dataset: str = "gauravtopre/bank-customer-churn-dataset",
    ):
        """
        Initialize the ChurnDataLoader with GCP and Kaggle configurations.
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            kaggle_dataset: Kaggle dataset identifier
        """
        self.project_id = project_id
        self.dataset_id = f"{project_id}.{dataset_id}"
        self.table_id = table_id
        self.full_table_id = f"{self.dataset_id}.{table_id}"
        self.kaggle_dataset = kaggle_dataset
        
        # Initialize clients
        self.bq_client = bigquery.Client()
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
    
    def setup_kaggle_credentials(self, kaggle_json: str) -> None:
        """
        Set up Kaggle credentials from environment variable.
        
        Args:
            kaggle_json: JSON string containing Kaggle credentials
        """
        kaggle_config_dir = os.path.expanduser("~/.config/kaggle")
        os.makedirs(kaggle_config_dir, exist_ok=True)
        
        kaggle_json_path = os.path.join(kaggle_config_dir, "kaggle.json")
        with open(kaggle_json_path, "w") as f:
            f.write(kaggle_json)
        
        os.chmod(kaggle_json_path, 0o600)
    
    def download_dataset(self, download_path: str = "/tmp") -> str:
        """
        Download the dataset from Kaggle.
        
        Args:
            download_path: Path where to save the downloaded dataset
            
        Returns:
            str: Path to the downloaded CSV file
        """
        self.kaggle_api.dataset_download_files(
            self.kaggle_dataset,
            path=download_path,
            unzip=True
        )
        
        # Find the downloaded CSV file
        csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found after download")
            
        return os.path.join(download_path, csv_files[0])
    
    def upload_to_bigquery(
        self,
        file_path: str,
        write_disposition: str = "WRITE_TRUNCATE"
    ) -> None:
        """
        Upload the dataset to BigQuery.
        
        Args:
            file_path: Path to the CSV file to upload
            write_disposition: BigQuery write disposition (WRITE_TRUNCATE or WRITE_APPEND)
        """
        # Ensure dataset exists
        self.bq_client.create_dataset(self.dataset_id, exists_ok=True)
        
        # Configure the upload job
        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            write_disposition=write_disposition
        )
        
        # Perform the upload
        with open(file_path, "rb") as source_file:
            job = self.bq_client.load_table_from_file(
                source_file,
                self.full_table_id,
                job_config=job_config
            )
        
        # Wait for the job to complete
        job.result()
        
        # Validate the upload
        table = self.bq_client.get_table(self.full_table_id)
        print(f"Loaded {table.num_rows} rows into {self.full_table_id}")
    
    def fetch_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from BigQuery using an optional custom query.
        
        Args:
            query: Custom SQL query to execute. If None, fetches all data.
            
        Returns:
            pd.DataFrame: The query results as a pandas DataFrame
        """
        if query is None:
            query = f"""
            SELECT *
            FROM `{self.full_table_id}`
            """
        
        return self.bq_client.query(query).to_dataframe()
    
    def validate_duplicates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Check for duplicate entries in the dataset.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing duplicate customer IDs
            and complete duplicate rows
        """
        # Check for duplicate customer IDs
        duplicates_query = f"""
        SELECT customer_id, COUNT(*) AS count
        FROM `{self.full_table_id}`
        GROUP BY customer_id
        HAVING count > 1
        """
        
        # Check for complete duplicate rows
        complete_duplicates_query = f"""
        SELECT 
            customer_id, credit_score, country, gender, age, tenure,
            balance, products_number, credit_card, active_member,
            estimated_salary, churn,
            COUNT(*) as duplicate_count
        FROM `{self.full_table_id}`
        GROUP BY 
            customer_id, credit_score, country, gender, age, tenure,
            balance, products_number, credit_card, active_member,
            estimated_salary, churn
        HAVING COUNT(*) > 1
        ORDER BY duplicate_count DESC
        """
        
        duplicates = self.bq_client.query(duplicates_query).to_dataframe()
        complete_duplicates = self.bq_client.query(complete_duplicates_query).to_dataframe()
        
        return duplicates, complete_duplicates
    
    def check_missing_values(self) -> pd.DataFrame:
        """
        Check for missing values in all columns.
        
        Returns:
            pd.DataFrame: Missing value counts for each column
        """
        # Get table schema
        table = self.bq_client.get_table(self.full_table_id)
        columns = [field.name for field in table.schema]
        
        # Generate and execute missing values query
        select_expr = [
            f"SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) AS missing_{col}"
            for col in columns
        ]
        
        query = f"""
        SELECT
            {', '.join(select_expr)}
        FROM `{self.full_table_id}`
        """
        
        return self.bq_client.query(query).to_dataframe()
