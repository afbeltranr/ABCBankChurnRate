"""Unit tests for the ChurnDataLoader class."""

import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from google.cloud import bigquery
from src.data.data_loader import ChurnDataLoader

class TestChurnDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.project_id = "test-project"
        self.dataset_id = "test_dataset"
        self.table_id = "test_table"
        
        # Create the loader with mocked clients
        with patch('google.cloud.bigquery.Client'), \
             patch('kaggle.api.kaggle_api_extended.KaggleApi'):
            self.loader = ChurnDataLoader(
                self.project_id,
                self.dataset_id,
                self.table_id
            )
    
    def test_setup_kaggle_credentials(self):
        """Test Kaggle credentials setup."""
        test_json = '{"username": "test", "key": "test-key"}'
        test_path = os.path.expanduser("~/.config/kaggle/kaggle.json")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        self.loader.setup_kaggle_credentials(test_json)
        
        # Check if the file exists and has correct permissions
        self.assertTrue(os.path.exists(test_path))
        self.assertEqual(os.stat(test_path).st_mode & 0o777, 0o600)
        
        # Clean up
        os.remove(test_path)
    
    @patch('kaggle.api.kaggle_api_extended.KaggleApi')
    def test_download_dataset(self, mock_kaggle_api):
        """Test dataset download from Kaggle."""
        # Create a temporary test file
        test_dir = "/tmp/test_download"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.csv")
        with open(test_file, "w") as f:
            f.write("test,data\n1,2\n")
        
        # Mock the Kaggle API download
        self.loader.kaggle_api = mock_kaggle_api
        
        # Test the download
        result = self.loader.download_dataset(test_dir)
        self.assertEqual(result, test_file)
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
    
    @patch('google.cloud.bigquery.Client')
    def test_upload_to_bigquery(self, mock_bq_client):
        """Test BigQuery upload functionality."""
        # Create a temporary test file
        test_file = "/tmp/test_upload.csv"
        with open(test_file, "w") as f:
            f.write("test,data\n1,2\n")
        
        # Mock BigQuery client methods
        mock_job = MagicMock()
        mock_table = MagicMock()
        mock_table.num_rows = 1
        
        mock_bq_client.load_table_from_file.return_value = mock_job
        mock_bq_client.get_table.return_value = mock_table
        
        self.loader.bq_client = mock_bq_client
        
        # Test the upload
        self.loader.upload_to_bigquery(test_file)
        
        # Verify the upload was called with correct parameters
        mock_bq_client.load_table_from_file.assert_called_once()
        mock_bq_client.get_table.assert_called_once()
        
        # Clean up
        os.remove(test_file)
    
    @patch('google.cloud.bigquery.Client')
    def test_fetch_data(self, mock_bq_client):
        """Test data fetching from BigQuery."""
        # Mock query result
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame({
            'customer_id': [1, 2],
            'churn': [0, 1]
        })
        
        mock_bq_client.query.return_value = mock_query_job
        self.loader.bq_client = mock_bq_client
        
        # Test data fetching
        result = self.loader.fetch_data()
        
        # Verify the result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue('customer_id' in result.columns)
        self.assertTrue('churn' in result.columns)

if __name__ == '__main__':
    unittest.main()
