#!/bin/bash
# Script to run tests for the churn prediction project

echo "Running unit tests for the churn prediction project..."

# Run tests with pytest
python -m pytest tests/ -v --tb=short

# If pytest is not available, run with unittest
if [ $? -ne 0 ]; then
    echo "Pytest not found, running with unittest..."
    python -m unittest discover tests/ -v
fi

echo "Test run completed!"
