# Bank Customer Churn Prediction

A production-ready machine learning project to predict customer churn in a banking context, comparing multiple algorithms including **TabM** (Tabular Model), **XGBoost**, **Random Forest**, **Logistic Regression**, **Naive Bayes**, and **AdaBoost**.

## 🚀 Quick Start

### Demo Mode (No BigQuery/Kaggle Required)
```bash
git clone <repository-url>
cd ABCBankChurnRate
pip install -r requirements.txt
python demo.py
```

### Production Mode (With Real Data)
```bash
# Set environment variables
export KAGGLE_JSON='{"username":"your_username","key":"your_key"}'
export GCP_SA_KEY='{"type":"service_account",...}'

# Run full pipeline
python main.py --download-new
```

## 📊 Key Results (Demo)

| Model | F1-Score | ROC-AUC | Training Time |
|-------|----------|---------|---------------|
| **XGBoost** | **0.749** | 0.623 | 0.07s |
| LogisticRegression | 0.748 | 0.661 | 0.004s |
| RandomForest | 0.736 | 0.642 | 0.20s |

## 🏗️ Project Structure

```
ABCBankChurnRate/
├── 📁 config/              # Configuration management
│   └── config.py           # Project settings
├── 📁 notebooks/           # Jupyter notebooks for exploration
│   ├── ExploratoryDA.ipynb
│   └── Modelling.ipynb
├── 📁 src/                 # Source code
│   ├── 📁 data/           # Data handling
│   │   ├── data_loader.py  # Kaggle & BigQuery integration
│   │   └── preprocessor.py # Feature engineering
│   ├── 📁 models/         # ML model implementations
│   │   ├── base_model.py   # Abstract base class
│   │   ├── logistic_regression.py
│   │   ├── naive_bayes.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── adaboost.py
│   │   ├── tabm.py         # TabM implementation
│   │   └── trainer.py      # Training pipeline
│   └── 📁 utils/          # Utilities
│       └── reporting.py    # Stakeholder reports
├── 📁 tests/              # Unit tests
├── demo.py                # Quick demo script
├── main.py                # Production pipeline
└── requirements.txt       # Dependencies
```

## 🔧 Features

### Data Pipeline
- ✅ **Kaggle Integration**: Automatic dataset download
- ✅ **BigQuery Storage**: Scalable cloud data warehouse
- ✅ **Data Validation**: Comprehensive duplicate and missing value checks
- ✅ **Feature Engineering**: Automated preprocessing pipeline

### Machine Learning
- ✅ **6 ML Algorithms**: Including advanced TabM model
- ✅ **Hyperparameter Tuning**: Grid search with cross-validation
- ✅ **Overfitting Prevention**: Stratified splits and validation
- ✅ **Model Persistence**: Automatic pickle saving/loading

### Production Ready
- ✅ **Comprehensive Testing**: Unit and integration tests
- ✅ **Stakeholder Reports**: Executive summaries and technical reports
- ✅ **Visualization**: Performance comparison plots
- ✅ **Error Handling**: Robust error management
- ✅ **Documentation**: Complete API documentation

## 📈 Model Comparison

### Traditional ML Models
- **Logistic Regression**: Fast, interpretable baseline
- **Naive Bayes**: Probabilistic approach
- **Random Forest**: Ensemble method with feature importance
- **AdaBoost**: Adaptive boosting classifier
- **XGBoost**: Gradient boosting with advanced optimization

### Advanced Models
- **TabM**: Neural network optimized for tabular data
  - Multi-layer perceptron with proper scaling
  - Adaptive learning rates
  - Early stopping to prevent overfitting

## 🎯 Business Impact

### Metrics Explained
- **F1-Score**: Balanced measure of precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes
- **Precision**: % of predicted churners who actually churn
- **Recall**: % of actual churners correctly identified

### Stakeholder Deliverables
1. **Executive Summary** (`demo_results/executive_summary.md`)
2. **Technical Report** (`demo_results/detailed_technical_report.csv`)
3. **Visualization Package** (PNG files for presentations)
4. **Feature Importance Analysis** (Key churn drivers)

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## ⚙️ Configuration

Edit `config/config.py` to customize:
- Feature engineering strategies
- Model hyperparameters
- Data processing settings
- File paths and BigQuery settings

## 📋 Dependencies

Core packages:
- `scikit-learn`: Traditional ML algorithms
- `xgboost`: Gradient boosting
- `pandas`: Data manipulation
- `google-cloud-bigquery`: Cloud data warehouse
- `kaggle`: Data source API

## 🔄 Workflow

### Data Flow
1. **Data Acquisition**: Kaggle → Local → BigQuery
2. **Validation**: Duplicates, missing values, outliers
3. **Preprocessing**: Scaling, encoding, feature engineering
4. **Training**: 6 models with hyperparameter tuning
5. **Evaluation**: Cross-validation and test set metrics
6. **Reporting**: Automated stakeholder deliverables

### Model Training
1. **Data Split**: 60% train, 20% validation, 20% test
2. **Hyperparameter Tuning**: Grid search with 5-fold CV
3. **Model Training**: Best parameters on full training set
4. **Evaluation**: Performance on held-out test set
5. **Persistence**: Save models for future use

## 🎁 Quick Results

After running `python demo.py`, check the `demo_results/` folder for:
- 📄 Executive summary for business stakeholders
- 📊 Performance comparison charts
- 📈 Feature importance analysis
- 📋 Technical metrics report

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `python -m pytest tests/`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for production ML workflows**
