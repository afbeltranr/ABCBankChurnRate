"""Visualization utilities for EDA and model results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent, professional plots
plt.style.use('default')
sns.set_palette("husl")

class ChurnVisualizer:
    """Create visualizations for churn analysis and model results."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize the visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib for high-quality plots
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_eda_dashboard(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create comprehensive EDA dashboard."""
        print("📊 Creating EDA visualizations...")
        
        plot_paths = {}
        
        # 1. Churn Distribution
        plot_paths['churn_distribution'] = self._plot_churn_distribution(df)
        
        # 2. Demographic Analysis
        plot_paths['demographic_analysis'] = self._plot_demographic_analysis(df)
        
        # 3. Financial Features Analysis
        plot_paths['financial_analysis'] = self._plot_financial_analysis(df)
        
        # 4. Correlation Heatmap
        plot_paths['correlation_heatmap'] = self._plot_correlation_heatmap(df)
        
        # 5. Feature Distributions by Churn
        plot_paths['feature_distributions'] = self._plot_feature_distributions(df)
        
        return plot_paths
    
    def create_model_results_dashboard(self, results: Dict[str, Any], 
                                     comparison_df: pd.DataFrame) -> Dict[str, str]:
        """Create model results visualization dashboard."""
        print("🎯 Creating model results visualizations...")
        
        plot_paths = {}
        
        # 1. Model Performance Comparison
        plot_paths['model_comparison'] = self._plot_model_comparison(comparison_df)
        
        # 2. Performance Metrics Radar Chart
        plot_paths['performance_radar'] = self._plot_performance_radar(comparison_df)
        
        # 3. Training Time vs Performance
        plot_paths['time_vs_performance'] = self._plot_time_vs_performance(comparison_df)
        
        # 4. ROC Curves Comparison
        if results:
            plot_paths['roc_curves'] = self._plot_roc_curves(results)
        
        return plot_paths
    
    def create_hyperparameter_analysis(self, tuning_results: Dict[str, Any]) -> Dict[str, str]:
        """Create hyperparameter tuning analysis visualizations."""
        print("🔧 Creating hyperparameter analysis visualizations...")
        
        plot_paths = {}
        
        # Create hyperparameter importance analysis
        plot_paths['hyperparameter_analysis'] = self._plot_hyperparameter_analysis(tuning_results)
        
        return plot_paths
    
    def _plot_churn_distribution(self, df: pd.DataFrame) -> str:
        """Plot churn distribution with insights."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Churn count
        churn_counts = df['churn'].value_counts()
        colors = ['#2E86AB', '#A23B72']
        ax1.pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        
        # Churn by country
        churn_by_country = df.groupby('country')['churn'].agg(['count', 'mean']).round(3)
        x_pos = np.arange(len(churn_by_country))
        bars = ax2.bar(x_pos, churn_by_country['mean'] * 100, color=['#F18F01', '#C73E1D', '#2E86AB'])
        ax2.set_xlabel('Country')
        ax2.set_ylabel('Churn Rate (%)')
        ax2.set_title('Churn Rate by Country', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(churn_by_country.index)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.output_dir / "churn_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_demographic_analysis(self, df: pd.DataFrame) -> str:
        """Plot demographic analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age distribution by churn
        sns.histplot(data=df, x='age', hue='churn', kde=True, ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by Churn Status', fontweight='bold')
        axes[0,0].legend(['Retained', 'Churned'])
        
        # Gender vs Churn
        gender_churn = pd.crosstab(df['gender'], df['churn'], normalize='index') * 100
        gender_churn.plot(kind='bar', ax=axes[0,1], color=['#2E86AB', '#A23B72'])
        axes[0,1].set_title('Churn Rate by Gender', fontweight='bold')
        axes[0,1].set_ylabel('Percentage (%)')
        axes[0,1].legend(['Retained', 'Churned'])
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # Tenure distribution
        sns.histplot(data=df, x='tenure', hue='churn', kde=True, ax=axes[1,0])
        axes[1,0].set_title('Tenure Distribution by Churn Status', fontweight='bold')
        axes[1,0].legend(['Retained', 'Churned'])
        
        # Products number vs Churn
        products_churn = pd.crosstab(df['products_number'], df['churn'], normalize='index') * 100
        products_churn.plot(kind='bar', ax=axes[1,1], color=['#2E86AB', '#A23B72'])
        axes[1,1].set_title('Churn Rate by Number of Products', fontweight='bold')
        axes[1,1].set_ylabel('Percentage (%)')
        axes[1,1].legend(['Retained', 'Churned'])
        axes[1,1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        save_path = self.output_dir / "demographic_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_financial_analysis(self, df: pd.DataFrame) -> str:
        """Plot financial features analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Credit Score vs Churn
        sns.boxplot(data=df, x='churn', y='credit_score', ax=axes[0,0])
        axes[0,0].set_title('Credit Score Distribution by Churn', fontweight='bold')
        axes[0,0].set_xticklabels(['Retained', 'Churned'])
        
        # Balance vs Churn
        # Remove zero balances for better visualization
        df_balance = df[df['balance'] > 0].copy()
        sns.boxplot(data=df_balance, x='churn', y='balance', ax=axes[0,1])
        axes[0,1].set_title('Account Balance Distribution by Churn\n(Non-zero balances)', fontweight='bold')
        axes[0,1].set_xticklabels(['Retained', 'Churned'])
        axes[0,1].ticklabel_format(style='plain', axis='y')
        
        # Estimated Salary vs Churn
        sns.boxplot(data=df, x='churn', y='estimated_salary', ax=axes[1,0])
        axes[1,0].set_title('Estimated Salary Distribution by Churn', fontweight='bold')
        axes[1,0].set_xticklabels(['Retained', 'Churned'])
        axes[1,0].ticklabel_format(style='plain', axis='y')
        
        # Balance categories
        df['balance_category'] = pd.cut(df['balance'], 
                                      bins=[0, 1, 50000, 100000, float('inf')],
                                      labels=['Zero', 'Low (1-50K)', 'Medium (50-100K)', 'High (>100K)'])
        balance_churn = pd.crosstab(df['balance_category'], df['churn'], normalize='index') * 100
        balance_churn.plot(kind='bar', ax=axes[1,1], color=['#2E86AB', '#A23B72'])
        axes[1,1].set_title('Churn Rate by Balance Category', fontweight='bold')
        axes[1,1].set_ylabel('Percentage (%)')
        axes[1,1].legend(['Retained', 'Churned'])
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.output_dir / "financial_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Plot correlation heatmap of numerical features."""
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        correlation_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        save_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_feature_distributions(self, df: pd.DataFrame) -> str:
        """Plot feature distributions by churn status."""
        # Select key features for visualization
        key_features = ['credit_score', 'age', 'balance', 'estimated_salary']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features):
            for churn_status in [0, 1]:
                data = df[df['churn'] == churn_status][feature]
                label = 'Churned' if churn_status == 1 else 'Retained'
                axes[i].hist(data, alpha=0.7, label=label, bins=30, density=True)
            
            axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution', fontweight='bold')
            axes[i].set_xlabel(feature.replace("_", " ").title())
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "feature_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> str:
        """Plot comprehensive model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1-Score comparison
        models = comparison_df['Model']
        f1_scores = comparison_df['F1-Score']
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars1 = axes[0,0].bar(models, f1_scores, color=colors)
        axes[0,0].set_title('F1-Score Comparison', fontweight='bold')
        axes[0,0].set_ylabel('F1-Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            axes[0,0].annotate(f'{height:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom')
        
        # ROC-AUC comparison
        roc_auc_scores = comparison_df['ROC-AUC']
        bars2 = axes[0,1].bar(models, roc_auc_scores, color=colors)
        axes[0,1].set_title('ROC-AUC Comparison', fontweight='bold')
        axes[0,1].set_ylabel('ROC-AUC')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            axes[0,1].annotate(f'{height:.3f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom')
        
        # Training Time comparison
        training_times = comparison_df['Training Time (s)']
        bars3 = axes[1,0].bar(models, training_times, color=colors)
        axes[1,0].set_title('Training Time Comparison', fontweight='bold')
        axes[1,0].set_ylabel('Training Time (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # Performance vs Speed scatter
        axes[1,1].scatter(training_times, f1_scores, c=range(len(models)), 
                         cmap='viridis', s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1,1].annotate(model, (training_times[i], f1_scores[i]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9)
        axes[1,1].set_xlabel('Training Time (seconds)')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].set_title('Performance vs Speed Trade-off', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_performance_radar(self, comparison_df: pd.DataFrame) -> str:
        """Create radar chart for model performance metrics."""
        from math import pi
        
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        models = comparison_df['Model'].tolist()
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Colors for different models
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Plot each model
        for i, model in enumerate(models):
            values = []
            for metric in metrics:
                values.append(comparison_df[comparison_df['Model'] == model][metric].iloc[0])
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        save_path = self.output_dir / "performance_radar.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_time_vs_performance(self, comparison_df: pd.DataFrame) -> str:
        """Plot training time vs performance analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        models = comparison_df['Model']
        training_times = comparison_df['Training Time (s)']
        f1_scores = comparison_df['F1-Score']
        roc_auc_scores = comparison_df['ROC-AUC']
        
        # F1-Score vs Training Time
        scatter1 = axes[0].scatter(training_times, f1_scores, c=range(len(models)), 
                                  cmap='viridis', s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[0].annotate(model, (training_times[i], f1_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_title('F1-Score vs Training Time', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # ROC-AUC vs Training Time
        scatter2 = axes[1].scatter(training_times, roc_auc_scores, c=range(len(models)), 
                                  cmap='plasma', s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1].annotate(model, (training_times[i], roc_auc_scores[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1].set_xlabel('Training Time (seconds)')
        axes[1].set_ylabel('ROC-AUC')
        axes[1].set_title('ROC-AUC vs Training Time', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        plt.tight_layout()
        save_path = self.output_dir / "time_vs_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_roc_curves(self, results: Dict[str, Any]) -> str:
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        for i, (model_name, result) in enumerate(results.items()):
            if 'roc_curve' in result:
                fpr, tpr, _ = result['roc_curve']
                auc_score = result['test_metrics']['roc_auc']
                plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        save_path = self.output_dir / "roc_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def _plot_hyperparameter_analysis(self, tuning_results: Dict[str, Any]) -> str:
        """Plot hyperparameter analysis."""
        n_models = len(tuning_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, data) in enumerate(tuning_results.items()):
            best_params = data['best_params']
            result = data['result']
            
            # Create a simple bar chart of parameter values (for categorical/discrete params)
            param_names = list(best_params.keys())
            param_values = []
            param_labels = []
            
            for param, value in best_params.items():
                if isinstance(value, (int, float)):
                    param_values.append(value)
                    param_labels.append(param)
                else:
                    # For categorical parameters, use index or simple encoding
                    param_labels.append(f"{param}\\n{value}")
                    param_values.append(1)  # Just for visualization
            
            if param_values:
                axes[i].bar(range(len(param_labels)), [1]*len(param_labels), 
                           color=plt.cm.viridis(np.linspace(0, 1, len(param_labels))))
                axes[i].set_xticks(range(len(param_labels)))
                axes[i].set_xticklabels(param_labels, rotation=45, ha='right')
                axes[i].set_title(f'{model_name}\\nF1: {result["test_metrics"]["f1_score"]:.3f}',
                                fontweight='bold')
                axes[i].set_ylabel('Selected')
        
        plt.suptitle('Best Hyperparameters by Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / "hyperparameter_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
