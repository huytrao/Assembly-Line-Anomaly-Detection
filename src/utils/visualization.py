"""
可视化工具模块
用于数据探索和结果可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataVisualizer:
    """DataVisualizer includes various methods for data visualization."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        initialize the visualizer
        
        Args:
            figsize: image size
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_target_distribution(self, df: pd.DataFrame, target_col: str = 'TARGET', binary: bool = False) -> None:
        """
        Draw the distribution of the target variable

        Args:
            df: DataFrame
            target_col: target column name
            binary: whether to collapse into Normal vs Default
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        target_counts = df[target_col].value_counts()
        axes[0].bar(target_counts.index, target_counts.values, color=self.colors[:len(target_counts)])
        axes[0].set_title('Target Variable Distribution')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Count')

        # Add percentage labels
        total = len(df)
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + total*0.01, f'{v}\n({v/total:.1%})',
                        ha='center', va='bottom')

        # --- Pie chart ---
        if binary:
            # Collapse into Normal vs Default (assume Default = Low or 0)
            binary_series = df[target_col].map(lambda x: 'Default' if x in ['Low', 0] else 'Normal')
            binary_counts = binary_series.value_counts()
            labels = binary_counts.index
            values = binary_counts.values
            colors = self.colors[:2]
        else:
            # Use all classes
            labels = target_counts.index
            values = target_counts.values
            colors = self.colors[:len(target_counts)]

        axes[1].pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
        axes[1].set_title('Target Variable Proportion')

        plt.tight_layout()
        plt.show()

    
    def plot_numerical_features(self, df: pd.DataFrame, features: List[str], 
                               target_col: str = 'TARGET', ncols: int = 3) -> None:
        """
        Draw the distribution of numerical features

        Args:
            df: DataFrame
            features: List of feature names
            target_col: target column name
            ncols: number of columns per row
        """
        nrows = (len(features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // ncols
            col = i % ncols
            
            if i < len(features):
                # Draw histograms for different target values
                for target in df[target_col].unique():
                    subset = df[df[target_col] == target][feature].dropna()
                    axes[row, col].hist(subset, alpha=0.7, bins=30, 
                                       label=f'TARGET={target}', density=True)

                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Density')
                axes[row, col].legend()
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_features(self, df: pd.DataFrame, features: List[str], 
                                 target_col: str = 'TARGET', ncols: int = 2) -> None:
        """
        Draw the distribution of categorical features

        Args:
            df: DataFrame
            features: List of feature names
            target_col: target column name
            ncols: number of columns per row
        """
        nrows = (len(features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows))
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row = i // ncols
            col = i % ncols
            
            if i < len(features):
                # Cross table
                ct = pd.crosstab(df[feature], df[target_col], normalize='index')
                ct.plot(kind='bar', ax=axes[row, col], color=self.colors[:2])
                axes[row, col].set_title(f'{feature} vs TARGET')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Proportion')
                axes[row, col].legend(['Normal', 'Default'])
                axes[row, col].tick_params(axis='x', rotation=45)
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str] = None, 
                               figsize: Tuple[int, int] = None) -> None:
        """
        Draw the correlation matrix

        Args:
            df: DataFrame
            features: List of feature names
            figsize: Figure size
        """
        if features is None:
            # Select numerical features
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if figsize is None:
            figsize = (max(10, len(features) * 0.8), max(8, len(features) * 0.6))

        # Compute the correlation matrix
        corr_matrix = df[features].corr()

        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Draw feature importance

        Args:
            importance_df: DataFrame containing feature importance
            top_n: number of top features to display
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color=self.colors[0])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Important Features')
        plt.gca().invert_yaxis()

        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float], 
                           title: str = 'Learning Curves') -> None:
        """
        Draw learning curves

        Args:
            train_scores: training scores
            val_scores: validation scores
            title: plot title
        """
        plt.figure(figsize=self.figsize)
        epochs = range(1, len(train_scores) + 1)

        plt.plot(epochs, train_scores, 'o-', label='Training Scores', color=self.colors[0])
        plt.plot(epochs, val_scores, 'o-', label='Validation Scores', color=self.colors[1])

        plt.xlabel('Epochs')
        plt.ylabel('Scores')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_distribution(self, predictions: np.ndarray, 
                                   true_labels: np.ndarray = None) -> None:
        """
        Draw the distribution of prediction results

        Args:
            predictions: prediction results
            true_labels: true labels
        """
        plt.figure(figsize=self.figsize)
        
        if true_labels is not None:
            
            for label in np.unique(true_labels):
                mask = true_labels == label
                plt.hist(predictions[mask], alpha=0.7, bins=50, 
                        label=f'True Label {label}', density=True)
        else:
            plt.hist(predictions, alpha=0.7, bins=50, density=True)
        
        plt.xlabel(' Predicted Values')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        if true_labels is not None:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_names: List[str] = None) -> None:
        """
        Draw ROC curves for multiple models
        
        Args:
            y_true: the true labels
            y_pred: predicted probabilities (can be the results of multiple models)
            model_names: list of model names
        """
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=self.figsize)
        
        # check if y_pred is 1D or 2D
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(y_pred.shape[1])]

        for i in range(y_pred.shape[1]):
            fpr, tpr, _ = roc_curve(y_true, y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_names[i]} (AUC = {roc_auc:.3f})',
                    color=self.colors[i % len(self.colors)])
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_cv_scores(self, cv_results: Dict[str, List[float]]) -> None:
        """
        Draw box plots for cross-validation scores comparison

        Args:
            cv_results: cross-validation results dictionary
        """
        models = list(cv_results.keys())
        scores = [cv_results[model] for model in models]
        
        plt.figure(figsize=self.figsize)
        
        # 箱线图
        box_plot = plt.boxplot(scores, labels=models, patch_artist=True)
        
        
        for patch, color in zip(box_plot['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.ylabel('AUC Score')
        plt.title('Cross-Validation Score Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_interactive_feature_importance(self, importance_df: pd.DataFrame, 
                                           top_n: int = 30) -> None:
        """
        Create an interactive feature importance plot
        
        Args:
            importance_df: DataFrame containing feature importance scores
            top_n: Number of top features to display
        """
        top_features = importance_df.head(top_n)
        
        fig = px.bar(top_features, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title=f'Top {top_n} Important Features',
                    labels={'importance': 'Importance', 'feature': 'Feature'})
        
        fig.update_layout(height=max(400, top_n * 20), yaxis={'categoryorder': 'total ascending'})
        fig.show()
    
    def create_interactive_correlation_heatmap(self, df: pd.DataFrame, 
                                             features: List[str] = None) -> None:
        """
        Create an interactive correlation heatmap
        
        Args:
            df: DataFrame to analyze
            features: List of features to include
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[features].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="特征相关性热力图",
                       color_continuous_scale='RdBu_r')
        
        fig.update_layout(height=max(400, len(features) * 25),
                         width=max(400, len(features) * 25))
        fig.show()


def plot_missing_values(df: pd.DataFrame, threshold: float = 0.01) -> None:
    """
    Plot missing values in the dataframe that exceed a certain threshold.
    
    Args:
        df: DataFrame to analyze
        threshold: Proportion threshold for displaying missing values
    """
    # Calculate missing value proportions
    missing_percent = df.isnull().sum() / len(df)
    missing_percent = missing_percent[missing_percent > threshold].sort_values(ascending=False)
    
    if len(missing_percent) == 0:
        print("No missing values exceed the threshold")
        return
    
    plt.figure(figsize=(12, max(6, len(missing_percent) * 0.3)))
    bars = plt.barh(range(len(missing_percent)), missing_percent.values, 
                   color='lightcoral')
    plt.yticks(range(len(missing_percent)), missing_percent.index)
    plt.xlabel('Missing Value Proportion')
    plt.title(f'Missing Value Analysis (>{threshold:.1%})')
    plt.gca().invert_yaxis()

    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', ha='left', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_data_overview(df: pd.DataFrame) -> None:
    """
    Plot an overview of the dataset including data types, missing values, and basic statistics.
    
    Args:
        df:   pd.DataFrame, the dataset to analyze
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Data type distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Data Type Distribution')

    # Missing values
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].head(20)
    if len(missing_counts) > 0:
        axes[0, 1].bar(range(len(missing_counts)), missing_counts.values)
        axes[0, 1].set_xticks(range(len(missing_counts)))
        axes[0, 1].set_xticklabels(missing_counts.index, rotation=45, ha='right')
        axes[0, 1].set_title('Top 20 Columns with Most Missing Values')
        axes[0, 1].set_ylabel('Missing Value Count')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=16)
        axes[0, 1].set_title('Missing Value Analysis')

    # Numeric feature distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        axes[1, 0].hist(df[sample_col].dropna(), bins=30, alpha=0.7)
        axes[1, 0].set_title(f'Numeric Feature Distribution: {sample_col}')
        axes[1, 0].set_xlabel(sample_col)
        axes[1, 0].set_ylabel('Frequency')

    # Dataset overview
    info_text = f"""Dataset Shape: {df.shape}
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}
Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}
Total Missing Values: {df.isnull().sum().sum()}"""
    axes[1, 1].text(0.1, 0.5, info_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='center')
    axes[1, 1].set_title('Dataset Overview')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show() 