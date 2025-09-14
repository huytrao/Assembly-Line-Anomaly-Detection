"""
# the code below in order to load and preprocess the Home Credit Default Risk dataset.
# It includes functions to load each individual data file, reduce memory usage,
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


class IntelligentManufacturingDataLoader:
    """Intelligent Manufacturing Data Loader"""
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the raw data
        """
        self.data_path = data_path
        self.datasets = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data
        
        Returns:
            datasets: dictionary of all loaded datasets
        """
        print("Loading all data...")

        # Load the manufacturing data
        self.load_manufacturing_data()
        
        print(f"len of datasets : {len(self.datasets)} datasets loaded.")

        return self.datasets

    def load_manufacturing_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the main manufacturing data

        Returns:
            train_df, test_df: training and testing datasets
        """
        print("Loading manufacturing data...")

        try:
            df = pd.read_csv(os.path.join(self.data_path, "manufacturing_6G_dataset.csv"))
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            print(f" check train shape : {train_df.shape}")
            print(f" check test shape : {test_df.shape}")

            self.datasets['manufacturing_train'] = train_df
            self.datasets['manufacturing_test'] = test_df
            mapping = {"Low": 0, "Medium": 1, "High": 2}
            train_df["Efficiency_Status_Num"] = train_df["Efficiency_Status"].map(mapping)
            test_df["Efficiency_Status_Num"] = test_df["Efficiency_Status"].map(mapping)
            
            return train_df, test_df
            
        except FileNotFoundError as e:
            print(f"print FileNotFoundError: {e}")
            print("data/raw/application_train.csv or data/raw/manufacturing_6G_dataset.csv not found!")
            return None, None

    def get_data_info(self) -> None:
        """
        Display basic information about each dataset
        """
        if not self.datasets:
            print("No datasets loaded.")
            return
        
        print("\n=== Dataset Information ===")
        for name, df in self.datasets.items():
            if df is not None:
                print(f"\n{name}:")
                print(f"  Shape: {df.shape}")
                print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                print(f"  Missing Values: {df.isnull().sum().sum()}")
    


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame
    
    Args:
        df: Input DataFrame
        verbose: Whether to display detailed information

    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype

        # Skip object and string types
        if col_type == object or pd.api.types.is_string_dtype(df[col]):
            continue

        # Only process numeric types
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                c_min = df[col].min()
                c_max = df[col].max()

                # Check for NaN values
                if pd.isna(c_min) or pd.isna(c_max):
                    continue
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                elif 'float' in str(col_type):
                    # For float types, handle more conservatively
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            except Exception as e:
                if verbose:
                    print(f"Skipping column {col}: {e}")
                continue
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df