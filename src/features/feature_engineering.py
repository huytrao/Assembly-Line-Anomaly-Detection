"""
the Feature Engineering class for Home Credit Default Risk dataset.
This class includes methods to create new features, aggregate features from
different datasets, and preprocess the data for modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Feature Engineering class for Home Credit Default Risk dataset."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
    
    import pandas as pd
import numpy as np

class ManufacturingFeatureEngineer:
    """Feature Engineering class for Manufacturing dataset."""

    def __init__(self):
        pass

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the manufacturing data.
        Args:
            df: manufacturing data
        Returns:
            Engineered manufacturing data
        """
        df = df.copy()

        # ===== TIME FEATURES =====
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['HOUR'] = df['Timestamp'].dt.hour
        df['DAY_OF_WEEK'] = df['Timestamp'].dt.dayofweek
        df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)

        # ===== RATIOS / NORMALIZATION =====
        # Energy efficiency
        df['ENERGY_PER_UNIT'] = df['Power_Consumption_kW'] / (df['Production_Speed_units_per_hr'] + 1e-5)
        # Error relative to speed
        df['ERROR_PER_SPEED'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'] + 1e-5)
        # Defect relative to speed
        df['DEFECT_PER_SPEED'] = df['Quality_Control_Defect_Rate_%'] / (df['Production_Speed_units_per_hr'] + 1e-5)
        # Network quality index
        df['NETWORK_QUALITY'] = (100 - df['Packet_Loss_%']) / (df['Network_Latency_ms'] + 1)

        # ===== STATISTICAL FEATURES (rolling windows by Machine_ID) =====
        df = df.sort_values(['Machine_ID','Timestamp'])
        for col in ['Temperature_C','Vibration_Hz','Power_Consumption_kW']:
            df[f'{col}_ROLL_MEAN'] = df.groupby('Machine_ID')[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
            df[f'{col}_ROLL_STD'] = df.groupby('Machine_ID')[col].transform(lambda x: x.rolling(window=5, min_periods=1).std())

        # ===== CATEGORICAL ENCODING =====
        df = pd.get_dummies(df, columns=['Operation_Mode'], drop_first=True)

        return df