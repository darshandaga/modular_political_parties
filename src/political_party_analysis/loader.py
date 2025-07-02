from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        return df.drop_duplicates()

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        # Drop non-feature columns if they exist in the dataframe
        cols_to_drop = [col for col in non_features if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Remove columns that are all NaN
        df = df.dropna(axis=1, how='all')
        
        # Handle index parameter - can be string or list
        if index is not None:
            if isinstance(index, str):
                index_cols = [index] if index in df.columns else []
            else:
                index_cols = [col for col in index if col in df.columns]
            
            if index_cols:
                df = df.set_index(index_cols)
        
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        # Fill NaN values with the median for numeric columns and mode for categorical columns
        df_filled = df.copy()
        
        # Fill numeric columns with median
        numeric_columns = df_filled.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        # Fill categorical columns with mode (most frequent value)
        categorical_columns = df_filled.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            mode_value = df_filled[col].mode()
            if not mode_value.empty:
                df_filled[col] = df_filled[col].fillna(mode_value[0])
        
        return df_filled

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        # Only scale numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            return df_scaled
        else:
            return df

    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        # Apply all preprocessing steps in sequence
        df = self.party_data.copy()
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Remove non-feature columns and set index
        df = self.remove_nonfeature_cols(df, self.non_features, self.index)
        
        # Step 3: Handle NaN values
        df = self.handle_NaN_values(df)
        
        # Step 4: Scale features
        df = self.scale_features(df)
        
        # Update the party_data with preprocessed data
        self.party_data = df
        
        return df

    def __str__(self):
        """Return string representation showing the DataFrame"""
        return f"DataLoader with party_data:\n{self.party_data}"
