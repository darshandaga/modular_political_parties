import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, method: str, data: pd.DataFrame, n_components: int = 2):
        self.method = method.upper()
        self.n_components = n_components
        self.data = data
        self.model = None
        self.explained_variance_ratio_ = None
        
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        # Only use numeric columns for dimensionality reduction
        self.numeric_data = data.select_dtypes(include=[np.number])
        
        if self.numeric_data.empty:
            raise ValueError("No numeric columns found in the data")

    def transform(self) -> pd.DataFrame:
        """Transform the high-dimensional data to lower dimensions"""
        
        if self.method == "PCA":
            return self._apply_pca()
        elif self.method == "TSNE":
            return self._apply_tsne()
        else:
            # Default to PCA if method not recognized
            print(f"Warning: Method '{self.method}' not recognized. Using PCA instead.")
            return self._apply_pca()
    
    def _apply_pca(self) -> pd.DataFrame:
        """Apply Principal Component Analysis"""
        self.model = PCA(n_components=self.n_components, random_state=42)
        
        # Fit and transform the data
        transformed_data = self.model.fit_transform(self.numeric_data)
        
        # Store explained variance ratio
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_
        
        # Create DataFrame with appropriate column names
        column_names = [f"PC{i+1}" for i in range(self.n_components)]
        
        return pd.DataFrame(
            transformed_data,
            index=self.data.index,
            columns=column_names
        )
    
    def _apply_tsne(self) -> pd.DataFrame:
        """Apply t-SNE (t-Distributed Stochastic Neighbor Embedding)"""
        # t-SNE parameters optimized for this type of data
        self.model = TSNE(
            n_components=self.n_components,
            random_state=42,
            perplexity=min(30, len(self.numeric_data) - 1),  # Adjust perplexity based on data size
            n_iter=1000
        )
        
        # Fit and transform the data
        transformed_data = self.model.fit_transform(self.numeric_data)
        
        # Create DataFrame with appropriate column names
        column_names = [f"tSNE{i+1}" for i in range(self.n_components)]
        
        return pd.DataFrame(
            transformed_data,
            index=self.data.index,
            columns=column_names
        )
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (only available for PCA)"""
        return self.explained_variance_ratio_
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance/loadings (only available for PCA)"""
        if self.model is None or self.method != "PCA":
            return None
            
        # Get the components (loadings)
        components = self.model.components_
        
        # Create DataFrame with feature names and component loadings
        feature_importance = pd.DataFrame(
            components.T,
            index=self.numeric_data.columns,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        
        return feature_importance
