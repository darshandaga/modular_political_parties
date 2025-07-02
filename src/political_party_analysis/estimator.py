import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names
        self.gmm_model = None
        self.n_components = 3  # Number of Gaussian components

    def fit_gaussian_mixture(self):
        """Fit a Gaussian Mixture Model to estimate the density of the data"""
        self.gmm_model = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42,
            max_iter=100
        )
        self.gmm_model.fit(self.data)
        return self.gmm_model

    def sample_from_distribution(self, n_samples=10):
        """Sample n_samples from the fitted Gaussian Mixture Model"""
        if self.gmm_model is None:
            raise ValueError("Model not fitted. Call fit_gaussian_mixture() first.")
        
        samples, labels = self.gmm_model.sample(n_samples)
        
        # Create DataFrame with same column names as original data
        sampled_df = pd.DataFrame(samples, columns=self.data.columns)
        
        return sampled_df, labels

    def map_to_high_dimension(self, low_dim_samples):
        """Map low-dimensional samples back to high-dimensional space using inverse transform"""
        if self.dim_reducer_model is None:
            raise ValueError("Dimensionality reducer model not available.")
        
        # For PCA, we can use inverse_transform
        if hasattr(self.dim_reducer_model, 'inverse_transform'):
            high_dim_samples = self.dim_reducer_model.inverse_transform(low_dim_samples)
            
            # Create DataFrame with original feature names
            high_dim_df = pd.DataFrame(high_dim_samples, columns=self.feature_names)
            return high_dim_df
        else:
            print("Warning: Inverse transform not available for this dimensionality reduction method.")
            return None

    def get_cluster_assignments(self):
        """Get cluster assignments for all data points"""
        if self.gmm_model is None:
            raise ValueError("Model not fitted. Call fit_gaussian_mixture() first.")
        
        return self.gmm_model.predict(self.data)

    def get_means_and_covariances(self):
        """Get the means and covariances of the fitted Gaussian components"""
        if self.gmm_model is None:
            raise ValueError("Model not fitted. Call fit_gaussian_mixture() first.")
        
        return self.gmm_model.means_, self.gmm_model.covariances_
