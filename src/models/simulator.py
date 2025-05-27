import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

class ModelSimulator:
    """
    A class for simulating statistical and machine learning models.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the simulator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_linear_data(self, n_samples: int = 1000, n_features: int = 5, 
                            noise: float = 0.1, coefficients: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from a linear model: y = X * beta + noise
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            noise: Standard deviation of Gaussian noise
            coefficients: True coefficients to use. If None, random coefficients are generated.
            
        Returns:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
        """
        # Generate random feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Generate or use provided coefficients
        if coefficients is None:
            beta = np.random.randn(n_features)
        else:
            beta = coefficients
            
        # Generate target with noise
        y = X.dot(beta) + noise * np.random.randn(n_samples)
        
        return X, y
    
    def generate_time_series(self, n_points: int = 1000, trend: float = 0.01, 
                            seasonality_amplitude: float = 2.0, seasonality_period: int = 365,
                            noise: float = 0.5) -> pd.Series:
        """
        Generate a time series with trend, seasonality, and noise.
        
        Args:
            n_points: Number of time points to generate
            trend: Slope of the linear trend
            seasonality_amplitude: Amplitude of the seasonal component
            seasonality_period: Period of the seasonal component
            noise: Standard deviation of the noise component
            
        Returns:
            Time series as a pandas Series
        """
        # Generate time index
        time_idx = pd.date_range(start='2020-01-01', periods=n_points, freq='D')
        
        # Generate components
        trend_component = np.arange(n_points) * trend
        seasonality_component = seasonality_amplitude * np.sin(2 * np.pi * np.arange(n_points) / seasonality_period)
        noise_component = noise * np.random.randn(n_points)
        
        # Combine components
        series = trend_component + seasonality_component + noise_component
        
        return pd.Series(series, index=time_idx, name='value')