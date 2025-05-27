import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame containing the data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def plot_distribution(data: pd.Series, title: str = None, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot the distribution of a variable.
    
    Args:
        data: Series containing the data to plot
        title: Plot title
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    plt.hist(data.dropna(), bins=30, alpha=0.7)
    plt.xlabel(data.name if data.name else 'Value')
    plt.ylabel('Frequency')
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def describe_with_percentiles(data: pd.DataFrame, percentiles: List[float] = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) -> pd.DataFrame:
    """
    Extend pandas describe with custom percentiles.
    
    Args:
        data: DataFrame to describe
        percentiles: List of percentiles to include
        
    Returns:
        DataFrame with descriptive statistics
    """
    return data.describe(percentiles=percentiles)