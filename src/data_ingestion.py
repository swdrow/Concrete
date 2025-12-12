"""
Data Ingestion Module
Handles downloading and loading the UCI Concrete Compressive Strength dataset
"""
import pandas as pd
import requests
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def download_data(force_download=False):
    """
    Download the concrete dataset from UCI repository

    Parameters:
    -----------
    force_download : bool
        If True, download even if file exists

    Returns:
    --------
    Path to downloaded file
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_DATA_FILE.exists() and not force_download:
        if config.VERBOSE:
            print(f"Data file already exists: {config.RAW_DATA_FILE}")
        return config.RAW_DATA_FILE

    if config.VERBOSE:
        print(f"Downloading data from {config.DATASET_URL}")

    try:
        response = requests.get(config.DATASET_URL, timeout=30)
        response.raise_for_status()

        with open(config.RAW_DATA_FILE, 'wb') as f:
            f.write(response.content)

        if config.VERBOSE:
            print(f"Data downloaded successfully to {config.RAW_DATA_FILE}")

        return config.RAW_DATA_FILE

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise


def load_data():
    """
    Load the concrete dataset from Excel file

    Returns:
    --------
    pd.DataFrame : Loaded dataset with proper column names
    """
    if not config.RAW_DATA_FILE.exists():
        if config.VERBOSE:
            print("Data file not found. Downloading...")
        download_data()

    if config.VERBOSE:
        print(f"Loading data from {config.RAW_DATA_FILE}")

    try:
        # Read Excel file
        df = pd.read_excel(config.RAW_DATA_FILE)

        # Rename columns to standard names
        df.columns = config.COLUMN_NAMES

        if config.VERBOSE:
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def get_data_summary(df):
    """
    Get summary statistics of the dataset

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset

    Returns:
    --------
    dict : Summary statistics and information
    """
    summary = {
        'n_samples': len(df),
        'n_features': len(config.FEATURE_COLUMNS),
        'target': config.TARGET_COLUMN,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'statistics': df.describe().to_dict()
    }

    return summary


def print_data_info(df):
    """
    Print detailed information about the dataset

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    """
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)

    print(f"\nShape: {df.shape[0]} samples × {df.shape[1]} features")

    print("\nColumn Names and Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(missing[missing > 0])

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nTarget Variable (Compressive Strength):")
    print(f"  Min:  {df[config.TARGET_COLUMN].min():.2f} MPa")
    print(f"  Max:  {df[config.TARGET_COLUMN].max():.2f} MPa")
    print(f"  Mean: {df[config.TARGET_COLUMN].mean():.2f} MPa")
    print(f"  Std:  {df[config.TARGET_COLUMN].std():.2f} MPa")

    print("\nFirst 5 rows:")
    print(df.head())

    print("="*80 + "\n")


def ingest_data(verbose=None):
    """
    Main function to ingest data
    Combines downloading, loading, and initial inspection

    Parameters:
    -----------
    verbose : bool, optional
        Override config.VERBOSE setting

    Returns:
    --------
    pd.DataFrame : The loaded concrete dataset
    """
    if verbose is not None:
        original_verbose = config.VERBOSE
        config.VERBOSE = verbose

    try:
        # Download if needed
        download_data()

        # Load data
        df = load_data()

        # Print information
        if config.VERBOSE:
            print_data_info(df)

        return df

    finally:
        if verbose is not None:
            config.VERBOSE = original_verbose


if __name__ == "__main__":
    # Test the module
    df = ingest_data()
    print(f"\nSuccessfully loaded {len(df)} samples")
