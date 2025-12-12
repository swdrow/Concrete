"""
Data Preparation Module
Handles feature engineering, scaling, and train/test splitting
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def engineer_features(df):
    """
    Create engineered features from the raw data

    Parameters:
    -----------
    df : pd.DataFrame
        The raw concrete dataset

    Returns:
    --------
    pd.DataFrame : Dataset with additional engineered features
    """
    df_eng = df.copy()

    if config.ENGINEER_FEATURES:
        if config.VERBOSE:
            print("Engineering features...")

        # Log transformation of age
        df_eng['log_age'] = np.log(df_eng['age'] + 1)

        # Water to cement ratio
        df_eng['water_cement_ratio'] = df_eng['water'] / (df_eng['cement'] + 1e-10)

        # SCM (Supplementary Cementitious Materials) fraction
        # Represents the proportion of binder that is slag + fly ash
        # This avoids collinearity with cement (unlike total binder content)
        total_binder = (df_eng['cement'] +
                       df_eng['blast_furnace_slag'] +
                       df_eng['fly_ash'])
        df_eng['scm_fraction'] = ((df_eng['blast_furnace_slag'] + df_eng['fly_ash']) /
                                  (total_binder + 1e-10))

        if config.VERBOSE:
            print(f"Added {len(config.ENGINEERED_FEATURES)} engineered features")
            print(f"  - log_age: log(age + 1)")
            print(f"  - water_cement_ratio: water / cement")
            print(f"  - scm_fraction: (slag + fly_ash) / total_binder")
            print(f"New shape: {df_eng.shape}")

    return df_eng


def split_data(df, test_size=None, random_seed=None):
    """
    Split data into train and test sets

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    test_size : float, optional
        Proportion for test set (default from config)
    random_seed : int, optional
        Random seed (default from config)

    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    # Get all feature columns (original + engineered)
    feature_cols = [col for col in df.columns if col != config.TARGET_COLUMN]

    X = df[feature_cols]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed
    )

    if config.VERBOSE:
        print(f"\nData split:")
        print(f"  Train set: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
        print(f"  Test set:  {len(X_test)} samples ({100*test_size:.0f}%)")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, save_scaler=True):
    """
    Standardize features using StandardScaler

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    save_scaler : bool
        Whether to save the fitted scaler

    Returns:
    --------
    tuple : (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    if config.VERBOSE:
        print("\nFeatures standardized (mean=0, std=1)")

    # Save scaler
    if save_scaler and config.SAVE_INTERMEDIATE:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        scaler_file = config.PROCESSED_DATA_DIR / "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        if config.VERBOSE:
            print(f"Saved scaler to: {scaler_file}")

    return X_train_scaled, X_test_scaled, scaler


def create_cv_folds(X, y, n_folds=None, random_seed=None):
    """
    Create cross-validation folds

    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_folds : int, optional
        Number of folds (default from config)
    random_seed : int, optional
        Random seed (default from config)

    Returns:
    --------
    list : List of (train_idx, val_idx) tuples
    """
    if n_folds is None:
        n_folds = config.N_FOLDS
    if random_seed is None:
        random_seed = config.RANDOM_SEED

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds = list(kfold.split(X))

    if config.VERBOSE:
        print(f"\nCreated {n_folds}-fold cross-validation splits")

    return folds


def plot_before_after_scaling(X_before, X_after, save=True):
    """
    Visualize distributions before and after scaling

    Parameters:
    -----------
    X_before : pd.DataFrame
        Features before scaling
    X_after : pd.DataFrame
        Features after scaling
    save : bool
        Whether to save the figure
    """
    # Select a subset of features to plot
    n_features_to_plot = min(6, len(X_before.columns))
    features = X_before.columns[:n_features_to_plot]

    fig, axes = plt.subplots(2, n_features_to_plot, figsize=(20, 8))

    for idx, feature in enumerate(features):
        # Before scaling
        axes[0, idx].hist(X_before[feature], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, idx].set_title(f'{feature}\n(Before)', fontsize=10)
        axes[0, idx].set_ylabel('Frequency' if idx == 0 else '', fontsize=9)
        axes[0, idx].grid(True, alpha=0.3)

        # After scaling
        axes[1, idx].hist(X_after[feature], bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1, idx].set_title(f'{feature}\n(After)', fontsize=10)
        axes[1, idx].set_ylabel('Frequency' if idx == 0 else '', fontsize=9)
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('Feature Distributions Before and After Standardization',
                 fontsize=14, y=1.00)
    plt.tight_layout()

    if save:
        config.FIGURES_PREP_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_PREP_DIR / "fig_before_after_scaling.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_engineered_features(df, save=True):
    """
    Visualize engineered features and their relationship with target

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with engineered features
    save : bool
        Whether to save the figure
    """
    if not config.ENGINEER_FEATURES:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Log age
    axes[0].scatter(df['log_age'], df[config.TARGET_COLUMN], alpha=0.5, s=20)
    axes[0].set_xlabel('Log(Age + 1)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[0].set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[0].set_title('Log Age vs Strength', fontsize=config.PLOT_CONFIG['title_size'])
    axes[0].grid(True, alpha=0.3)
    corr = df['log_age'].corr(df[config.TARGET_COLUMN])
    axes[0].text(0.05, 0.95, f'r = {corr:.3f}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Water/Cement ratio
    axes[1].scatter(df['water_cement_ratio'], df[config.TARGET_COLUMN], alpha=0.5, s=20)
    axes[1].set_xlabel('Water/Cement Ratio', fontsize=config.PLOT_CONFIG['label_size'])
    axes[1].set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[1].set_title('Water/Cement Ratio vs Strength', fontsize=config.PLOT_CONFIG['title_size'])
    axes[1].grid(True, alpha=0.3)
    corr = df['water_cement_ratio'].corr(df[config.TARGET_COLUMN])
    axes[1].text(0.05, 0.95, f'r = {corr:.3f}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # SCM fraction (replaces binder content to avoid collinearity)
    axes[2].scatter(df['scm_fraction'], df[config.TARGET_COLUMN], alpha=0.5, s=20)
    axes[2].set_xlabel('SCM Fraction', fontsize=config.PLOT_CONFIG['label_size'])
    axes[2].set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[2].set_title('SCM Fraction vs Strength\n(slag + fly_ash) / total_binder',
                      fontsize=config.PLOT_CONFIG['title_size'])
    axes[2].grid(True, alpha=0.3)
    corr = df['scm_fraction'].corr(df[config.TARGET_COLUMN])
    axes[2].text(0.05, 0.95, f'r = {corr:.3f}',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save:
        config.FIGURES_PREP_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_PREP_DIR / "fig_engineered_features.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def prepare_data(df, save_figures=True):
    """
    Complete data preparation pipeline

    Parameters:
    -----------
    df : pd.DataFrame
        Raw concrete dataset
    save_figures : bool
        Whether to save visualization figures

    Returns:
    --------
    dict : Dictionary containing prepared data and objects
        Keys: 'X_train', 'X_test', 'y_train', 'y_test',
              'X_train_scaled', 'X_test_scaled', 'scaler', 'cv_folds'
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80 + "\n")

    # 1. Engineer features
    df_eng = engineer_features(df)

    # Plot engineered features
    if save_figures:
        plot_engineered_features(df_eng, save=True)

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(df_eng)

    # 3. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Plot before/after scaling
    if save_figures:
        plot_before_after_scaling(X_train, X_train_scaled, save=True)

    # 4. Create CV folds
    cv_folds = create_cv_folds(X_train_scaled, y_train)

    # Save processed data
    if config.SAVE_INTERMEDIATE:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save splits
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(config.PROCESSED_DATA_DIR / "train_data.csv", index=False)
        test_data.to_csv(config.PROCESSED_DATA_DIR / "test_data.csv", index=False)

        # Save scaled data
        train_scaled = pd.concat([X_train_scaled, y_train], axis=1)
        test_scaled = pd.concat([X_test_scaled, y_test], axis=1)
        train_scaled.to_csv(config.PROCESSED_DATA_DIR / "train_scaled.csv", index=False)
        test_scaled.to_csv(config.PROCESSED_DATA_DIR / "test_scaled.csv", index=False)

        if config.VERBOSE:
            print(f"\nSaved processed data to: {config.PROCESSED_DATA_DIR}")

    if config.VERBOSE:
        print("\n" + "="*80)
        print("DATA PREPARATION COMPLETE")
        print("="*80 + "\n")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'cv_folds': cv_folds
    }


if __name__ == "__main__":
    # Test the module
    from data_ingestion import ingest_data
    df = ingest_data(verbose=False)
    prepared_data = prepare_data(df, save_figures=True)
    print(f"Prepared data with {len(prepared_data['X_train_scaled'].columns)} features")
