"""
Gaussian Process Regression Model
Uses scikit-learn for GP regression with RBF kernel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def build_gp_model(use_ard=None):
    """
    Build Gaussian Process Regression model

    Model specification:
        f ~ GP(0, k(x, x'))

        Kernel function with explicit noise term:
        k(x, x') = σ²_f × k_RBF(x, x') + σ²_n × δ(x, x')

        where:
        - k_RBF(x, x') = exp(-½ Σᵢ (xᵢ - x'ᵢ)² / ℓᵢ²)  [RBF/squared exponential]
        - σ²_f = signal variance (amplitude of function variations)
        - ℓᵢ = length scales (one per feature with ARD)
        - σ²_n = noise variance (observation noise)
        - δ(x, x') = Kronecker delta (1 if x=x', 0 otherwise)

        Observations:
        y = f(x) + ε, where ε ~ N(0, σ²_n)

    Note: In scikit-learn, the noise term σ²_n is handled via the 'alpha'
    parameter rather than an explicit WhiteKernel, for numerical stability.

    Parameters:
    -----------
    use_ard : bool, optional
        Whether to use Automatic Relevance Determination (different length scale per feature)

    Returns:
    --------
    GaussianProcessRegressor : Sklearn GP model
    """
    if use_ard is None:
        use_ard = config.GP_CONFIG['use_ard']

    # Determine length scale dimensions
    if use_ard:
        # ARD: one length scale per feature (will be learned)
        length_scale = 1.0
        length_scale_bounds = (1e-2, 1e2)
    else:
        # Isotropic: single length scale for all features
        length_scale = 1.0
        length_scale_bounds = (1e-2, 1e2)

    # Define kernel
    # Constant kernel for signal variance × RBF kernel + White kernel for noise
    kernel = (C(1.0, (1e-3, 1e3)) *
              RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds))

    # Build GP model
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=config.GP_CONFIG['n_restarts_optimizer'],
        alpha=config.GP_CONFIG['alpha'],
        random_state=config.GP_CONFIG['random_seed'],
        normalize_y=True  # Normalize target values
    )

    if config.VERBOSE:
        print(f"\nBuilding Gaussian Process Regression model...")
        print(f"  Kernel: {config.GP_CONFIG['kernel_type']}")
        print(f"  ARD (feature-specific length scales): {use_ard}")
        print(f"  Number of optimizer restarts: {config.GP_CONFIG['n_restarts_optimizer']}")

    return gp


def fit_gp_model(gp, X_train, y_train):
    """
    Fit Gaussian Process model

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        GP model
    X_train : pd.DataFrame or np.ndarray
        Training features (scaled)
    y_train : pd.Series or np.ndarray
        Training target

    Returns:
    --------
    GaussianProcessRegressor : Fitted GP model
    """
    # Convert to numpy arrays
    if isinstance(X_train, pd.DataFrame):
        X = X_train.values
        feature_names = X_train.columns.tolist()
    else:
        X = X_train
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    if isinstance(y_train, pd.Series):
        y = y_train.values
    else:
        y = y_train

    if config.VERBOSE:
        print(f"\nFitting Gaussian Process model...")
        print(f"  Number of training samples: {len(y)}")
        print(f"  Number of features: {X.shape[1]}")

    # Fit the model
    gp.fit(X, y)

    if config.VERBOSE:
        print("\nGP model fitted successfully!")
        print(f"\nOptimized kernel parameters:")
        print(f"  {gp.kernel_}")
        print(f"\nLog marginal likelihood: {gp.log_marginal_likelihood():.2f}")

    return gp, feature_names


def get_kernel_params(gp, feature_names):
    """
    Extract and interpret kernel hyperparameters

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        Fitted GP model
    feature_names : list
        Names of features

    Returns:
    --------
    dict : Kernel parameters
    """
    kernel = gp.kernel_

    # Extract length scales - try multiple access patterns
    length_scales = None
    for attr in ['k2', 'k1']:
        if hasattr(kernel, attr):
            subkernel = getattr(kernel, attr)
            if hasattr(subkernel, 'length_scale'):
                length_scales = np.atleast_1d(subkernel.length_scale)
                break

    # Extract signal variance
    signal_variance = None
    try:
        # Try different kernel structures
        if hasattr(kernel, 'k1') and hasattr(kernel.k1, 'constant_value'):
            signal_variance = kernel.k1.constant_value
        elif hasattr(kernel, 'constant_value'):
            signal_variance = kernel.constant_value
    except:
        signal_variance = 1.0  # Default

    params = {
        'signal_variance': signal_variance if signal_variance is not None else 1.0,
        'length_scales': length_scales,
        'feature_names': feature_names
    }

    if config.VERBOSE:
        print("\n" + "="*80)
        print("KERNEL HYPERPARAMETERS")
        print("="*80)
        print(f"\nSignal variance (σ²_f): {signal_variance:.4f}")

        if length_scales is not None and len(length_scales) > 1:
            print("\nLength scales (ARD - per feature):")
            for i, (name, ls) in enumerate(zip(feature_names, length_scales)):
                print(f"  {name}: {ls:.4f}")

            # Feature importance: inverse of length scale
            importance = 1.0 / length_scales
            importance_normalized = importance / importance.sum()
            print("\nFeature importance (normalized inverse length scales):")
            sorted_idx = np.argsort(importance_normalized)[::-1]
            for idx in sorted_idx:
                print(f"  {feature_names[idx]}: {importance_normalized[idx]:.4f}")
        else:
            print(f"\nLength scale (isotropic): {length_scales[0]:.4f}")

        print("="*80 + "\n")

    return params


def plot_kernel_params(kernel_params, save=True):
    """
    Visualize kernel hyperparameters

    Parameters:
    -----------
    kernel_params : dict
        Kernel parameters from get_kernel_params()
    save : bool
        Whether to save the figure
    """
    length_scales = kernel_params['length_scales']
    feature_names = kernel_params['feature_names']

    if length_scales is None or len(length_scales) == 1:
        if config.VERBOSE:
            print("Skipping kernel params plot (isotropic kernel)")
        return

    # Calculate feature importance
    importance = 1.0 / length_scales
    importance_normalized = importance / importance.sum()

    # Sort by importance
    sorted_idx = np.argsort(importance_normalized)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Length scales
    ax = axes[0]
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, length_scales[sorted_idx], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Length Scale', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('GP Kernel Length Scales\n(smaller = more sensitive)',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Feature importance
    ax = axes[1]
    ax.barh(y_pos, importance_normalized[sorted_idx], color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Normalized Importance', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('Feature Importance from GP\n(1 / length_scale, normalized)',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save:
        config.FIGURES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_MODELS_DIR / "fig_gp_kernel_params.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def predict_gp(gp, X_test, return_std=True, return_cov=False):
    """
    Make predictions with GP model

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        Fitted GP model
    X_test : pd.DataFrame or np.ndarray
        Test features
    return_std : bool
        Whether to return standard deviation
    return_cov : bool
        Whether to return full covariance matrix

    Returns:
    --------
    tuple : (y_mean, y_std) or (y_mean, y_cov) or just y_mean
    """
    if isinstance(X_test, pd.DataFrame):
        X = X_test.values
    else:
        X = X_test

    if return_std or return_cov:
        y_mean, y_std = gp.predict(X, return_std=True)
        if return_cov:
            y_cov = gp.predict(X, return_cov=True)[1]
            return y_mean, y_cov
        return y_mean, y_std
    else:
        y_mean = gp.predict(X)
        return y_mean


def sample_from_gp(gp, X_test, n_samples=1000):
    """
    Sample from GP posterior predictive distribution

    Parameters:
    -----------
    gp : GaussianProcessRegressor
        Fitted GP model
    X_test : pd.DataFrame or np.ndarray
        Test features
    n_samples : int
        Number of samples to draw

    Returns:
    --------
    np.ndarray : Samples with shape (n_samples, n_test_points)
    """
    if isinstance(X_test, pd.DataFrame):
        X = X_test.values
    else:
        X = X_test

    # Get predictive mean and std
    y_mean, y_std = gp.predict(X, return_std=True)

    # Sample from independent Gaussians at each test point
    samples = np.random.randn(n_samples, len(X)) * y_std + y_mean

    return samples


if __name__ == "__main__":
    # Test the model
    print("Testing Gaussian Process Regression model...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0]**2 + X[:, 1] - X[:, 2] +
         np.random.randn(n_samples) * 0.2)

    # Build and fit model
    gp = build_gp_model(use_ard=True)
    gp_fitted, feature_names = fit_gp_model(gp, X, y)

    # Get kernel parameters
    kernel_params = get_kernel_params(gp_fitted, feature_names)

    # Make predictions
    X_test = np.random.randn(10, n_features)
    y_pred, y_std = predict_gp(gp_fitted, X_test)

    print("\nTest predictions:")
    print(f"  Mean: {y_pred[:5]}")
    print(f"  Std:  {y_std[:5]}")

    print("\nTest complete!")
