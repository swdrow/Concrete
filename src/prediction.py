"""
Prediction Module
Handles predictions and uncertainty quantification for both models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def predict_blr(trace, X_test, n_samples=None):
    """
    Generate posterior predictive samples for Bayesian Linear Regression

    Parameters:
    -----------
    trace : arviz.InferenceData
        BLR posterior samples
    X_test : pd.DataFrame or np.ndarray
        Test features (scaled)
    n_samples : int, optional
        Number of posterior samples to use

    Returns:
    --------
    dict : Predictions with mean, std, and samples
    """
    # Convert to numpy
    if isinstance(X_test, pd.DataFrame):
        X = X_test.values
    else:
        X = X_test

    if n_samples is None:
        n_samples = config.VALIDATION_CONFIG['n_posterior_samples']

    # Extract posterior samples
    alpha_samples = trace.posterior['alpha'].values  # Intercept
    beta_samples = trace.posterior['beta'].values
    sigma_samples = trace.posterior['sigma'].values

    # Reshape to (n_total_samples, n_features)
    n_chains, n_draws, n_features = beta_samples.shape
    alpha_samples = alpha_samples.flatten()
    beta_samples = beta_samples.reshape(-1, n_features)
    sigma_samples = sigma_samples.flatten()

    # Randomly select samples if needed
    if n_samples < len(beta_samples):
        idx = np.random.choice(len(beta_samples), size=n_samples, replace=False)
        alpha_samples = alpha_samples[idx]
        beta_samples = beta_samples[idx]
        sigma_samples = sigma_samples[idx]

    # Generate predictions: y = alpha + X @ beta + noise
    y_pred_samples = []
    for alpha, beta, sigma in zip(alpha_samples, beta_samples, sigma_samples):
        y_mean = alpha + X @ beta  # Include intercept!
        y_sample = y_mean + np.random.randn(len(X)) * sigma
        y_pred_samples.append(y_sample)

    y_pred_samples = np.array(y_pred_samples)

    # Compute statistics
    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred_std = y_pred_samples.std(axis=0)

    # Compute credible intervals
    percentiles = [2.5, 25, 50, 75, 97.5]
    y_pred_quantiles = np.percentile(y_pred_samples, percentiles, axis=0)

    predictions = {
        'mean': y_pred_mean,
        'std': y_pred_std,
        'samples': y_pred_samples,
        'quantiles': {
            '2.5': y_pred_quantiles[0],
            '25': y_pred_quantiles[1],
            '50': y_pred_quantiles[2],
            '75': y_pred_quantiles[3],
            '97.5': y_pred_quantiles[4]
        }
    }

    return predictions


def predict_gp(gp_model, X_test, n_samples=None):
    """
    Generate posterior predictive samples for Gaussian Process

    Parameters:
    -----------
    gp_model : GaussianProcessRegressor
        Fitted GP model
    X_test : pd.DataFrame or np.ndarray
        Test features (scaled)
    n_samples : int, optional
        Number of samples to draw from predictive distribution

    Returns:
    --------
    dict : Predictions with mean, std, and samples
    """
    # Convert to numpy
    if isinstance(X_test, pd.DataFrame):
        X = X_test.values
    else:
        X = X_test

    if n_samples is None:
        n_samples = config.VALIDATION_CONFIG['n_posterior_samples']

    # Get predictive mean and std
    y_pred_mean, y_pred_std = gp_model.predict(X, return_std=True)

    # Generate samples from predictive distribution
    y_pred_samples = []
    for _ in range(n_samples):
        y_sample = y_pred_mean + np.random.randn(len(X)) * y_pred_std
        y_pred_samples.append(y_sample)

    y_pred_samples = np.array(y_pred_samples)

    # Compute quantiles
    percentiles = [2.5, 25, 50, 75, 97.5]
    y_pred_quantiles = np.percentile(y_pred_samples, percentiles, axis=0)

    predictions = {
        'mean': y_pred_mean,
        'std': y_pred_std,
        'samples': y_pred_samples,
        'quantiles': {
            '2.5': y_pred_quantiles[0],
            '25': y_pred_quantiles[1],
            '50': y_pred_quantiles[2],
            '75': y_pred_quantiles[3],
            '97.5': y_pred_quantiles[4]
        }
    }

    return predictions


def plot_predictions(y_test, predictions_blr, predictions_gp, save=True):
    """
    Plot predictions vs actual for both models

    Parameters:
    -----------
    y_test : pd.Series or np.ndarray
        True test values
    predictions_blr : dict
        BLR predictions
    predictions_gp : dict
        GP predictions
    save : bool
        Whether to save the figure
    """
    if isinstance(y_test, pd.Series):
        y_true = y_test.values
    else:
        y_true = y_test

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # BLR predictions
    ax = axes[0]
    ax.scatter(y_true, predictions_blr['mean'], alpha=0.6, s=30, label='Predictions')
    ax.errorbar(y_true, predictions_blr['mean'], yerr=2*predictions_blr['std'],
                fmt='none', alpha=0.2, color='blue', label='95% CI (±2σ)')

    # Perfect prediction line
    min_val = min(y_true.min(), predictions_blr['mean'].min())
    max_val = max(y_true.max(), predictions_blr['mean'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('True Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Predicted Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('Bayesian Linear Regression\nPredictions vs Actual',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # Compute R²
    ss_res = np.sum((y_true - predictions_blr['mean'])**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.05, 0.95, f'R² = {r2:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # GP predictions
    ax = axes[1]
    ax.scatter(y_true, predictions_gp['mean'], alpha=0.6, s=30, label='Predictions')
    ax.errorbar(y_true, predictions_gp['mean'], yerr=2*predictions_gp['std'],
                fmt='none', alpha=0.2, color='green', label='95% CI (±2σ)')

    # Perfect prediction line
    min_val = min(y_true.min(), predictions_gp['mean'].min())
    max_val = max(y_true.max(), predictions_gp['mean'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('True Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Predicted Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('Gaussian Process Regression\nPredictions vs Actual',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # Compute R²
    ss_res = np.sum((y_true - predictions_gp['mean'])**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    ax.text(0.05, 0.95, f'R² = {r2:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_VALIDATION_DIR / "fig_predictions.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_residuals(y_test, predictions_blr, predictions_gp, save=True):
    """
    Plot residual analysis for both models

    Parameters:
    -----------
    y_test : pd.Series or np.ndarray
        True test values
    predictions_blr : dict
        BLR predictions
    predictions_gp : dict
        GP predictions
    save : bool
        Whether to save the figure
    """
    if isinstance(y_test, pd.Series):
        y_true = y_test.values
    else:
        y_true = y_test

    # Calculate residuals
    residuals_blr = y_true - predictions_blr['mean']
    residuals_gp = y_true - predictions_gp['mean']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # BLR residuals vs predicted
    ax = axes[0, 0]
    ax.scatter(predictions_blr['mean'], residuals_blr, alpha=0.6, s=30)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('BLR: Residuals vs Predicted', fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3)

    # BLR residual histogram
    ax = axes[0, 1]
    ax.hist(residuals_blr, bins=30, edgecolor='black', alpha=0.7, density=True)
    ax.set_xlabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Density', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('BLR: Residual Distribution', fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3)
    # Add normal curve
    from scipy.stats import norm
    x = np.linspace(residuals_blr.min(), residuals_blr.max(), 100)
    ax.plot(x, norm.pdf(x, residuals_blr.mean(), residuals_blr.std()),
            'r-', linewidth=2, label='Normal fit')
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])

    # GP residuals vs predicted
    ax = axes[1, 0]
    ax.scatter(predictions_gp['mean'], residuals_gp, alpha=0.6, s=30, color='green')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('GP: Residuals vs Predicted', fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3)

    # GP residual histogram
    ax = axes[1, 1]
    ax.hist(residuals_gp, bins=30, edgecolor='black', alpha=0.7, density=True, color='green')
    ax.set_xlabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Density', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('GP: Residual Distribution', fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3)
    # Add normal curve
    x = np.linspace(residuals_gp.min(), residuals_gp.max(), 100)
    ax.plot(x, norm.pdf(x, residuals_gp.mean(), residuals_gp.std()),
            'r-', linewidth=2, label='Normal fit')
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])

    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_VALIDATION_DIR / "fig_residuals.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def make_all_predictions(fitted_models, prepared_data, save_figures=True):
    """
    Generate predictions from all fitted models

    Parameters:
    -----------
    fitted_models : dict
        Dictionary containing fitted models from fitting module
    prepared_data : dict
        Dictionary containing prepared data
    save_figures : bool
        Whether to save prediction figures

    Returns:
    --------
    dict : Predictions from both models
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS")
        print("="*80 + "\n")

    X_test_scaled = prepared_data['X_test_scaled']
    y_test = prepared_data['y_test']

    # BLR predictions
    if config.VERBOSE:
        print("Generating BLR predictions...")
    predictions_blr = predict_blr(fitted_models['blr']['trace'], X_test_scaled)

    # GP predictions
    if config.VERBOSE:
        print("Generating GP predictions...")
    predictions_gp = predict_gp(fitted_models['gp']['model'], X_test_scaled)

    # Plot predictions
    if save_figures:
        plot_predictions(y_test, predictions_blr, predictions_gp, save=True)
        plot_residuals(y_test, predictions_blr, predictions_gp, save=True)

    # Save predictions
    if config.SAVE_INTERMEDIATE:
        config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        pred_df = pd.DataFrame({
            'y_true': y_test.values if isinstance(y_test, pd.Series) else y_test,
            'blr_mean': predictions_blr['mean'],
            'blr_std': predictions_blr['std'],
            'gp_mean': predictions_gp['mean'],
            'gp_std': predictions_gp['std']
        })
        pred_df.to_csv(config.PREDICTIONS_DIR / "predictions.csv", index=False)

        if config.VERBOSE:
            print(f"\nSaved predictions to: {config.PREDICTIONS_DIR / 'predictions.csv'}")

    if config.VERBOSE:
        print("\n" + "="*80)
        print("PREDICTIONS COMPLETE")
        print("="*80 + "\n")

    return {
        'blr': predictions_blr,
        'gp': predictions_gp
    }


if __name__ == "__main__":
    print("Testing prediction module...")
    # This would require fitted models to test
    print("Module loaded successfully")
