"""
Model Fitting Module
Orchestrates the fitting of both BLR and GP models
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

from src.models.bayesian_linear_regression import (
    build_blr_model, fit_blr_model, check_convergence,
    plot_trace, plot_posterior, get_posterior_summary
)
from src.models.gaussian_process import (
    build_gp_model, fit_gp_model, get_kernel_params, plot_kernel_params
)


def fit_all_models(prepared_data, save_models=True, save_figures=True):
    """
    Fit both Bayesian Linear Regression and Gaussian Process models

    Parameters:
    -----------
    prepared_data : dict
        Dictionary containing prepared data from data_preparation module
    save_models : bool
        Whether to save fitted models
    save_figures : bool
        Whether to save diagnostic figures

    Returns:
    --------
    dict : Dictionary containing fitted models and diagnostics
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("MODEL FITTING")
        print("="*80 + "\n")

    # Extract data
    X_train_scaled = prepared_data['X_train_scaled']
    y_train = prepared_data['y_train']

    results = {}

    # ========================================================================
    # 1. FIT BAYESIAN LINEAR REGRESSION
    # ========================================================================
    if config.VERBOSE:
        print("\n" + "-"*80)
        print("FITTING BAYESIAN LINEAR REGRESSION")
        print("-"*80)

    # Build model
    blr_model, blr_feature_names = build_blr_model(X_train_scaled, y_train)

    # Fit model
    blr_trace = fit_blr_model(blr_model)

    # Check convergence
    blr_diagnostics = check_convergence(blr_trace, blr_feature_names)

    # Get posterior summary
    blr_summary = get_posterior_summary(blr_trace, blr_feature_names)

    # Plot diagnostics
    if save_figures:
        plot_trace(blr_trace, blr_feature_names, save=True)
        plot_posterior(blr_trace, blr_feature_names, save=True)

    # Save model
    if save_models and config.SAVE_INTERMEDIATE:
        config.POSTERIORS_DIR.mkdir(parents=True, exist_ok=True)
        blr_file = config.POSTERIORS_DIR / "blr_trace.nc"
        blr_trace.to_netcdf(blr_file)
        if config.VERBOSE:
            print(f"\nSaved BLR trace to: {blr_file}")

        # Save feature names (model object cannot be pickled)
        model_file = config.POSTERIORS_DIR / "blr_feature_names.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({'feature_names': blr_feature_names}, f)

    results['blr'] = {
        'model': blr_model,
        'trace': blr_trace,
        'feature_names': blr_feature_names,
        'diagnostics': blr_diagnostics,
        'summary': blr_summary
    }

    # ========================================================================
    # 2. FIT GAUSSIAN PROCESS
    # ========================================================================
    if config.VERBOSE:
        print("\n" + "-"*80)
        print("FITTING GAUSSIAN PROCESS REGRESSION")
        print("-"*80)

    # Build model
    gp_model = build_gp_model()

    # Fit model
    gp_fitted, gp_feature_names = fit_gp_model(gp_model, X_train_scaled, y_train)

    # Get kernel parameters
    gp_kernel_params = get_kernel_params(gp_fitted, gp_feature_names)

    # Plot kernel parameters
    if save_figures:
        plot_kernel_params(gp_kernel_params, save=True)

    # Save model
    if save_models and config.SAVE_INTERMEDIATE:
        model_file = config.POSTERIORS_DIR / "gp_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({'model': gp_fitted, 'feature_names': gp_feature_names}, f)
        if config.VERBOSE:
            print(f"\nSaved GP model to: {model_file}")

    results['gp'] = {
        'model': gp_fitted,
        'feature_names': gp_feature_names,
        'kernel_params': gp_kernel_params
    }

    if config.VERBOSE:
        print("\n" + "="*80)
        print("MODEL FITTING COMPLETE")
        print("="*80 + "\n")

    return results


def run_prior_sensitivity_analysis(X_train_scaled, y_train, save_figures=True):
    """
    Run prior sensitivity analysis for BLR model
    Test different prior scales for beta to see how it affects predictions

    Parameters:
    -----------
    X_train_scaled : pd.DataFrame
        Scaled training features
    y_train : pd.Series
        Training target
    save_figures : bool
        Whether to save figures

    Returns:
    --------
    dict : Results for each prior scale
    """
    import matplotlib.pyplot as plt

    prior_scales = config.PRIOR_SENSITIVITY_SCALES

    if config.VERBOSE:
        print("\n" + "-"*80)
        print("PRIOR SENSITIVITY ANALYSIS")
        print("-"*80)
        print(f"Testing prior scales: {prior_scales}")

    results = {}

    for scale in prior_scales:
        if config.VERBOSE:
            print(f"\n  Fitting BLR with beta prior scale = {scale}...")

        # Build model with specific prior scale (no hyperprior for sensitivity analysis)
        model, feature_names = build_blr_model(
            X_train_scaled, y_train,
            prior_beta_std=scale,
            use_hyperprior=False
        )

        # Fit with fewer samples for speed
        trace = fit_blr_model(model, sample_kwargs={
            'draws': 1000,
            'tune': 500,
            'chains': 2
        })

        # Get posterior summary
        summary = get_posterior_summary(trace, feature_names)

        # Store results
        results[scale] = {
            'trace': trace,
            'summary': summary,
            'feature_names': feature_names
        }

        if config.VERBOSE:
            beta_mean = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
            beta_std = trace.posterior['beta'].std(dim=['chain', 'draw']).values
            print(f"    Beta posterior means: {beta_mean.mean():.3f} (avg)")
            print(f"    Beta posterior stds: {beta_std.mean():.3f} (avg)")

    # Plot comparison
    if save_figures and len(results) > 0:
        _plot_prior_sensitivity(results, save=True)

    return results


def _plot_prior_sensitivity(results, save=True):
    """
    Plot prior sensitivity analysis results

    Parameters:
    -----------
    results : dict
        Results from run_prior_sensitivity_analysis
    save : bool
        Whether to save figure
    """
    import matplotlib.pyplot as plt

    scales = list(results.keys())
    n_features = len(results[scales[0]]['feature_names'])
    feature_names = results[scales[0]]['feature_names']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot 1: Beta coefficient means across prior scales
    ax = axes[0]
    width = 0.25
    x = np.arange(min(6, n_features))  # Plot first 6 features

    for idx, scale in enumerate(scales):
        trace = results[scale]['trace']
        beta_means = trace.posterior['beta'].mean(dim=['chain', 'draw']).values[:6]
        ax.bar(x + idx*width, beta_means, width, label=f'σ_β={scale}', alpha=0.7)

    ax.set_xlabel('Feature', fontsize=10)
    ax.set_ylabel('Posterior Mean of β', fontsize=10)
    ax.set_title('Effect of Prior Scale on β Coefficients', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f[:8] for f in feature_names[:6]], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Beta coefficient stds across prior scales
    ax = axes[1]
    for idx, scale in enumerate(scales):
        trace = results[scale]['trace']
        beta_stds = trace.posterior['beta'].std(dim=['chain', 'draw']).values[:6]
        ax.bar(x + idx*width, beta_stds, width, label=f'σ_β={scale}', alpha=0.7)

    ax.set_xlabel('Feature', fontsize=10)
    ax.set_ylabel('Posterior Std of β', fontsize=10)
    ax.set_title('Effect of Prior Scale on β Uncertainty', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f[:8] for f in feature_names[:6]], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Sigma (noise) posterior across scales
    ax = axes[2]
    for scale in scales:
        trace = results[scale]['trace']
        sigma_samples = trace.posterior['sigma'].values.flatten()
        ax.hist(sigma_samples, bins=30, alpha=0.5, label=f'σ_β={scale}', density=True)

    ax.set_xlabel('σ (noise std)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Posterior of Noise σ Across Prior Scales', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax = axes[3]
    metrics = ['σ posterior mean', 'σ posterior std', 'Avg |β| mean']
    x_metrics = np.arange(len(metrics))
    width = 0.25

    for idx, scale in enumerate(scales):
        trace = results[scale]['trace']
        sigma_mean = trace.posterior['sigma'].mean().values
        sigma_std = trace.posterior['sigma'].std().values
        beta_abs_mean = np.abs(trace.posterior['beta'].mean(dim=['chain', 'draw']).values).mean()

        values = [sigma_mean, sigma_std, beta_abs_mean]
        ax.bar(x_metrics + idx*width, values, width, label=f'σ_β={scale}', alpha=0.7)

    ax.set_xlabel('Metric', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Summary Statistics Across Prior Scales', fontsize=12)
    ax.set_xticks(x_metrics + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Prior Sensitivity Analysis: Effect of β Prior Scale', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        config.FIGURES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_MODELS_DIR / "fig_prior_sensitivity.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def load_fitted_models():
    """
    Load previously fitted models from disk

    Returns:
    --------
    dict : Dictionary containing loaded models
    """
    results = {}

    # Load BLR
    try:
        import arviz as az
        blr_file = config.POSTERIORS_DIR / "blr_trace.nc"
        blr_trace = az.from_netcdf(blr_file)

        model_file = config.POSTERIORS_DIR / "blr_model.pkl"
        with open(model_file, 'rb') as f:
            blr_data = pickle.load(f)

        results['blr'] = {
            'model': blr_data['model'],
            'trace': blr_trace,
            'feature_names': blr_data['feature_names']
        }
        if config.VERBOSE:
            print("Loaded BLR model")
    except Exception as e:
        print(f"Could not load BLR model: {e}")

    # Load GP
    try:
        model_file = config.POSTERIORS_DIR / "gp_model.pkl"
        with open(model_file, 'rb') as f:
            gp_data = pickle.load(f)

        results['gp'] = {
            'model': gp_data['model'],
            'feature_names': gp_data['feature_names']
        }
        if config.VERBOSE:
            print("Loaded GP model")
    except Exception as e:
        print(f"Could not load GP model: {e}")

    return results


if __name__ == "__main__":
    # Test the module
    from data_ingestion import ingest_data
    from data_preparation import prepare_data

    print("Testing fitting module...")

    # Load and prepare data
    df = ingest_data(verbose=False)
    prepared_data = prepare_data(df, save_figures=False)

    # Fit models
    fitted_models = fit_all_models(prepared_data, save_models=True, save_figures=True)

    print("\nFitting test complete!")
    print(f"Fitted models: {list(fitted_models.keys())}")
