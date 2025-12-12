"""
Validation Module
Compute metrics, calibration, and posterior predictive checks
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    dict : Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }


def compute_calibration(y_true, predictions, confidence_levels=None):
    """
    Compute calibration: do X% prediction intervals contain X% of observations?

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    predictions : dict
        Predictions dictionary with 'samples' or quantiles
    confidence_levels : list, optional
        Confidence levels to check

    Returns:
    --------
    dict : Calibration results
    """
    if confidence_levels is None:
        confidence_levels = config.VALIDATION_CONFIG['confidence_levels']

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    calibration_results = {}

    # Use samples to compute empirical coverage
    samples = predictions['samples']  # shape: (n_samples, n_test_points)

    for level in confidence_levels:
        # Compute percentiles for this confidence level
        lower_percentile = (1 - level) / 2 * 100
        upper_percentile = (1 + level) / 2 * 100

        lower = np.percentile(samples, lower_percentile, axis=0)
        upper = np.percentile(samples, upper_percentile, axis=0)

        # Count how many true values fall within the interval
        coverage = np.mean((y_true >= lower) & (y_true <= upper))

        calibration_results[f'{int(level*100)}%'] = {
            'expected': level,
            'observed': coverage,
            'difference': coverage - level
        }

    return calibration_results


def plot_calibration(calibration_blr, calibration_gp, save=True):
    """
    Plot calibration curves

    Parameters:
    -----------
    calibration_blr : dict
        BLR calibration results
    calibration_gp : dict
        GP calibration results
    save : bool
        Whether to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data
    expected_blr = [v['expected'] for v in calibration_blr.values()]
    observed_blr = [v['observed'] for v in calibration_blr.values()]

    expected_gp = [v['expected'] for v in calibration_gp.values()]
    observed_gp = [v['observed'] for v in calibration_gp.values()]

    # Plot
    ax.plot(expected_blr, observed_blr, 'o-', linewidth=2, markersize=10,
            label='BLR', color='blue')
    ax.plot(expected_gp, observed_gp, 's-', linewidth=2, markersize=10,
            label='GP', color='green')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

    ax.set_xlabel('Expected Coverage', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Observed Coverage', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('Calibration Plot: Prediction Interval Coverage',
                 fontsize=config.PLOT_CONFIG['title_size'] + 2)
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'] + 2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add text annotations
    for exp, obs_b, obs_g in zip(expected_blr, observed_blr, observed_gp):
        ax.annotate(f'{int(exp*100)}%', xy=(exp, obs_b), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='blue')
        ax.annotate(f'{int(exp*100)}%', xy=(exp, obs_g), xytext=(5, -10),
                    textcoords='offset points', fontsize=8, color='green')

    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_VALIDATION_DIR / "fig_calibration_plot.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def posterior_predictive_check(y_true, predictions, model_name='Model', save=True, filename=None):
    """
    Posterior predictive check: compare observed data to predictions

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    predictions : dict
        Predictions with samples
    model_name : str
        Name of the model for title
    save : bool
        Whether to save the figure
    filename : str, optional
        Custom filename
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    samples = predictions['samples']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Distribution comparison
    ax = axes[0]
    ax.hist(y_true, bins=30, alpha=0.7, label='Observed data', color='blue', edgecolor='black', density=True)

    # Plot multiple posterior predictive samples
    for i in range(min(50, samples.shape[0])):
        ax.hist(samples[i], bins=30, alpha=0.02, color='red', density=True)

    # Plot mean of posterior predictive
    ax.hist(samples.mean(axis=0), bins=30, alpha=0.5, label='Posterior predictive mean',
            color='orange', edgecolor='black', density=True)

    ax.set_xlabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Density', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title(f'{model_name}: Posterior Predictive Distribution',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # Plot 2: Predictive bands
    ax = axes[1]
    sorted_idx = np.argsort(y_true)
    x_axis = np.arange(len(y_true))

    # Plot true values
    ax.scatter(x_axis, y_true[sorted_idx], s=30, alpha=0.7, color='blue', label='Observed', zorder=3)

    # Plot predictive mean
    pred_mean = predictions['mean'][sorted_idx]
    ax.plot(x_axis, pred_mean, 'r-', linewidth=2, label='Predicted mean', zorder=2)

    # Plot prediction intervals
    lower_50 = predictions['quantiles']['25'][sorted_idx]
    upper_50 = predictions['quantiles']['75'][sorted_idx]
    ax.fill_between(x_axis, lower_50, upper_50, alpha=0.3, color='red', label='50% interval')

    lower_95 = predictions['quantiles']['2.5'][sorted_idx]
    upper_95 = predictions['quantiles']['97.5'][sorted_idx]
    ax.fill_between(x_axis, lower_95, upper_95, alpha=0.15, color='red', label='95% interval')

    ax.set_xlabel('Test Sample (sorted by true value)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title(f'{model_name}: Predictive Intervals',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"fig_posterior_predictive_{model_name.lower().replace(' ', '_')}.png"
        filepath = config.FIGURES_VALIDATION_DIR / filename
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_residuals_vs_fitted(y_true, predictions, model_name='Model', save=True):
    """
    Plot residuals vs fitted values to check for heteroscedasticity and patterns

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    predictions : dict
        Predictions with mean
    model_name : str
        Name of the model for title
    save : bool
        Whether to save the figure
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    y_pred = predictions['mean']
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Residuals vs Fitted Values
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted Values (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title(f'{model_name}: Residuals vs Fitted Values',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3)

    # Add lowess smoother to detect patterns
    try:
        from scipy.ndimage import uniform_filter1d
        sorted_idx = np.argsort(y_pred)
        smoothed = uniform_filter1d(residuals[sorted_idx], size=20)
        ax.plot(y_pred[sorted_idx], smoothed, 'orange', linewidth=2,
                label='Smoothed trend')
        ax.legend()
    except ImportError:
        pass

    # Plot 2: Histogram of residuals (check normality)
    ax = axes[1]
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, density=True)
    ax.set_xlabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Density', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title(f'{model_name}: Residual Distribution',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.grid(True, alpha=0.3)

    # Add normal curve overlay
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy import stats
    norm_pdf = stats.norm.pdf(x_norm, loc=residuals.mean(), scale=residuals.std())
    ax.plot(x_norm, norm_pdf, 'r-', linewidth=2, label='Normal fit')
    ax.legend()

    # Add statistics text
    textstr = f'Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}'
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"fig_residuals_fitted_{model_name.lower().replace(' ', '_')}.png"
        filepath = config.FIGURES_VALIDATION_DIR / filename
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_residuals_vs_covariates(y_true, predictions, X_test, model_name='Model',
                                  key_features=None, save=True):
    """
    Plot residuals vs key covariates to check for systematic patterns

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    predictions : dict
        Predictions with mean
    X_test : pd.DataFrame
        Test features (unscaled)
    model_name : str
        Name of the model
    key_features : list, optional
        Features to plot against (default: cement, water, age, w/c ratio)
    save : bool
        Whether to save the figure
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    y_pred = predictions['mean']
    residuals = y_true - y_pred

    # Default key features to examine
    if key_features is None:
        key_features = ['cement', 'water', 'age', 'water_cement_ratio']
        # Filter to only features that exist
        key_features = [f for f in key_features if f in X_test.columns]

    n_features = len(key_features)
    if n_features == 0:
        return

    n_cols = min(2, n_features)
    n_rows = (n_features + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        x_vals = X_test[feature].values

        ax.scatter(x_vals, residuals, alpha=0.6, s=30,
                   edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(feature.replace('_', ' ').title(),
                      fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_ylabel('Residuals (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'Residuals vs {feature.replace("_", " ").title()}',
                     fontsize=config.PLOT_CONFIG['title_size'])
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(x_vals, residuals)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{model_name}: Residuals vs Key Covariates',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"fig_residuals_covariates_{model_name.lower().replace(' ', '_')}.png"
        filepath = config.FIGURES_VALIDATION_DIR / filename
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_model_comparison(metrics_blr, metrics_gp, save=True):
    """
    Create side-by-side comparison of model metrics

    Parameters:
    -----------
    metrics_blr : dict
        BLR metrics
    metrics_gp : dict
        GP metrics
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = ['RMSE', 'MAE', 'R2']
    models = ['BLR', 'GP']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        values = [metrics_blr[metric], metrics_gp[metric]]
        colors = ['steelblue', 'seagreen']

        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')

        ax.set_ylabel(metric, fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'{metric} Comparison', fontsize=config.PLOT_CONFIG['title_size'])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

        # For R2, higher is better; for RMSE and MAE, lower is better
        if metric == 'R2':
            better_idx = np.argmax(values)
        else:
            better_idx = np.argmin(values)

        bars[better_idx].set_edgecolor('gold')
        bars[better_idx].set_linewidth(3)

    plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        config.FIGURES_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_VALIDATION_DIR / "fig_model_comparison.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def validate_models(predictions, prepared_data, save_figures=True):
    """
    Complete validation pipeline for both models

    Parameters:
    -----------
    predictions : dict
        Predictions from both models
    prepared_data : dict
        Prepared data with test set
    save_figures : bool
        Whether to save validation figures

    Returns:
    --------
    dict : Validation results for both models
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("MODEL VALIDATION")
        print("="*80 + "\n")

    y_test = prepared_data['y_test']
    predictions_blr = predictions['blr']
    predictions_gp = predictions['gp']

    # Compute metrics
    if config.VERBOSE:
        print("Computing metrics...")

    metrics_blr = compute_metrics(y_test, predictions_blr['mean'])
    metrics_gp = compute_metrics(y_test, predictions_gp['mean'])

    if config.VERBOSE:
        print("\nBayesian Linear Regression Metrics:")
        for key, value in metrics_blr.items():
            print(f"  {key}: {value:.4f}")

        print("\nGaussian Process Metrics:")
        for key, value in metrics_gp.items():
            print(f"  {key}: {value:.4f}")

    # Compute calibration
    if config.VERBOSE:
        print("\nComputing calibration...")

    calibration_blr = compute_calibration(y_test, predictions_blr)
    calibration_gp = compute_calibration(y_test, predictions_gp)

    if config.VERBOSE:
        print("\nBLR Calibration:")
        for level, result in calibration_blr.items():
            print(f"  {level}: Expected {result['expected']:.2%}, Observed {result['observed']:.2%}, "
                  f"Difference {result['difference']:+.2%}")

        print("\nGP Calibration:")
        for level, result in calibration_gp.items():
            print(f"  {level}: Expected {result['expected']:.2%}, Observed {result['observed']:.2%}, "
                  f"Difference {result['difference']:+.2%}")

    # Generate validation plots
    if save_figures:
        if config.VERBOSE:
            print("\nGenerating validation plots...")

        plot_calibration(calibration_blr, calibration_gp, save=True)
        posterior_predictive_check(y_test, predictions_blr, model_name='BLR',
                                   save=True, filename='fig_ppc_blr.png')
        posterior_predictive_check(y_test, predictions_gp, model_name='GP',
                                   save=True, filename='fig_ppc_gp.png')
        plot_model_comparison(metrics_blr, metrics_gp, save=True)

        # Residual diagnostic plots
        if config.VERBOSE:
            print("\nGenerating residual diagnostic plots...")

        plot_residuals_vs_fitted(y_test, predictions_blr, model_name='BLR', save=True)
        plot_residuals_vs_fitted(y_test, predictions_gp, model_name='GP', save=True)

        # Residuals vs covariates (need unscaled X_test)
        X_test_unscaled = prepared_data['X_test']
        plot_residuals_vs_covariates(y_test, predictions_blr, X_test_unscaled,
                                     model_name='BLR', save=True)
        plot_residuals_vs_covariates(y_test, predictions_gp, X_test_unscaled,
                                     model_name='GP', save=True)

    # Save metrics to file
    if config.SAVE_INTERMEDIATE:
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Model': ['BLR', 'GP'],
            'RMSE': [metrics_blr['RMSE'], metrics_gp['RMSE']],
            'MAE': [metrics_blr['MAE'], metrics_gp['MAE']],
            'R2': [metrics_blr['R2'], metrics_gp['R2']],
            'MAPE': [metrics_blr['MAPE'], metrics_gp['MAPE']]
        })
        metrics_df.to_csv(config.METRICS_DIR / "metrics.csv", index=False)

        # Save calibration results
        calib_data = []
        for level in calibration_blr.keys():
            calib_data.append({
                'Model': 'BLR',
                'Confidence_Level': level,
                'Expected': calibration_blr[level]['expected'],
                'Observed': calibration_blr[level]['observed'],
                'Difference': calibration_blr[level]['difference']
            })
            calib_data.append({
                'Model': 'GP',
                'Confidence_Level': level,
                'Expected': calibration_gp[level]['expected'],
                'Observed': calibration_gp[level]['observed'],
                'Difference': calibration_gp[level]['difference']
            })

        calib_df = pd.DataFrame(calib_data)
        calib_df.to_csv(config.METRICS_DIR / "calibration.csv", index=False)

        if config.VERBOSE:
            print(f"\nSaved metrics to: {config.METRICS_DIR}")

    if config.VERBOSE:
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80 + "\n")

    return {
        'blr': {
            'metrics': metrics_blr,
            'calibration': calibration_blr
        },
        'gp': {
            'metrics': metrics_gp,
            'calibration': calibration_gp
        }
    }


if __name__ == "__main__":
    print("Testing validation module...")
    print("Module loaded successfully")
