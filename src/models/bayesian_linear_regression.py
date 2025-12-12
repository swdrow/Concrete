"""
Bayesian Linear Regression Model
Uses PyMC for MCMC inference
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def build_blr_model(X_train, y_train, prior_beta_std=None, use_hyperprior=None):
    """
    Build Bayesian Linear Regression model using PyMC

    Model specification (without hyperprior):
        y = α + Xβ + ε
        α ~ Normal(μ_y, 20)             # Intercept centered near data mean
        β ~ Normal(0, σ_β²)
        σ ~ HalfNormal(σ_prior)
        y | X, α, β, σ ~ Normal(α + Xβ, σ²)

    Model specification (with hyperprior on τ):
        y = α + Xβ + ε
        α ~ Normal(μ_y, 20)             # Intercept centered near data mean
        τ ~ HalfNormal(τ_prior)         # Hyperprior on shrinkage
        β | τ ~ Normal(0, τ²)           # Coefficients depend on τ
        σ ~ HalfNormal(σ_prior)
        y | X, α, β, σ ~ Normal(α + Xβ, σ²)

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features (scaled)
    y_train : pd.Series or np.ndarray
        Training target
    prior_beta_std : float, optional
        Override prior std for beta (for sensitivity analysis)
    use_hyperprior : bool, optional
        Override whether to use hyperprior on tau

    Returns:
    --------
    pm.Model : PyMC model object
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

    n_features = X.shape[1]

    # Use provided values or defaults from config
    if prior_beta_std is None:
        prior_beta_std = config.BLR_CONFIG['prior_beta_std']
    if use_hyperprior is None:
        use_hyperprior = config.BLR_CONFIG.get('use_hyperprior', False)

    if config.VERBOSE:
        print(f"\nBuilding Bayesian Linear Regression model...")
        print(f"  Number of features: {n_features}")
        print(f"  Number of samples: {len(y)}")
        print(f"  Using hyperprior on τ: {use_hyperprior}")

    # Compute y mean for intercept prior
    y_mean = float(np.mean(y))

    with pm.Model() as model:
        # Intercept prior - centered near the data mean
        alpha = pm.Normal('alpha', mu=y_mean, sigma=20.0)

        if use_hyperprior:
            # Hyperprior on tau (global shrinkage)
            tau = pm.HalfNormal('tau',
                                sigma=config.BLR_CONFIG.get('tau_prior_std', 10.0))

            # Priors for regression coefficients (depend on tau)
            beta = pm.Normal('beta',
                             mu=0,
                             sigma=tau,
                             shape=n_features)
        else:
            # Fixed prior on beta (original specification)
            beta = pm.Normal('beta',
                             mu=0,
                             sigma=prior_beta_std,
                             shape=n_features)

        # Prior for noise standard deviation
        sigma = pm.HalfNormal('sigma',
                              sigma=config.BLR_CONFIG['prior_sigma_std'])

        # Linear model WITH INTERCEPT
        mu = alpha + pm.math.dot(X, beta)

        # Likelihood
        y_obs = pm.Normal('y_obs',
                          mu=mu,
                          sigma=sigma,
                          observed=y)

        if config.VERBOSE:
            print("\nModel specification:")
            print(f"  Prior on α (intercept): Normal({y_mean:.2f}, 20)")
            if use_hyperprior:
                print(f"  Hyperprior on τ: HalfNormal({config.BLR_CONFIG.get('tau_prior_std', 10.0)})")
                print(f"  Prior on β | τ: Normal(0, τ²)")
            else:
                print(f"  Prior on β: Normal(0, {prior_beta_std}²)")
            print(f"  Prior on σ: HalfNormal({config.BLR_CONFIG['prior_sigma_std']})")
            print(f"  Likelihood: y ~ Normal(α + Xβ, σ²)")

    return model, feature_names


def fit_blr_model(model, sample_kwargs=None):
    """
    Fit Bayesian Linear Regression using MCMC

    Parameters:
    -----------
    model : pm.Model
        PyMC model object
    sample_kwargs : dict, optional
        Additional kwargs for pm.sample()

    Returns:
    --------
    arviz.InferenceData : Posterior samples and diagnostics
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    # Default sampling parameters from config
    default_kwargs = {
        'draws': config.BLR_CONFIG['n_draws'],
        'tune': config.BLR_CONFIG['n_tune'],
        'chains': config.BLR_CONFIG['n_chains'],
        'target_accept': config.BLR_CONFIG['target_accept'],
        'random_seed': config.BLR_CONFIG['random_seed'],
        'return_inferencedata': True,
        'progressbar': config.VERBOSE
    }
    default_kwargs.update(sample_kwargs)

    if config.VERBOSE:
        print("\nRunning MCMC sampling...")
        print(f"  Draws per chain: {default_kwargs['draws']}")
        print(f"  Tuning samples: {default_kwargs['tune']}")
        print(f"  Number of chains: {default_kwargs['chains']}")

    with model:
        trace = pm.sample(**default_kwargs)

    if config.VERBOSE:
        print("\nMCMC sampling complete!")

    return trace


def check_convergence(trace, feature_names):
    """
    Check MCMC convergence diagnostics

    Parameters:
    -----------
    trace : arviz.InferenceData
        Posterior samples
    feature_names : list
        Names of features

    Returns:
    --------
    dict : Convergence diagnostics
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("CONVERGENCE DIAGNOSTICS")
        print("="*80)

    # R-hat statistic
    rhat = az.rhat(trace)
    rhat_beta = rhat['beta'].values
    rhat_sigma = rhat['sigma'].values
    rhat_alpha = rhat['alpha'].values

    # Effective sample size
    ess = az.ess(trace)
    ess_beta = ess['beta'].values
    ess_sigma = ess['sigma'].values
    ess_alpha = ess['alpha'].values

    diagnostics = {
        'rhat_alpha': rhat_alpha,
        'rhat_beta': rhat_beta,
        'rhat_sigma': rhat_sigma,
        'ess_alpha': ess_alpha,
        'ess_beta': ess_beta,
        'ess_sigma': ess_sigma
    }

    # Check for tau (hyperprior) if present
    has_tau = 'tau' in rhat.data_vars
    if has_tau:
        rhat_tau = rhat['tau'].values
        ess_tau = ess['tau'].values
        diagnostics['rhat_tau'] = rhat_tau
        diagnostics['ess_tau'] = ess_tau

    if config.VERBOSE:
        print("\nR-hat (should be < 1.01):")
        print(f"  α (intercept): {rhat_alpha:.4f}")
        print(f"  σ: {rhat_sigma:.4f}")
        if has_tau:
            print(f"  τ: {rhat_tau:.4f}")
        for i, name in enumerate(feature_names):
            print(f"  β[{name}]: {rhat_beta[i]:.4f}")

        print("\nEffective Sample Size (should be > 400):")
        print(f"  α (intercept): {ess_alpha:.0f}")
        print(f"  σ: {ess_sigma:.0f}")
        if has_tau:
            print(f"  τ: {ess_tau:.0f}")
        for i, name in enumerate(feature_names):
            print(f"  β[{name}]: {ess_beta[i]:.0f}")

        # Check if any issues
        rhat_issues = (rhat_beta > 1.01).sum() + (rhat_sigma > 1.01) + (rhat_alpha > 1.01)
        ess_issues = (ess_beta < 400).sum() + (ess_sigma < 400) + (ess_alpha < 400)
        if has_tau:
            rhat_issues += (rhat_tau > 1.01)
            ess_issues += (ess_tau < 400)

        if rhat_issues > 0 or ess_issues > 0:
            print("\nWARNING: Convergence issues detected!")
            if rhat_issues > 0:
                print(f"  {rhat_issues} parameters have R-hat > 1.01")
            if ess_issues > 0:
                print(f"  {ess_issues} parameters have ESS < 400")
        else:
            print("\nAll convergence diagnostics look good!")

        print("="*80 + "\n")

    return diagnostics


def plot_trace(trace, feature_names, save=True):
    """
    Plot MCMC trace plots

    Parameters:
    -----------
    trace : arviz.InferenceData
        Posterior samples
    feature_names : list
        Names of features
    save : bool
        Whether to save the figure
    """
    # Use arviz's built-in plotting with all variables (including alpha intercept)
    az.plot_trace(trace, var_names=['alpha', 'beta', 'sigma'],
                  figsize=(15, max(12, (len(feature_names) + 2) * 1.5)),
                  show=False)

    plt.suptitle('MCMC Trace Plots - Bayesian Linear Regression',
                 fontsize=14, y=1.00)
    plt.tight_layout()

    if save:
        config.FIGURES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_MODELS_DIR / "fig_blr_trace.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_posterior(trace, feature_names, save=True):
    """
    Plot posterior distributions of coefficients

    Parameters:
    -----------
    trace : arviz.InferenceData
        Posterior samples
    feature_names : list
        Names of features
    save : bool
        Whether to save the figure
    """
    # Extract posterior samples
    beta_samples = trace.posterior['beta'].values.reshape(-1, len(feature_names))

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        ax.hist(beta_samples[:, i], bins=50, edgecolor='black', alpha=0.7, density=True)
        ax.set_xlabel(f'β[{name}]', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Posterior: β[{name}]', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean = beta_samples[:, i].mean()
        std = beta_samples[:, i].std()
        ci_low, ci_high = np.percentile(beta_samples[:, i], [2.5, 97.5])

        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
        ax.axvline(ci_low, color='green', linestyle=':', linewidth=1.5)
        ax.axvline(ci_high, color='green', linestyle=':', linewidth=1.5, label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
        ax.legend(fontsize=8)

    # Plot sigma
    sigma_samples = trace.posterior['sigma'].values.flatten()
    ax = axes[len(feature_names)]
    ax.hist(sigma_samples, bins=50, edgecolor='black', alpha=0.7, density=True, color='coral')
    ax.set_xlabel('σ (noise std)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Posterior: σ', fontsize=10)
    ax.grid(True, alpha=0.3)

    mean_sigma = sigma_samples.mean()
    ci_low, ci_high = np.percentile(sigma_samples, [2.5, 97.5])
    ax.axvline(mean_sigma, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sigma:.3f}')
    ax.axvline(ci_low, color='green', linestyle=':', linewidth=1.5)
    ax.axvline(ci_high, color='green', linestyle=':', linewidth=1.5, label=f'95% CI: [{ci_low:.3f}, {ci_high:.3f}]')
    ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(len(feature_names) + 1, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Posterior Distributions - Bayesian Linear Regression',
                 fontsize=14, y=1.00)
    plt.tight_layout()

    if save:
        config.FIGURES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_MODELS_DIR / "fig_blr_posterior.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def get_posterior_summary(trace, feature_names):
    """
    Get summary statistics of posterior distributions

    Parameters:
    -----------
    trace : arviz.InferenceData
        Posterior samples
    feature_names : list
        Names of features

    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    summary = az.summary(trace, var_names=['alpha', 'beta', 'sigma'])

    # Add feature names to beta coefficients
    beta_idx = summary.index.str.startswith('beta')
    summary.loc[beta_idx, 'feature'] = feature_names

    return summary


if __name__ == "__main__":
    # Test the model
    print("Testing Bayesian Linear Regression model...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([2.0, -1.5, 0.5])
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    # Build and fit model
    model, feature_names = build_blr_model(X, y)
    trace = fit_blr_model(model)

    # Check convergence
    diagnostics = check_convergence(trace, feature_names)

    print("\nTest complete!")
