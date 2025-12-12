"""
Decision Analysis Module
Reliability-based decisions using posterior predictive distributions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


def compute_reliability_probability(predictions, threshold):
    """
    Compute P(strength >= threshold) for each test point

    Parameters:
    -----------
    predictions : dict
        Predictions with samples
    threshold : float
        Minimum strength threshold (MPa)

    Returns:
    --------
    np.ndarray : Reliability probabilities for each test point
    """
    samples = predictions['samples']  # shape: (n_samples, n_test_points)

    # For each test point, compute fraction of samples above threshold
    reliability_probs = np.mean(samples >= threshold, axis=0)

    return reliability_probs


def make_certification_decision(reliability_probs, p_target):
    """
    Make certification decisions based on reliability threshold

    Parameters:
    -----------
    reliability_probs : np.ndarray
        Reliability probabilities for each mixture
    p_target : float
        Target reliability (e.g., 0.95)

    Returns:
    --------
    np.ndarray : Boolean array (True = approved, False = rejected)
    """
    decisions = reliability_probs >= p_target
    return decisions


def compute_mixture_cost(X, feature_names=None):
    """
    Compute cost of concrete mixtures based on ingredient prices

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Features (mixture compositions)
    feature_names : list, optional
        Names of features (required if X is np.ndarray)

    Returns:
    --------
    np.ndarray : Cost for each mixture (relative units)
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        X_array = X

    costs = config.DECISION_CONFIG['cost_weights']

    # Calculate cost for each mixture
    mixture_costs = np.zeros(len(X_array))

    for i, feature in enumerate(feature_names):
        # Only consider original features (not engineered ones)
        if feature in costs:
            mixture_costs += X_array[:, i] * costs[feature]

    return mixture_costs


def analyze_cost_reliability_tradeoff(X_test, predictions, y_test=None, feature_names=None):
    """
    Analyze cost vs reliability tradeoff

    Parameters:
    -----------
    X_test : pd.DataFrame or np.ndarray
        Test features (original scale, not scaled)
    predictions : dict
        Predictions with samples
    y_test : np.ndarray, optional
        True strengths (for visualization)
    feature_names : list, optional
        Feature names

    Returns:
    --------
    pd.DataFrame : Cost-reliability analysis results
    """
    threshold = config.DECISION_CONFIG['s_min']
    p_target = config.DECISION_CONFIG['p_target']

    # Compute reliability probabilities
    reliability_probs = compute_reliability_probability(predictions, threshold)

    # Compute costs
    costs = compute_mixture_cost(X_test, feature_names)

    # Make certification decisions
    approved = make_certification_decision(reliability_probs, p_target)

    # Create results DataFrame
    results = pd.DataFrame({
        'reliability_prob': reliability_probs,
        'cost': costs,
        'approved': approved,
        'expected_strength': predictions['mean'],
        'strength_std': predictions['std']
    })

    if y_test is not None:
        if isinstance(y_test, pd.Series):
            results['true_strength'] = y_test.values
        else:
            results['true_strength'] = y_test

    return results


def plot_reliability_distribution(reliability_probs_blr, reliability_probs_gp,
                                   p_target, save=True):
    """
    Plot distribution of reliability probabilities

    Parameters:
    -----------
    reliability_probs_blr : np.ndarray
        BLR reliability probabilities
    reliability_probs_gp : np.ndarray
        GP reliability probabilities
    p_target : float
        Target reliability threshold
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # BLR histogram
    ax = axes[0]
    ax.hist(reliability_probs_blr, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(p_target, color='red', linestyle='--', linewidth=2,
               label=f'Target: {p_target:.0%}')
    ax.set_xlabel('P(Strength ≥ 35 MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Count', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('BLR: Reliability Probability Distribution',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # Add statistics
    n_approved = (reliability_probs_blr >= p_target).sum()
    approval_rate = n_approved / len(reliability_probs_blr)
    ax.text(0.05, 0.95, f'Approval rate: {approval_rate:.1%}\n({n_approved}/{len(reliability_probs_blr)} mixtures)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # GP histogram
    ax = axes[1]
    ax.hist(reliability_probs_gp, bins=30, edgecolor='black', alpha=0.7, color='seagreen')
    ax.axvline(p_target, color='red', linestyle='--', linewidth=2,
               label=f'Target: {p_target:.0%}')
    ax.set_xlabel('P(Strength ≥ 35 MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Count', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('GP: Reliability Probability Distribution',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # Add statistics
    n_approved = (reliability_probs_gp >= p_target).sum()
    approval_rate = n_approved / len(reliability_probs_gp)
    ax.text(0.05, 0.95, f'Approval rate: {approval_rate:.1%}\n({n_approved}/{len(reliability_probs_gp)} mixtures)',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save:
        config.FIGURES_DECISION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_DECISION_DIR / "fig_reliability_probabilities.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_certification_decisions(results_blr, results_gp, save=True):
    """
    Visualize certification decisions

    Parameters:
    -----------
    results_blr : pd.DataFrame
        BLR decision results
    results_gp : pd.DataFrame
        GP decision results
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # BLR scatter plot
    ax = axes[0]
    approved_blr = results_blr[results_blr['approved']]
    rejected_blr = results_blr[~results_blr['approved']]

    ax.scatter(approved_blr['expected_strength'], approved_blr['reliability_prob'],
               c='green', s=50, alpha=0.6, label=f'Approved (n={len(approved_blr)})', edgecolors='black')
    ax.scatter(rejected_blr['expected_strength'], rejected_blr['reliability_prob'],
               c='red', s=50, alpha=0.6, label=f'Rejected (n={len(rejected_blr)})', edgecolors='black')

    ax.axhline(config.DECISION_CONFIG['p_target'], color='blue', linestyle='--',
               linewidth=2, label=f"Reliability threshold: {config.DECISION_CONFIG['p_target']:.0%}")
    ax.axvline(config.DECISION_CONFIG['s_min'], color='orange', linestyle='--',
               linewidth=2, label=f"Strength threshold: {config.DECISION_CONFIG['s_min']} MPa")

    ax.set_xlabel('Expected Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Reliability Probability', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('BLR: Certification Decisions', fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # GP scatter plot
    ax = axes[1]
    approved_gp = results_gp[results_gp['approved']]
    rejected_gp = results_gp[~results_gp['approved']]

    ax.scatter(approved_gp['expected_strength'], approved_gp['reliability_prob'],
               c='green', s=50, alpha=0.6, label=f'Approved (n={len(approved_gp)})', edgecolors='black')
    ax.scatter(rejected_gp['expected_strength'], rejected_gp['reliability_prob'],
               c='red', s=50, alpha=0.6, label=f'Rejected (n={len(rejected_gp)})', edgecolors='black')

    ax.axhline(config.DECISION_CONFIG['p_target'], color='blue', linestyle='--',
               linewidth=2, label=f"Reliability threshold: {config.DECISION_CONFIG['p_target']:.0%}")
    ax.axvline(config.DECISION_CONFIG['s_min'], color='orange', linestyle='--',
               linewidth=2, label=f"Strength threshold: {config.DECISION_CONFIG['s_min']} MPa")

    ax.set_xlabel('Expected Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Reliability Probability', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('GP: Certification Decisions', fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        config.FIGURES_DECISION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_DECISION_DIR / "fig_certification_decisions.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_uncertainty_impact(results_blr, results_gp, save=True):
    """
    Show how uncertainty affects certification decisions

    Parameters:
    -----------
    results_blr : pd.DataFrame
        BLR decision results
    results_gp : pd.DataFrame
        GP decision results
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # BLR: Uncertainty vs decision boundary
    ax = axes[0]
    approved_blr = results_blr[results_blr['approved']]
    rejected_blr = results_blr[~results_blr['approved']]

    ax.scatter(approved_blr['expected_strength'], approved_blr['strength_std'],
               c='green', s=50, alpha=0.6, label='Approved', edgecolors='black')
    ax.scatter(rejected_blr['expected_strength'], rejected_blr['strength_std'],
               c='red', s=50, alpha=0.6, label='Rejected', edgecolors='black')

    ax.axvline(config.DECISION_CONFIG['s_min'], color='orange', linestyle='--',
               linewidth=2, label=f"Strength threshold: {config.DECISION_CONFIG['s_min']} MPa")

    ax.set_xlabel('Expected Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Prediction Std Dev (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('BLR: How Uncertainty Affects Decisions',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    # GP: Uncertainty vs decision boundary
    ax = axes[1]
    approved_gp = results_gp[results_gp['approved']]
    rejected_gp = results_gp[~results_gp['approved']]

    ax.scatter(approved_gp['expected_strength'], approved_gp['strength_std'],
               c='green', s=50, alpha=0.6, label='Approved', edgecolors='black')
    ax.scatter(rejected_gp['expected_strength'], rejected_gp['strength_std'],
               c='red', s=50, alpha=0.6, label='Rejected', edgecolors='black')

    ax.axvline(config.DECISION_CONFIG['s_min'], color='orange', linestyle='--',
               linewidth=2, label=f"Strength threshold: {config.DECISION_CONFIG['s_min']} MPa")

    ax.set_xlabel('Expected Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_ylabel('Prediction Std Dev (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title('GP: How Uncertainty Affects Decisions',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        config.FIGURES_DECISION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_DECISION_DIR / "fig_uncertainty_decision.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def input_sensitivity_analysis(X_test, predictions, scaler, model_predict_func,
                                perturbation=None, save=True):
    """
    Analyze how sensitive decisions are to input perturbations
    Perturb each ingredient by ±X% and measure change in P(strength >= 35)

    Parameters:
    -----------
    X_test : pd.DataFrame
        Test features (unscaled)
    predictions : dict
        Original predictions
    scaler : StandardScaler
        Fitted scaler for transforming inputs
    model_predict_func : callable
        Function that takes scaled X and returns predictions dict with samples
    perturbation : float
        Perturbation factor (default from config, e.g., 0.05 for ±5%)
    save : bool
        Whether to save results

    Returns:
    --------
    pd.DataFrame : Sensitivity analysis results
    """
    if perturbation is None:
        perturbation = config.SENSITIVITY_CONFIG['input_perturbation']

    threshold = config.DECISION_CONFIG['s_min']
    ingredients = config.SENSITIVITY_CONFIG['ingredients_to_perturb']

    # Get baseline reliability
    baseline_reliability = compute_reliability_probability(predictions, threshold)

    sensitivity_results = []

    for ingredient in ingredients:
        if ingredient not in X_test.columns:
            continue

        for direction in ['increase', 'decrease']:
            # Create perturbed input
            X_perturbed = X_test.copy()
            if direction == 'increase':
                X_perturbed[ingredient] = X_perturbed[ingredient] * (1 + perturbation)
            else:
                X_perturbed[ingredient] = X_perturbed[ingredient] * (1 - perturbation)

            # Re-engineer features if needed
            if 'water_cement_ratio' in X_perturbed.columns:
                X_perturbed['water_cement_ratio'] = (
                    X_perturbed['water'] / (X_perturbed['cement'] + 1e-10)
                )
            if 'scm_fraction' in X_perturbed.columns:
                total_binder = (X_perturbed['cement'] +
                               X_perturbed['blast_furnace_slag'] +
                               X_perturbed['fly_ash'])
                X_perturbed['scm_fraction'] = (
                    (X_perturbed['blast_furnace_slag'] + X_perturbed['fly_ash']) /
                    (total_binder + 1e-10)
                )

            # Scale and predict
            try:
                X_perturbed_scaled = scaler.transform(X_perturbed)
                perturbed_preds = model_predict_func(X_perturbed_scaled)
                perturbed_reliability = compute_reliability_probability(
                    perturbed_preds, threshold
                )

                # Compute change in reliability
                reliability_change = perturbed_reliability - baseline_reliability

                sensitivity_results.append({
                    'ingredient': ingredient,
                    'direction': direction,
                    'perturbation_pct': perturbation * 100,
                    'mean_reliability_change': reliability_change.mean(),
                    'std_reliability_change': reliability_change.std(),
                    'max_reliability_change': reliability_change.max(),
                    'min_reliability_change': reliability_change.min()
                })
            except Exception as e:
                if config.VERBOSE:
                    print(f"Warning: Could not compute sensitivity for {ingredient}: {e}")

    results_df = pd.DataFrame(sensitivity_results)

    if save and config.SAVE_INTERMEDIATE:
        results_dir = config.RESULTS_DIR / "sensitivity"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_dir / "input_sensitivity.csv", index=False)

    return results_df


def plot_input_sensitivity(sensitivity_results, model_name='Model', save=True):
    """
    Plot input sensitivity analysis results

    Parameters:
    -----------
    sensitivity_results : pd.DataFrame
        Results from input_sensitivity_analysis
    model_name : str
        Name of model for title
    save : bool
        Whether to save figure
    """
    if sensitivity_results.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique ingredients
    ingredients = sensitivity_results['ingredient'].unique()
    x = np.arange(len(ingredients))
    width = 0.35

    # Plot increase and decrease effects
    increase_data = sensitivity_results[sensitivity_results['direction'] == 'increase']
    decrease_data = sensitivity_results[sensitivity_results['direction'] == 'decrease']

    # Match order
    increase_vals = [increase_data[increase_data['ingredient'] == i]['mean_reliability_change'].values[0]
                     if i in increase_data['ingredient'].values else 0 for i in ingredients]
    decrease_vals = [decrease_data[decrease_data['ingredient'] == i]['mean_reliability_change'].values[0]
                     if i in decrease_data['ingredient'].values else 0 for i in ingredients]

    bars1 = ax.bar(x - width/2, increase_vals, width, label='+5%', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, decrease_vals, width, label='-5%', color='red', alpha=0.7)

    ax.set_ylabel('Mean Change in Reliability P(strength ≥ 35)',
                  fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_xlabel('Ingredient', fontsize=config.PLOT_CONFIG['label_size'])
    ax.set_title(f'{model_name}: Input Sensitivity Analysis (±5% perturbation)',
                 fontsize=config.PLOT_CONFIG['title_size'])
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace('_', '\n') for i in ingredients], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        config.FIGURES_DECISION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_DECISION_DIR / f"fig_input_sensitivity_{model_name.lower()}.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def threshold_sensitivity_analysis(predictions_blr, predictions_gp, save=True):
    """
    Analyze how certification decisions change with different thresholds

    Parameters:
    -----------
    predictions_blr : dict
        BLR predictions
    predictions_gp : dict
        GP predictions
    save : bool
        Whether to save results

    Returns:
    --------
    pd.DataFrame : Sensitivity results for different threshold combinations
    """
    threshold_values = config.SENSITIVITY_CONFIG['threshold_values']
    reliability_targets = config.SENSITIVITY_CONFIG['reliability_targets']

    results = []

    for s_min in threshold_values:
        for p_target in reliability_targets:
            # BLR
            rel_probs_blr = compute_reliability_probability(predictions_blr, s_min)
            decisions_blr = make_certification_decision(rel_probs_blr, p_target)
            approval_rate_blr = decisions_blr.mean()

            # GP
            rel_probs_gp = compute_reliability_probability(predictions_gp, s_min)
            decisions_gp = make_certification_decision(rel_probs_gp, p_target)
            approval_rate_gp = decisions_gp.mean()

            results.append({
                's_min': s_min,
                'p_target': p_target,
                'blr_approval_rate': approval_rate_blr,
                'gp_approval_rate': approval_rate_gp,
                'decision_agreement': (decisions_blr == decisions_gp).mean()
            })

    results_df = pd.DataFrame(results)

    if save and config.SAVE_INTERMEDIATE:
        results_dir = config.RESULTS_DIR / "sensitivity"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_dir / "threshold_sensitivity.csv", index=False)

    return results_df


def plot_threshold_sensitivity(threshold_results, save=True):
    """
    Plot threshold sensitivity analysis

    Parameters:
    -----------
    threshold_results : pd.DataFrame
        Results from threshold_sensitivity_analysis
    save : bool
        Whether to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get unique values
    threshold_values = sorted(threshold_results['s_min'].unique())
    reliability_targets = sorted(threshold_results['p_target'].unique())

    # Heatmap data
    for idx, (model, col) in enumerate([('BLR', 'blr_approval_rate'),
                                         ('GP', 'gp_approval_rate'),
                                         ('Agreement', 'decision_agreement')]):
        ax = axes[idx]

        # Create matrix
        matrix = np.zeros((len(reliability_targets), len(threshold_values)))
        for i, p in enumerate(reliability_targets):
            for j, s in enumerate(threshold_values):
                row = threshold_results[(threshold_results['s_min'] == s) &
                                        (threshold_results['p_target'] == p)]
                if len(row) > 0:
                    matrix[i, j] = row[col].values[0]

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Labels
        ax.set_xticks(range(len(threshold_values)))
        ax.set_xticklabels([f'{s}' for s in threshold_values])
        ax.set_yticks(range(len(reliability_targets)))
        ax.set_yticklabels([f'{p:.0%}' for p in reliability_targets])
        ax.set_xlabel('Strength Threshold (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_ylabel('Reliability Target', fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'{model} Approval Rate' if model != 'Agreement'
                     else 'BLR-GP Decision Agreement',
                     fontsize=config.PLOT_CONFIG['title_size'])

        # Add text annotations
        for i in range(len(reliability_targets)):
            for j in range(len(threshold_values)):
                text = ax.text(j, i, f'{matrix[i, j]:.0%}',
                              ha='center', va='center', color='black', fontsize=9)

        plt.colorbar(im, ax=ax, label='Rate')

    plt.suptitle('Sensitivity of Decisions to Threshold Choices', fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        config.FIGURES_DECISION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_DECISION_DIR / "fig_threshold_sensitivity.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def define_trust_region(X_train, method=None, margin=None):
    """
    Define the trust region (model validity region) based on training data

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features (unscaled)
    method : str
        Method for defining trust region ('feature_ranges' or 'mahalanobis')
    margin : float
        Margin beyond observed ranges

    Returns:
    --------
    dict : Trust region definition with bounds for each feature
    """
    if method is None:
        method = config.TRUST_REGION_CONFIG['method']
    if margin is None:
        margin = config.TRUST_REGION_CONFIG['range_margin']

    trust_region = {
        'method': method,
        'features': {}
    }

    if method == 'feature_ranges':
        for col in X_train.columns:
            min_val = X_train[col].min()
            max_val = X_train[col].max()
            range_val = max_val - min_val

            trust_region['features'][col] = {
                'observed_min': min_val,
                'observed_max': max_val,
                'trust_min': min_val - margin * range_val,
                'trust_max': max_val + margin * range_val
            }

    elif method == 'mahalanobis':
        # Store mean and covariance for Mahalanobis distance
        trust_region['mean'] = X_train.mean().values
        trust_region['cov'] = X_train.cov().values
        trust_region['threshold'] = config.TRUST_REGION_CONFIG['mahalanobis_threshold']

    return trust_region


def check_in_trust_region(X_test, trust_region):
    """
    Check which test points fall within the trust region

    Parameters:
    -----------
    X_test : pd.DataFrame
        Test features (unscaled)
    trust_region : dict
        Trust region definition from define_trust_region

    Returns:
    --------
    np.ndarray : Boolean array indicating if each point is in trust region
    """
    method = trust_region['method']

    if method == 'feature_ranges':
        in_region = np.ones(len(X_test), dtype=bool)
        for col in X_test.columns:
            if col in trust_region['features']:
                bounds = trust_region['features'][col]
                in_region &= (X_test[col] >= bounds['trust_min'])
                in_region &= (X_test[col] <= bounds['trust_max'])
        return in_region

    elif method == 'mahalanobis':
        from scipy.spatial.distance import mahalanobis
        mean = trust_region['mean']
        cov = trust_region['cov']
        threshold = trust_region['threshold']

        try:
            cov_inv = np.linalg.inv(cov)
            distances = np.array([mahalanobis(x, mean, cov_inv)
                                  for x in X_test.values])
            return distances <= threshold
        except np.linalg.LinAlgError:
            # If covariance is singular, fall back to feature ranges
            return np.ones(len(X_test), dtype=bool)

    return np.ones(len(X_test), dtype=bool)


def plot_trust_region_summary(trust_region, X_test, in_region, save=True):
    """
    Visualize trust region and which test points fall within it

    Parameters:
    -----------
    trust_region : dict
        Trust region definition
    X_test : pd.DataFrame
        Test features
    in_region : np.ndarray
        Boolean array from check_in_trust_region
    save : bool
        Whether to save figure
    """
    # Select key features for visualization
    key_features = ['cement', 'water', 'age']
    key_features = [f for f in key_features if f in X_test.columns]

    if len(key_features) < 2:
        return

    fig, axes = plt.subplots(1, len(key_features), figsize=(6*len(key_features), 5))
    if len(key_features) == 1:
        axes = [axes]

    for idx, feature in enumerate(key_features):
        ax = axes[idx]

        # Plot all test points
        ax.scatter(X_test[feature][in_region], np.arange(in_region.sum()),
                   c='green', alpha=0.6, label='In trust region', s=30)
        ax.scatter(X_test[feature][~in_region], np.arange((~in_region).sum()),
                   c='red', alpha=0.6, label='Outside trust region', s=30)

        # Plot trust region bounds
        if feature in trust_region.get('features', {}):
            bounds = trust_region['features'][feature]
            ax.axvline(bounds['observed_min'], color='blue', linestyle='--',
                       label='Observed range')
            ax.axvline(bounds['observed_max'], color='blue', linestyle='--')
            ax.axvline(bounds['trust_min'], color='orange', linestyle=':',
                       label='Trust region')
            ax.axvline(bounds['trust_max'], color='orange', linestyle=':')

        ax.set_xlabel(feature.replace('_', ' ').title(),
                      fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_ylabel('Test Sample Index', fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'Trust Region: {feature}',
                     fontsize=config.PLOT_CONFIG['title_size'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Model Validity Region ({in_region.sum()}/{len(in_region)} points in region)',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        config.FIGURES_DECISION_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_DECISION_DIR / "fig_trust_region.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def analyze_decisions(predictions, prepared_data, save_figures=True):
    """
    Complete decision analysis pipeline

    Parameters:
    -----------
    predictions : dict
        Predictions from both models
    prepared_data : dict
        Prepared data
    save_figures : bool
        Whether to save decision figures

    Returns:
    --------
    dict : Decision analysis results
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("DECISION ANALYSIS")
        print("="*80 + "\n")

    predictions_blr = predictions['blr']
    predictions_gp = predictions['gp']

    X_test = prepared_data['X_test']  # Original scale
    y_test = prepared_data['y_test']

    threshold = config.DECISION_CONFIG['s_min']
    p_target = config.DECISION_CONFIG['p_target']

    if config.VERBOSE:
        print(f"Decision criteria:")
        print(f"  Minimum strength threshold: {threshold} MPa")
        print(f"  Target reliability: {p_target:.0%}")

    # Compute reliability probabilities
    if config.VERBOSE:
        print("\nComputing reliability probabilities...")

    reliability_probs_blr = compute_reliability_probability(predictions_blr, threshold)
    reliability_probs_gp = compute_reliability_probability(predictions_gp, threshold)

    # Analyze cost-reliability tradeoff
    if config.VERBOSE:
        print("Analyzing cost-reliability tradeoff...")

    results_blr = analyze_cost_reliability_tradeoff(
        X_test, predictions_blr, y_test, X_test.columns.tolist()
    )
    results_gp = analyze_cost_reliability_tradeoff(
        X_test, predictions_gp, y_test, X_test.columns.tolist()
    )

    # Print summary statistics
    if config.VERBOSE:
        print("\nBLR Decision Summary:")
        print(f"  Approved mixtures: {results_blr['approved'].sum()} / {len(results_blr)} ({100*results_blr['approved'].mean():.1f}%)")
        print(f"  Mean reliability prob: {reliability_probs_blr.mean():.3f}")
        print(f"  Median reliability prob: {np.median(reliability_probs_blr):.3f}")

        print("\nGP Decision Summary:")
        print(f"  Approved mixtures: {results_gp['approved'].sum()} / {len(results_gp)} ({100*results_gp['approved'].mean():.1f}%)")
        print(f"  Mean reliability prob: {reliability_probs_gp.mean():.3f}")
        print(f"  Median reliability prob: {np.median(reliability_probs_gp):.3f}")

    # Generate decision plots
    if save_figures:
        if config.VERBOSE:
            print("\nGenerating decision plots...")

        plot_reliability_distribution(reliability_probs_blr, reliability_probs_gp, p_target, save=True)
        plot_certification_decisions(results_blr, results_gp, save=True)
        plot_uncertainty_impact(results_blr, results_gp, save=True)

    # Threshold sensitivity analysis
    if config.VERBOSE:
        print("\nRunning threshold sensitivity analysis...")

    threshold_results = threshold_sensitivity_analysis(predictions_blr, predictions_gp, save=True)

    if config.VERBOSE:
        print("\nThreshold Sensitivity Results:")
        for _, row in threshold_results.iterrows():
            print(f"  s_min={row['s_min']}, p_target={row['p_target']:.0%}: "
                  f"BLR approval={row['blr_approval_rate']:.1%}, "
                  f"GP approval={row['gp_approval_rate']:.1%}, "
                  f"Agreement={row['decision_agreement']:.1%}")

    if save_figures:
        plot_threshold_sensitivity(threshold_results, save=True)

    # Trust region analysis
    if config.VERBOSE:
        print("\nDefining trust region...")

    X_train = prepared_data['X_train']
    trust_region = define_trust_region(X_train)
    in_region = check_in_trust_region(X_test, trust_region)

    if config.VERBOSE:
        print(f"  Trust region method: {trust_region['method']}")
        print(f"  Test points in trust region: {in_region.sum()}/{len(in_region)} "
              f"({100*in_region.mean():.1f}%)")

    if save_figures:
        plot_trust_region_summary(trust_region, X_test, in_region, save=True)

    # Save results
    if config.SAVE_INTERMEDIATE:
        results_dir = config.RESULTS_DIR / "decisions"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_blr.to_csv(results_dir / "decisions_blr.csv", index=False)
        results_gp.to_csv(results_dir / "decisions_gp.csv", index=False)

        if config.VERBOSE:
            print(f"\nSaved decision results to: {results_dir}")

    if config.VERBOSE:
        print("\n" + "="*80)
        print("DECISION ANALYSIS COMPLETE")
        print("="*80 + "\n")

    return {
        'blr': {
            'reliability_probs': reliability_probs_blr,
            'decisions': results_blr
        },
        'gp': {
            'reliability_probs': reliability_probs_gp,
            'decisions': results_gp
        },
        'threshold_sensitivity': threshold_results,
        'trust_region': trust_region,
        'in_trust_region': in_region
    }


if __name__ == "__main__":
    print("Testing decision module...")
    print("Module loaded successfully")
