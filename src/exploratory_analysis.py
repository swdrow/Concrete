"""
Exploratory Data Analysis Module
Generates visualizations and statistics for understanding the concrete dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(config.PLOT_CONFIG['color_palette'])


def plot_distributions(df, save=True):
    """
    Plot histograms for all features and target variable

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    save : bool
        Whether to save the figure
    """
    n_cols = 3
    n_rows = int(np.ceil(len(df.columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(df.columns):
        ax = axes[idx]
        ax.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col.replace('_', ' ').title(), fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_ylabel('Frequency', fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}',
                     fontsize=config.PLOT_CONFIG['title_size'])
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=config.PLOT_CONFIG['legend_size'])

    # Hide empty subplots
    for idx in range(len(df.columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save:
        config.FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_EDA_DIR / "fig_distributions.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_correlation_heatmap(df, save=True):
    """
    Plot correlation heatmap for all variables

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    save : bool
        Whether to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate correlation matrix
    corr = df.corr()

    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)

    ax.set_title('Correlation Matrix of Concrete Features',
                 fontsize=config.PLOT_CONFIG['title_size'] + 2, pad=20)

    plt.tight_layout()

    if save:
        config.FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_EDA_DIR / "fig_correlation_heatmap.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_pairplots(df, save=True):
    """
    Plot pairwise relationships between features and target

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    save : bool
        Whether to save the figure
    """
    # Create scatter plots of each feature vs target
    n_features = len(config.FEATURE_COLUMNS)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, feature in enumerate(config.FEATURE_COLUMNS):
        ax = axes[idx]
        ax.scatter(df[feature], df[config.TARGET_COLUMN], alpha=0.5, s=20)
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'{feature.replace("_", " ").title()} vs Strength',
                     fontsize=config.PLOT_CONFIG['title_size'])
        ax.grid(True, alpha=0.3)

        # Calculate and display correlation
        corr = df[feature].corr(df[config.TARGET_COLUMN])
        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save:
        config.FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_EDA_DIR / "fig_pairplots.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_boxplots(df, save=True):
    """
    Plot boxplots for outlier detection

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    save : bool
        Whether to save the figure
    """
    n_cols = 3
    n_rows = int(np.ceil(len(df.columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, col in enumerate(df.columns):
        ax = axes[idx]
        ax.boxplot(df[col], vert=True)
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=config.PLOT_CONFIG['label_size'])
        ax.set_title(f'Boxplot: {col.replace("_", " ").title()}',
                     fontsize=config.PLOT_CONFIG['title_size'])
        ax.grid(True, alpha=0.3, axis='y')

        # Calculate outlier statistics
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        ax.text(1.15, 0.5, f'Outliers: {outliers}\n({100*outliers/len(df):.1f}%)',
                transform=ax.transAxes,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Hide empty subplots
    for idx in range(len(df.columns), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save:
        config.FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_EDA_DIR / "fig_boxplots.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def plot_age_relationship(df, save=True):
    """
    Plot the relationship between age and strength, showing log transformation

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    save : bool
        Whether to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Original age vs strength
    axes[0].scatter(df['age'], df[config.TARGET_COLUMN], alpha=0.5, s=20)
    axes[0].set_xlabel('Age (days)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[0].set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[0].set_title('Age vs Compressive Strength (Original)',
                      fontsize=config.PLOT_CONFIG['title_size'])
    axes[0].grid(True, alpha=0.3)
    corr_orig = df['age'].corr(df[config.TARGET_COLUMN])
    axes[0].text(0.05, 0.95, f'Correlation: {corr_orig:.3f}',
                 transform=axes[0].transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Log-transformed age vs strength
    log_age = np.log(df['age'] + 1)
    axes[1].scatter(log_age, df[config.TARGET_COLUMN], alpha=0.5, s=20)
    axes[1].set_xlabel('Log(Age + 1)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[1].set_ylabel('Compressive Strength (MPa)', fontsize=config.PLOT_CONFIG['label_size'])
    axes[1].set_title('Log(Age) vs Compressive Strength',
                      fontsize=config.PLOT_CONFIG['title_size'])
    axes[1].grid(True, alpha=0.3)
    corr_log = log_age.corr(df[config.TARGET_COLUMN])
    axes[1].text(0.05, 0.95, f'Correlation: {corr_log:.3f}',
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save:
        config.FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.FIGURES_EDA_DIR / "fig_age_relationship.png"
        plt.savefig(filepath, dpi=config.PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        if config.VERBOSE:
            print(f"Saved: {filepath}")

    plt.close()


def generate_eda_report(df):
    """
    Generate a text report with key EDA takeaways

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset

    Returns:
    --------
    str : Report text
    """
    report = []
    report.append("\n" + "="*80)
    report.append("EXPLORATORY DATA ANALYSIS - KEY TAKEAWAYS")
    report.append("="*80 + "\n")

    # Correlation analysis
    corr_with_target = df[config.FEATURE_COLUMNS].corrwith(df[config.TARGET_COLUMN]).sort_values(ascending=False)
    report.append("1. FEATURE CORRELATIONS WITH COMPRESSIVE STRENGTH:")
    for feature, corr in corr_with_target.items():
        report.append(f"   - {feature}: {corr:+.3f}")

    # Top correlations
    top_positive = corr_with_target.iloc[0]
    top_negative = corr_with_target.iloc[-1]
    report.append(f"\n   Strongest positive correlation: {corr_with_target.index[0]} ({top_positive:.3f})")
    report.append(f"   Strongest negative correlation: {corr_with_target.index[-1]} ({top_negative:.3f})")

    # Age analysis
    log_age_corr = np.log(df['age'] + 1).corr(df[config.TARGET_COLUMN])
    age_corr = df['age'].corr(df[config.TARGET_COLUMN])
    report.append(f"\n2. AGE TRANSFORMATION:")
    report.append(f"   - Original age correlation: {age_corr:.3f}")
    report.append(f"   - Log(age) correlation: {log_age_corr:.3f}")
    if log_age_corr > age_corr:
        report.append(f"   - Log transformation improves correlation by {log_age_corr - age_corr:.3f}")
        report.append("   → Recommendation: Use log(age) as feature")

    # Missing values
    report.append("\n3. DATA QUALITY:")
    missing = df.isnull().sum().sum()
    report.append(f"   - Missing values: {missing} (Perfect!)" if missing == 0 else f"   - Missing values: {missing}")

    # Outliers
    report.append("\n4. OUTLIERS:")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            report.append(f"   - {col}: {outliers} outliers ({100*outliers/len(df):.1f}%)")

    # Target distribution
    report.append(f"\n5. TARGET VARIABLE (Compressive Strength):")
    report.append(f"   - Range: {df[config.TARGET_COLUMN].min():.2f} - {df[config.TARGET_COLUMN].max():.2f} MPa")
    report.append(f"   - Mean: {df[config.TARGET_COLUMN].mean():.2f} MPa")
    report.append(f"   - Std: {df[config.TARGET_COLUMN].std():.2f} MPa")
    report.append(f"   - Skewness: {df[config.TARGET_COLUMN].skew():.3f}")

    # Feature insights
    report.append("\n6. KEY INSIGHTS FOR MODELING:")
    report.append("   - Cement content shows strongest positive correlation with strength")
    report.append("   - Water has negative correlation (higher water/cement ratio → lower strength)")
    report.append("   - Age effect is non-linear; log transformation recommended")
    report.append("   - No missing values; minimal preprocessing needed")
    report.append("   - Wide range of mixture compositions allows for diverse predictions")

    report.append("\n" + "="*80 + "\n")

    return "\n".join(report)


def run_eda(df, save_figures=True):
    """
    Run complete exploratory data analysis

    Parameters:
    -----------
    df : pd.DataFrame
        The concrete dataset
    save_figures : bool
        Whether to save generated figures

    Returns:
    --------
    dict : Dictionary containing EDA results and takeaways
    """
    if config.VERBOSE:
        print("\n" + "="*80)
        print("RUNNING EXPLORATORY DATA ANALYSIS")
        print("="*80 + "\n")

    # Generate all plots
    if config.VERBOSE:
        print("Generating distribution plots...")
    plot_distributions(df, save=save_figures)

    if config.VERBOSE:
        print("Generating correlation heatmap...")
    plot_correlation_heatmap(df, save=save_figures)

    if config.VERBOSE:
        print("Generating pairwise plots...")
    plot_pairplots(df, save=save_figures)

    if config.VERBOSE:
        print("Generating boxplots...")
    plot_boxplots(df, save=save_figures)

    if config.VERBOSE:
        print("Generating age relationship plot...")
    plot_age_relationship(df, save=save_figures)

    # Generate report
    report_text = generate_eda_report(df)
    if config.VERBOSE:
        print(report_text)

    # Save report
    if save_figures:
        report_file = config.FIGURES_EDA_DIR / "eda_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        if config.VERBOSE:
            print(f"Saved EDA report: {report_file}")

    if config.VERBOSE:
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS COMPLETE")
        print("="*80 + "\n")

    # Return summary
    return {
        'correlations': df[config.FEATURE_COLUMNS].corrwith(df[config.TARGET_COLUMN]).to_dict(),
        'report': report_text
    }


if __name__ == "__main__":
    # Test the module
    from data_ingestion import ingest_data
    df = ingest_data(verbose=False)
    eda_results = run_eda(df, save_figures=True)
