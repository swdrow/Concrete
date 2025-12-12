"""
Configuration file for MEM 679 Concrete Mixture Design Project
Contains all hyperparameters, paths, and constants
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Figure subdirectories
FIGURES_EDA_DIR = FIGURES_DIR / "eda"
FIGURES_PREP_DIR = FIGURES_DIR / "preparation"
FIGURES_MODELS_DIR = FIGURES_DIR / "models"
FIGURES_VALIDATION_DIR = FIGURES_DIR / "validation"
FIGURES_DECISION_DIR = FIGURES_DIR / "decision"

# Results subdirectories
METRICS_DIR = RESULTS_DIR / "metrics"
POSTERIORS_DIR = RESULTS_DIR / "posteriors"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# ============================================================================
# DATA PARAMETERS
# ============================================================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
RAW_DATA_FILE = RAW_DATA_DIR / "Concrete_Data.xls"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "concrete_processed.csv"

# Column names from the UCI dataset
COLUMN_NAMES = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "compressive_strength"
]

TARGET_COLUMN = "compressive_strength"
FEATURE_COLUMNS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age"
]

# ============================================================================
# DATA SPLITTING PARAMETERS
# ============================================================================
TEST_SIZE = 0.2
RANDOM_SEED = 42
N_FOLDS = 5  # For cross-validation

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
ENGINEER_FEATURES = True
ENGINEERED_FEATURES = [
    "log_age",              # log(age + 1)
    "water_cement_ratio",   # water / cement
    "scm_fraction"          # (slag + fly_ash) / (cement + slag + fly_ash)
    # Note: Replaced "binder_content" with "scm_fraction" to avoid collinearity
    # binder_content is highly correlated with cement (sum of cement + components)
]

# ============================================================================
# BAYESIAN LINEAR REGRESSION PARAMETERS
# ============================================================================
BLR_CONFIG = {
    "prior_beta_std": 10.0,      # Standard deviation for weight priors (baseline)
    "prior_sigma_std": 10.0,     # Standard deviation for noise prior
    "use_hyperprior": True,      # Whether to use hyperprior on tau (shrinkage)
    "tau_prior_std": 10.0,       # Scale for HalfNormal prior on tau
    "n_draws": 2000,             # Number of MCMC samples per chain
    "n_tune": 1000,              # Number of tuning samples
    "n_chains": 4,               # Number of MCMC chains
    "target_accept": 0.95,       # Target acceptance rate for NUTS
    "random_seed": RANDOM_SEED
}

# Prior sensitivity analysis: test different beta prior scales
PRIOR_SENSITIVITY_SCALES = [5.0, 10.0, 20.0]

# ============================================================================
# GAUSSIAN PROCESS PARAMETERS
# ============================================================================
GP_CONFIG = {
    "kernel_type": "RBF",        # RBF (Radial Basis Function) kernel
    "use_ard": True,             # Automatic Relevance Determination
    "n_restarts_optimizer": 10,  # Number of random restarts for optimization
    "alpha": 0.1,                # Noise regularization (variance of observation noise)
    "random_seed": RANDOM_SEED
}

# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================
VALIDATION_CONFIG = {
    "confidence_levels": [0.50, 0.68, 0.90, 0.95],  # Credible interval levels
    "n_posterior_samples": 1000,  # For posterior predictive checks
    "calibration_bins": 20        # Number of bins for calibration plot
}

# ============================================================================
# DECISION ANALYSIS PARAMETERS
# ============================================================================
DECISION_CONFIG = {
    "s_min": 35.0,               # Minimum strength threshold (MPa)
    "p_target": 0.95,            # Target reliability (95%)
    "cost_weights": {            # Relative cost per kg
        "cement": 1.0,
        "blast_furnace_slag": 0.5,
        "fly_ash": 0.3,
        "water": 0.01,
        "superplasticizer": 5.0,
        "coarse_aggregate": 0.05,
        "fine_aggregate": 0.05
    }
}

# ============================================================================
# SENSITIVITY ANALYSIS PARAMETERS
# ============================================================================
SENSITIVITY_CONFIG = {
    # Input sensitivity: perturbation factors
    "input_perturbation": 0.05,  # ±5% perturbation on ingredients
    "ingredients_to_perturb": [
        "cement", "water", "blast_furnace_slag", "fly_ash",
        "superplasticizer", "coarse_aggregate", "fine_aggregate"
    ],
    # Threshold sensitivity
    "threshold_values": [30.0, 35.0, 40.0],  # s_min values to test
    "reliability_targets": [0.90, 0.95, 0.99],  # p_target values to test
}

# ============================================================================
# TRUST REGION / MODEL VALIDITY PARAMETERS
# ============================================================================
# Trust region defines where model predictions are reliable
# Based on convex hull of training data or feature ranges
TRUST_REGION_CONFIG = {
    "method": "feature_ranges",  # "feature_ranges" or "mahalanobis"
    "range_margin": 0.1,         # 10% margin beyond observed ranges
    "mahalanobis_threshold": 3.0  # Chi-squared threshold for Mahalanobis
}

# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================
PLOT_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "figure_dpi": 300,
    "figure_format": "png",
    "font_size": 10,
    "title_size": 12,
    "label_size": 10,
    "legend_size": 9,
    "color_palette": "Set2"
}

# ============================================================================
# MISC PARAMETERS
# ============================================================================
VERBOSE = True  # Print progress messages
SAVE_INTERMEDIATE = True  # Save intermediate results
