# MEM 679 Final Project: Concrete Mixture Design Under Uncertainty

## Project Overview

This project implements a Bayesian approach to concrete mixture design, using probabilistic regression models to predict compressive strength and make reliability-based certification decisions. The analysis follows the complete data science pipeline from data ingestion through decision-making under uncertainty.

**Engineering Question**: Can we reliably predict concrete compressive strength from mixture compositions and quantify uncertainty to make safe design decisions?

**Dataset**: UCI Concrete Compressive Strength Dataset (1,030 samples, 8 features, 1 target)

**Models**:
- Bayesian Linear Regression (baseline)
- Gaussian Process Regression (advanced)

**Decision Task**: Certify concrete mixtures that meet P(strength ≥ 35 MPa) ≥ 95%

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Navigate to the project directory:
```bash
cd /home/swd/mem679_concrete_project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

**Note**: PyMC installation may take several minutes as it includes JAX and other dependencies.

---

## Usage

### Running the Complete Pipeline

To execute the entire analysis pipeline:

```bash
python main.py
```

This will:
1. Download and load the concrete dataset
2. Perform exploratory data analysis
3. Engineer features and split data
4. Fit Bayesian Linear Regression and Gaussian Process models
5. Generate predictions with uncertainty quantification
6. Validate models with calibration checks
7. Perform reliability-based decision analysis

**Expected runtime**: 5-15 minutes depending on hardware

### Output

The pipeline generates:
- **Figures** (~18 plots): Saved in `figures/` subdirectories
- **Processed Data**: Training/test splits in `data/processed/`
- **Fitted Models**: Posterior samples and GP models in `results/posteriors/`
- **Metrics**: Performance metrics and calibration results in `results/metrics/`
- **Decisions**: Certification decisions in `results/decisions/`

---

## Project Structure

```
mem679_concrete_project/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration parameters
├── main.py                    # Main execution script
│
├── data/
│   ├── raw/                   # Downloaded dataset
│   └── processed/             # Prepared data files
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── data_ingestion.py      # Data loading
│   ├── exploratory_analysis.py # EDA plots and statistics
│   ├── data_preparation.py    # Feature engineering, scaling
│   ├── models/
│   │   ├── bayesian_linear_regression.py
│   │   └── gaussian_process.py
│   ├── fitting.py             # Model training
│   ├── prediction.py          # Posterior predictive inference
│   ├── validation.py          # Metrics and calibration
│   └── decision.py            # Reliability-based decisions
│
├── figures/                   # Generated plots
│   ├── eda/                   # Exploratory analysis
│   ├── preparation/           # Data transformation
│   ├── models/                # Model diagnostics
│   ├── validation/            # Performance evaluation
│   └── decision/              # Decision analysis
│
├── results/                   # Numerical results
│   ├── metrics/               # Performance metrics CSV
│   ├── posteriors/            # Saved model objects
│   ├── predictions/           # Prediction outputs
│   └── decisions/             # Decision results
│
└── notebooks/                 # (Optional) Jupyter notebooks
```

---

## Data Analysis Pipeline

The project follows the standard data science pipeline as required:

### 1. Ingest
- Downloads UCI Concrete dataset
- Loads into pandas DataFrame
- Performs initial inspection

### 2. Split
- 80/20 train-test split
- Stratified sampling
- 5-fold cross-validation setup

### 3. Transform
- Feature engineering: log(age), water/cement ratio, binder content
- Standardization (mean=0, std=1)
- Before/after visualizations

### 4. Define
- **BLR**: β ~ Normal(0, 10²), σ ~ HalfNormal(10)
- **GP**: RBF kernel with ARD, optimized hyperparameters

### 5. Fit
- BLR: MCMC sampling (4 chains × 2000 draws)
- GP: Marginal likelihood optimization
- Convergence diagnostics (R-hat, ESS)

### 6. Predict
- Posterior predictive distributions
- Uncertainty quantification (epistemic + aleatoric)
- Credible intervals

### 7. Validate
- Metrics: RMSE, MAE, R²
- Calibration: Do 90% intervals contain 90% of observations?
- Posterior predictive checks

### 8. Decide
- Reliability: P(strength ≥ 35 MPa)
- Certification: Approve if P ≥ 95%
- Cost-reliability tradeoffs

---

## Configuration

Key parameters can be modified in `config.py`:

```python
# Decision thresholds
DECISION_CONFIG = {
    's_min': 35.0,        # Minimum strength (MPa)
    'p_target': 0.95,     # Target reliability
}

# Model hyperparameters
BLR_CONFIG = {
    'n_draws': 2000,      # MCMC samples
    'n_chains': 4,        # Parallel chains
}

GP_CONFIG = {
    'use_ard': True,      # Feature-specific length scales
}
```

---

## Key Results (Example)

After running the pipeline, you should see output similar to:

```
Model Performance:
  BLR - RMSE: 5.234, MAE: 3.876, R²: 0.912
  GP  - RMSE: 4.891, MAE: 3.542, R²: 0.927

Calibration:
  90% intervals contain 91.2% (BLR), 89.8% (GP) of observations

Certification Results:
  BLR: 168/206 mixtures approved (81.6%)
  GP:  172/206 mixtures approved (83.5%)
```

---

## Generated Figures

The pipeline produces approximately 18 figures organized by analysis stage:

### Exploratory Data Analysis (5 figures)
1. `fig_distributions.png` - Feature and target distributions
2. `fig_correlation_heatmap.png` - Correlation matrix
3. `fig_pairplots.png` - Features vs. strength scatter plots
4. `fig_boxplots.png` - Outlier detection
5. `fig_age_relationship.png` - Age transformation analysis

### Data Preparation (2 figures)
6. `fig_before_after_scaling.png` - Standardization effect
7. `fig_engineered_features.png` - New feature relationships

### Model Diagnostics (3 figures)
8. `fig_blr_trace.png` - MCMC convergence
9. `fig_blr_posterior.png` - Coefficient posteriors
10. `fig_gp_kernel_params.png` - GP hyperparameters

### Validation (5 figures)
11. `fig_predictions.png` - Predicted vs. actual
12. `fig_residuals.png` - Residual analysis
13. `fig_calibration_plot.png` - Interval coverage
14. `fig_ppc_blr.png` - BLR posterior predictive check
15. `fig_ppc_gp.png` - GP posterior predictive check
16. `fig_model_comparison.png` - Side-by-side metrics

### Decision Analysis (3 figures)
17. `fig_reliability_probabilities.png` - P(strength ≥ 35) distribution
18. `fig_certification_decisions.png` - Approved vs. rejected mixtures
19. `fig_uncertainty_decision.png` - How uncertainty affects decisions

---

## Troubleshooting

### Installation Issues

**Problem**: PyMC installation fails
```bash
# Try installing with conda instead
conda install -c conda-forge pymc
```

**Problem**: NumPy/JAX compatibility errors
```bash
# Update to compatible versions
pip install --upgrade numpy jax jaxlib
```

### Runtime Issues

**Problem**: MCMC sampling is very slow
- Reduce `n_draws` in `config.py` (e.g., to 1000)
- Reduce `n_chains` to 2

**Problem**: Out of memory during GP fitting
- Dataset is small enough that this shouldn't occur
- If it does, check for other memory-intensive processes

**Problem**: Figures not displaying properly
- Ensure matplotlib backend is set correctly
- Figures are saved to disk regardless of display

---

## Extending the Project

### Adding a New Model

1. Create a new file in `src/models/`
2. Implement `build_model()` and `fit_model()` functions
3. Add prediction logic to `src/prediction.py`
4. Update `src/fitting.py` to include the new model

### Changing Decision Criteria

Modify `config.py`:
```python
DECISION_CONFIG = {
    's_min': 40.0,        # Higher strength requirement
    'p_target': 0.99,     # Stricter reliability
}
```

### Using Different Data

1. Place your data in `data/raw/`
2. Update column names in `config.py`
3. Modify `src/data_ingestion.py` to load your format

---

## References

- **Dataset**: Yeh, I-Cheng. "Concrete Compressive Strength." UCI Machine Learning Repository, 2007.
- **PyMC**: Salvatier et al. "Probabilistic programming in Python using PyMC3." PeerJ Computer Science 2:e55, 2016.
- **Gaussian Processes**: Rasmussen & Williams. "Gaussian Processes for Machine Learning." MIT Press, 2006.

---

## Author

MEM 679 Student
Duke University
December 2025

---

## License

This project is for educational purposes as part of the MEM 679 course requirement.
