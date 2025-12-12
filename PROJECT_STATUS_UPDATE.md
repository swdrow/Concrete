# MEM 679 Final Project - Status Update
**Date**: Wednesday, December 3rd, 2025
**Project**: Concrete Mixture Design Under Uncertainty
**Student**: [Your Name]

---

## Project Overview

This document summarizes my progress and planned approach for the final project. I've begun implementing the analysis pipeline and have made substantial progress on the data processing and modeling components.

---

## 1. TASK

**High-Level Goal**: I'm building a probabilistic regression framework to predict concrete compressive strength from mixture compositions, then using these predictions with uncertainty quantification to make reliability-based certification decisions for structural applications.

**Specific Objectives**:
- Predict compressive strength (MPa) given mixture components and age
- Quantify uncertainty in predictions using Bayesian methods
- Make certification decisions: approve mixtures where P(strength ≥ 35 MPa) ≥ 95%

**Why This Matters**: In structural engineering, under-strength concrete can lead to catastrophic failures. A probabilistic approach allows us to account for uncertainty and make conservative, safe decisions while avoiding unnecessarily rejecting viable mixtures.

---

## 2. DATA SET

**Source**: UCI Concrete Compressive Strength Dataset (1,030 samples)

**Input Features** (8 variables):
- Cement (kg/m³): [102, 540], always positive, continuous
- Blast Furnace Slag (kg/m³): [0, 359.4], non-negative, continuous
- Fly Ash (kg/m³): [0, 200.1], non-negative, continuous
- Water (kg/m³): [121.8, 247], always positive, continuous
- Superplasticizer (kg/m³): [0, 32.2], non-negative, continuous
- Coarse Aggregate (kg/m³): [801, 1145], always positive, continuous
- Fine Aggregate (kg/m³): [594, 992.6], always positive, continuous
- Age (days): [1, 365], positive integer, discrete but treated as continuous

**Output**:
- Compressive Strength (MPa): [2.33, 82.60], always positive, continuous

**Domain Characteristics**:
- All features are physically constrained to be non-negative
- No missing values in the dataset
- Some features span 2-3 orders of magnitude (e.g., age: 1-365 days)

**Data Split**:
- Training set: 824 samples (80%)
- Test set: 206 samples (20%)
- Random split with fixed seed for reproducibility

**Dimensionality**:
- Input space: ℝ⁸₊ (8-dimensional, non-negative reals)
- After feature engineering: ℝ¹¹ (will add 3 engineered features)
- Output space: ℝ₊ (positive reals)

---

## 3. TRANSFORMATION

**Pre-processing Pipeline**:

### 3.1 Feature Engineering
I'm creating 3 additional features based on domain knowledge:

1. **log(age + 1)**: Exploratory analysis shows age has a non-linear, saturating effect on strength. The log transformation better captures the diminishing returns of aging:
   ```
   log_age = log(age + 1)
   ```

2. **Water/Cement Ratio**: A fundamental parameter in concrete science - higher ratios typically reduce strength:
   ```
   w/c = water / cement
   ```

3. **Total Binder Content**: Sum of all cementitious materials:
   ```
   binder = cement + blast_furnace_slag + fly_ash
   ```

**Rationale**: These features capture known physical relationships and improve model interpretability.

### 3.2 Standardization
I'm applying z-score standardization to all features:
```
x_scaled = (x - μ_train) / σ_train
```

**Why**:
- Puts all features on the same scale for Bayesian Linear Regression (makes prior specification easier)
- Improves numerical stability in optimization
- After standardization: x ∈ ℝ, typically |x| < 3

**Critical**: Fit scaler on training data only, then transform both train and test to avoid data leakage.

### 3.3 Target Variable
I'm keeping the target (strength) in its original units (MPa) rather than log-transforming because:
- Values don't span many orders of magnitude (only ~40×)
- Loss functions and prediction intervals are more interpretable in original units
- Normal likelihood is reasonable for this range

---

## 4. MODEL SETUP

I'm implementing two models: a baseline Bayesian Linear Regression and an advanced Gaussian Process.

### 4.1 Bayesian Linear Regression (Baseline)

**Model Specification**:
```
y = X β + ε
```

**Prior Distributions**:
```
β ~ Normal(0, σ²_β I)     where σ_β = 10
σ ~ HalfNormal(σ_prior)   where σ_prior = 10
```

**Likelihood**:
```
y | X, β, σ ~ Normal(X β, σ²)
```

**Parameters**:
- β ∈ ℝ¹¹: regression coefficients (11-dimensional after feature engineering)
- σ ∈ ℝ₊: observation noise standard deviation

**Hyperparameters**:
- σ_β = 10: Prior standard deviation for coefficients (fixed, not learned)
- σ_prior = 10: Scale parameter for noise prior (fixed)

**Assumptions**:
- Linear relationship between standardized features and strength
- Gaussian noise with constant variance
- Weak prior information (σ_β = 10 is large relative to standardized scale)

**Why This Prior**: After standardization, coefficients typically have |β| < 3. A prior with σ_β = 10 is weakly informative - it regularizes extreme values but lets the data dominate.

### 4.2 Gaussian Process Regression (Advanced)

**Model Specification**:
```
f ~ GP(0, k(x, x'))
y = f(x) + ε
ε ~ Normal(0, σ²_n)
```

**Kernel Function**:
```
k(x, x') = σ²_f · k_RBF(x, x') + σ²_n · δ(x, x')

k_RBF(x, x') = exp(-½ Σᵢ (xᵢ - x'ᵢ)² / ℓᵢ²)
```

**Hyperparameters** (to be learned):
- σ²_f ∈ ℝ₊: Signal variance (amplitude of function variations)
- ℓᵢ ∈ ℝ₊¹¹: Length scales, one per feature (ARD - Automatic Relevance Determination)
- σ²_n ∈ ℝ₊: Noise variance

**Assumptions**:
- Smooth function relating inputs to outputs
- Different features may have different relevance (via ARD)
- Stationary covariance structure

**Why GP**: Captures non-linear relationships without specifying functional form. ARD automatically performs feature selection by learning which features matter (small length scale) vs. don't matter (large length scale).

---

## 5. FIT THE MODEL

### 5.1 Bayesian Linear Regression Fitting

**Method**: Markov Chain Monte Carlo (MCMC) using the No-U-Turn Sampler (NUTS)

**Implementation**:
- 4 chains running in parallel
- 2,000 draws per chain (8,000 total posterior samples)
- 1,000 tuning/warmup samples per chain (discarded)
- Target acceptance rate: 0.95

**Why MCMC**: Gets full posterior distribution p(β, σ | data), not just point estimates. This allows proper uncertainty quantification in predictions.

**Convergence Diagnostics** I'll check:
- R̂ (R-hat) < 1.01 for all parameters (measures chain convergence)
- Effective Sample Size (ESS) > 400 for all parameters
- Visual inspection of trace plots for mixing

### 5.2 Gaussian Process Fitting

**Method**: Maximum Likelihood Estimation of hyperparameters

**Implementation**:
- Optimize log marginal likelihood: log p(y | X, θ) where θ = {σ²_f, ℓ₁...ℓ₁₁, σ²_n}
- Use L-BFGS-B optimizer with multiple random restarts (10 restarts)
- Restarts help avoid local optima

**Why MLE for GP**:
- Computing full Bayesian posterior over GP hyperparameters is computationally expensive
- Marginal likelihood provides principled way to select hyperparameters
- Still get full posterior predictive distribution p(y* | x*, data) at test points

**Bounds on Hyperparameters**:
- σ²_f: [10⁻³, 10³] (prevents numerical issues)
- ℓᵢ: [10⁻², 10²] (reasonable range after standardization)
- σ²_n: [10⁻¹⁰, 10⁰] (small noise floor)

---

## 6. VALIDATION

### 6.1 Performance Metrics

**Pointwise Accuracy** (on test set):
- RMSE: √(1/n Σ(yᵢ - ŷᵢ)²) - penalizes large errors
- MAE: 1/n Σ|yᵢ - ŷᵢ| - more robust to outliers
- R²: 1 - (SS_res / SS_tot) - fraction of variance explained

**Success Criteria**: I'm aiming for R² > 0.85, which would indicate the model captures most of the variation in concrete strength.

### 6.2 Uncertainty Calibration

**Calibration Check**: For each confidence level α, check if α% of observations fall within α% prediction intervals.

For example, 90% prediction intervals should contain ~90% of test observations.

**Metric**: Calibration error = |observed coverage - expected coverage|

I'll compute this for multiple levels: 50%, 68%, 90%, 95%

### 6.3 Posterior Predictive Checks (PPC)

**Visual Checks**:
1. **Predictive distribution overlay**: Plot observed test data against samples from posterior predictive
2. **Residual analysis**: Check if residuals are approximately Normal with mean 0
3. **Prediction interval coverage**: Visualize which points fall outside intervals

**Quantitative PPC**:
- Compare moments: E[y_pred] vs E[y_obs], Var[y_pred] vs Var[y_obs]
- Test statistics: Compare distribution of actual residuals to posterior predictive residuals

### 6.4 Model Comparison

Compare BLR vs GP on:
- Predictive accuracy (RMSE, MAE, R²)
- Calibration quality
- Computational cost
- Interpretability

---

## 7. PREDICTION

### 7.1 Quantities of Interest (QoIs)

For each test mixture, I'm predicting:
1. **Compressive strength** (point estimate and full distribution)
2. **Reliability probability**: P(strength ≥ s_min | mixture, data)

### 7.2 Prediction Approach

**For BLR**:
```
1. Sample (β, σ) from posterior ~ p(β, σ | data)  [use MCMC samples]
2. For each sample:
   y_pred ~ Normal(X_test β, σ²)
3. Aggregate samples to get posterior predictive distribution
```

**For GP**:
```
Posterior predictive is analytically available:
p(y* | x*, X, y) = Normal(μ*, σ*²)

where:
μ* = k(x*, X) K⁻¹ y
σ*² = k(x*, x*) - k(x*, X) K⁻¹ k(X, x*) + σ²_n
```

I'll sample from these distributions to get 1,000 posterior predictive samples per test point.

### 7.3 Uncertainty Reporting

For each prediction, I'll report:
- **Point estimate**: Posterior predictive mean
- **Uncertainty**: Standard deviation of posterior predictive
- **Intervals**:
  - 50% credible interval (interquartile range)
  - 90% credible interval
  - 95% credible interval

**Visualization**: Scatter plot of predicted vs. actual with error bars showing 90% intervals

### 7.4 Uncertainty Decomposition

I'll distinguish between:
- **Epistemic uncertainty**: Uncertainty about the model (reducible with more data)
- **Aleatoric uncertainty**: Inherent noise in observations (irreducible)

For BLR, this can be separated as:
- Epistemic: Var[X_test β] (variation due to uncertainty in β)
- Aleatoric: σ² (observation noise)

---

## 8. DECISION

### 8.1 Decision Framework

**Engineering Problem**: Certify concrete mixtures for structural use requiring minimum strength.

**Decision Rule**: Approve mixture if P(strength ≥ s_min) ≥ p_target

**Parameters**:
- s_min = 35 MPa (common requirement for structural concrete)
- p_target = 0.95 (95% reliability - conservative safety factor)

### 8.2 Computing Reliability

For each test mixture, use posterior predictive samples:
```
P(strength ≥ 35 | mixture, data) = 1/N Σᵢ I(y_pred^(i) ≥ 35)
```
where y_pred^(i) are samples from posterior predictive.

### 8.3 Decision Incorporation of Uncertainty

**Key Insight**: High uncertainty reduces reliability probability even if expected strength is high.

Example:
- Mixture A: E[strength] = 40 MPa, SD = 2 MPa  →  P(≥35) ≈ 99%  ✓ Approve
- Mixture B: E[strength] = 40 MPa, SD = 8 MPa  →  P(≥35) ≈ 73%  ✗ Reject

This is appropriate for safety-critical applications where we must account for worst-case scenarios.

### 8.4 Decision Outcomes

I'll classify each test mixture as:
- **Approved** (green): P(strength ≥ 35) ≥ 95%
- **Rejected** (red): P(strength ≥ 35) < 95%

And analyze:
- Approval rate overall
- How model uncertainty affects approval
- Characteristics of approved vs. rejected mixtures

### 8.5 Cost-Reliability Tradeoff

I'll explore the tradeoff between mixture cost and reliability:
- Cement is the most expensive ingredient
- Higher cement generally increases strength and reliability
- Can optimize: minimize cost subject to reliability ≥ 95%

### 8.6 Sensitivity Analysis

Examine how decisions change with:
- Different thresholds (s_min = 30, 35, 40 MPa)
- Different reliability targets (p_target = 0.90, 0.95, 0.99)
- Model choice (BLR vs GP)

---

## Current Progress Summary

**Completed**:
- ✅ Data ingestion and initial exploration
- ✅ Feature engineering implementation
- ✅ Data preprocessing and train/test split
- ✅ BLR model implementation in PyMC
- ✅ GP model implementation in scikit-learn
- ✅ Basic MCMC sampling and convergence checks

**In Progress**:
- 🔄 Finalizing prediction pipeline
- 🔄 Implementing validation metrics
- 🔄 Creating visualizations for report

**Next Steps**:
- Run complete pipeline on full dataset
- Generate all figures for report
- Perform posterior predictive checks
- Execute decision analysis
- Write final report

---

## Questions for Feedback

1. Is my choice of s_min = 35 MPa and p_target = 95% reasonable for structural applications?

2. Should I consider log-transforming the target variable given it's always positive, or is keeping it in original units better for interpretability?

3. For the GP, would you recommend trying other kernels (e.g., Matérn) or is RBF with ARD sufficient?

4. Any suggestions for additional validation metrics or posterior predictive checks I should include?

---

## References

- Yeh, I-C. "Modeling of strength of high-performance concrete using artificial neural networks." Cement and Concrete Research 28.12 (1998): 1797-1808.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
- Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.
