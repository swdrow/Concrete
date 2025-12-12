# MEM 679 Final Project Report Writing Guide

## Project: Concrete Mixture Design Under Uncertainty

This guide provides a comprehensive template for writing the final project report. The analysis pipeline has been updated to include sensitivity analyses, trust region definitions, and improved residual diagnostics based on feedback.

---

## CRITICAL: WRITING STYLE GUIDANCE

**To ensure your report reads naturally and reflects your own understanding, follow these guidelines:**

### Sentence Structure
- Mix short and long sentences. Don't fall into a rhythm.
- Start sentences differently. Avoid beginning multiple sentences with "The" or "This."
- Break up complex ideas across multiple sentences rather than using semicolons.

### Voice and Tone
- Write in first person where it makes sense: "I chose to use log(age) because..." or "We observe that..."
- Include your own reasoning: "At first I considered using total binder content, but noticed the collinearity issue when examining the correlation matrix."
- Express uncertainty naturally: "The GP appears to perform better, though the calibration issues suggest we should interpret these results cautiously."

### Specific Language Patterns to Avoid
- Don't start paragraphs with "In this section, we will discuss..."
- Avoid phrases like "It is important to note that..." or "As mentioned previously..."
- Don't use "Moreover," "Furthermore," "Additionally" to start every other sentence.
- Avoid "comprehensive," "robust," "leveraging," "utilize" - use simpler words.
- Don't write "In order to" - just write "To."

### Making It Personal
- Reference specific observations: "Looking at Figure 3, the cement correlation (r=0.498) stands out..."
- Describe your process: "After running the initial model, I noticed the negative R-squared, which led me to investigate..."
- Include trial and error: "My first approach was to use all features, but the collinearity between cement and total binder caused issues."

### Technical Writing Tips
- Define terms when first used, then use them naturally.
- Round numbers appropriately (35.82 MPa, not 35.823456 MPa).
- Use tables for comparisons, prose for explanations.
- Reference figures directly: "Figure 5 shows..." not "As can be seen in Figure 5..."

### Paragraph Structure
- Vary paragraph length. Some can be 2 sentences, others 5-6.
- Don't follow a rigid formula for every section.
- Let the content dictate structure, not a template.

---

## REPORT STRUCTURE

Your report should follow the 8-step pipeline. Below is guidance for each section.

---

## 1. INTRODUCTION (1-2 pages)

**What to cover:**
- The engineering problem: certifying concrete for structural use
- Why uncertainty matters in safety-critical decisions
- Your approach: Bayesian regression plus reliability-based certification

**Sample opening (adapt in your own words):**

> Concrete strength varies. Even with the same mixture design, batches can differ by 10-15% due to aggregate properties, mixing conditions, and curing environment. For structural applications requiring minimum 35 MPa strength, we need more than just a point prediction - we need to know how confident we are in that prediction. This project uses Bayesian regression to provide full predictive distributions, then computes P(strength >= 35 MPa) to make certification decisions.

---

## 2. DATA DESCRIPTION (1-2 pages)

**Dataset**: UCI Concrete Compressive Strength
- 1,030 samples
- 8 input features (cement, slag, fly ash, water, superplasticizer, coarse/fine aggregate, age)
- 1 output (compressive strength, 2.33-82.60 MPa)
- No missing values

**Key observations to mention:**
- Cement shows strongest correlation with strength (r=0.50)
- Water/cement ratio is physically meaningful (higher ratio = lower strength)
- Age distribution is right-skewed (many young samples)

**Figures:**
- `figures/eda/fig_target_distribution.png`
- `figures/eda/fig_correlation_matrix.png`

---

## 3. TRANSFORMATIONS & FEATURE ENGINEERING (1-2 pages)

**Three engineered features:**

1. **log(age + 1)**: Age has diminishing returns on strength. Correlation improves from 0.33 to 0.55 with log transform.

2. **water/cement ratio**: Known physical relationship. Higher ratio reduces strength.

3. **SCM fraction**: (slag + fly_ash) / (cement + slag + fly_ash)
   - *Important*: Replaced "total binder" to avoid collinearity
   - Total binder = cement + slag + fly_ash would be highly correlated with cement itself

**On standardization:**
- Z-score normalization (mean=0, std=1)
- Fit only on training data to prevent leakage

**Figures:**
- `figures/preparation/fig_engineered_features.png`
- `figures/preparation/fig_before_after_scaling.png`

---

## 4. MODEL SPECIFICATION (2-3 pages)

### Bayesian Linear Regression

**Model with intercept and hyperprior:**
```
alpha ~ Normal(y_mean, 20)     # Intercept centered at data mean
tau ~ HalfNormal(10)           # Hyperprior on shrinkage
beta | tau ~ Normal(0, tau^2)  # Coefficients depend on tau
sigma ~ HalfNormal(10)         # Noise std
y | X, alpha, beta, sigma ~ Normal(alpha + X*beta, sigma^2)
```

The intercept (alpha) is centered at the training data mean (~35.8 MPa) since features are standardized to mean zero. Without this intercept, predictions would center around zero instead of the true strength range.

The hyperprior on tau lets the data inform how much shrinkage is appropriate, rather than fixing it arbitrarily.

### Gaussian Process Regression

**Kernel with explicit noise term:**
```
k(x, x') = sigma_f^2 * RBF(x, x') + sigma_n^2 * delta(x, x')

where RBF(x, x') = exp(-0.5 * sum_i ((x_i - x'_i)^2 / ell_i^2))
```

The delta(x, x') term is the Kronecker delta - it adds noise variance only on the diagonal (observation noise).

**Figures:**
- `figures/models/fig_blr_trace.png`
- `figures/models/fig_blr_posterior.png`
- `figures/models/fig_gp_kernel_params.png`
- `figures/models/fig_prior_sensitivity.png` (NEW)

---

## 5. MODEL FITTING (1-2 pages)

### BLR Fitting
- MCMC with NUTS sampler
- 4 chains, 2000 draws each
- Check R-hat < 1.01 and ESS > 400

### Prior Sensitivity Analysis (NEW)
Tested three prior scales for beta: sigma_beta in {5, 10, 20}

Report how the posterior changes with prior scale. If results are similar across scales, the data is informative. If they differ substantially, discuss implications.

### GP Fitting
- Maximum marginal likelihood
- 10 random restarts
- Report optimized length scales (feature importance)

---

## 6. VALIDATION (2-3 pages)

### Metrics
| Metric | BLR | GP |
|--------|-----|-----|
| RMSE | 6.49 MPa | 5.28 MPa |
| MAE | 5.25 MPa | 3.73 MPa |
| R-squared | 0.836 | 0.892 |

The GP outperforms BLR across all metrics, which makes sense given its ability to capture nonlinear relationships. However, both models provide useful uncertainty estimates for decision-making.

### Calibration
Check if X% intervals contain X% of observations.

### Residual Diagnostics (NEW)

**Residuals vs Fitted Values:**
- Look for heteroscedasticity (variance changing with prediction)
- Look for patterns (suggesting missing nonlinearity)

**Residuals vs Key Covariates:**
- Check cement, water, age, water/cement ratio
- Patterns indicate the model misses something

**Figures:**
- `figures/validation/fig_calibration_plot.png`
- `figures/validation/fig_model_comparison.png`
- `figures/validation/fig_residuals_fitted_blr.png` (NEW)
- `figures/validation/fig_residuals_fitted_gp.png` (NEW)
- `figures/validation/fig_residuals_covariates_blr.png` (NEW)
- `figures/validation/fig_residuals_covariates_gp.png` (NEW)

---

## 7. DECISION ANALYSIS (2-3 pages)

### Framework
- Approve if P(strength >= 35 MPa) >= 95%
- Compute reliability from posterior predictive samples

### Threshold Sensitivity Analysis (NEW)
Test combinations:
- s_min in {30, 35, 40} MPa
- p_target in {90%, 95%, 99%}

Show how approval rates change. This helps understand how conservative the decision rule is.

### Input Sensitivity Analysis (NEW)
Perturb each ingredient by +/-5% and measure change in reliability.

| Ingredient | +5% Effect | -5% Effect |
|------------|------------|------------|
| Cement | [change] | [change] |
| Water | [change] | [change] |
| ... | ... | ... |

Which ingredients have the biggest impact? This informs quality control priorities.

### Model Sensitivity (NEW)
Do BLR and GP agree on decisions? Report the agreement rate and characterize disagreements.

### Trust Region (NEW)
Where is the model reliable? Define based on training data ranges with 10% margin.

Report: X% of test points fall within the trust region. Be cautious about predictions outside this region.

**Figures:**
- `figures/decision/fig_reliability_probabilities.png`
- `figures/decision/fig_certification_decisions.png`
- `figures/decision/fig_threshold_sensitivity.png` (NEW)
- `figures/decision/fig_input_sensitivity_blr.png` (NEW)
- `figures/decision/fig_trust_region.png` (NEW)

---

## 8. CONCLUSIONS (1 page)

Summarize:
- Which model performed better and why
- How uncertainty affected decisions
- Sensitivity of decisions to thresholds and inputs
- Practical implications for concrete certification
- Limitations and what you'd do differently

---

## COMPLETE FIGURE LIST

### EDA (4 figures)
1. fig_target_distribution.png
2. fig_feature_distributions.png
3. fig_correlation_matrix.png
4. fig_pairplot.png

### Preparation (2 figures)
5. fig_engineered_features.png
6. fig_before_after_scaling.png

### Models (4 figures)
7. fig_blr_trace.png
8. fig_blr_posterior.png
9. fig_gp_kernel_params.png
10. fig_prior_sensitivity.png (NEW)

### Validation (8 figures)
11. fig_calibration_plot.png
12. fig_ppc_blr.png
13. fig_ppc_gp.png
14. fig_model_comparison.png
15. fig_residuals_fitted_blr.png (NEW)
16. fig_residuals_fitted_gp.png (NEW)
17. fig_residuals_covariates_blr.png (NEW)
18. fig_residuals_covariates_gp.png (NEW)

### Decision (6 figures)
19. fig_reliability_probabilities.png
20. fig_certification_decisions.png
21. fig_uncertainty_decision.png
22. fig_threshold_sensitivity.png (NEW)
23. fig_input_sensitivity_blr.png (NEW)
24. fig_trust_region.png (NEW)

---

## WRITING CHECKLIST

Before submitting, verify:

- [ ] Sentences vary in length and structure
- [ ] First person used where appropriate ("I chose...", "We observe...")
- [ ] Specific figures and numbers referenced directly
- [ ] Reasoning explained, not just results stated
- [ ] Limitations discussed honestly
- [ ] Avoided generic phrases ("It is important to note...", "In this section...")
- [ ] Technical terms defined on first use
- [ ] Tables used for comparisons, prose for explanations
- [ ] Paragraphs vary in length
- [ ] Active voice predominates

---

## RUNNING THE UPDATED PIPELINE

```bash
cd /home/swd/mem679_concrete_project
python main.py
```

This runs all analyses including the new sensitivity analyses, residual diagnostics, and trust region definition.

---

## KEY EQUATIONS

**BLR with Hyperprior:**
```
tau ~ HalfNormal(10)
beta | tau ~ Normal(0, tau^2 * I)
sigma ~ HalfNormal(10)
y | X, beta, sigma ~ Normal(X*beta, sigma^2)
```

**GP Kernel:**
```
k(x, x') = sigma_f^2 * exp(-0.5 * sum_i ((x_i - x'_i)^2 / ell_i^2)) + sigma_n^2 * delta(x, x')
```

**Decision Rule:**
```
Approve if P(strength >= 35 | data) >= 0.95
```

**Reliability Computation:**
```
P(strength >= s_min) = (1/N) * sum_i I(y_pred_i >= s_min)
```

---

## REFERENCES

1. Yeh, I-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. Cement and Concrete Research, 28(12), 1797-1808.

2. Rasmussen & Williams (2006). Gaussian Processes for Machine Learning. MIT Press.

3. Gelman et al. (2013). Bayesian Data Analysis, 3rd ed. CRC Press.

---

*Updated to include sensitivity analyses, trust region, and residual diagnostics based on project feedback.*
