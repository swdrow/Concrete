#!/usr/bin/env python3
"""
MEM 679 Final Project: Concrete Mixture Design Under Uncertainty
Main execution script

This script runs the complete data analysis pipeline:
Ingest → Split → Transform → Define → Fit → Predict → Validate → Decide

Author: MEM 679 Student
Date: December 2025
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from src.data_ingestion import ingest_data
from src.exploratory_analysis import run_eda
from src.data_preparation import prepare_data
from src.fitting import fit_all_models, run_prior_sensitivity_analysis
from src.prediction import make_all_predictions
from src.validation import validate_models
from src.decision import analyze_decisions


def print_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """
    Main pipeline execution
    """
    start_time = time.time()

    print_header("MEM 679 FINAL PROJECT")
    print("Concrete Mixture Design Under Uncertainty")
    print("Bayesian Regression and Decision Analysis\n")

    try:
        # ====================================================================
        # 1. DATA INGESTION
        # ====================================================================
        print_header("STEP 1: DATA INGESTION")
        print("Loading UCI Concrete Compressive Strength Dataset...")

        df = ingest_data()

        print(f"\n✓ Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")

        # ====================================================================
        # 2. EXPLORATORY DATA ANALYSIS
        # ====================================================================
        print_header("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("Generating exploratory visualizations and statistics...")

        eda_results = run_eda(df, save_figures=True)

        print("\n✓ Exploratory analysis complete")
        print(f"  Generated {len(list(config.FIGURES_EDA_DIR.glob('*.png')))} figures")

        # ====================================================================
        # 3. DATA PREPARATION
        # ====================================================================
        print_header("STEP 3: DATA PREPARATION")
        print("Feature engineering, scaling, and train/test split...")

        prepared_data = prepare_data(df, save_figures=True)

        print("\n✓ Data preparation complete")
        print(f"  Training set: {len(prepared_data['X_train'])} samples")
        print(f"  Test set: {len(prepared_data['X_test'])} samples")
        print(f"  Number of features: {len(prepared_data['X_train'].columns)}")

        # ====================================================================
        # 4. MODEL DEFINITION & FITTING
        # ====================================================================
        print_header("STEP 4: MODEL DEFINITION & FITTING")
        print("Fitting Bayesian Linear Regression and Gaussian Process models...")

        fitted_models = fit_all_models(prepared_data, save_models=True, save_figures=True)

        print("\n✓ Model fitting complete")
        print(f"  Fitted models: {list(fitted_models.keys())}")

        # ====================================================================
        # 4b. PRIOR SENSITIVITY ANALYSIS
        # ====================================================================
        print_header("STEP 4b: PRIOR SENSITIVITY ANALYSIS")
        print("Testing BLR with different prior scales (5, 10, 20)...")

        prior_sensitivity_results = run_prior_sensitivity_analysis(
            prepared_data['X_train_scaled'],
            prepared_data['y_train'],
            save_figures=True
        )

        print("\n✓ Prior sensitivity analysis complete")
        print(f"  Tested {len(prior_sensitivity_results)} prior scale configurations")

        # ====================================================================
        # 5. PREDICTION
        # ====================================================================
        print_header("STEP 5: PREDICTION")
        print("Generating posterior predictive distributions...")

        predictions = make_all_predictions(fitted_models, prepared_data, save_figures=True)

        print("\n✓ Predictions generated")
        print(f"  BLR mean prediction: {predictions['blr']['mean'].mean():.2f} ± {predictions['blr']['std'].mean():.2f} MPa")
        print(f"  GP mean prediction: {predictions['gp']['mean'].mean():.2f} ± {predictions['gp']['std'].mean():.2f} MPa")

        # ====================================================================
        # 6. VALIDATION
        # ====================================================================
        print_header("STEP 6: VALIDATION & UNCERTAINTY QUANTIFICATION")
        print("Computing metrics, calibration, and posterior predictive checks...")

        validation_results = validate_models(predictions, prepared_data, save_figures=True)

        print("\n✓ Validation complete")
        print("\nModel Performance:")
        print(f"  BLR - RMSE: {validation_results['blr']['metrics']['RMSE']:.3f}, "
              f"MAE: {validation_results['blr']['metrics']['MAE']:.3f}, "
              f"R²: {validation_results['blr']['metrics']['R2']:.3f}")
        print(f"  GP  - RMSE: {validation_results['gp']['metrics']['RMSE']:.3f}, "
              f"MAE: {validation_results['gp']['metrics']['MAE']:.3f}, "
              f"R²: {validation_results['gp']['metrics']['R2']:.3f}")

        # ====================================================================
        # 7. DECISION ANALYSIS
        # ====================================================================
        print_header("STEP 7: DECISION ANALYSIS")
        print(f"Reliability-based certification decisions...")
        print(f"  Threshold: s_min = {config.DECISION_CONFIG['s_min']} MPa")
        print(f"  Target reliability: {config.DECISION_CONFIG['p_target']:.0%}")

        decision_results = analyze_decisions(predictions, prepared_data, save_figures=True)

        print("\n✓ Decision analysis complete")
        print(f"\nCertification Results:")
        print(f"  BLR: {decision_results['blr']['decisions']['approved'].sum()} / "
              f"{len(decision_results['blr']['decisions'])} mixtures approved "
              f"({100*decision_results['blr']['decisions']['approved'].mean():.1f}%)")
        print(f"  GP:  {decision_results['gp']['decisions']['approved'].sum()} / "
              f"{len(decision_results['gp']['decisions'])} mixtures approved "
              f"({100*decision_results['gp']['decisions']['approved'].mean():.1f}%)")

        # Print trust region info
        if 'in_trust_region' in decision_results:
            in_region = decision_results['in_trust_region']
            print(f"\nTrust Region Analysis:")
            print(f"  Test points in trust region: {in_region.sum()}/{len(in_region)} "
                  f"({100*in_region.mean():.1f}%)")

        # ====================================================================
        # SUMMARY
        # ====================================================================
        elapsed_time = time.time() - start_time

        print_header("PIPELINE COMPLETE")

        print("✓ All steps completed successfully!\n")

        print("Generated Outputs:")
        print(f"  Figures: {len(list(config.FIGURES_DIR.rglob('*.png')))} plots")
        print(f"  Data files: {len(list(config.PROCESSED_DATA_DIR.glob('*.csv')))} CSV files")
        print(f"  Models: {len(list(config.POSTERIORS_DIR.glob('*')))} saved models")
        print(f"  Metrics: {len(list(config.METRICS_DIR.glob('*.csv')))} metric files")

        print(f"\nTotal execution time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

        print("\n" + "="*80)
        print("Next steps:")
        print("1. Review figures in the 'figures/' directory")
        print("2. Examine metrics in 'results/metrics/'")
        print("3. Write the project report using the generated results")
        print("="*80 + "\n")

        return {
            'data': df,
            'prepared_data': prepared_data,
            'fitted_models': fitted_models,
            'prior_sensitivity': prior_sensitivity_results,
            'predictions': predictions,
            'validation_results': validation_results,
            'decision_results': decision_results
        }

    except Exception as e:
        print(f"\n❌ Error in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()
