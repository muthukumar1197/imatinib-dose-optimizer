# Imatinib Dose Optimization using PK Simulation and Neural Network Surrogate

This repository contains the full pipeline used to simulate pharmacokinetic (PK) profiles of imatinib using a two-compartment model (Yi-Han Chien), train a neural network surrogate model, and predict optimal personalized doses based on simulated AUC targets.

## 🚀 Project Overview

- **PK Model**: Simulates AUC using a physiologically-based two-compartment model with transit absorption.
- **Neural Network Model**: Trained to predict AUC from patient features + dose.
- **Validation**: Five virtual patients + sensitivity analysis.
- **Uncertainty**: Monte Carlo dropout for AUC prediction uncertainty.
- **Visualization**: Dose–AUC plots, heatmaps, summary tables.

## 📂 Files

- `pk_model.py`: Generates virtual patient dose–AUC data.
- `Neural Network.py`: Neural network model with dose prediction, uncertainty estimation, and plots.
- `dose_auc_simulated_data.csv`: Output from PK model (can be regenerated).
- `summary_exposure_metrics.csv`: Summary AUC metrics.
- `virtual_patient_dose_predictions.csv`: Dose recommendations.
- `dose_sensitivity_analysis.csv`: Results of sensitivity testing.

## 💾 Requirements

Install with pip:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tensorflow
