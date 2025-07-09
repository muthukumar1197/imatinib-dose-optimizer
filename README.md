# 🎯 Personalized Imatinib Dosing using PK Simulation + Neural Network Surrogate

This repository provides a full pipeline for simulating pharmacokinetics (PK) of imatinib using a physiologically based two-compartment model, training a neural network surrogate model, and recommending individualized doses based on predicted AUC (Area Under Curve).

## 🧠 Project Summary

- **PK Simulation**: Based on the Yi-Han Chien two-compartment model with transit absorption.
- **Neural Network Surrogate**: Trained on simulated dose–AUC data to predict exposure from patient covariates.
- **Uncertainty Estimation**: Monte Carlo dropout is used to estimate prediction uncertainty.
- **Realistic Validation**: 5 virtual patients used for case-based dose recommendations.
- **Sensitivity Analysis**: Shows how changes in age, weight, height affect dose and AUC.
- **Target AUC**: `32.5 mg·h/L` — optimal therapeutic exposure for imatinib.

---

## 📁 Files Overview

| File / Folder                       | Description |
|------------------------------------|-------------|
| `pk_model.py`                      | Simulates imatinib exposure using PBPK model |
| `Neural Network.py`        | Trains neural net, predicts dose, performs uncertainty & sensitivity analysis |
| `dose_auc_simulated_data.csv`      | Dose–AUC dataset generated from PK simulations |
| `virtual_patient_dose_predictions.csv` | Recommended doses for 5 virtual patients |
| `dose_sensitivity_analysis.csv`    | Sensitivity of dose to covariate variations |
| `dose_auc_uncertainty_patient1.png`| Uncertainty band around dose–AUC curve |
| `summary_dose_recommendations_table.png` | Tabular summary of 5 patient recommendations |
| `requirements.txt`                 | Python dependencies to run the code |

---

## 🧪 How to Use (Colab Recommended)

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

