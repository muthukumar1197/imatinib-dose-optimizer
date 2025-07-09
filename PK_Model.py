# Improved Two-Compartment Model with Transit Absorption (Yi-Han Chien Model)
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import trapezoid  # For AUC computation

# --- Baseline Parameters from Yi-Han Chien Model ---
base_params = {
    "Dose": 400,          # mg
    "CL_F": 13.2,         # Apparent clearance (L/h)
    "V1_F": 172,          # Central compartment volume (L)
    "Q_F": 3.75,          # Intercompartmental clearance (L/h)
    "V2_F": 43.6,         # Peripheral compartment volume (L)
    "Ka": 1.22,           # Absorption rate constant (h^-1)
    "MTT": 0.55,          # Mean transit time (h)
    "N": 3.5              # Number of transit compartments
}

# --- Helper Functions ---
def generate_virtual_population(n):
    np.random.seed(42)
    ages = np.random.uniform(35, 83, n)
    weights = np.random.uniform(43, 98, n)
    heights = np.random.uniform(150, 193, n)
    bmis = weights / ((heights / 100) ** 2)
    bsas = np.sqrt((heights * weights) / 3600)
    return ages, weights, heights, bmis, bsas

def apply_allometric_scaling(weight, age, bsa, base_params):
    eta_CL = np.random.normal(0, 0.2)
    eta_V1 = np.random.normal(0, 0.2)
    eta_Q = np.random.normal(0, 0.15)
    eta_V2 = np.random.normal(0, 0.15)
    eta_Ka = np.random.normal(0, 0.1)
    eta_MTT = np.random.normal(0, 0.1)

    F = 0.98  # Oral bioavailability of imatinib
    CL = (base_params["CL_F"] / F) * (weight / 70.0)**0.75 * np.exp(eta_CL)
    V1 = base_params["V1_F"] * (weight / 70.0)**1.0 * np.exp(eta_V1)
    Q = base_params["Q_F"] * (bsa / 1.8)**1.0 * np.exp(eta_Q)
    V2 = base_params["V2_F"] * (weight / 70.0)**1.0 * np.exp(eta_V2)
    Ka = base_params["Ka"] * (age / 50.0)**-0.1 * np.exp(eta_Ka)
    MTT = base_params["MTT"] * (bsa / 1.8)**0.2 * np.exp(eta_MTT)
    N = base_params["N"]

    return {"CL": CL, "V1": V1, "Q": Q, "V2": V2, "Ka": Ka, "MTT": MTT, "N": N}

def pk_model(t, y, params, Ktr, n_transit):
    A_gut = y[0]
    A_transit = y[1:n_transit + 1]
    A_central = y[n_transit + 1]
    A_peripheral = y[n_transit + 2]

    dA_gut = -params["Ka"] * A_gut
    dA_transit = np.zeros(n_transit)
    dA_transit[0] = params["Ka"] * A_gut - Ktr * A_transit[0]
    for i in range(1, n_transit):
        dA_transit[i] = Ktr * (A_transit[i - 1] - A_transit[i])

    dA_central = (
        Ktr * A_transit[-1]
        - (params["CL"] / params["V1"]) * A_central
        - (params["Q"] / params["V1"]) * A_central
        + (params["Q"] / params["V2"]) * A_peripheral
    )
    dA_peripheral = (params["Q"] / params["V1"]) * A_central - (params["Q"] / params["V2"]) * A_peripheral

    return [dA_gut] + list(dA_transit) + [dA_central, dA_peripheral]

def compute_auc(time, conc):
    return trapezoid(conc, time)

# --- Simulation Settings ---
num_patients = 100
time_points = np.linspace(0, 72, 100)
ages, weights, heights, bmis, bsas = generate_virtual_population(num_patients)

# --- Run Simulations ---
patient_records = []
summary = []

for i in range(num_patients):
    patient_covs = apply_allometric_scaling(weights[i], ages[i], bsas[i], base_params)
    n_transit = int(np.ceil(patient_covs["N"]))
    Ktr = (n_transit + 1) / patient_covs["MTT"]
    initial_conditions = [base_params["Dose"]] + [0] * n_transit + [0, 0]

    try:
        sol = solve_ivp(
            fun=lambda t, y: pk_model(t, y, patient_covs, Ktr, n_transit),
            t_span=(0, max(time_points)),
            y0=initial_conditions,
            t_eval=time_points,
            method="LSODA"
        )
    except Exception as e:
        print(f"Simulation failed for patient {i + 1}: {e}")
        continue

    conc_central = sol.y[n_transit + 1] / patient_covs["V1"]
    conc_obs = conc_central * (1 + np.random.normal(0, 0.1, len(conc_central)))

    auc = compute_auc(sol.t, conc_obs)
    cmax = np.max(conc_obs)
    tmax = sol.t[np.argmax(conc_obs)]
    summary.append((i + 1, cmax, tmax, auc))

    for t, conc in zip(sol.t, conc_obs):
        patient_records.append([
            i + 1, ages[i], weights[i], heights[i], bmis[i], bsas[i], t, conc,
            patient_covs["CL"], patient_covs["V1"], patient_covs["Q"], patient_covs["V2"],
            patient_covs["Ka"], patient_covs["MTT"]
        ])

# --- Create DataFrames ---
df = pd.DataFrame(patient_records, columns=[
    "Patient_ID", "Age", "Weight", "Height", "BMI", "BSA", "Time", "Concentration",
    "CL", "V1", "Q", "V2", "Ka", "MTT"
])

df_summary = pd.DataFrame(summary, columns=["Patient_ID", "Cmax", "Tmax", "AUC_0_72"])

# Save to CSV
df.to_csv("improved_simulated_patient_data.csv", index=False)
df_summary.to_csv("summary_exposure_metrics.csv", index=False)
print("Simulation complete. Data saved to 'improved_simulated_patient_data.csv' and exposure metrics to 'summary_exposure_metrics.csv'")
