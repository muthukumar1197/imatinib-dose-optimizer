# full_dose_auc_pipeline.py
# This script runs the complete pipeline: model training, 5 patient validation, sensitivity analysis, and AUC uncertainty plotting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# --- Load Input Data ---
df = pd.read_csv("dose_auc_simulated_data.csv")

# --- Derive Summary Exposure Metrics ---
if 'AUC' in df.columns:
    df_summary_auc = (
        df.groupby("Patient_ID")
        .apply(lambda g: pd.Series({
            "Cmax": np.nan,
            "Tmax": np.nan,
            "AUC_0_72": g["AUC"].values[0]  # One AUC per dose per patient
        }))
        .reset_index()
    )
    df_summary_auc.to_csv("summary_exposure_metrics.csv", index=False)
    print("✅ Saved: summary_exposure_metrics.csv")
else:
    raise ValueError("No 'AUC' column found in input data. Ensure your CSV contains dose–AUC data.")

# --- Prepare Dataset ---
df_input = df[['Patient_ID', 'Age', 'Weight', 'Height', 'BMI', 'Dose']].copy()
merged = pd.merge(df_input, df_summary_auc[['Patient_ID', 'AUC_0_72']], on='Patient_ID')

X = merged[['Age', 'Weight', 'Height', 'BMI', 'Dose']].values
y = merged['AUC_0_72'].values

# --- Normalize Features ---
scaler = StandardScaler()
X[:, :-1] = scaler.fit_transform(X[:, :-1])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model with MC Dropout ---
class MCDropoutModel(Model):
    def __init__(self, input_dim, dropout_rate=0.2):
        super(MCDropoutModel, self).__init__()
        self.d1 = Dense(128, activation='relu')
        self.drop1 = Dropout(dropout_rate)
        self.d2 = Dense(64, activation='relu')
        self.drop2 = Dropout(dropout_rate)
        self.out = Dense(1)

    def call(self, x, training=False):
        x = self.d1(x)
        x = self.drop1(x, training=training)
        x = self.d2(x)
        x = self.drop2(x, training=training)
        return self.out(x)

model = MCDropoutModel(input_dim=X.shape[1])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# --- MC Dropout Prediction ---
def predict_with_uncertainty(f_model, X, n_iter=50):
    preds = np.array([f_model(X, training=True).numpy().flatten() for _ in range(n_iter)])
    return preds.mean(axis=0), preds.std(axis=0)

# --- Five Realistic Virtual Patients ---
virtual_patients = [
    {"Patient_ID": 1, "Age": 45, "Weight": 65, "Height": 165},
    {"Patient_ID": 2, "Age": 62, "Weight": 80, "Height": 170},
    {"Patient_ID": 3, "Age": 55, "Weight": 70, "Height": 160},
    {"Patient_ID": 4, "Age": 72, "Weight": 60, "Height": 158},
    {"Patient_ID": 5, "Age": 50, "Weight": 85, "Height": 175},
]

for p in virtual_patients:
    p["BMI"] = p["Weight"] / ((p["Height"] / 100) ** 2)

def recommend_dose_curve_uncertainty(patient, model, scaler, doses=np.arange(100, 701, 100), n_iter=50):
    X_dose = []
    for d in doses:
        x = [patient['Age'], patient['Weight'], patient['Height'], patient['BMI']]
        x = scaler.transform([x])[0]
        X_dose.append(np.append(x, d))
    X_dose = np.array(X_dose)
    mean_preds, std_preds = predict_with_uncertainty(model, X_dose, n_iter)
    best_idx = np.argmin(np.abs(mean_preds - 32.5))
    return doses, mean_preds, std_preds, doses[best_idx], mean_preds[best_idx]

# --- Run Prediction + Sensitivity Analysis ---
sensitivity_records = []
summary_recommendations = []

for patient in virtual_patients:
    doses, auc_mean, auc_std, rec_dose, rec_auc = recommend_dose_curve_uncertainty(patient, model, scaler)
    summary_recommendations.append({"Patient_ID": patient['Patient_ID'], "Recommended Dose": rec_dose, "Predicted AUC": round(rec_auc, 2)})

    for param in ["Age", "Weight", "Height"]:
        for pct in [-0.1, 0.1]:
            modified = patient.copy()
            modified[param] = modified[param] * (1 + pct)
            modified["BMI"] = modified["Weight"] / ((modified["Height"] / 100) ** 2)
            _, _, _, new_dose, new_auc = recommend_dose_curve_uncertainty(modified, model, scaler)
            sensitivity_records.append({
                "Patient_ID": patient['Patient_ID'],
                "Parameter": param,
                "Change": f"{int(pct*100)}%",
                "Modified Dose": new_dose,
                "Predicted AUC": round(new_auc, 2),
                "ΔDose": new_dose - rec_dose,
                "ΔAUC": round(new_auc - rec_auc, 2)
            })

# --- Save Outputs ---
df_summary = pd.DataFrame(summary_recommendations)
df_sensitivity = pd.DataFrame(sensitivity_records)

df_summary.to_csv("virtual_patient_dose_predictions.csv", index=False)
df_sensitivity.to_csv("dose_sensitivity_analysis.csv", index=False)

print("\n✅ Dose Recommendations:")
print(df_summary)
print("\n✅ Sensitivity Analysis (first few rows):")
print(df_sensitivity.head())

# --- Save Summary Table Plot ---
plt.figure(figsize=(8, 2))
plt.axis('off')
table = plt.table(cellText=df_summary.values, colLabels=df_summary.columns, cellLoc='center', loc='center')
table.scale(1, 1.5)
plt.savefig("summary_dose_recommendations_table.png", bbox_inches='tight')
print("📄 Saved: summary_dose_recommendations_table.png")

# --- Save Sensitivity Heatmap ---
pivot = df_sensitivity.pivot_table(index=['Parameter', 'Change'], columns='Patient_ID', values='ΔDose')
plt.figure(figsize=(10, 4))
sns.heatmap(pivot, annot=True, cmap="coolwarm", fmt=".0f")
plt.title("Sensitivity of Dose to Parameter Changes")
plt.tight_layout()
plt.savefig("sensitivity_heatmap_dose.png")
plt.show()
print("📊 Saved: sensitivity_heatmap_dose.png")

# --- Plot for Patient 1 ---
p0 = virtual_patients[0]
d, a_mean, a_std, _, _ = recommend_dose_curve_uncertainty(p0, model, scaler)
plt.figure(figsize=(8, 5))
plt.plot(d, a_mean, marker='o', label='Predicted AUC')
plt.fill_between(d, a_mean - a_std, a_mean + a_std, alpha=0.3, label='±1 SD')
plt.axhline(32.5, color='red', linestyle='--', label='Target AUC')
plt.xlabel("Dose (mg)")
plt.ylabel("AUC (mg·h/L)")
plt.title("Dose–AUC with Uncertainty (Patient 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dose_auc_uncertainty_patient1.png")
plt.show()
print("\n📊 Saved: dose_auc_uncertainty_patient1.png")