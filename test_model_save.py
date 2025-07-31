import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Small dummy dataset (very fast)
X = pd.DataFrame({
    "tenure": [1, 2, 3],
    "MonthlyCharges": [50, 70, 90],
    "SeniorCitizen": [0, 1, 0]
})
y = [0, 1, 0]

# Train tiny model
model = RandomForestClassifier()
model.fit(X, y)

# Try saving model
joblib.dump(model, "test_rf_model.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "test_model_features.pkl")

print("✅ Test model and features saved successfully.")
