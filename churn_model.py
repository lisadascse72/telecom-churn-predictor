import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# STEP 1: Load cleaned data
df = pd.read_csv("cleaned_churn_data.csv")

# STEP 2: Check for duplicates
print(f"Total duplicate rows in dataset: {df.duplicated().sum()}")

# STEP 3: Convert TotalCharges to numeric and drop rows with missing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# STEP 4: Encode target manually
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# DROP LEAKAGE COLUMN
if 'Churn_Flag' in df.columns:
    print("Dropping 'Churn_Flag' column to avoid data leakage.")
    df.drop('Churn_Flag', axis=1, inplace=True)

# STEP 5: Correlation check with churn (only numeric)
print("\nCorrelation of numeric features with Churn:")
numeric_df = df.select_dtypes(include=['number'])
print(numeric_df.corr()['Churn'].sort_values(ascending=False))

# STEP 6: Drop ID column
df.drop('customerID', axis=1, inplace=True)

# STEP 7: Split features and target before encoding to avoid leakage
X = df.drop('Churn', axis=1)
y = df['Churn']

# STEP 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n📊 Churn Distribution in Training Data:")
print(y_train.value_counts(normalize=True))

# STEP 9: One-hot encoding categorical columns - fit on train only, apply to test
cat_cols = X_train.select_dtypes(include='object').columns.tolist()

X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align test set columns to train set columns (fill missing columns with 0)
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# STEP 10: Baseline dummy classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_enc, y_train)
y_dummy_pred = dummy_clf.predict(X_test_enc)
print("\nBaseline Dummy Classifier Accuracy:", accuracy_score(y_test, y_dummy_pred))

# STEP 11: Logistic Regression with cross-validation
log_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_model.fit(X_train_enc, y_train)
y_pred_log = log_model.predict(X_test_enc)
y_pred_log_proba = log_model.predict_proba(X_test_enc)[:, 1]

print("\n📊 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_pred_log_proba))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Cross-validation accuracy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(log_model, X_train_enc, y_train, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# STEP 12: Random Forest with cross-validation
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train_enc, y_train)
y_pred_rf = rf_model.predict(X_test_enc)
y_pred_rf_proba = rf_model.predict_proba(X_test_enc)[:, 1]

print("\n🌲 Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_pred_rf_proba))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

cv_scores_rf = cross_val_score(rf_model, X_train_enc, y_train, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy Scores RF: {cv_scores_rf}")
print(f"Mean CV Accuracy RF: {cv_scores_rf.mean():.4f}")

# STEP 13: Feature Importance from Random Forest
importances = rf_model.feature_importances_
features = X_train_enc.columns  # <-- this was missing earlier

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='lightgreen')
plt.xlabel("Importance")
plt.title("📌 Feature Importance from Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("✅ Feature importance plot saved as 'feature_importance.png'")

# Add predictions to test set and save
X_test['Actual_Churn'] = y_test.values
X_test['Predicted_Churn_RF'] = y_pred_rf
X_test.to_csv('rf_predictions.csv', index=False)
print("📁 Predictions saved to rf_predictions.csv")



