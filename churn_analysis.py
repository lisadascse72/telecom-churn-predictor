# Step 1: Import necessary libraries
import pandas as pd

# Step 2: Load the CSV file (corrected)
df = pd.read_csv("Telco-Customer-Churn.csv")  # Make sure the filename is exact

# Step 3: Preview top rows
print("📊 First 5 rows:")
print(df.head())

# Step 4: Column types
print("\n🧾 Column Types:")
print(df.dtypes)

# Step 5: Missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

print("\n🔧 Cleaning & Feature Engineering...")

# 1. Convert TotalCharges to numeric, coerce errors (if any non-numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Check again for missing values
print("\nNulls after conversion:")
print(df.isnull().sum())

# 3. Drop rows with missing TotalCharges
df = df.dropna(subset=['TotalCharges'])

# 4. Create Churn_Flag: 1 if Churn = Yes, 0 if No
df['Churn_Flag'] = df['Churn'].apply(lambda x: 1 if x.strip().lower() == 'yes' else 0)

# 5. Export cleaned data
df.to_csv("cleaned_churn_data.csv", index=False)
print("\n✅ Cleaned data exported as 'cleaned_churn_data.csv'")
