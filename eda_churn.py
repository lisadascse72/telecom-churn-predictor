import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# Load cleaned churn data
df = pd.read_csv("cleaned_churn_data.csv")

print("\n📊 Churn Value Counts:")
print(df['Churn'].value_counts())

# 🔹 1. Churn Count
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title("Churn vs Non-Churn Customers")
plt.tight_layout()
plt.show()

# 🔹 2. Churn by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', hue='Churn', data=df, palette='Set1')
plt.title("Churn by Gender")
plt.tight_layout()
plt.show()

# 🔹 3. Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='pastel')
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.show()

# 🔹 4. Monthly Charges vs Churn (Boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='coolwarm')
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.show()

# 🔹 5. Tenure Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['tenure'], bins=30, kde=True, color='skyblue')
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.tight_layout()
plt.show()

# 🔹 6. Churn by Internet Service
plt.figure(figsize=(8, 5))
sns.countplot(x='InternetService', hue='Churn', data=df, palette='pastel')
plt.title('Churn by Internet Service Type')
plt.tight_layout()
plt.show()

# 🔹 7. Tenure Distribution by Churn
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, palette='Set2')
plt.title('Tenure Distribution by Churn Status')
plt.tight_layout()
plt.show()

# 🔹 8. Monthly Charges Distribution by Churn
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette='Set2')
plt.title('Monthly Charges Distribution by Churn')
plt.tight_layout()
plt.show()

# 🔹 9. Total Charges vs Churn (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='TotalCharges', data=df, palette='coolwarm')
plt.title('Total Charges vs Churn')
plt.tight_layout()
plt.show()

# 🔹 10. Churn by Paperless Billing
plt.figure(figsize=(8, 5))
sns.countplot(x='PaperlessBilling', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Paperless Billing')
plt.tight_layout()
plt.show()

# 🔹 11. Heatmap of Numeric Correlation
plt.figure(figsize=(10, 6))
corr = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 🔹 12. Monthly Charges vs Tenure (Scatter)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, palette='Set1')
plt.title("Monthly Charges vs Tenure (Colored by Churn)")
plt.tight_layout()
plt.show()

# 🔹 13. Extra: Histogram of Tenure by Churn (again for visual clarity)
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', kde=False, palette='Accent')
plt.title('Tenure Distribution by Churn')
plt.tight_layout()
plt.show()
