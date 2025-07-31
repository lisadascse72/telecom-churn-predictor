import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_churn_data.csv")

print("\n📊 Churn Value Counts:")
print(df['Churn'].value_counts())

# Bar chart: Churn count
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title("Churn vs Non-Churn Customers")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Bar chart: Churn by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', hue='Churn', data=df, palette='Set1')
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Contract Type vs Churn
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='pastel')
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Boxplot: Monthly Charges vs Churn
plt.figure(figsize=(6, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='coolwarm')
plt.title("Monthly Charges by Churn")
plt.tight_layout()
plt.show()

# Histogram: Tenure Distribution
plt.figure(figsize=(7, 4))
sns.histplot(df['tenure'], bins=30, kde=True)
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (months)")
plt.tight_layout()
plt.show()

# 🔍 Churn by Contract Type
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title("Churn Count by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.legend(title="Churn")
plt.tight_layout()
plt.show()


# Plot 2: Churn by Internet Service Type
plt.figure(figsize=(8,5))
sns.countplot(x='InternetService', hue='Churn', data=df, palette='pastel')
plt.title('Churn by Internet Service Type')
plt.xlabel('Internet Service')
plt.ylabel('Customer Count')
plt.legend(title='Churn')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of Tenure by Churn
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set1', bins=30)
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# Plot 4: Monthly Charges distribution by Churn
plt.figure(figsize=(10,6))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette='Set2')
plt.title('Monthly Charges Distribution by Churn Status')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Plot 5: Contract Type vs Churn
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.tight_layout()
plt.show()

# Plot 6: Tenure Distribution by Churn
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set2', bins=30)
plt.title('Churn Distribution by Tenure')
plt.xlabel('Customer Tenure (in months)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# Plot 7: Monthly Charges Distribution by Churn
plt.figure(figsize=(10,6))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='Set2')
plt.title('Monthly Charges Distribution by Churn')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Plot 8: Total Charges vs Churn (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Churn', y='TotalCharges', data=df, palette='coolwarm')
plt.title('Total Charges vs Churn')
plt.xlabel('Churn')
plt.ylabel('Total Charges')
plt.tight_layout()
plt.show()

# Plot 9: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set3')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# Plot 10: Churn by Internet Service Type
plt.figure(figsize=(8, 5))
sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Internet Service')
plt.xlabel('Internet Service')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# Plot 11: Heatmap of correlations
plt.figure(figsize=(12, 8))
# Select only numeric columns
corr_matrix = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Plot 12: Scatterplot of Monthly Charges vs Tenure
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=df, palette='Set1')
plt.title('Monthly Charges vs Tenure (Churn Colored)')
plt.xlabel('Tenure (Months)')
plt.ylabel('Monthly Charges')
plt.tight_layout()
plt.show()

# Plot 13: Distribution of Tenure
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', kde=False, palette='Accent')
plt.title('Distribution of Customer Tenure by Churn')
plt.xlabel('Tenure (Months)')
plt.tight_layout()
plt.show()

# Plot 14: Churn vs Paperless Billing
plt.figure(figsize=(8, 5))
sns.countplot(x='PaperlessBilling', hue='Churn', data=df, palette='Set2')
plt.title('Churn by Paperless Billing')
plt.xlabel('Paperless Billing')
plt.tight_layout()
plt.show()

