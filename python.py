# ============================================================
# LOAN DEFAULT & CREDIT RISK ANALYSIS
# ============================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# STEP 2: LOAD DATA
# In Google Colab, upload the CSV first and use its filename here.
df = pd.read_csv("Loan_Credit_Risk_Dataset.csv")

print("First 5 records:")
print(df.head())

print("\nLast 5 records:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# ============================================================
# PART 2: DATA EXPLORATION & CLEANING
# ============================================================

print("\nData types:")
print(df.dtypes)

print("\nStatistical summary:")
print(df.describe(include="all").T)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates().copy()

# Unique values
for col in df.select_dtypes(include="object").columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

# Handle missing values
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include="object").columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Outlier check using IQR
def outlier_count(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((data[column] < lower) | (data[column] > upper)).sum()

print("\nOutlier counts:")
for col in ["Age", "Annual_Income", "Credit_Score", "Loan_Amount",
            "Interest_Rate", "Debt_to_Income_Ratio"]:
    print(col, ":", outlier_count(df, col))

print("\nCleaned dataset shape:")
print(df.shape)

# Create analysis groups
df["Credit_Score_Group"] = pd.cut(
    df["Credit_Score"],
    bins=[0, 579, 669, 739, 799, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    include_lowest=True
)

df["Income_Group"] = pd.qcut(
    df["Annual_Income"],
    q=4,
    labels=["Low", "Medium", "High", "Very High"],
    duplicates="drop"
)

# ============================================================
# PART 3: DATA ANALYSIS
# ============================================================

total_customers = df["Customer_ID"].nunique()
total_loans = df["Loan_ID"].nunique()
defaulted_loans = (df["Default_Status"] == "Default").sum()
non_defaulted_loans = (df["Default_Status"] == "Non-Default").sum()
default_rate = defaulted_loans / total_loans * 100

print("\n===== KEY KPIs =====")
print("Total Customers:", total_customers)
print("Total Loans:", total_loans)
print("Defaulted Loans:", defaulted_loans)
print("Non-Defaulted Loans:", non_defaulted_loans)
print("Overall Default Rate: {:.2f}%".format(default_rate))
print("Average Annual Income:", df["Annual_Income"].mean())
print("Average Credit Score:", df["Credit_Score"].mean())
print("Average Loan Amount:", df["Loan_Amount"].mean())
print("Average Interest Rate:", df["Interest_Rate"].mean())
print("Average Debt-to-Income Ratio:", df["Debt_to_Income_Ratio"].mean())

# Loan count by loan type
loan_count_type = df["Loan_Type"].value_counts()
print("\nLoans by Loan Type:")
print(loan_count_type)

# Loan count by region
loan_count_region = df["Region"].value_counts()
print("\nLoans by Region:")
print(loan_count_region)

# Helper function for default rate
def default_rate_by(column):
    result = df.groupby(column, observed=False)["Default_Status"].apply(
        lambda x: (x == "Default").mean() * 100
    ).sort_values(ascending=False)
    return result

print("\nDefault Rate by Loan Type:")
print(default_rate_by("Loan_Type"))

print("\nDefault Rate by Employment Status:")
print(default_rate_by("Employment_Status"))

print("\nDefault Rate by Education:")
print(default_rate_by("Education"))

print("\nDefault Rate by Property Ownership:")
print(default_rate_by("Property_Ownership"))

print("\nDefault Rate by Credit Score Group:")
print(default_rate_by("Credit_Score_Group"))

print("\nDefault Rate by Income Group:")
print(default_rate_by("Income_Group"))

# Below-average credit score
below_avg_credit = df[df["Credit_Score"] < df["Credit_Score"].mean()]
print("\nCustomers with below-average credit scores:")
print(below_avg_credit[["Customer_ID", "Credit_Score", "Default_Status"]].head(20))

# Above-average loan amount
above_avg_loan = df[df["Loan_Amount"] > df["Loan_Amount"].mean()]
print("\nCustomers with above-average loan amounts:")
print(above_avg_loan[["Customer_ID", "Loan_Amount", "Default_Status"]].head(20))

# High DTI
high_dti = df[df["Debt_to_Income_Ratio"] > 0.50]
print("\nCustomers with high DTI (> 0.50):")
print(high_dti[["Customer_ID", "Debt_to_Income_Ratio", "Default_Status"]].head(20))

# Top 10 customers by loan amount
top10_loans = df.nlargest(10, "Loan_Amount")
print("\nTop 10 customers by loan amount:")
print(top10_loans[["Customer_ID", "Loan_ID", "Loan_Type", "Loan_Amount", "Default_Status"]])

# Highest default rate categories
loan_type_default = default_rate_by("Loan_Type")
region_default = default_rate_by("Region")

print("\nLoan type with highest default rate:")
print(loan_type_default.index[0], loan_type_default.iloc[0])

print("\nRegion with highest default rate:")
print(region_default.index[0], region_default.iloc[0])

# Cross-analysis
print("\nCredit Score vs Default:")
print(df.groupby("Default_Status")["Credit_Score"].agg(["mean", "median"]))

print("\nIncome vs Default:")
print(df.groupby("Default_Status")["Annual_Income"].agg(["mean", "median"]))

print("\nLoan Amount vs Default:")
print(df.groupby("Default_Status")["Loan_Amount"].agg(["mean", "median"]))

print("\nPrevious Defaults vs Current Default:")
print(pd.crosstab(df["Previous_Defaults"], df["Default_Status"], normalize="index") * 100)

# ============================================================
# PART 4: PYTHON VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid")

# 1. Loan Default Distribution - Pie Chart
default_counts = df["Default_Status"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(default_counts, labels=default_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Loan Default Distribution")
plt.show()

# 2. Loan Count by Loan Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Loan_Type", order=df["Loan_Type"].value_counts().index)
plt.title("Loan Count by Loan Type")
plt.xlabel("Loan Type")
plt.ylabel("Number of Loans")
plt.xticks(rotation=30)
plt.show()

# 3. Default Rate by Loan Type
rate = default_rate_by("Loan_Type").reset_index(name="Default_Rate")
plt.figure(figsize=(8, 5))
sns.barplot(data=rate, x="Loan_Type", y="Default_Rate")
plt.title("Default Rate by Loan Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30)
plt.show()

# 4. Default Rate by Employment Status
rate = default_rate_by("Employment_Status").reset_index(name="Default_Rate")
plt.figure(figsize=(8, 5))
sns.barplot(data=rate, x="Employment_Status", y="Default_Rate")
plt.title("Default Rate by Employment Status")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30)
plt.show()

# 5. Default Rate by Education
rate = default_rate_by("Education").reset_index(name="Default_Rate")
plt.figure(figsize=(8, 5))
sns.barplot(data=rate, x="Education", y="Default_Rate")
plt.title("Default Rate by Education")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30)
plt.show()

# 6. Credit Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Credit_Score", bins=25, kde=True)
plt.title("Credit Score Distribution")
plt.show()

# 7. Loan Amount Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Loan_Amount", bins=30, kde=True)
plt.title("Loan Amount Distribution")
plt.show()

# 8. Annual Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Annual_Income", bins=30, kde=True)
plt.title("Annual Income Distribution")
plt.show()

# 9. Credit Score vs Loan Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Credit_Score", y="Loan_Amount", hue="Default_Status")
plt.title("Credit Score vs Loan Amount")
plt.show()

# 10. Income vs Loan Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Annual_Income", y="Loan_Amount", hue="Default_Status")
plt.title("Income vs Loan Amount")
plt.show()

# 11. Loan Amount by Default Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Default_Status", y="Loan_Amount")
plt.title("Loan Amount by Default Status")
plt.show()

# 12. Credit Score by Default Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Default_Status", y="Credit_Score")
plt.title("Credit Score by Default Status")
plt.show()

# 13. DTI by Default Status
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Default_Status", y="Debt_to_Income_Ratio")
plt.title("Debt-to-Income Ratio by Default Status")
plt.show()

# 14. Default Rate by Income Group
rate = default_rate_by("Income_Group").reset_index(name="Default_Rate")
plt.figure(figsize=(8, 5))
sns.barplot(data=rate, x="Income_Group", y="Default_Rate")
plt.title("Default Rate by Income Group")
plt.ylabel("Default Rate (%)")
plt.show()

# 15. Correlation Heatmap
corr_cols = [
    "Age", "Annual_Income", "Credit_Score", "Existing_Loans",
    "Loan_Amount", "Loan_Term_Months", "Interest_Rate",
    "Monthly_Installment", "Debt_to_Income_Ratio",
    "Employment_Years", "Previous_Defaults", "Credit_History"
]
plt.figure(figsize=(12, 9))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
# BUSINESS INSIGHTS
# ============================================================

print("\n===== BUSINESS INSIGHTS =====")
print("1. Overall default rate: {:.2f}%".format(default_rate))
print("2. Highest loan-type default rate:", loan_type_default.index[0])
print("3. Highest region default rate:", region_default.index[0])

avg_credit_default = df.loc[df["Default_Status"] == "Default", "Credit_Score"].mean()
avg_credit_nondefault = df.loc[df["Default_Status"] == "Non-Default", "Credit_Score"].mean()
print("4. Average credit score - Default:", round(avg_credit_default, 2))
print("   Average credit score - Non-Default:", round(avg_credit_nondefault, 2))

avg_income_default = df.loc[df["Default_Status"] == "Default", "Annual_Income"].mean()
avg_income_nondefault = df.loc[df["Default_Status"] == "Non-Default", "Annual_Income"].mean()
print("5. Average income - Default:", round(avg_income_default, 2))
print("   Average income - Non-Default:", round(avg_income_nondefault, 2))

avg_dti_default = df.loc[df["Default_Status"] == "Default", "Debt_to_Income_Ratio"].mean()
avg_dti_nondefault = df.loc[df["Default_Status"] == "Non-Default", "Debt_to_Income_Ratio"].mean()
print("6. Average DTI - Default:", round(avg_dti_default, 3))
print("   Average DTI - Non-Default:", round(avg_dti_nondefault, 3))

prev_default_rate = pd.crosstab(
    df["Previous_Defaults"], df["Default_Status"], normalize="index"
).fillna(0) * 100
print("\n7. Previous defaults vs current default:")
print(prev_default_rate)

print("\n8. Important risk factors to monitor:")
print("- Credit score")
print("- Debt-to-income ratio")
print("- Previous defaults")
print("- Existing loans")
print("- Loan amount relative to annual income")
print("- Employment status")
print("- Interest rate")

# ============================================================
# SAVE CLEANED DATASET
# ============================================================
df.to_csv("Loan_Credit_Risk_Cleaned.csv", index=False)
print("\nCleaned dataset saved as Loan_Credit_Risk_Cleaned.csv")
