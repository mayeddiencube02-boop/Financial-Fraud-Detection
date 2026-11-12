""" FRAUD DETECTION MODEL

Financial fraud detection is a critical aspect of banking and finance. This project aims to develop a machine
learning model to detect fraudulent transactions using real-world financial data. Given the imbalanced nature
of fraud cases, careful preprocessing and model selection are essential.



Step 1 - First Load and explore the fraud dataset

Install some important libraries:

pip install pandas numpy matplotlib seaborn


"""
import os
import pandas as pd


PROJECT_DIR = r"C:\Users\HomePC\Desktop\Financial_Fraud_Detection_Model"
FILE_NAME = "financial_fraud_detection_dataset.csv"
path = os.path.join(PROJECT_DIR, FILE_NAME)

df = pd.read_csv(path)

# 2. First look 

print("Shape (rows, columns):", df.shape)
print("\nColumns and data types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

#3. Data quality 

print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df.describe(include="all").T)

# 4. Class balance (fraud vs legit) 

possible_targets = [c for c in df.columns if "fraud" in c.lower() or c.lower() == "class"]
print("\nPossible target columns:", possible_targets)

if possible_targets:
    target = possible_targets[0]
    print(f"\nClass distribution for '{target}':")
    print(df[target].value_counts())
    print((df[target].value_counts(normalize=True) * 100).round(3).astype(str) + " %")

#  5. Save a sample so we can share/inspect it 

df.head(20).to_csv(os.path.join(PROJECT_DIR, "sample_20_rows.csv"), index=False)

print("\nDone. Copy the output above and paste it back to me.")
