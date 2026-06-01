import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_retail.csv")

# Create total spent per row
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# CUSTOMER LEVEL AGGREGATION
customer_df = df.groupby("CustomerID").agg({
    "Revenue": "sum",
    "Quantity": "sum",
    "InvoiceNo": "nunique"
}).reset_index()

# Rename columns
customer_df.columns = [
    "CustomerID",
    "TotalSpent",
    "TotalItems",
    "TotalOrders"
]

# Create prediction target (High value customer)
customer_df["HighValue"] = (
    customer_df["TotalSpent"] > customer_df["TotalSpent"].median()
).astype(int)

print(customer_df.head())
print("Shape:", customer_df.shape)

# Save dataset for ML
customer_df.to_csv("data/customer_features.csv", index=False)

print("Saved customer_features.csv")