import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_retail.csv", encoding="latin1")

# Ensure datetime format
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Create total revenue
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Set snapshot date (day after last purchase)
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# RFM table
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                                   # Frequency
    "Revenue": "sum"                                          # Monetary
})

# Rename columns
rfm.columns = ["Recency", "Frequency", "Monetary"]

# Reset index
rfm = rfm.reset_index()

print(rfm.head())
print("Shape:", rfm.shape)

# Save dataset
rfm.to_csv("data/rfm_data.csv", index=False)

print("Saved rfm_data.csv")