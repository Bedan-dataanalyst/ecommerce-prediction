import pandas as pd

# Load RFM data
df = pd.read_csv("data/rfm_data.csv")

# Create segment rules
def segment_customer(row):
    if row["Monetary"] > 2000 and row["Frequency"] > 5:
        return "VIP"
    elif row["Frequency"] > 3:
        return "Loyal"
    elif row["Recency"] <= 30:
        return "New"
    elif row["Recency"] > 90:
        return "At Risk"
    else:
        return "Regular"

df["Segment"] = df.apply(segment_customer, axis=1)

print(df["Segment"].value_counts())

# Save
df.to_csv("data/rfm_segmented.csv", index=False)

print("Saved rfm_segmented.csv")