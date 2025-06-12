import pandas as pd

# Load dataset
file_path = "gaze.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Display column names
print("\nColumn Names in the Dataset:")
print(df.columns)
