import pandas as pd

# Load dataset
file_path = "gaze.csv"
df = pd.read_csv(file_path)

# Display a sample of 'eye_details' and 'eye_region_details'
print("Sample of 'eye_details' Column:")
print(df["eye_details"].head())

print("\nSample of 'eye_region_details' Column:")
print(df["eye_region_details"].head())
