import pandas as pd
import ast  # To convert string to dictionary

# Load dataset
file_path = "gaze.csv"
df = pd.read_csv(file_path)

# Convert string representation of dictionaries into actual dictionaries
df["eye_details"] = df["eye_details"].apply(ast.literal_eval)
df["eye_region_details"] = df["eye_region_details"].apply(ast.literal_eval)

# Print available keys in 'eye_details'
print("Keys in 'eye_details':", df["eye_details"].iloc[0].keys())

# Print available keys in 'eye_region_details'
print("Keys in 'eye_region_details':", df["eye_region_details"].iloc[0].keys())
