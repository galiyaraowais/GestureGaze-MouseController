import pandas as pd
import ast  # To safely convert string to list

# Load the dataset
df = pd.read_csv("gaze.csv")  # Make sure this file exists

# Extract the 'look_vec' from 'eye_details'
df["look_vec"] = df["eye_details"].apply(lambda x: ast.literal_eval(x)["look_vec"] if pd.notna(x) else None)

# Save extracted data
df[["look_vec"]].to_csv("extracted_look_vec.csv", index=False)

print("✅ Extracted look_vec saved as 'extracted_look_vec.csv'!")
