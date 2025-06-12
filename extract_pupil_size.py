import pandas as pd
import ast  # Convert string to dictionary

# Load dataset
file_path = "gaze.csv"
df = pd.read_csv(file_path)

# Convert string representation of dictionaries into actual dictionaries
df["eye_details"] = df["eye_details"].apply(ast.literal_eval)

# Extract pupil size
df["pupil_size"] = df["eye_details"].apply(lambda x: x["pupil_size"])

# Print sample pupil sizes
print("Sample Pupil Sizes:\n", df["pupil_size"].head())

# Save the extracted data
df[["pupil_size"]].to_csv("pupil_size_data.csv", index=False)
            