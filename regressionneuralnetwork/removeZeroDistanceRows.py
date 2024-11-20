import pandas as pd

# Read the CSV file
input_file = "finalNewData.csv"  # Replace with your input file name
output_file = "finalNewDataNoZeros.csv"  # Replace with your desired output file name

# Load the CSV file into a DataFrame
df = pd.read_csv(input_file, header=None)  # Assuming no headers in the file

# Filter rows where the 8th column (index 7) is not 0.0
filtered_df = df[df[7] != 0.0]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file, index=False, header=False)

print(f"Filtered data saved to {output_file}")
