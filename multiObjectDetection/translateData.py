import csv

# File paths
input_file = 'finalNewData.csv'
output_file = 'test.csv'

# Read the CSV data
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    data = list(reader)

# Process each row to remove the last three columns and append 1.0
for i in range(len(data)):
    if len(data[i]) > 3:  # Ensure the row has more than three columns
        data[i] = data[i][:-3]  # Remove the last three columns
    data[i].append('1.0')  # Append 1.0 to the modified row

# Write the modified data to a new CSV file
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data)

print(f"Processed data has been saved to {output_file}")