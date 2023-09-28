import os
import pandas as pd

# Path to the data folder
data_folder = 'PreprocessedNepaliData/'

# Get a list of all CSV files in the data folder
csv_files = [file for file in os.listdir(data_folder) if file.endswith('.csv')]

# Initialize a variable to keep track of the total row count
total_rows = 0

# Loop through each CSV file and count the rows
for csv_file in csv_files:
    csv_path = os.path.join(data_folder, csv_file)
    
    data = pd.read_csv(csv_path, skiprows=1, header=None)
    
    # Count the rows in the current CSV file
    num_rows = data.shape[0]
    
    # Add the row count of the current file to the total count
    total_rows += num_rows

# Print the total number of rows across all CSV files
print("Total number of articles across all CSV files:", total_rows)