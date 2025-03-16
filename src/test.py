import pandas as pd

# Read the dataset from the CSV file
df = pd.read_csv('../data/processed/gjobat-all.csv')

# Check the total number of rows in the dataset
total_rows = len(df)
print(f"Total rows: {total_rows}")

# Loop through each column in the DataFrame
for column in df.columns:
    print(f"Checking column: {column}")  # Debugging line
    # Check if the column is categorical (object dtype or boolean)
    if df[column].dtype == 'object' or df[column].dtype == 'bool':
        print(f"Processing column '{column}'")  # Debugging line
        # Count the occurrences of each category in the column
        category_counts = df[column].value_counts()
        
        # Show the counts of each category
        print(f"Category counts for '{column}':\n", category_counts)

        # Verify if the total count matches the total rows
        category_count_total = category_counts.sum()

        # Compare the total rows with the sum of the category counts
        if total_rows == category_count_total:
            print(f"\nThe total number of rows matches the sum of the categories in '{column}'. Total rows: {total_rows}\n")
        else:
            print(f"\nThere is a mismatch in the total number of rows and category counts for '{column}'. Total rows: {total_rows}, Total categories sum: {category_count_total}\n")
