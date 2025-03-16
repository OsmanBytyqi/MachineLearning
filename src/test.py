import pandas as pd

# Read the dataset from the CSV file
# Replace 'your_dataset.csv' with the path to your actual CSV file
df = pd.read_csv('../data/processed/gjobat-all.csv')

# Check the total number of rows in the dataset
total_rows = len(df)

# Count the number of occurrences of each category in "Përshkrimi i Sektorit"
category_counts = df['Përshkrimi i Sektorit'].value_counts()

# Show the counts of each category
print("Category counts:\n", category_counts)

# Verify if the total count matches the total rows
category_count_total = category_counts.sum()

# Compare the total rows with the sum of the category counts
if total_rows == category_count_total:
    print(f"\nThe total number of rows matches the sum of the categories. Total rows: {total_rows}")
else:
    print(f"\nThere is a mismatch in the total number of rows and category counts. Total rows: {total_rows}, Total categories sum: {category_count_total}")
