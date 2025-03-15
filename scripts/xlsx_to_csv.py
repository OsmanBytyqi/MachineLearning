import pandas as pd
import os
import glob

def convert_and_combine_xlsx_to_csv(input_files, output_file):
    """
    Converts multiple Excel (.xlsx) files to a single CSV file, combining their data.

    Args:
        input_files (str or list): Path(s) to the input .xlsx file(s) (supports wildcards like "*.xlsx").
        output_file (str): Path where the combined CSV file will be saved.

    Returns:
        str: Path of the combined CSV file.
    """
    if isinstance(input_files, str):  # If a single string is provided
        input_files = glob.glob(input_files)  # Expand wildcard patterns like "*.xlsx"

    if not input_files:
        raise FileNotFoundError("No matching Excel files found.")

    combined_df = pd.DataFrame()  # Initialize an empty DataFrame

    for input_file in input_files:
        df = pd.read_excel(input_file, sheet_name=0, engine="openpyxl")

        # Drop empty rows at the top and reset the index
        df = df.dropna(how="all").reset_index(drop=True)

        # Rename columns if they are Unnamed
        df.columns = [col if "Unnamed" not in col else "" for col in df.columns]

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # Append the DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined data into a single CSV
    combined_df.to_csv(output_file, index=False, sep=",", quoting=1, encoding="utf-8")
    print(f"Combined data from all files and saved to {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine multiple XLSX files into one CSV")
    parser.add_argument("input_files", nargs="+", help="Path(s) to the input .xlsx file(s) (supports wildcards)")
    parser.add_argument("-o", "--output_file", help="Path to the output combined CSV file")

    args = parser.parse_args()

    convert_and_combine_xlsx_to_csv(args.input_files, args.output_file)
