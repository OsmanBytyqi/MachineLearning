import pandas as pd
import os

def convert_xlsx_to_csv(input_file, output_file=None):
    """
    Converts an Excel (.xlsx) file to CSV (.csv).
    
    Args:
        input_file (str): Path to the input .xlsx file.
        output_file (str, optional): Path to the output .csv file. Defaults to the same name as input.
    
    Returns:
        str: Path of the generated CSV file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    df = pd.read_excel(input_file, sheet_name=0)

    if output_file is None:
        output_file = input_file.replace(".xlsx", ".csv")

    df.to_csv(output_file, index=False)
    
    print(f"Converted {input_file} → {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert XLSX to CSV")
    parser.add_argument("input_file", help="Path to the input .xlsx file")
    parser.add_argument("-o", "--output", help="Path to the output .csv file (optional)")

    args = parser.parse_args()
    
    convert_xlsx_to_csv(args.input_file, args.output)
