# download_data.py
# This script downloads datasets from a given URL and saves them locally.

import requests

def download_file(url, output_path):
    """
    Downloads a file from the internet and saves it.

    Args:
        url (str): File URL.
        output_path (str): Path to save the file.
    """
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)
    print(f"File downloaded: {output_path}")

if __name__ == "__main__":
    DATA_URL = "https://example.com/dataset.csv"
    OUTPUT_PATH = "../data/raw/dataset.csv"
    download_file(DATA_URL, OUTPUT_PATH)
