import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:
    def __init__(self, df):
        """
        Initialize the DataPreprocessing class with a DataFrame.
        Args:
            df (pd.DataFrame): The dataset to be preprocessed.
        """
        self.df = df

    def drop_missing_values(self):
        """Removes rows with missing values."""
        self.df = self.df.dropna()
        return self.df

    def drop_duplicates(self):
        """Removes duplicate rows from the dataset."""
        self.df = self.df.drop_duplicates()
        return self.df

    def encode_categorical(self, columns, encoding_type="label"):
        """
        Converts categorical features into numerical format.

        Args:
            columns (list): List of categorical column names to encode.
            encoding_type (str): "label" for label encoding, "onehot" for one-hot encoding.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        if encoding_type == "label":
            label_encoders = {}
            for col in columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                label_encoders[col] = le
        elif encoding_type == "onehot":
            self.df = pd.get_dummies(self.df, columns=columns)
        else:
            raise ValueError("Invalid encoding type. Use 'label' or 'onehot'.")

        return self.df

    def check_and_drop_years(self):
        """Checks if all years are between 2019 and 2024 and drops rows with invalid years."""
        # Ensure that 'Viti' is numeric
        self.df['Viti'] = pd.to_numeric(self.df['Viti'], errors='coerce')  # Convert to numeric, invalid entries become NaN
        initial_count = len(self.df)
        self.df = self.df[self.df['Viti'].between(2019, 2024, inclusive=True)]  # Drop rows with invalid years
        final_count = len(self.df)
        print(f"Dropped {initial_count - final_count} rows with invalid years outside 2019-2024.")

    def check_and_drop_months(self):
        """Checks if all months are between 1 and 12 and drops rows with invalid months."""
        # Ensure that 'Muaji' is numeric
        self.df['Muaji'] = pd.to_numeric(self.df['Muaji'], errors='coerce')  # Convert to numeric, invalid entries become NaN
        initial_count = len(self.df)
        self.df = self.df[self.df['Muaji'].between(1, 12, inclusive=True)]  # Drop rows with invalid months
        final_count = len(self.df)
        print(f"Dropped {initial_count - final_count} rows with invalid months outside 1-12.")

    def check_and_drop_komuna(self):
        """Checks if all Komuna values are between 1 and 38 and drops rows with invalid Komuna values."""
        # Ensure that 'Komuna' is numeric
        self.df['Komuna'] = pd.to_numeric(self.df['Komuna'], errors='coerce')  # Convert to numeric, invalid entries become NaN
        initial_count = len(self.df)
        self.df = self.df[self.df['Komuna'].between(1, 38, inclusive=True)]  # Drop rows with invalid Komuna values
        final_count = len(self.df)
        print(f"Dropped {initial_count - final_count} rows with invalid Komuna values outside 1-38.")

    def save_to_csv(self, filename):
        """Saves the cleaned DataFrame to a CSV file."""
        self.df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/gjobat-all.csv", encoding='utf-8')

    # Clean up column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Verify if all the specified columns exist in the DataFrame
    required_columns = ["Përshkrimi i Sektorit", "Komuna", "Statusi i Regjistrimit", "Përshkrimi i Gjobave në bazë të Ligjit", "Viti", "Muaji"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        # Add missing columns with placeholder values (e.g., 'Unknown')
        for col in missing_columns:
            df[col] = 'Unknown'

    # Fill missing values in categorical columns with a placeholder
    df[required_columns] = df[required_columns].fillna('Unknown')

    # Create an instance of the DataPreprocessing class
    processor = DataPreprocessing(df)

    # Check and drop rows with invalid years
    processor.check_and_drop_years()

    # Check and drop rows with invalid months
    processor.check_and_drop_months()

    # Check and drop rows with invalid Komuna values
    processor.check_and_drop_komuna()

    # Drop missing values and duplicates
    processor.drop_missing_values()
    processor.drop_duplicates()

    # Encode categorical columns
    processor.encode_categorical(required_columns, encoding_type="label")

    # Save the cleaned data to a CSV file
    processor.save_to_csv("../data/processed/gjobat-all.csv")
