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

    def save_to_csv(self, filename):
        """Saves the cleaned DataFrame to a CSV file."""
        self.df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    df = pd.read_csv("../data/raw/gjobat-all.csv")
    processor = DataPreprocessing(df)

    processor.drop_missing_values()
    processor.drop_duplicates()
    processor.encode_categorical(["Përshkrimi i Sektorit", "Komuna", "Statusi i Regjistrimit", "Përshkrimi i Gjobave"], encoding_type="label")

    processor.save_to_csv("cleaned_dataset.csv")
