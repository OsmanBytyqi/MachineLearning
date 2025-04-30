import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
import os
import category_encoders as ce
from sklearn.kernel_approximation import RBFSampler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def __init__(self):
        self.df = None
        self.df_processed = None
        self.label_encoders = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, file_path):
        """Load the dataset from CSV file."""
        self.df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data with shape: {self.df.shape}")
        return self.df


    def preprocess_data(self, target_column='Vlera e Gjobave të Lëshuara'):
        """Preprocess the dataset with enhanced feature engineering."""
        df = self.df.copy()
        logging.info(f"Initial data shape: {df.shape}")

        # Handle missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logging.info(f"Missing values per column:\n{missing_values[missing_values > 0]}")
            
            # Fill numeric columns with median
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])

        categorical_columns = df.select_dtypes(include=['object']).columns
        original_cat_columns = categorical_columns.tolist()

        df_encoded = df.copy()

        important_cat_cols = ['Përshkrimi i Gjobave në bazë të Ligjit', 'Komuna']
        remaining_cat_cols = [col for col in original_cat_columns if col not in important_cat_cols]

        if len(important_cat_cols) > 0:
            try:
                target_encoder = ce.TargetEncoder(cols=important_cat_cols)
                target_encoder.fit(df_encoded[important_cat_cols], df_encoded[target_column])
                df_encoded[important_cat_cols] = target_encoder.transform(df_encoded[important_cat_cols])
                logging.info(f"Applied target encoding to {len(important_cat_cols)} important categorical columns")
            except Exception as e:
                logging.error(f"Error in target encoding: {str(e)}")

        for col in original_cat_columns:
            freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
            df_encoded[f'{col}_freq'] = df_encoded[col].map(freq_map)
