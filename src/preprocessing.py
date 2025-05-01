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



        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            col_name = f"{column}_label" 
            df_encoded[col_name] = label_encoders[column].fit_transform(df[column].astype(str))

        if 'Numri i Gjobave të Lëshuara' in df.columns and 'Numri i Tatimpaguesve' in df.columns:
            df_encoded['gjobat_per_tatimpagues'] = df_encoded['Numri i Gjobave të Lëshuara'] / (df_encoded['Numri i Tatimpaguesve'] + 0.001)
            df_encoded['tatimpagues_per_gjobe'] = df_encoded['Numri i Tatimpaguesve'] / (df_encoded['Numri i Gjobave të Lëshuara'] + 0.001)
            df_encoded['gjoba_tatimpagues_ratio_log'] = np.log1p(df_encoded['gjobat_per_tatimpagues'])




        if 'Viti' in df.columns and 'Muaji' in df.columns:
            df_encoded['date_numeric'] = df_encoded['Viti'] + df_encoded['Muaji']/12
            df_encoded['season'] = ((df_encoded['Muaji'] % 12) // 3 + 1).astype(int)
            df_encoded['quarter'] = ((df_encoded['Muaji'] - 1) // 3 + 1).astype(int)
            df_encoded['is_summer'] = ((df_encoded['Muaji'] >= 6) & (df_encoded['Muaji'] <= 8)).astype(int)
            df_encoded['is_winter'] = ((df_encoded['Muaji'] == 12) | (df_encoded['Muaji'] <= 2)).astype(int)
            df_encoded['month_sin'] = np.sin(2 * np.pi * df_encoded['Muaji']/12)
            df_encoded['month_cos'] = np.cos(2 * np.pi * df_encoded['Muaji']/12)


            
        if df_encoded.shape[0] > 10:
            numeric_cols = df_encoded.select_dtypes(include=['number']).columns
            numeric_cols = [col for col in numeric_cols if col != target_column]
            if len(numeric_cols) > 1:
                clustering_data = df_encoded[numeric_cols].copy()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clustering_data)
                for n_clusters in [3, 5, 8]:
                    try:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(scaled_data)
                        df_encoded[f'cluster_kmeans_{n_clusters}'] = cluster_labels
                        distances = kmeans.transform(scaled_data)
                        for i in range(n_clusters):
                            df_encoded[f'dist_to_cluster_{n_clusters}_{i}'] = distances[:, i]
                    except Exception as e:
                        logging.error(f"Error in creating cluster features: {str(e)}")

        if target_column in df_encoded.columns:
            skewness = df_encoded[target_column].skew()
            if skewness > 1.0:
                df_encoded[target_column] = np.log1p(df_encoded[target_column])
                logging.info(f"Applied log transformation to target column: {target_column}")

        if df_encoded.shape[1] > 5:
            numeric_cols = df_encoded.select_dtypes(include=['number']).columns
            numeric_cols = [col for col in numeric_cols if col != target_column]
            if len(numeric_cols) > 5:
                pca_data = df_encoded[numeric_cols].copy()
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(pca_data)
                n_components = min(10, len(numeric_cols) - 1)
                pca = PCA(n_components=n_components, random_state=42)
                pca_result = pca.fit_transform(scaled_data)
                for i in range(pca_result.shape[1]):
                    df_encoded[f'pca_{i+1}'] = pca_result[:, i]

        df_encoded.drop(columns=original_cat_columns, inplace=True)

        self.df_processed = df_encoded
        self.label_encoders = label_encoders
        logging.info(f"Enhanced preprocessing completed successfully. Shape: {self.df_processed.shape}")
        return self.df_processed, self.label_encoders
