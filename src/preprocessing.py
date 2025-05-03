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
        
        # Store original categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        original_cat_columns = categorical_columns.tolist()
        
        # Categorical encoding
        important_cat_cols = ['Përshkrimi i Gjobave në bazë të Ligjit', 'Komuna']
        remaining_cat_cols = [col for col in original_cat_columns if col not in important_cat_cols]
        df_encoded = df.copy()
        
        # Target encoding for important categorical columns
        if len(important_cat_cols) > 0:
            try:
                target_encoder = ce.TargetEncoder(cols=important_cat_cols)
                target_encoder.fit(df_encoded[important_cat_cols], df_encoded[target_column])
                df_encoded[important_cat_cols] = target_encoder.transform(df_encoded[important_cat_cols])
                logging.info(f"Applied target encoding to {len(important_cat_cols)} important categorical columns")
            except Exception as e:
                logging.error(f"Error in target encoding: {str(e)}")
        
        # Frequency encoding
        for col in original_cat_columns:
            freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
            df_encoded[f'{col}_freq'] = df_encoded[col].map(freq_map)
        
        # Label encoding
        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            col_name = f"{column}_label" 
            df_encoded[col_name] = label_encoders[column].fit_transform(df[column].astype(str))
        
        # Ratio features
        if 'Numri i Gjobave të Lëshuara' in df.columns and 'Numri i Tatimpaguesve' in df.columns:
            df_encoded['gjobat_per_tatimpagues'] = df_encoded['Numri i Gjobave të Lëshuara'] / (df_encoded['Numri i Tatimpaguesve'] + 0.001)
            df_encoded['tatimpagues_per_gjobe'] = df_encoded['Numri i Tatimpaguesve'] / (df_encoded['Numri i Gjobave të Lëshuara'] + 0.001)
            df_encoded['gjoba_tatimpagues_ratio_log'] = np.log1p(df_encoded['gjobat_per_tatimpagues'])
        
        # Date-based features
        if 'Viti' in df.columns and 'Muaji' in df.columns:
            df_encoded['date_numeric'] = df_encoded['Viti'] + df_encoded['Muaji']/12
            df_encoded['season'] = ((df_encoded['Muaji'] % 12) // 3 + 1).astype(int)
            df_encoded['quarter'] = ((df_encoded['Muaji'] - 1) // 3 + 1).astype(int)
            df_encoded['is_summer'] = ((df_encoded['Muaji'] >= 6) & (df_encoded['Muaji'] <= 8)).astype(int)
            df_encoded['is_winter'] = ((df_encoded['Muaji'] == 12) | (df_encoded['Muaji'] <= 2)).astype(int)
            df_encoded['month_sin'] = np.sin(2 * np.pi * df_encoded['Muaji']/12)
            df_encoded['month_cos'] = np.cos(2 * np.pi * df_encoded['Muaji']/12)
        
        # Group-based statistics
        important_cat_cols = []
        if 'Përshkrimi i Gjobave në bazë të Ligjit_label' in df_encoded.columns:
            important_cat_cols.append('Përshkrimi i Gjobave në bazë të Ligjit_label')
        if 'Komuna_label' in df_encoded.columns:
            important_cat_cols.append('Komuna_label')
        
        for cat_col in important_cat_cols:
            key_numeric_cols = ['Numri i Gjobave të Lëshuara', 'Numri i Tatimpaguesve']
            for num_col in key_numeric_cols:
                group_aggs = df_encoded.groupby(cat_col)[num_col]
                df_encoded[f'mean_{num_col}_by_{cat_col}'] = group_aggs.transform('mean')
                df_encoded[f'std_{num_col}_by_{cat_col}'] = group_aggs.transform('std').fillna(0)
                df_encoded[f'min_{num_col}_by_{cat_col}'] = group_aggs.transform('min')
                df_encoded[f'max_{num_col}_by_{cat_col}'] = group_aggs.transform('max')
                df_encoded[f'median_{num_col}_by_{cat_col}'] = group_aggs.transform('median')
                df_encoded[f'dev_from_mean_{num_col}_by_{cat_col}'] = df_encoded[num_col] - df_encoded[f'mean_{num_col}_by_{cat_col}']
                df_encoded[f'rank_pct_{num_col}_by_{cat_col}'] = group_aggs.transform(lambda x: (x.rank(method='min') / len(x)))
            
            if cat_col != target_column:
                target_aggs = df_encoded.groupby(cat_col)[target_column]
                df_encoded[f'target_mean_by_{cat_col}'] = target_aggs.transform('mean')
                df_encoded[f'target_median_by_{cat_col}'] = target_aggs.transform('median')
                df_encoded[f'target_std_by_{cat_col}'] = target_aggs.transform('std').fillna(0)
                df_encoded[f'target_dev_from_mean_by_{cat_col}'] = df_encoded[target_column] - df_encoded[f'target_mean_by_{cat_col}']
        
        # Mathematical transformations
        important_numeric_cols = ['Numri i Tatimpaguesve', 'Numri i Gjobave të Lëshuara']
        for num_col in important_numeric_cols:
            if (df_encoded[num_col] > 0).all():
                df_encoded[f'log_{num_col}'] = np.log1p(df_encoded[num_col])
                df_encoded[f'sqrt_{num_col}'] = np.sqrt(df_encoded[num_col])
                df_encoded[f'qcut_{num_col}'] = pd.qcut(df_encoded[num_col], q=10, labels=False, duplicates='drop')
        
        # Polynomial features
        if 'Numri i Gjobave të Lëshuara' in df_encoded.columns and 'Numri i Tatimpaguesve' in df_encoded.columns:
            df_encoded['gjobat_squared'] = df_encoded['Numri i Gjobave të Lëshuara'] ** 2
            df_encoded['tatimpaguesve_squared'] = df_encoded['Numri i Tatimpaguesve'] ** 2
            df_encoded['gjobat_cubed'] = df_encoded['Numri i Gjobave të Lëshuara'] ** 3
            df_encoded['gjobat_x_tatimpaguesve'] = df_encoded['Numri i Gjobave të Lëshuara'] * df_encoded['Numri i Tatimpaguesve']
            if 'Viti' in df_encoded.columns:
                df_encoded['gjobat_x_viti'] = df_encoded['Numri i Gjobave të Lëshuara'] * df_encoded['Viti']
                df_encoded['tatimpaguesve_x_viti'] = df_encoded['Numri i Tatimpaguesve'] * df_encoded['Viti']
        
        # RBF features
        key_numeric_features = ['Numri i Gjobave të Lëshuara', 'Numri i Tatimpaguesve']
        if all(col in df_encoded.columns for col in key_numeric_features):
            X_rbf = df_encoded[key_numeric_features].copy()
            scaler = RobustScaler()
            X_rbf_scaled = scaler.fit_transform(X_rbf)
            for gamma in [0.1, 0.5, 1.0, 5.0]:
                rbf_sampler = RBFSampler(gamma=gamma, n_components=5, random_state=42)
                rbf_features = rbf_sampler.fit_transform(X_rbf_scaled)
                for i in range(rbf_features.shape[1]):
                    df_encoded[f'rbf_g{gamma}_{i}'] = rbf_features[:, i]
        
        # Clustering-based features
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
        
        # Target transformation
        if target_column in df_encoded.columns:
            skewness = df_encoded[target_column].skew()
            if skewness > 1.0:
                df_encoded[target_column] = np.log1p(df_encoded[target_column])
                logging.info(f"Applied log transformation to target column: {target_column}")
        
        # PCA features
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
        
        # Drop original categorical columns
        df_encoded.drop(columns=original_cat_columns, inplace=True)
        
        self.df_processed = df_encoded
        self.label_encoders = label_encoders
        logging.info(f"Enhanced preprocessing completed successfully. Shape: {self.df_processed.shape}")
        return self.df_processed, self.label_encoders

    def prepare_train_test_data(self, target_column='Vlera e Gjobave të Lëshuara', test_size=0.2, random_state=42):
        """Split data into train and test sets with feature selection."""
        X = self.df_processed.drop(columns=[target_column])
        y = self.df_processed[target_column]
        
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        
        # Feature selection with mutual information
        k1 = min(50, X.shape[1])
        mi_selector = SelectKBest(mutual_info_regression, k=k1)
        _ = mi_selector.fit_transform(X, y)
        
        # Stratified split using binned target
        y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y_binned
        )
        
        # Scaling
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        logging.info(f"Train set shape: {self.X_train.shape}")
        logging.info(f"Test set shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test, self.scaler

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    
    data_path = "data/raw/gjobat-all.csv"
    preprocessor = DataPreprocessor()
    preprocessor.load_data(data_path)
    df_processed, label_encoders = preprocessor.preprocess_data(target_column='Vlera e Gjobave të Lëshuara')
    
    processed_data_path = "data/processed/gjobat_processed.csv"
    df_processed.to_csv(processed_data_path, index=False)
    
    print(f"Data preprocessing completed successfully!")
    print(f"Processed data saved to {processed_data_path}")
