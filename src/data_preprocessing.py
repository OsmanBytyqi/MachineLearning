import os
import unicodedata
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer as DataFrameKNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler as DataFrameRobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.utils.validation import check_is_fitted
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

# class DataFrameKNNImputer(KNNImputer):
#     def transform(self, X):
#         X_trans = super().transform(X)
#         return pd.DataFrame(X_trans, columns=X.columns, index=X.index)

# class DataFrameRobustScaler(RobustScaler):
#     def transform(self, X):
#         X_trans = super().transform(X)
#         return pd.DataFrame(X_trans, columns=X.columns, index=X.index)

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)

class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, year_col='Year', month_col='Month', cycle_precision=3):
        self.year_col = year_col
        self.month_col = month_col
        self.cycle_precision = cycle_precision
        self.feature_names_ = None  # Add to track feature names

    def fit(self, X, y=None):
        # Validate input columns
        if self.year_col not in X.columns or self.month_col not in X.columns:
            raise ValueError("Missing year/month columns")
        # Use dummy data to determine output columns
        X_dummy = pd.DataFrame({self.year_col: [2000], self.month_col: [1]}, columns=X.columns)
        X_trans = self.transform(X_dummy)
        self.feature_names_ = X_trans.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Your existing validation checks
        if not X[self.month_col].between(1,12).all():
            raise ValueError(f"{self.month_col} contains invalid month values")
        if X[self.year_col].lt(2000).any():
            raise ValueError("Year values before 2000 are not supported")
            
        # Existing date processing
        X['Date'] = pd.to_datetime(
            X[self.year_col].astype(str) + '-' + 
            X[self.month_col].astype(str).str.zfill(2) + '-01'
        )
        
        if X['Date'].isna().any():
            bad_indices = X[X['Date'].isna()].index.tolist()
            raise ValueError(f"Invalid date components at indices: {bad_indices}")
            
        # Feature engineering
        X['Days_in_month'] = X['Date'].dt.days_in_month
        X['Quarter'] = X['Date'].dt.quarter
        
        # Cyclic encoding
        X['Month_sin'] = np.sin(2 * np.pi * (X[self.month_col]-1)/12
                               ).round(self.cycle_precision)
        X['Month_cos'] = np.cos(2 * np.pi * (X[self.month_col]-1)/12
                               ).round(self.cycle_precision)
        
        X['Quarter_sin'] = np.sin(2 * np.pi * (X['Quarter']-1)/4
                                 ).round(self.cycle_precision)
        X['Quarter_cos'] = np.cos(2 * np.pi * (X['Quarter']-1)/4
                                 ).round(self.cycle_precision)
        
        # Column removal
        cols_to_remove = {'Date', 'Quarter', self.year_col, self.month_col}
        X = X.drop(columns=list(cols_to_remove.intersection(X.columns)))
        
        # Preserve feature names
        self.feature_names_ = X.columns.tolist()
        return X


class LegalComponentExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_threshold=0.3):
        self.unknown_threshold = unknown_threshold
        self.pattern = r"""
            (?xi)
            (?:Nd\.?\s*Neni\.?\s*|Neni\s+)
            (?:\(\s*)? 
            (?P<Article>
                (?:[a-z]+\s+)?
                [\d\s\.\/()%-]+  # Added % to match entries like 15.3%
                (?:-[a-z]+)?
                (?:\)?\s*[a-z]*)?
                (?:\s+(?:dhe|and)\s+[\d\s\.\/()%-]+)*
            )
            (?:\)\s*)?[-–]\s*  # Handle different hyphen types
            (?:Ligji\s*(?:nr\.?)?\s*|Law\s*(?:No\.?)?\s*)?  # More flexible prefix handling
            (?P<Law>
                (?:\d+/)?
                (?:L-?)? 
                \d+[\d\/L-]*  # Better handle complex codes like 03/L-222
                (?:/\w+)?
            )
        """

        self.compiled_pattern = re.compile(self.pattern, re.VERBOSE)
        self.feature_names_ = None  # Track feature names

    def fit(self, X, y=None):
        # Capture initial feature names
        self.feature_names_ = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        legal_components = X["Legal_Description"].str.extract(self.compiled_pattern)
        
        # Your existing processing logic
        legal_components["Article"] = (
            legal_components["Article"]
            .str.replace(r'(?i)(\d+)\s+(dhe|and)\s+(\d+)', r'\1-\3', regex=True)
            .str.replace(r'[()\s]', '', regex=True)
            .str.replace(r'\.(?=\d)', '-', regex=True)  # Use hyphen instead of underscore
            .str.replace(r'[^a-zA-Z0-9.-]', '', regex=True)
            .str.replace(r'\.+', '.', regex=True)
            .str.rstrip('.')
        )

        legal_components["Law"] = (
            legal_components["Law"]
            .str.upper()
            .str.replace(r'(?i)^(ligji|law)', '', regex=True)
            .str.replace(r'\s+', '', regex=True)
            .str.replace(r'^L(?=\d)', '', regex=True)
            .str.replace(r'/?L(?=-)', '/L', regex=True)  # Better normalization
        )
        
        legal_components["Law"] = legal_components["Law"].fillna(
            X["Legal_Description"].str.extract(
                r'(?i)(Ligji|Law)[\s-]*(\d+\/L-?\d+)',  # Fixed escape sequence
                flags=re.I
            )[1]  # Capture group 2 (the law number)
        )

        legal_components["Article"] = legal_components["Article"].fillna(
            X["Legal_Description"].str.extract(r'Neni[\s.]*(\d+[\d\.()%-]*)', flags=re.I)[0]
        )

        X["Law_Article"] = (
            legal_components["Law"] 
            + "_Art" 
            + legal_components["Article"]
        ).fillna("UNKNOWN")
        
        # Update feature names
        self.feature_names_ = X.columns.tolist()
        
        unknown_rate = (X["Law_Article"] == "UNKNOWN").mean()
        if unknown_rate > self.unknown_threshold:
            warnings.warn(
                f"High unknown rate ({unknown_rate:.1%}) in legal component extraction. "
                f"Verify pattern matches for: \n"
                f"{X.loc[X['Law_Article'] == 'UNKNOWN', 'Legal_Description'].sample(3).tolist()}"
            )
            
        return X



class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=3, smoothing_factor=0.5, handle_unknown='smooth'):
        self.min_freq = min_freq
        self.smoothing_factor = smoothing_factor
        self.handle_unknown = handle_unknown
        self.freq_map_ = None
        self.total_count_ = None
        self.feature_names_ = None  # Track feature names

    def fit(self, X, y=None):
        # Capture initial feature names
        self.feature_names_ = X.columns.tolist()
        
        freq_series = X['Law_Article'].value_counts()
        self.total_count_ = len(X)
        
        smoothed_counts = freq_series + self.smoothing_factor
        self.freq_map_ = smoothed_counts / (self.total_count_ + 
                                          self.smoothing_factor * len(freq_series))
        
        min_freq_value = (self.min_freq + self.smoothing_factor) / \
                       (self.total_count_ + self.smoothing_factor * len(freq_series))
        self.freq_map_ = self.freq_map_.where(
            freq_series >= self.min_freq, 
            other=min_freq_value
        ).to_dict()
        
        return self

    def transform(self, X):
        check_is_fitted(self, ['freq_map_', 'total_count_'])
        X = X.copy()
        
        X['Law_Article_Freq'] = X['Law_Article'].map(self.freq_map_)
        
        if self.handle_unknown == 'smooth':
            default_value = self.smoothing_factor / (self.total_count_ + 
                                                    self.smoothing_factor)
        elif self.handle_unknown == 'min_freq':
            default_value = (self.min_freq + self.smoothing_factor) / \
                          (self.total_count_ + self.smoothing_factor)
        else:
            raise ValueError(f"Invalid handle_unknown: {self.handle_unknown}")
            
        X['Law_Article_Freq'] = X['Law_Article_Freq'].fillna(default_value)
        
        unknown_count = (X['Law_Article'] == 'UNKNOWN').sum()
        if unknown_count > 0:
            warnings.warn(f"{unknown_count} entries with UNKNOWN Law_Article detected")
        
        # Update feature names after transformation
        self.feature_names_ = X.columns.tolist()
        return X


class DataPreprocessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self._create_column_map()
        self.column_descriptions = {
            'Year': 'The year when the fine was issued',
            'Month': 'The month when the fine was issued (1-12)',
            'Sector': 'Economic sector/industry of the taxpayer',
            'Municipality': 'Geographic municipality of registration',
            'Registration_Status': 'Legal entity type (LLC, Individual, etc.)',
            'Legal_Description': 'Legal basis citation for the fine',
            'Taxpayers_Count': 'Number of registered taxpayers in category',
            'Fines_Issued': 'Total number of fines issued',
            'Fine_Amount': 'Monetary value of fines in euros (€)',
            'Days_in_month': 'Number of days in the reported month',
            'Quarter': 'Calendar quarter (1-4)',
            'Month_sin': 'Sine-transformed month for cyclical encoding',
            'Month_cos': 'Cosine-transformed month for cyclical encoding',
            'Quarter_sin': 'Sine-transformed quarter for cyclical encoding',
            'Quarter_cos': 'Cosine-transformed quarter for cyclical encoding',
            'Law_Article': 'Extracted law/article identifier combination',
            'Law_Article_Freq': 'Frequency-encoded law/article occurrence'
        }
        
        self.column_types = {
            'Year': 'Categorical',
            'Month': 'Categorical',
            'Sector': 'Categorical',
            'Municipality': 'Categorical',
            'Registration_Status': 'Categorical',
            'Legal_Description': 'Text',
            'Taxpayers_Count': 'Numerical (Discrete)',
            'Fines_Issued': 'Numerical (Discrete)',
            'Fine_Amount': 'Numerical (Continuous)',
            'Days_in_month': 'Numerical (Discrete)',
            'Quarter': 'Categorical',
            'Month_sin': 'Numerical (Continuous)',
            'Month_cos': 'Numerical (Continuous)',
            'Quarter_sin': 'Numerical (Continuous)',
            'Quarter_cos': 'Numerical (Continuous)',
            'Law_Article': 'Categorical',
            'Law_Article_Freq': 'Numerical (Continuous)'
        }
        
    def _create_column_map(self):
        self.column_map = {
            'Viti': 'Year',
            'Muaji': 'Month',
            'Përshkrimi i Sektorit': 'Sector',
            'Komuna': 'Municipality',
            'Statusi i Regjistrimit': 'Registration_Status',
            'Përshkrimi i Gjobave në bazë të Ligjit': 'Legal_Description',
            'Numri i Tatimpaguesve': 'Taxpayers_Count',
            'Numri i Gjobave të Lëshuara': 'Fines_Issued',
            'Vlera e Gjobave të Lëshuara': 'Fine_Amount'
        }
        
    def _base_clean(self, df):
        df = (df.rename(columns=self.column_map)
              .pipe(self._clean_text)
              .pipe(self._convert_numeric)
              .pipe(self._handle_missing)
              .pipe(self._handle_outliers)
              .pipe(self._remove_duplicates))
        return df
    
    def _clean_text(self, df):
        text_cols = ['Sector', 'Municipality', 'Registration_Status']
        df[text_cols] = df[text_cols].apply(
            lambda x: x.str.strip().str.upper()
        )
        df = self._normalize_municipalities(df)
        return df
    
    def _normalize_municipalities(self, df):
        df['Municipality'] = df['Municipality'].apply(
            lambda x: unicodedata.normalize('NFKD', x)
            .encode('ascii', 'ignore')
            .decode('utf-8')
        )
        return df
    
    def _convert_numeric(self, df):
        numeric_cols = ['Taxpayers_Count', 'Fines_Issued', 'Fine_Amount']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        return df
    
    def _handle_missing(self, df):
        return df.dropna(subset=['Fine_Amount'])
    
    def _remove_duplicates(self, df):
        return df.drop_duplicates()
    
    def _handle_outliers(self, df):
        """Clip Fine_Amount using IQR bounds."""
        if 'Fine_Amount' not in df.columns:
            return df

        # Calculate IQR bounds
        q1 = df['Fine_Amount'].quantile(0.25)
        q3 = df['Fine_Amount'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Store bounds for consistency (optional)
        self.iqr_bounds = {'lower': lower_bound, 'upper': upper_bound}

        # Clip values
        df['Fine_Amount'] = df['Fine_Amount'].clip(
            lower=lower_bound, 
            upper=upper_bound
        )
        return df
    
    def _enhance_features(self, df):
        sector_map = {
            r'TR\.ME SHUMICE\.': 'WHOLESALE_RETAIL',
            r'NDERTIMTARIA': 'CONSTRUCTION',
            r'INDUSTRIA PERPUNUESE': 'PROCESSING_INDUSTRY',
            r'AKOMODIMI\.': 'HOSPITALITY',
            r'ARTET\.': 'ARTS_RECREATION',
            r'PERSON FIZIK': 'INDIVIDUAL',
            r'MUNGON AKTIVITETI': 'INACTIVE',
            r'INFORMIMI DHE KOMUNIKIMI': 'INFORMATION_COMMUNICATION',
            r'INDUSTRIA NXJERRESE': 'EXTRACTIVE_INDUSTRY'
        }

        for pattern, replacement in sector_map.items():
            df['Sector'] = df['Sector'].replace(
                regex=pattern, 
                value=replacement
            )

        # Enhanced registration status mapping
        df['Registration_Status'] = df['Registration_Status'].replace({
            'SH.P.K.': 'LLC',
            'PERSON FIZIK': 'INDIVIDUAL',
            'INDIVIDUAL': 'INDIVIDUAL',
            'SHOQËRI AKCIONARE': 'CORPORATION',
            'KOMPANI E HUAJ': 'FOREIGN_COMPANY',
            'OJQ': 'NGO'
        })
        
        return df
    
    def drop_unused_columns(self, df):
        return df.drop(
            columns=['Year', 'Month', 'Legal_Description', 'Law_Article', 'Registration_Status'],
            errors='ignore'
        )

    def drop_column_names(self, _, input_features):
        drop_cols = {'Year', 'Month', 'Legal_Description', 'Law_Article', 'Registration_Status'}
        return [col for col in input_features if col not in drop_cols]

    def to_dataframe_with_columns(self, x, columns):
        return pd.DataFrame(x, columns=columns)

    
    def build_preprocessing_pipeline(self):
        drop_transformer = FunctionTransformer(
            self.drop_unused_columns,
            feature_names_out=self.drop_column_names
        )

        feature_engineering = Pipeline([
            ('date_features', DateEncoder()),
            ('legal_extractor', LegalComponentExtractor()),
            ('frequency_encoder', FrequencyEncoder(min_freq=5)),
            ('drop_columns', drop_transformer)
        ])

        numeric_features = ['Taxpayers_Count', 'Days_in_month']
        cyclic_features = ['Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos']
        categorical_features = ['Sector', 'Municipality']

        # Numeric pipeline with DataFrame output
        numeric_transformer = Pipeline([
            ('num_impute', DataFrameKNNImputer(n_neighbors=5)),
            ('num_scale', DataFrameRobustScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('to_df', DataFrameWrapper(columns=categorical_features))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cyclic_scale', DataFrameRobustScaler(), cyclic_features),
            ('target_enc', categorical_transformer, categorical_features),
            ('freq_enc', 'passthrough', ['Law_Article_Freq'])
        ], remainder='passthrough')

        full_pipeline = Pipeline([
            ('feat_eng', feature_engineering),
            ('target_encoder', TargetEncoder(cols=categorical_features, smoothing=1.0)),
            ('preprocessor', preprocessor),
            # ('log_transform', log_transformer),
            ('feature_selection', SelectKBest(mutual_info_regression, k=10))
        ])

        return full_pipeline

    def preprocess(self):
        """Main method to execute full preprocessing pipeline"""
        df = self._base_clean(self.df)
        df = self._enhance_features(df)
        return df
    
    def to_csv(self, output_path, processed=True):
        """Save the data to a CSV file"""
        data = self.preprocess() if processed else self.df
        data.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
    def plot_time_series(self, df=None, title='Time Series of Fines', figsize=(12, 6), save_path=None):
        """Plot monthly total fines over time."""
        if df is None:
            df = self.preprocess()
        
        # Create Date from Year and Month
        df['Date'] = pd.to_datetime(
            df['Year'].astype(str) + '-' + 
            df['Month'].astype(str).str.zfill(2) + '-01'
        )
        
        plt.figure(figsize=figsize)
        sns.lineplot(
            data=df.groupby('Date')['Fine_Amount'].sum().reset_index(),
            x='Date', y='Fine_Amount'
        )
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Total Fine Amount (€)')
        plt.grid(True)
        self._handle_plot_output(save_path)
        
    def generate_data_quality_report(self, df=None):
        if df is None:
            df = self.preprocess()
        
        report = {
            'Basic Info': {
                'Total Records': len(df),
                'Total Features': len(df.columns)
            },
            'Missing Values': (df.isnull().sum() / len(df) * 100
                            ).round(2).to_dict(),
            'Exact Duplicates': df.duplicated().sum(),
            'Near Duplicates': self._find_near_duplicates(df),
            'Zero Values': (df.select_dtypes(include=np.number) == 0
                        ).sum().to_dict(),
            'Constant Features': self._find_constant_features(df),
            'Data Types': df.dtypes.value_counts().to_dict()
        }
        
        print("\nDATA QUALITY REPORT")
        print("===================")
        print(f"Total Records: {report['Basic Info']['Total Records']}")
        print(f"Total Features: {report['Basic Info']['Total Features']}\n")
        
        print("Missing Values (%):")
        print(pd.Series(report['Missing Values']).to_string())
        
        print(f"\nExact Duplicates: {report['Exact Duplicates']}")
        print(f"Near Duplicates: {report['Near Duplicates']}")
        
        print("\nZero Values in Numerical Columns:")
        print(pd.Series(report['Zero Values']).to_string())
        
        print("\nConstant Features:")
        print(", ".join(report['Constant Features']) if report['Constant Features'] else "None")
        
        print("\nData Types Distribution:")
        print(pd.Series(report['Data Types']).to_string())
        
        return report

    def _find_near_duplicates(self, df, threshold=0.95):
        """Identify near-duplicates using hashing"""
        sample_hash = df.head(1000).apply(lambda x: hash(tuple(x)), axis=1)
        return int(sample_hash.nunique() / len(sample_hash) < threshold)

    def _find_constant_features(self, df):
        """Identify features with <2 unique values"""
        return df.columns[df.nunique() == 1].tolist()

    def plot_distributions(self, df=None, columns=None, figsize=(15, 5), save_path=None):
        """Plot distributions of numerical features."""
        if df is None:
            df = self.preprocess()
        columns = columns or ['Fine_Amount', 'Taxpayers_Count', 'Fines_Issued']
        
        plt.figure(figsize=figsize)
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        self._handle_plot_output(save_path)
        
    def get_key_statistics(self, df=None):
        """Generate key numerical/categorical statistics"""
        if df is None:
            df = self.preprocess()
        
        stats = {
            'Numerical': df.select_dtypes(include=np.number).agg(
                ['mean', 'median', 'std', 'min', 'max']
            ).T,
            'Categorical': pd.DataFrame({
                'Unique Count': df.select_dtypes(exclude=np.number).nunique(),
                'Top Category': df.select_dtypes(exclude=np.number).mode().iloc[0],
                'Top Frequency': df.select_dtypes(exclude=np.number).apply(
                    lambda x: x.value_counts().max() / len(x)
                ).round(2)
            })
        }
        
        print("\nKEY STATISTICS")
        print("==============")
        print("\nNumerical Features:")
        print(stats['Numerical'].to_string())
        
        print("\n\nCategorical Features:")
        print(stats['Categorical'].to_string())
        
        return stats
    
    def temporal_analysis_report(self, df=None):
        if df is None:
            df = self.preprocess()
        
        temporal = {
            'Yearly': df.groupby('Year')['Fine_Amount'].agg(['sum', 'mean', 'count']),
            'Monthly': df.groupby('Month')['Fine_Amount'].mean(),
            # 'Seasonal': df.groupby('Quarter')['Fine_Amount'].sum()
        }
        
        print("\nTEMPORAL ANALYSIS")
        print("=================")
        print("\nYearly Trends:")
        print(temporal['Yearly'].to_string())
        
        print("\nMonthly Averages:")
        print(temporal['Monthly'].to_string())
        
        # print("\nSeasonal Totals:")
        # print(temporal['Seasonal'].to_string())
        
        return temporal
    
    def categorical_distribution_report(self, df=None, top_n=5):
        if df is None:
            df = self.preprocess()
        
        cat_cols = df.select_dtypes(exclude=np.number).columns
        report = {}
        
        for col in cat_cols:
            dist = df[col].value_counts(normalize=True).head(top_n)
            report[col] = dist.to_dict()
        
        print("\nCATEGORICAL DISTRIBUTIONS")
        print("========================")
        for col, dist in report.items():
            print(f"\n{col} (Top {top_n}):")
            print(pd.Series(dist).to_string(header=False))
        
        return report
    
    def legal_component_summary(self, df=None):
        """Analyze extracted legal components"""
        if df is None:
            df = self.preprocess()
        print("\nthis is df: ", df.head())
        summary = {
            'Total Unique Laws': df['Legal_Description'].nunique(),
            'Most Common Law': df['Legal_Description'].value_counts().idxmax(),
            'Law Coverage': (
                df['Legal_Description'].value_counts(normalize=True)
                .cumsum()
                .head(10)
                .to_dict()
            )
        }
        
        print("\nLEGAL COMPONENT ANALYSIS")
        print("=======================")
        print(f"Total Unique Laws: {summary['Total Unique Laws']}")
        print(f"Most Common Law: {summary['Most Common Law']}")
        print("\nTop 10 Law Coverage:")
        print(pd.Series(summary['Law Coverage']).to_string())
        
        return summary
    
    def outlier_analysis_report(self, df=None):
        """Detailed outlier statistics"""
        if df is None:
            df = self.preprocess()
        
        num_cols = df.select_dtypes(include=np.number).columns
        report = {}
        
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            report[col] = {
                'Outliers Count': len(outliers),
                'Outliers %': len(outliers)/len(df)*100,
                'Min Outlier': outliers.min() if not outliers.empty else None,
                'Max Outlier': outliers.max() if not outliers.empty else None
            }
        
        print("\nOUTLIER ANALYSIS")
        print("================")
        print(pd.DataFrame(report).T.to_string())
        
        return report
    
    def correlation_report(self, df=None, threshold=0.7):
        if df is None:
            df = self.preprocess()
        
        corr_matrix = df.select_dtypes(include=np.number).corr().abs()
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if mask[i,j]:
                    pairs.append((
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i,j]
                    ))
        
        high_corr = pd.Series(
            {f"{p[0]} vs {p[1]}": p[2] for p in pairs if p[2] > threshold},
            name='Correlation'
        ).sort_values(ascending=False)
        
        print("\nFEATURE CORRELATIONS")
        print("====================")
        if not high_corr.empty:
            print("Highly Correlated Features (|r| > {}):".format(threshold))
            print(high_corr.to_string())
        else:
            print("No significant correlations above {}".format(threshold))
        
        return high_corr
    
    def dataset_version_info(self, df=None):
        if df is None:
            df = self.preprocess()
        
        info = {
            'Shape': df.shape,
            'Columns Hash': hash(tuple(sorted(df.columns))),
            'Data Hash': hash(pd.util.hash_pandas_object(df).sum()),
            'Last Modified': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("\nDATASET VERSION INFO")
        print("====================")
        print(f"Shape: {info['Shape']}")
        print(f"Columns Hash: {info['Columns Hash']}")
        print(f"Data Hash: {info['Data Hash']}")
        print(f"Last Modified: {info['Last Modified']}")
        
        return info
        
    def distribution_metrics_report(self, df=None):
        """Show skewness and kurtosis for numerical features"""
        if df is None:
            df = self.preprocess()
        
        num_cols = df.select_dtypes(include=np.number).columns
        report = pd.DataFrame({
            'Skewness': df[num_cols].skew(),
            'Kurtosis': df[num_cols].kurt()
        }).sort_values('Skewness', ascending=False)
        
        print("\nDISTRIBUTION METRICS")
        print("====================")
        print(report.to_string())
        
        return report

    def plot_categorical_counts(self, df=None, columns=None, top_n=10, figsize=(15, 5), save_path=None):
        """Plot top categories for categorical features."""
        if df is None:
            df = self.preprocess()
        columns = columns or ['Sector', 'Municipality', 'Registration_Status']
        
        plt.figure(figsize=figsize)
        for i, col in enumerate(columns, 1):
            plt.subplot(1, len(columns), i)
            counts = df[col].value_counts().nlargest(top_n)
            sns.barplot(x=counts.values, y=counts.index)
            plt.title(f'Top {top_n} {col} Categories')
            plt.xlabel('Count')
            plt.ylabel('')
        plt.tight_layout()
        self._handle_plot_output(save_path)

    def plot_outliers_comparison(self, figsize=(12, 6), save_path=None):
        """Compare distribution before/after outlier handling."""
        original = self._base_clean(self.df.copy())
        processed = self.preprocess()
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.boxplot(y=original['Fine_Amount'])
        plt.title('Original Fine Amount')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(y=processed['Fine_Amount'])
        plt.title('Processed Fine Amount')
        self._handle_plot_output(save_path)

    def plot_correlations(self, df=None, figsize=(12, 10), save_path=None):
        """Plot correlation heatmap of numerical features."""
        if df is None:
            df = self.preprocess()
        
        numerical_cols = df.select_dtypes(include=np.number).columns
        corr_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            mask=np.triu(np.ones_like(corr_matrix))
        )
        plt.title('Feature Correlation Matrix')
        self._handle_plot_output(save_path)
        
    def print_column_explanations(self):
        """Print formatted table of column metadata"""
        meta_df = pd.DataFrame({
            'Description': self.column_descriptions,
            'Data Type': self.column_types
        })
        print("\nCOLUMN METADATA")
        print(meta_df.to_string())
        
    def generate_data_profile(self, df=None, display_sample=True):
        """
        Generate comprehensive data profile report including:
        - Data sample
        - Data types
        - Missing values
        - Basic statistics
        - Outlier analysis
        - Skewness analysis
        """
        if df is None:
            df = self.preprocess()
            
        profile = {}
        
        # Data sample
        if display_sample:
            print("\nDATA SAMPLE")
            print(df.head(3).to_string(index=False))
            print("...")

        # Data types and null values
        type_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().mean() * 100).round(2)
        })

        # Unique values
        type_info['Unique Values'] = df.nunique()

        # Numerical analysis
        num_cols = df.select_dtypes(include=np.number).columns
        num_stats = df[num_cols].agg(['min', 'max', 'mean', 'std']).T
        num_stats = num_stats.rename(columns={
            'min': 'Minimum',
            'max': 'Maximum',
            'mean': 'Mean',
            'std': 'Std Dev'
        })

        # Outlier analysis
        num_stats['Outliers'] = self._calculate_outliers(df[num_cols])

        # Skewness
        num_stats['Skewness'] = df[num_cols].skew().round(3)

        # Combine all information
        profile['type_info'] = type_info
        profile['num_stats'] = num_stats
        
        print("\nDATA PROFILE REPORT")
        print("===================")
        print("\nDATA TYPES AND NULL VALUES")
        print(type_info.to_string())
        
        print("\n\nNUMERICAL FEATURE STATISTICS")
        print(num_stats.to_string())
        
        return profile
        
    def _calculate_outliers(self, df, method='iqr'):
        """Calculate number of outliers using specified method"""
        outliers = []
        
        if method == 'iqr':
            for col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = ((df[col] < lower) | (df[col] > upper)).sum()
                outliers.append(count)
                
        elif method == 'zscore':
            for col in df.columns:
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                count = (z > 3).sum()
                outliers.append(count)
                
        return pd.Series(outliers, index=df.columns)
        
    def plot_skewness(self, df=None, figsize=(10, 6), save_path=None):
        """Visualize skewness of numerical features"""
        if df is None:
            df = self.preprocess()
            
        num_cols = df.select_dtypes(include=np.number).columns
        skewness = df[num_cols].skew().sort_values()

        plt.figure(figsize=figsize)
        sns.barplot(x=skewness.values, y=skewness.index, palette='viridis')
        plt.axvline(0, color='k', linestyle='--')
        plt.title('Feature Skewness Analysis')
        plt.xlabel('Skewness Value')
        plt.ylabel('Features')
        self._handle_plot_output(save_path)

    def plot_missing_values(self, df=None, figsize=(10, 6), save_path=None):
        """Visualize missing values in raw data."""
        df = df if df is not None else self.df
        missing = df.isnull().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        
        plt.figure(figsize=figsize)
        sns.barplot(x=missing.values, y=missing.index, orient='h')
        plt.title('Missing Values Distribution')
        plt.xlabel('Proportion Missing')
        plt.ylabel('Features')
        self._handle_plot_output(save_path)

    def _handle_plot_output(self, save_path):
        """Handle plot display/saving."""
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()