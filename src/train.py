import os
import sys
import warnings
import pandas as pd
import joblib
import numpy as np


class ModelTrainer:
    def __init__(self, data_path='data/raw/gjobat-all.csv', model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model = None
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.evaluation_results = []  # Stores metrics until final print

        self._load_and_prepare_data()
        # self._time_based_split()

    def _load_and_prepare_data(self):
        print("📥 Loading data...")
        data = pd.read_csv(self.data_path)
        
        self.preprocessor = DataPreprocessor(data)
        self.preprocessor.print_column_explanations()
        profile = self.preprocessor.generate_data_profile()
        print("\nNumerical Statistics:")
        print(profile['num_stats'])
        self.preprocessor.generate_data_quality_report()
        self.preprocessor.temporal_analysis_report()
        self.preprocessor.correlation_report(threshold=0.6)
        self.preprocessor.distribution_metrics_report()
        self.preprocessor.legal_component_summary()
        self.preprocessor.dataset_version_info()
        # Generate skewness visualization
        self.preprocessor.plot_skewness(save_path='images/skewness.png')
        self.preprocessor.plot_time_series(save_path='images/time_series.png')
        self.preprocessor.plot_distributions(save_path='images/distributions.png')
        self.preprocessor.plot_categorical_counts(top_n=5, save_path='images/categorical_counts.png')
        self.preprocessor.plot_correlations(figsize=(10, 8), save_path='images/correlation_matrix.png')
        df_processed = self.preprocessor.preprocess()
        self.df_clean = df_processed.sort_values('Year')
        print("✅ Data loaded and preprocessed.")
      def _time_based_split(self):
        print("⏳ Creating time-based split...")
        split_year = 2023
        train_mask = self.df_clean['Year'] < split_year
        
        self.X_train = self.df_clean[train_mask].drop('Fine_Amount', axis=1)
        self.y_train = self.df_clean[train_mask]['Fine_Amount']
        self.X_test = self.df_clean[~train_mask].drop('Fine_Amount', axis=1)
        self.y_test = self.df_clean[~train_mask]['Fine_Amount']
        print(f"✅ Split complete - Train: {len(self.X_train)}, Test: {len(self.X_test)}")

    def _evaluate_model(self, model):
        """Calculate metrics and return formatted results"""
        try:
            check_is_fitted(model)
        except Exception as e:
            return [f"⚠️ Model evaluation failed: {e}"]
        
        metrics = []
        sets = [('Train', self.X_train, self.y_train), ('Test', self.X_test, self.y_test)]
        
        for set_name, X, y in sets:
            y_pred = model.predict(X)
            metrics.append(
                f"{set_name} Performance:\n"
                f"R²: {r2_score(y, y_pred):.3f}, "
                f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}, "
                f"MAE: {mean_absolute_error(y, y_pred):.2f}"
            )
        return metrics
