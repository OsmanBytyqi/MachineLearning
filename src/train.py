import os
import sys
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from data_preprocessor import DataPreprocessor
from sklearn.utils.validation import check_is_fitted

# Targeted warning suppression for pipeline fitting messages
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=".*Pipeline instance is not fitted yet.*")

class ModelTrainer:
    def __init__(self, data_path='data/raw/gjobat-all.csv', model_dir='models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model = None
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.evaluation_results = []  # Stores metrics until final print
        
        self.metric_df = pd.DataFrame(columns=['Model', 'Dataset', 'R2', 'RMSE', 'MAE'])
        self.feature_importance = {}

        self._load_and_prepare_data()
        self._time_based_split()

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
        metrics = []
        sets = [('Train', self.X_train, self.y_train), 
                ('Test', self.X_test, self.y_test)]
        
        for set_name, X, y in sets:
            y_pred = model.predict(X)
            metrics.append({
                'Model': type(model.named_steps['model']).__name__,
                'Dataset': set_name,
                'R2': r2_score(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred)
            })
            
            if set_name == 'Test':
                self._plot_predictions(y, y_pred, model)
                
        return metrics
    
    def _plot_predictions(self, y_true, y_pred, model):
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 'k--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predictions')
        plt.title('Actual vs Predicted Values')
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title('Residual Distribution')
        
        model_name = type(model.named_steps['model']).__name__
        plt.suptitle(f'{model_name} Prediction Analysis')
        plt.tight_layout()
        plt.savefig(f'images/{model_name}_predictions.png')
        plt.close()
        
    def _plot_feature_importance(self, model):
        try:
            preprocessor = model.named_steps['preprocessing']
            feature_names = preprocessor.get_feature_names_out()
            
            model_name = type(model.named_steps['model']).__name__
            importance = model.named_steps['model'].feature_importances_
            
            # Get preprocessed test data
            X_test_preprocessed = preprocessor.transform(self.X_test)
            
            # Validate array lengths
            if len(feature_names) != len(importance):
                print(f"Mismatch: {len(feature_names)} features vs {len(importance)} importance values")
                min_length = min(len(feature_names), len(importance))
                feature_names = feature_names[:min_length]
                importance = importance[:min_length]

            # Calculate permutation importance with correct input
            result = permutation_importance(
                model.named_steps['model'], 
                X_test_preprocessed, 
                self.y_test,
                n_repeats=5, 
                random_state=42,
                n_jobs=-1
            )
            
            # Validate permutation importance length
            perm_importance = result.importances_mean
            if len(perm_importance) != len(importance):
                print(f"Truncating permutation importance from {len(perm_importance)} to {len(importance)}")
                perm_importance = perm_importance[:len(importance)]

            # Create DataFrame with validated lengths
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Gain Importance': importance,
                'Permutation Importance': perm_importance
            }).sort_values('Gain Importance', ascending=False).head(10)

            # Plotting code remains the same
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Gain Importance', y='Feature', data=fi_df, color='b', label='Gain')
            sns.barplot(x='Permutation Importance', y='Feature', data=fi_df, 
                    color='r', alpha=0.5, label='Permutation')
            plt.title(f'{model_name} Feature Importance')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'images/{model_name}_feature_importance.png')
            plt.close()
            
            self.feature_importance[model_name] = fi_df

        except Exception as e:
            print(f"\n⚠️ Feature importance visualization failed: {str(e)}")
            print("Continuing without feature importance plot...")

    def train_model(self, model_type='random_forest'):
        preprocessing_pipeline = self.preprocessor.build_preprocessing_pipeline()

        models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'model__n_estimators': [200, 300],
                    'model__max_depth': [None, 10],
                    'model__min_samples_split': [2, 5]
                }
            },
            'xgboost': {
                'model': XGBRegressor(random_state=42),
                'params': {
                    'model__learning_rate': [0.05, 0.1],
                    'model__max_depth': [4, 6],
                    'model__n_estimators': [200, 300]
                }
            },
            'catboost': {
                'model': CatBoostRegressor(verbose=0, random_state=42),
                'params': {
                    'model__iterations': [500, 1000],
                    'model__depth': [6, 8],
                    'model__learning_rate': [0.01, 0.05]
                }
            }
        }

        print(f"\n🚀 Training {model_type.replace('_', ' ').title()}...")
        pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('model', models[model_type]['model'])
        ])

        try:
            Xt = pipeline[:-1].fit_transform(self.X_train)
            if np.isnan(Xt).any():
                nan_count = np.isnan(Xt).sum()
                nan_cols = np.where(np.isnan(Xt).any(axis=0))[0]
                print(f"⚠️ Preprocessed data contains {nan_count} NaNs in columns: {nan_cols}")
                print("Adding additional imputation...")
                pipeline.steps.insert(-1, ('emergency_imputer', SimpleImputer()))
        except Exception as e:
            print(f"Pre-transform validation failed: {e}")

        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            pipeline,
            param_grid=models[model_type]['params'],
            cv=tscv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        try:
            grid_search.fit(self.X_train, self.y_train)
        except ValueError as e:
            print(f"🔥 Critical pipeline error: {e}")
            print("Running emergency imputation...")
            self._emergency_imputation()
            grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        self._plot_feature_importance(grid_search.best_estimator_)
        self.evaluation_results.extend(
            [f"\n----- {model_type.replace('_', ' ').title()} Metrics -----"] 
            + self._evaluate_model(self.model)
        )
        self._save_model(f'{model_type}_model.pkl')

    def _emergency_imputation(self):
        print("Applying emergency imputation...")
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        self.y_train = self.y_train.fillna(0)
        
    def _validate_feature_names(self, pipeline):
        try:
            print("\nPipeline Feature Name Debug:")
            for step_name, step in pipeline.named_steps.items():
                print(f"\n{step_name} ({type(step).__name__}):")
                if hasattr(step, 'get_feature_names_out'):
                    print("Output features:", step.get_feature_names_out()[:5])
                elif hasattr(step, 'feature_names_in_'):
                    print("Input features:", step.feature_names_in_[:5])
                else:
                    print("Feature name tracking not available")
        except Exception as e:
            print(f"Feature validation failed: {e}")
        
    def generate_reports(self):
        # Create metric table
        metric_table = self.metric_df.round(3)
        metric_table.to_markdown('images/metric_table.md', index=False)
        
        # Create feature importance tables
        for model_name, fi_df in self.feature_importance.items():
            fi_df.round(3).to_markdown(
                f'images/{model_name}_feature_importance.md', 
                index=False
            )
        
        print("📊 Reports generated in images/ directory")

    def _save_model(self, filename):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, model_path)
        print(f"\n💾 Model saved to {model_path}")

    def print_metrics(self):
        print("\n📊 FINAL MODEL METRICS:")
        print("\n".join(str(result) for result in self.evaluation_results))

if __name__ == "__main__":
    trainer = ModelTrainer()
    os.makedirs('images', exist_ok=True)
    # Train models and collect metrics
    for model_type in ['random_forest', 'xgboost', 'catboost']:
        trainer.train_model(model_type)
    
    # Print all metrics after training completes
    trainer.generate_reports()
    trainer.print_metrics()