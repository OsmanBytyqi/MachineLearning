import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from typing import Dict, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    """Class for evaluating and comparing machine learning models."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the model evaluator."""
        self.results_dir = results_dir
        self.metrics_file = os.path.join(results_dir, "summary.txt")
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model performance and return metrics."""
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        if model_name:
            self.logger.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Compare multiple models and return all metrics."""
        results = {}
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics
        
        return results
    
    def get_best_model(self, results: Dict[str, Dict[str, float]], 
                      metric: str = 'r2') -> Tuple[str, Dict[str, float]]:
        """Identify the best model based on the specified metric."""
        if not results:
            return None, {}
        
        if metric == 'r2':
            best_model_name = max(results.keys(), key=lambda k: results[k].get('r2', 0))
        else:
            best_model_name = min(results.keys(), key=lambda k: results[k].get(metric, float('inf')))
        
        return best_model_name, results[best_model_name]
    
    def calculate_feature_importance(self, model: Any, X_test: pd.DataFrame, 
                                    y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
        """Calculate permutation feature importance for a model."""
        feature_names = X_test.columns.tolist()
        
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=n_repeats, 
            random_state=42, 
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_results(self, results: Dict[str, Dict[str, float]], 
                    train_results: Dict[str, Dict[str, float]] = None) -> None:
        """Save evaluation results to a file."""
        with open(self.metrics_file, 'w') as f:
            f.write("# MODEL PERFORMANCE SUMMARY\n\n")
            
            for model_name, metrics in results.items():
                f.write(f"\n{model_name.upper()} Results (TEST):\n")
                for metric_name, value in metrics.items():
                    if metric_name != 'predictions':
                        f.write(f"{metric_name.upper()}: {value:.4f}\n")
                
                # Include training results if available
                if train_results and model_name in train_results:
                    f.write(f"\n{model_name.upper()} Results (TRAIN):\n")
                    for metric_name, value in train_results[model_name].items():
                        if metric_name != 'predictions':
                            f.write(f"{metric_name.upper()}: {value:.4f}\n")
                    
                    # Calculate and show gap
                    if 'r2' in metrics and 'r2' in train_results[model_name]:
                        r2_gap = train_results[model_name]['r2'] - metrics['r2']
                        f.write(f"R2_GAP (TRAIN-TEST): {r2_gap:.4f}\n")
                
                f.write("-" * 80 + "\n")
            
            best_model_name, best_metrics = self.get_best_model(results)
            f.write(f"\n\nBEST MODEL PERFORMANCE: {best_model_name.upper()} with TEST R2: {best_metrics.get('r2', 0):.4f}\n")
            
            best_r2 = best_metrics.get('r2', 0)
            if best_r2 >= 0.9:
                f.write("\nTARGET ACHIEVED: R² score of 90% or better has been reached!\n")
            else:
                f.write(f"\nTARGET NOT YET ACHIEVED: R² score is {best_r2:.4f}, " 
                        f"which is {(0.9 - best_r2) * 100:.2f}% away from the 90% target.\n")
            
        self.logger.info(f"Results saved to {self.metrics_file}")
    
    def load_results(self) -> Dict[str, Dict[str, float]]:
        """Load evaluation results from the summary file."""
        results = {}
        
        if not os.path.exists(self.metrics_file):
            self.logger.warning(f"Results file not found: {self.metrics_file}")
            return results
        
        current_model = None
        with open(self.metrics_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.endswith("Results:"):
                    current_model = line.split()[0].lower()
                    results[current_model] = {}
                    continue
                
                if current_model and any(m in line for m in ["MAE:", "RMSE:", "R2:"]):
                    parts = line.split(":")
                    if len(parts) == 2:
                        metric_name = parts[0].strip().lower()
                        metric_value = float(parts[1].strip())
                        results[current_model][metric_name] = metric_value
        
        return results

    def model_error_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Analyze prediction errors to identify patterns."""
        y_pred = model.predict(X_test)
        
        error_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred,
            'AbsError': np.abs(y_test - y_pred),
            'PercentError': 100 * np.abs((y_test - y_pred) / (y_test + 1e-10))
        })
        
        return error_df

if __name__ == "__main__":
    # Load existing models
    models_dir = "models"
    models = {}
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".joblib"):
            model_name = model_file.split(".")[0]
            model_path = os.path.join(models_dir, model_file)
            models[model_name] = joblib.load(model_path)
    
    # Load preprocessed data
    from preprocessing import DataPreprocessor
    
    target_column = 'Vlera e Gjobave të Lëshuara'
    preprocessor = DataPreprocessor()
    
    # Load the processed data
    df_processed = preprocessor.load_data("data/processed/gjobat_processed.csv")
    preprocessor.df_processed = df_processed
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_train_test_data(
        target_column=target_column
    )
    
    # Convert numpy arrays to pandas DataFrames with column names
    feature_names = df_processed.drop(columns=[target_column]).columns.tolist()
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_test = pd.Series(y_test)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    results = evaluator.compare_models(models, X_test, y_test)
    evaluator.save_results(results)
    
    # Print the best model
    best_model_name, best_metrics = evaluator.get_best_model(results)
    print(f"Best model: {best_model_name} with R² score: {best_metrics['r2']:.4f}") 