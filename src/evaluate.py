import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.metrics_file = os.path.join(results_dir, "summary.txt")
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: Optional[str] = None) -> Dict[str, float]:
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

    def compare_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        results = {}
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = metrics
        return results

    def get_best_model(self, results: Dict[str, Dict[str, float]], metric: str = 'r2') -> Tuple[str, Dict[str, float]]:
        if not results:
            return None, {}
        if metric == 'r2':
            best_model_name = max(results.keys(), key=lambda k: results[k].get('r2', 0))
        else:
            best_model_name = min(results.keys(), key=lambda k: results[k].get(metric, float('inf')))
        return best_model_name, results[best_model_name]

    def calculate_feature_importance(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
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

    def model_error_analysis(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        y_pred = model.predict(X_test)
        error_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred,
            'AbsError': np.abs(y_test - y_pred),
            'PercentError': 100 * np.abs((y_test - y_pred) / (y_test + 1e-10))
        })
        return error_df

    def save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        with open(self.metrics_file, 'w') as f:
            f.write("# MODEL PERFORMANCE SUMMARY\n\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name.upper()} Results:\n")
                for metric_name, value in metrics.items():
                    if metric_name != 'predictions':
                        f.write(f"{metric_name.upper()}: {value:.4f}\n")
                f.write("-" * 80 + "\n")
            
            best_model_name, best_metrics = self.get_best_model(results)
            f.write(f"\n\nBEST MODEL PERFORMANCE: {best_model_name.upper()} with R2: {best_metrics.get('r2', 0):.4f}\n")
            
            best_r2 = best_metrics.get('r2', 0)
            if best_r2 >= 0.9:
                f.write("\nTARGET ACHIEVED: R² score of 90% or better has been reached!\n")
            else:
                f.write(f"\nTARGET NOT YET ACHIEVED: R² score is {best_r2:.4f}, " 
                        f"which is {(0.9 - best_r2) * 100:.2f}% away from the 90% target.\n")
        
        self.logger.info(f"Results saved to {self.metrics_file}")

    def load_results(self) -> Dict[str, Dict[str, float]]:
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
