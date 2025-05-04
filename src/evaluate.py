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
