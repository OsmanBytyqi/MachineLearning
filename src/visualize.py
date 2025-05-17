import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Visualizer:
    """Class for creating visualizations of models and data."""
    
    def __init__(self, plots_dir: str = "results/plots"):
        """Initialize the visualizer."""
        self.plots_dir = plots_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.plots_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str, 
                               top_n: int = 20) -> str:
        """Plot feature importance for a model."""
        # Get the top N features
        top_features = importance_df.head(top_n)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.barh(range(len(top_features)), top_features['Importance'], align='center')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(self.plots_dir, f"{model_name}_feature_importance.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_predictions(self, y_test: pd.Series, y_pred: np.ndarray, 
                        model_name: str) -> Tuple[str, str, str]:
        """Create scatter plot of predictions vs actual values."""
        # Create prediction scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values - {model_name}')
        plt.tight_layout()
        
        predictions_path = os.path.join(self.plots_dir, f"{model_name}_predictions.png")
        plt.savefig(predictions_path)
        plt.close()
        
        # Create residual plot
        plt.figure(figsize=(10, 8))
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=min_val, xmax=max_val, colors='r', linestyles='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - {model_name}')
        plt.tight_layout()
        
        residuals_path = os.path.join(self.plots_dir, f"{model_name}_residuals.png")
        plt.savefig(residuals_path)
        plt.close()
        
        # Create histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution - {model_name}')
        plt.tight_layout()
        
        residual_dist_path = os.path.join(self.plots_dir, f"{model_name}_residual_dist.png")
        plt.savefig(residual_dist_path)
        plt.close()
        
        return predictions_path, residuals_path, residual_dist_path
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> List[str]:
        """Create comparison plots for all models."""
        output_paths = []
        
        # Extract model names and metrics
        model_names = list(results.keys())
        r2_values = [results[model].get('r2', 0) for model in model_names]
        rmse_values = [results[model].get('rmse', 0) for model in model_names]
        mae_values = [results[model].get('mae', 0) for model in model_names]
        
        # Plot R² comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, r2_values, color=['#2C8ECF', '#FF9A13', '#4CAF50'][:len(model_names)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)  # R² is typically between 0 and 1
        plt.axhline(y=0.9, color='r', linestyle='--', label='Target R² = 0.9')
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Model R² Performance Comparison')
        plt.legend()
        plt.tight_layout()
        
        r2_path = os.path.join(self.plots_dir, "r2_comparison.png")
        plt.savefig(r2_path)
        plt.close()
        output_paths.append(r2_path)
        
        # Plot RMSE comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, rmse_values, color=['#2C8ECF', '#FF9A13', '#4CAF50'][:len(model_names)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.title('Model RMSE Comparison')
        plt.tight_layout()
        
        rmse_path = os.path.join(self.plots_dir, "rmse_comparison.png")
        plt.savefig(rmse_path)
        plt.close()
        output_paths.append(rmse_path)
        
        # Plot MAE comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, mae_values, color=['#2C8ECF', '#FF9A13', '#4CAF50'][:len(model_names)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.title('Model MAE Comparison')
        plt.tight_layout()
        
        mae_path = os.path.join(self.plots_dir, "mae_comparison.png")
        plt.savefig(mae_path)
        plt.close()
        output_paths.append(mae_path)
        
        return output_paths
    
    def plot_error_analysis(self, error_df: pd.DataFrame, model_name: str) -> List[str]:
        """Create visualizations for error analysis."""
        output_paths = []
        
        # Error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(error_df['Error'], bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - {model_name}')
        plt.tight_layout()
        
        error_dist_path = os.path.join(self.plots_dir, f"{model_name}_error_distribution.png")
        plt.savefig(error_dist_path)
        plt.close()
        output_paths.append(error_dist_path)
        
        # Error vs Predicted Value
        plt.figure(figsize=(10, 6))
        plt.scatter(error_df['Predicted'], error_df['Error'], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Value')
        plt.ylabel('Error')
        plt.title(f'Error vs Predicted Value - {model_name}')
        plt.tight_layout()
        
        error_pred_path = os.path.join(self.plots_dir, f"{model_name}_error_vs_predicted.png")
        plt.savefig(error_pred_path)
        plt.close()
        output_paths.append(error_pred_path)
        
        # Percent Error Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(error_df['PercentError'], bins=30, alpha=0.7)
        plt.xlabel('Percent Error')
        plt.ylabel('Frequency')
        plt.title(f'Percent Error Distribution - {model_name}')
        plt.tight_layout()
        
        pct_error_path = os.path.join(self.plots_dir, f"{model_name}_percent_error.png")
        plt.savefig(pct_error_path)
        plt.close()
        output_paths.append(pct_error_path)
        
        return output_paths
    
    def plot_feature_distributions(self, df: pd.DataFrame, target_column: str, 
                                  top_n_features: int = 5) -> List[str]:
        """Plot distributions of the most important features."""
        output_paths = []
        
        # Get numeric columns excluding the target
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Get correlation with target
        correlations = df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
        top_features = correlations.head(top_n_features).index.tolist()
        
        # Plot distributions for top features
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.tight_layout()
            
            output_path = os.path.join(self.plots_dir, f"dist_{feature.replace(' ', '_')}.png")
            plt.savefig(output_path)
            plt.close()
            output_paths.append(output_path)
            
            # Scatter plot with target
            plt.figure(figsize=(10, 6))
            plt.scatter(df[feature], df[target_column], alpha=0.5)
            plt.title(f'{feature} vs {target_column}')
            plt.xlabel(feature)
            plt.ylabel(target_column)
            plt.tight_layout()
            
            scatter_path = os.path.join(self.plots_dir, f"scatter_{feature.replace(' ', '_')}_vs_target.png")
            plt.savefig(scatter_path)
            plt.close()
            output_paths.append(scatter_path)
        
        return output_paths
    
    def plot_correlation_matrix(self, df: pd.DataFrame, target_column: str) -> str:
        """Plot correlation matrix of features with target highlighted."""
        # Get numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                  square=True, linewidths=.5, annot=False, fmt='.2f', cbar_kws={"shrink": .5})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        output_path = os.path.join(self.plots_dir, "correlation_matrix.png")
        plt.savefig(output_path)
        plt.close()
        
        # Also create a simplified correlation with target
        if target_column in numeric_df.columns:
            target_corr = numeric_df.corrwith(numeric_df[target_column]).sort_values(ascending=False)
            
            # Filter out the target itself and low correlations
            target_corr = target_corr[target_corr.index != target_column]
            
            # Take top 15 correlations by magnitude (positive or negative)
            target_corr = target_corr.abs().sort_values(ascending=False).head(15).index
            target_corr = numeric_df[target_corr].corrwith(numeric_df[target_column]).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            target_corr.plot(kind='barh', color=plt.cm.RdYlGn(np.linspace(0, 1, len(target_corr))))
            plt.title(f'Feature Correlations with {target_column}')
            plt.xlabel('Correlation')
            plt.tight_layout()
            
            target_corr_path = os.path.join(self.plots_dir, "target_correlations.png")
            plt.savefig(target_corr_path)
            plt.close()
        
        return output_path

if __name__ == "__main__":
    # Example usage
    from preprocessing import DataPreprocessor
    from evaluate import ModelEvaluator
    import joblib
    import os
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Load data and preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("data/raw/gjobat-all.csv")
    
    # Preprocess data
    target_column = 'Vlera e Gjobave të Lëshuara'
    df_processed, _ = preprocessor.preprocess_data(df, target_column=target_column)
    
    # Create feature distribution plots
    visualizer.plot_feature_distributions(df_processed, target_column)
    
    # Create correlation matrix plot
    visualizer.plot_correlation_matrix(df_processed, target_column)
    
    # Load evaluation results
    evaluator = ModelEvaluator()
    results = evaluator.load_results()
    
    # Create model comparison plots
    visualizer.plot_model_comparison(results)
    
    # Load models for more detailed plots
    models_dir = "models"
    if os.path.exists(models_dir):
        for model_file in os.listdir(models_dir):
            if model_file.endswith(".joblib"):
                model_name = model_file.split(".")[0]
                model_path = os.path.join(models_dir, model_file)
                model = joblib.load(model_path)
                
                # Get test data
                _, X_test, _, y_test, _ = preprocessor.prepare_train_test_data(
                    df_processed, target_column=target_column, test_size=0.2
                )
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Create prediction plots
                visualizer.plot_predictions(y_test, y_pred, model_name)
                
                # Create error analysis plots
                error_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred,
                    'Error': y_test - y_pred,
                    'AbsError': np.abs(y_test - y_pred),
                    'PercentError': 100 * np.abs((y_test - y_pred) / (y_test + 1e-10))
                })
                visualizer.plot_error_analysis(error_df, model_name)
    
    print("Visualization completed.") 