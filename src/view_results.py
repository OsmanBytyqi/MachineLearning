import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_available_models(model_dir="models"):
    """Load all available models from the models directory."""
    if not os.path.exists(model_dir):
        logging.error(f"Model directory '{model_dir}' not found.")
        return {}
    
    models = {}
    for file in os.listdir(model_dir):
        if file.endswith(".joblib"):
            model_name = file.split(".")[0]
            model_path = os.path.join(model_dir, file)
            
            try:
                model = joblib.load(model_path)
                models[model_name] = model
                logging.info(f"Loaded model: {model_name}")
            except Exception as e:
                logging.error(f"Error loading model {model_name}: {e}")
    
    return models

def load_model_metrics(results_file="results/summary.txt"):
    """Load model metrics from the summary file."""
    metrics = {}
    
    if not os.path.exists(results_file):
        logging.error(f"Results file '{results_file}' not found.")
        return metrics
    
    current_model = None
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect model section headers
            if line.endswith("Results:"):
                current_model = line.split()[0].lower()
                metrics[current_model] = {}
                continue
                
            # Parse metrics
            if current_model and any(m in line for m in ["MAE:", "RMSE:", "R2:"]):
                parts = line.split(":")
                if len(parts) == 2:
                    metric_name = parts[0].strip().lower()
                    metric_value = float(parts[1].strip())
                    metrics[current_model][metric_name] = metric_value
    
    return metrics

def visualize_model_comparison(metrics, output_dir="results/plots"):
    """Create visualizations comparing all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not metrics:
        logging.error("No metrics available for visualization.")
        return
    
    # Extract model names and key metrics
    model_names = list(metrics.keys())
    r2_scores = [metrics[model].get("r2", 0) for model in model_names]
    rmse_scores = [metrics[model].get("rmse", 0) for model in model_names]
    mae_scores = [metrics[model].get("mae", 0) for model in model_names]
    
    # Create bar chart for R² values
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, r2_scores, color=['#2C8ECF', '#FF9A13', '#4CAF50'][:len(model_names)])
    
    # Add value labels on top of bars
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
    plt.savefig(f"{output_dir}/r2_comparison.png")
    plt.close()
    
    # Create bar chart for RMSE and MAE
    metrics_data = {
        'RMSE': rmse_scores,
        'MAE': mae_scores
    }
    
    for metric_name, values in metrics_data.items():
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, values, color=['#2C8ECF', '#FF9A13', '#4CAF50'][:len(model_names)])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel(metric_name)
        plt.title(f'Model {metric_name} Comparison')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_name.lower()}_comparison.png")
        plt.close()
    
    # Create summary table
    fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    table_data = []
    for model in model_names:
        table_data.append([
            model, 
            f"{metrics[model].get('mae', 0):.4f}", 
            f"{metrics[model].get('rmse', 0):.4f}", 
            f"{metrics[model].get('r2', 0):.4f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'MAE', 'RMSE', 'R²'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Model Performance Metrics Comparison')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_table.png")
    plt.close()

def main():
    try:
        logging.info("Starting results viewer...")
        
        # Create output directory
        os.makedirs("results/plots", exist_ok=True)
        
        # Load model metrics
        metrics = load_model_metrics()
        
        if not metrics:
            logging.error("No model metrics found in results file.")
            return
        
        # Visualize model comparison
        visualize_model_comparison(metrics)
        
        # Load available models
        models = load_available_models()
        
        if not models:
            logging.warning("No trained models found in models directory.")
            
        # Print summary
        logging.info("\nModel Performance Summary:")
        for model_name, model_metrics in metrics.items():
            logging.info(f"\n{model_name.upper()}:")
            for metric, value in model_metrics.items():
                logging.info(f"  {metric.upper()}: {value:.4f}")
        
        # Identify best model
        best_model = max(metrics.items(), key=lambda x: x[1].get('r2', 0))
        logging.info(f"\nBest model: {best_model[0].upper()} with R²: {best_model[1].get('r2', 0):.4f}")
        
        logging.info("\nResults visualizations have been saved to results/plots/")
        
    except Exception as e:
        logging.error(f"Error in results viewer: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 