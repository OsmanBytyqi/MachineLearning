import logging
import os
import numpy as np
import pandas as pd
from preprocessing import DataPreprocessor
from train import ModelTrainer
from evaluate import ModelEvaluator
from visualize import Visualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLPipeline:
    def __init__(self):
        pass
        
    def train_evaluate_save_model(self, trainer, model_type, X_train, y_train, X_test, y_test, results):
        """Train, evaluate, save a model and add metrics to results dictionary."""
        logging.info(f"\nTraining {model_type} model...")
        
        model, mae, rmse, r2 = trainer.train_model(
            model_type, X_train, y_train, X_test, y_test
        )
        
        # Get model name (handle special cases like RandomForest ensemble)
        model_name = model_type.lower()
        if model_type.lower() == 'randomforest':
            if hasattr(model, 'rf_model') and hasattr(model, 'et_model'):
                model_name = "random_forest_ensemble"
                logging.info("Using an ensemble of RandomForest and ExtraTrees")
            else:
                model_name = "random_forest"
        
        results[model_name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
        
        trainer.save_model(model, model_name)
        logging.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        return model
    
    def run(self):
        try:
            # Create necessary directories
            os.makedirs("results", exist_ok=True)
            os.makedirs("results/plots", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            
            logging.info("Starting ML pipeline...")
            
            # Data loading and preprocessing
            data_path = "data/raw/gjobat-all.csv"
            target_column = 'Vlera e Gjobave të Lëshuara'
            
            preprocessor = DataPreprocessor()
            df = preprocessor.load_data(data_path)
            preprocessor.df = df
            df_processed, label_encoders = preprocessor.preprocess_data(target_column=target_column)
            
            # Save processed data
            processed_data_path = "data/processed/gjobat_processed.csv"
            df_processed.to_csv(processed_data_path, index=False)
            logging.info(f"Saved processed data to {processed_data_path}")

            # Prepare train and test data
            X_train_np, X_test_np, y_train_np, y_test_np, scaler = preprocessor.prepare_train_test_data(
                target_column=target_column
            )
            
            # Convert numpy arrays to pandas DataFrames with column names
            feature_names = df_processed.drop(columns=[target_column]).columns.tolist()
            X_train = pd.DataFrame(X_train_np, columns=feature_names)
            X_test = pd.DataFrame(X_test_np, columns=feature_names)
            y_train = pd.Series(y_train_np)
            y_test = pd.Series(y_test_np)
            
            # Model training
            trainer = ModelTrainer()
            results = {}
            
            # Train models
            rf_model = self.train_evaluate_save_model(trainer, 'randomforest', X_train, y_train, X_test, y_test, results)
            xgb_model = self.train_evaluate_save_model(trainer, 'xgboost', X_train, y_train, X_test, y_test, results)
            cat_model = self.train_evaluate_save_model(trainer, 'catboost', X_train, y_train, X_test, y_test, results)
            
            # Evaluation
            evaluator = ModelEvaluator()
            evaluator.save_results(results)
            best_model_name, best_metrics = evaluator.get_best_model(results)
            best_r2 = best_metrics.get('r2', 0)
            
            # Visualization
            visualizer = Visualizer()
            visualizer.plot_model_comparison(results)
            
            # Create plots for each model
            models = {
                "random_forest": rf_model,
                "xgboost": xgb_model,
                "catboost": cat_model
            }
            
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                
                # Create prediction plots
                visualizer.plot_predictions(y_test, y_pred, model_name)
                
                # Create error analysis
                error_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_pred,
                    'Error': y_test - y_pred,
                    'AbsError': np.abs(y_test - y_pred),
                    'PercentError': 100 * np.abs((y_test - y_pred) / (y_test + 1e-10))
                })
                
                visualizer.plot_error_analysis(error_df, model_name)
            
            # Additional visualizations
            visualizer.plot_correlation_matrix(df_processed, target_column)
            visualizer.plot_feature_distributions(df_processed, target_column)
            
            # Log results
            logging.info(f"\nBest model: {best_model_name.upper()} with R²: {best_r2:.4f}")
            
            if best_r2 >= 0.9:
                logging.info("TARGET ACHIEVED! R² score of 90% or better has been reached.")
                logging.info("\nModel comparison:")
                for model_name, result in results.items():
                    logging.info(f"- {model_name}: R²={result['r2']:.4f}, MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
            else:
                logging.info(f"TARGET NOT YET ACHIEVED: R² score is {best_r2:.4f}, "
                            f"which is {(0.9 - best_r2) * 100:.2f}% away from the 90% target.")
            
            logging.info("\nTraining completed successfully!")
            logging.info("Detailed metrics saved to results/summary.txt")
            logging.info("Visualizations have been saved to results/plots/")

        except Exception as e:
            logging.error(f"Error in main: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run() 