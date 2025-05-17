import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import joblib
import os
import logging
from sklearn.model_selection import KFold, train_test_split
from sklearn.inspection import permutation_importance
import time
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from typing import Tuple, Optional, Any

from preprocessing import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

N_ITER = 20
class ModelTrainer:
    """Class for training and evaluating various ML models."""
    
    class EnsembleModel:
        """Model that combines Random Forest and Extra Trees predictions."""
        def __init__(self, rf_model, et_model, rf_weight=0.7):
            self.rf_model = rf_model
            self.et_model = et_model
            self.rf_weight = rf_weight
            self.et_weight = 1 - rf_weight
            self._feature_names = rf_model._feature_names
            
        def predict(self, X):
            return self.rf_weight * self.rf_model.predict(X) + self.et_weight * self.et_model.predict(X)
        
        @property
        def feature_importances_(self):
            return self.rf_weight * self.rf_model.feature_importances_ + self.et_weight * self.et_model.feature_importances_
    
    def __init__(self, models_dir: str = "models", random_state: int = 42):
        """Initialize the model trainer with standard settings."""
        self.models_dir = models_dir
        self.random_state = random_state
        self.trained_models = {}
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, float, float, float]:
        """Train a model of the specified type and return model and metrics."""
        model_type = model_type.lower()
        
        if model_type == 'randomforest':
            return self.train_random_forest(X_train, y_train, X_test, y_test)
        elif model_type == 'xgboost':
            return self.train_xgboost(X_train, y_train, X_test, y_test)
        elif model_type == 'catboost':
            return self.train_catboost(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: Optional[str] = None) -> Tuple[float, float, float, np.ndarray]:
        """Evaluate a model and return performance metrics."""
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        if model_name:
            self.logger.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return mae, rmse, r2, y_pred
    
    def save_model(self, model: Any, model_name: str) -> str:
        """Save the trained model and return the path."""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        if hasattr(model, 'feature_importances_'):
            feature_names = getattr(model, '_feature_names', [f"feature_{i}" for i in range(len(model.feature_importances_))])
            importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            importances.to_csv(os.path.join(self.models_dir, f"{model_name}_importances.csv"), index=False)
            
        return model_path
    
    def train_extra_trees_fast(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               X_test: pd.DataFrame, y_test: pd.Series) -> Any:
        """Train and evaluate Extra Trees Regressor with minimal tuning for speed."""
        et = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        et.fit(X_train, y_train)
        mae, rmse, r2, _ = self.evaluate_model(et, X_test, y_test, "Extra Trees (Fast)")
        
        return et
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, float, float, float]:
        """Train and evaluate Random Forest Regressor with optimized hyperparameter tuning."""
        start_time = time.time()
        
        feature_names = X_train.columns.tolist()
        
        # Baseline model for feature importance
        baseline_rf = RandomForestRegressor(
            n_estimators=200, 
            max_depth=20,
            random_state=self.random_state, 
            n_jobs=-1
        )
        baseline_rf.fit(X_train, y_train)
        baseline_rf._feature_names = feature_names
        
        # Feature importance analysis
        perm_importance = permutation_importance(
            baseline_rf, X_test, y_test, 
            n_repeats=5, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        
        perm_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        top_k_features = min(50, len(feature_names))
        top_features = perm_imp_df.head(top_k_features)['Feature'].values
        
        # Hyperparameter optimization
        param_space = {
            'n_estimators': Integer(400, 800),
            'max_depth': Integer(20, 40),
            'min_samples_split': Integer(2, 15),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'bootstrap': Categorical([True]),
            'max_samples': Real(0.7, 0.9),
            'criterion': Categorical(['squared_error'])
        }
        
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        bayes_search = BayesSearchCV(
            estimator=rf,
            search_spaces=param_space,
            n_iter=N_ITER, # change this for better results
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=self.random_state,
            scoring='r2'
        )
        
        bayes_search.fit(X_train, y_train)
        best_rf = bayes_search.best_estimator_
        best_rf._feature_names = feature_names
        
        # Evaluate on test set
        mae, rmse, r2, y_pred = self.evaluate_model(best_rf, X_test, y_test, "Random Forest (Best)")
        
        # Create ensemble with Extra Trees
        try:
            et_model = self.train_extra_trees_fast(X_train, y_train, X_test, y_test)
            et_model._feature_names = feature_names
            
            ensemble_y_pred = 0.7 * best_rf.predict(X_test) + 0.3 * et_model.predict(X_test)
            
            ensemble_mae = mean_absolute_error(y_test, ensemble_y_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_y_pred))
            ensemble_r2 = r2_score(y_test, ensemble_y_pred)
            
            if ensemble_r2 > r2:
                ensemble_model = self.EnsembleModel(best_rf, et_model)
                self.trained_models['random_forest_ensemble'] = ensemble_model
                return ensemble_model, ensemble_mae, ensemble_rmse, ensemble_r2
            
            self.trained_models['random_forest'] = best_rf
            return best_rf, mae, rmse, r2
            
        except Exception as e:
            self.logger.error(f"Error in ensemble creation: {str(e)}")
            self.trained_models['random_forest'] = best_rf
            return best_rf, mae, rmse, r2
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                     X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, float, float, float]:
        """Train and evaluate XGBoost Regressor with simplified hyperparameter tuning."""
        feature_names = X_train.columns.tolist()
        
        # Split some validation data for early stopping
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state)
        
        # Parameter combinations to test
        test_params = [
            {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 300, 'min_child_weight': 1,
             'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
            
            {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 200, 'min_child_weight': 5,
             'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 5.0},
             
            {'max_depth': 9, 'learning_rate': 0.01, 'n_estimators': 300, 'min_child_weight': 10,
             'subsample': 1.0, 'colsample_bytree': 1.0, 'gamma': 1.0, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
        ]
        
        best_score = -float('inf')
        best_params = None
        best_model = None
        
        # Test each parameter combination
        for params in test_params:
            params['random_state'] = self.random_state
            params['verbosity'] = 0
            params['n_jobs'] = -1
            
            model = xgb.XGBRegressor(**params)
            eval_set = [(X_val_xgb, y_val_xgb)]
            
            try:
                model.fit(
                    X_train_xgb, y_train_xgb,
                    eval_set=eval_set,
                    early_stopping_rounds=50,
                    verbose=False
                )
            except TypeError:
                model.fit(X_train_xgb, y_train_xgb)
            
            y_pred = model.predict(X_val_xgb)
            score = r2_score(y_val_xgb, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
        
        # Train final model on full training data
        final_model = xgb.XGBRegressor(**best_params)
        
        try:
            final_model.fit(
                X_train, y_train,
                early_stopping_rounds=50,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        except TypeError:
            final_model.fit(X_train, y_train)
        
        final_model._feature_names = feature_names
        
        # Evaluate final model
        mae, rmse, r2, _ = self.evaluate_model(final_model, X_test, y_test, "XGBoost (Best)")
        
        self.trained_models['xgboost'] = final_model
        return final_model, mae, rmse, r2
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, float, float, float]:
        """Train and evaluate CatBoost Regressor with optimized parameter tuning."""
        feature_names = X_train.columns.tolist()
        
        # Create validation pool for early stopping
        eval_pool = Pool(X_test, y_test)
        
        # Parameter combinations to test
        test_params = [
            {'iterations': 500, 'learning_rate': 0.03, 'depth': 6, 'l2_leaf_reg': 3,
             'bagging_temperature': 1, 'random_strength': 1},
            
            {'iterations': 300, 'learning_rate': 0.1, 'depth': 8, 'l2_leaf_reg': 5,
             'bagging_temperature': 0.5, 'random_strength': 0.5},
             
            {'iterations': 1000, 'learning_rate': 0.01, 'depth': 10, 'l2_leaf_reg': 1,
             'bagging_temperature': 0.7, 'random_strength': 0.8}
        ]
        
        best_score = -float('inf')
        best_params = None
        best_model = None
        
        # Test each parameter combination
        for params in test_params:
            params['random_seed'] = self.random_state
            params['thread_count'] = -1
            params['verbose'] = False
            params['loss_function'] = 'RMSE'
            
            model = CatBoostRegressor(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=eval_pool,
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
        
        best_model._feature_names = feature_names
        
        # Final evaluation
        mae, rmse, r2, _ = self.evaluate_model(best_model, X_test, y_test, "CatBoost (Best)")
        
        self.trained_models['catboost'] = best_model
        return best_model, mae, rmse, r2

if __name__ == "__main__":
    # Example usage
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("data/raw/gjobat-all.csv")
    
    # Preprocess data
    target_column = 'Vlera e Gjobave të Lëshuara'
    df_processed, label_encoders = preprocessor.preprocess_data(df, target_column=target_column)
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_train_test_data(
        df_processed, target_column=target_column
    )
    
    # Initialize and use trainer
    trainer = ModelTrainer()
    
    # Train and evaluate RandomForest
    rf_model, rf_mae, rf_rmse, rf_r2 = trainer.train_model('randomforest', X_train, y_train, X_test, y_test)
    trainer.save_model(rf_model, 'random_forest')
    
    # Train and evaluate XGBoost
    xgb_model, xgb_mae, xgb_rmse, xgb_r2 = trainer.train_model('xgboost', X_train, y_train, X_test, y_test)
    trainer.save_model(xgb_model, 'xgboost')
    
    # Train and evaluate CatBoost
    cat_model, cat_mae, cat_rmse, cat_r2 = trainer.train_model('catboost', X_train, y_train, X_test, y_test)
    trainer.save_model(cat_model, 'catboost')
    
    print("Model training completed successfully!") 