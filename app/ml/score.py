import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import asyncio

from app.core.config import settings
from app.core.model_registry import model_registry
from app.core.logging import ml_logger


class ScorePredictor:
    """Score prediction using LogisticRegression or XGBoost pipeline."""
    ## xgboost -> multi class (many segmentation)
    ## decision_tree_classifier -> multi classification
    ## logistic_regression -> performant when it comes to 2 classes
    def __init__(self, model_type: str = "logistic_regression", model_path: Optional[str] = None):
        self.model_type = model_type  # "logistic_regression" or "xgboost"
        self.model_path = model_path or settings.score_model_path
        self.pipeline = None
        self.feature_columns = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from registry."""
        try:
            model_key = f"score_{self.model_type}"
            model_data = model_registry.load_model(model_key, version="latest")
            if model_data:
                self.pipeline = model_data.get("pipeline")
                self.feature_columns = model_data.get("feature_columns")
                ml_logger.info(f"Score model ({self.model_type}) loaded successfully")
        except Exception as e:
            ml_logger.warning(f"Could not load score model: {e}")
            self.pipeline = None
            self.feature_columns = None
    
    def _prepare_features(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """Prepare features for model input."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # If we have trained feature columns, ensure consistency
        if self.feature_columns:
            # Add missing columns with default values
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Reorder columns to match training
            df = df[self.feature_columns]
        
        return df
    
    def _create_logistic_regression_pipeline(self, hyperparameters: Dict[str, Any]) -> Pipeline:
        """Create Logistic Regression pipeline."""
        
        # To adjust
        # TPOT -> help us choose the appropriate model for our Data
        lr_params = {
            'C': hyperparameters.get('C', 1.0),
            'penalty': hyperparameters.get('penalty', 'l2'),
            'solver': hyperparameters.get('solver', 'liblinear'),
            'max_iter': hyperparameters.get('max_iter', 1000),
            'random_state': 42
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**lr_params))
        ])
        
        return pipeline
    
    def _create_xgboost_pipeline(self, hyperparameters: Dict[str, Any]) -> Pipeline:
        """Create XGBoost pipeline."""
        
        xgb_params = {
            'n_estimators': hyperparameters.get('n_estimators', 100),
            'max_depth': hyperparameters.get('max_depth', 6),
            'learning_rate': hyperparameters.get('learning_rate', 0.1),
            'subsample': hyperparameters.get('subsample', 1.0),
            'colsample_bytree': hyperparameters.get('colsample_bytree', 1.0),
            'reg_alpha': hyperparameters.get('reg_alpha', 0),
            'reg_lambda': hyperparameters.get('reg_lambda', 1),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(**xgb_params))
        ])
        
        return pipeline
    
    def _create_pipeline(self, hyperparameters: Dict[str, Any]) -> Pipeline:
        """Create pipeline based on model type."""
        if self.model_type == "logistic_regression":
            return self._create_logistic_regression_pipeline(hyperparameters)
        elif self.model_type == "xgboost":
            return self._create_xgboost_pipeline(hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    async def train(
        self, 
        data_path: str, 
        hyperparameters: Dict[str, Any] = None,
        validation_split: float = 0.2,
        target_column: str = 'score'
    ) -> Dict[str, float]:
        """
        Train the score prediction model.
        
        Args:
            data_path: Path to training data CSV file
            hyperparameters: Model hyperparameters
            validation_split: Fraction for validation data
            target_column: Name of the target column
            
        Returns:
            Training metrics dictionary
        """
        ml_logger.info(f"Starting score model training with {self.model_type}")
        
        hyperparameters = hyperparameters or {}
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Prepare features and target
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Store feature columns for consistency
            self.feature_columns = X.columns.tolist()
            
            # Handle categorical variables if any
            categorical_columns = X.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
                self.feature_columns = X.columns.tolist()
            
            # Convert target to binary if needed (assuming score > threshold = positive class)
            score_threshold = hyperparameters.get('score_threshold', 0.5)
            if y.dtype == 'object' or len(y.unique()) > 2:
                # If target is continuous, convert to binary
                if y.dtype in ['float64', 'int64'] and len(y.unique()) > 2:
                    y = (y > score_threshold).astype(int)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Create and train pipeline
            self.pipeline = self._create_pipeline(hyperparameters)
            
            # Hyperparameter tuning if enabled
            if hyperparameters.get('tune_hyperparameters', False):
                if self.model_type == "logistic_regression":
                    param_grid = {
                        'classifier__C': [0.01, 0.1, 1.0, 10.0],
                        'classifier__penalty': ['l1', 'l2'],
                        'classifier__solver': ['liblinear', 'saga']
                    }
                else:  # xgboost
                    param_grid = {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__max_depth': [3, 6, 9],
                        'classifier__learning_rate': [0.01, 0.1, 0.2]
                    }
                
                grid_search = GridSearchCV(
                    self.pipeline, 
                    param_grid, 
                    cv=3, 
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                self.pipeline = grid_search.best_estimator_
                ml_logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                self.pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.pipeline.predict(X_val)
            y_pred_proba = self.pipeline.predict_proba(X_val)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted'),
                'f1_score': f1_score(y_val, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            # Save model
            model_data = {
                'pipeline': self.pipeline,
                'feature_columns': self.feature_columns
            }
            
            metadata = {
                'model_type': f'score_predictor_{self.model_type}',
                'algorithm': self.model_type.replace('_', ' ').title(),
                'metrics': metrics,
                'hyperparameters': hyperparameters,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'num_features': len(self.feature_columns),
                'feature_columns': self.feature_columns
            }
            
            model_key = f"score_{self.model_type}"
            model_registry.save_model(
                model=model_data,
                model_name=model_key,
                metadata=metadata
            )
            
            ml_logger.info(f"Score model ({self.model_type}) training completed with AUC: {metrics['roc_auc']:.4f}")
            return metrics
            
        except Exception as e:
            ml_logger.error(f"Score model training failed: {e}")
            raise
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict score for input features.
        
        Args:
            features: Input features dictionary
            
        Returns:
            Dictionary with predicted score and probability
        """
        if not self.pipeline:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Prepare features
            df = self._prepare_features(features)
            
            # Get prediction and probabilities
            prediction = self.pipeline.predict(df)[0]
            probabilities = self.pipeline.predict_proba(df)[0]
            
            result = {
                'score': float(prediction),
                'probability': float(probabilities[1] if len(probabilities) > 1 else probabilities[0]),
                'probabilities': {
                    'negative': float(probabilities[0]),
                    'positive': float(probabilities[1] if len(probabilities) > 1 else 1 - probabilities[0])
                }
            }
            
            ml_logger.debug(f"Score prediction: {result}")
            return result
            
        except Exception as e:
            ml_logger.error(f"Score prediction failed: {e}")
            raise
    
    async def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict scores for multiple feature sets.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        if not self.pipeline:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Prepare features
            df = pd.DataFrame(features_list)
            df = self._prepare_features(df)
            
            # Get predictions and probabilities
            predictions = self.pipeline.predict(df)
            probabilities = self.pipeline.predict_proba(df)
            
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                result = {
                    'features': features_list[i],
                    'score': float(pred),
                    'probability': float(probs[1] if len(probs) > 1 else probs[0]),
                    'probabilities': {
                        'negative': float(probs[0]),
                        'positive': float(probs[1] if len(probs) > 1 else 1 - probs[0])
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            ml_logger.error(f"Batch score prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.pipeline or not self.feature_columns:
            raise ValueError("Model not trained or loaded")
        
        try:
            classifier = self.pipeline.named_steps['classifier']
            
            if hasattr(classifier, 'feature_importances_'):
                # XGBoost has feature_importances_
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Logistic Regression has coefficients
                importances = np.abs(classifier.coef_[0])
            else:
                raise ValueError("Model does not support feature importance")
            
            feature_importance = dict(zip(self.feature_columns, importances))
            
            # Sort by importance
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return feature_importance
            
        except Exception as e:
            ml_logger.error(f"Failed to get feature importance: {e}")
            raise
