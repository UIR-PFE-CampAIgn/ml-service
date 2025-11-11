from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.logging import ml_logger
from app.core.model_registry import model_registry


class ScorePredictor:
    """
    Multi-class lead scoring predictor (cold / warm / hot).
    Supports LogisticRegression (fallback) and XGBoost (recommended).
    """

    def __init__(self, model_type: str = "xgboost", model_path: Optional[str] = None):
        self.model_type = model_type.lower()  # "xgboost" or "logistic_regression"
        self.model_path = model_path or settings.score_model_path
        self.pipeline = None
        self.label_encoder = None
        self.feature_columns = None

    def _load_model(self):
        """Load pipeline + label_encoder from model registry."""
        try:
            model_key = "lead_scoring"
            ml_logger.info(f"Loading model: {model_key}")
            model_data = model_registry.load_model(model_key, version="latest")
            if model_data:
                self.pipeline = model_data.get("pipeline")
                self.label_encoder = model_data.get("label_encoder")
                self.feature_columns = model_data.get("feature_columns")
                if self.label_encoder is None:
                    ml_logger.warning("label_encoder missing in model data")
                ml_logger.info("Lead scoring model loaded successfully")
                return

            # Fallback
            model_key = f"score_{self.model_type}"
            model_data = model_registry.load_model(model_key, version="latest")
            if model_data:
                self.pipeline = model_data.get("pipeline")
                self.label_encoder = model_data.get("label_encoder")
                self.feature_columns = model_data.get("feature_columns")
                ml_logger.info(f"Score model ({self.model_type}) loaded")
        except Exception as e:
            ml_logger.warning(f"Failed to load model: {e}")
            self.pipeline = None
            self.label_encoder = None
            self.feature_columns = None

    def _prepare_features(
        self, data: Union[Dict[str, Any], pd.DataFrame]
    ) -> pd.DataFrame:
        """Convert input to DataFrame. No auto-fill."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        return df

    def _create_logistic_regression_pipeline(
        self, hyperparameters: Dict[str, Any]
    ) -> Pipeline:
        lr_params = {
            "C": hyperparameters.get("C", 1.0),
            "penalty": hyperparameters.get("penalty", "l2"),
            "solver": hyperparameters.get("solver", "lbfgs"),
            "max_iter": hyperparameters.get("max_iter", 1000),
            "random_state": 42,
            "multi_class": "multinomial",  # Multi-class support
        }
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    xgb.XGBClassifier(**lr_params),
                ),  # Will be replaced in training
            ]
        )

    def _create_xgboost_pipeline(self, hyperparameters: Dict[str, Any]) -> Pipeline:
        xgb_params = {
            "n_estimators": hyperparameters.get("n_estimators", 200),
            "max_depth": hyperparameters.get("max_depth", 6),
            "learning_rate": hyperparameters.get("learning_rate", 0.1),
            "subsample": hyperparameters.get("subsample", 0.8),
            "colsample_bytree": hyperparameters.get("colsample_bytree", 0.8),
            "reg_alpha": hyperparameters.get("reg_alpha", 0),
            "reg_lambda": hyperparameters.get("reg_lambda", 1),
            "random_state": 42,
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": 3,  # cold, warm, hot
        }
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", xgb.XGBClassifier(**xgb_params)),
            ]
        )

    def _create_pipeline(self, hyperparameters: Dict[str, Any]) -> Pipeline:
        if self.model_type == "xgboost":
            return self._create_xgboost_pipeline(hyperparameters)
        elif self.model_type == "logistic_regression":
            # Use sklearn LogisticRegression for binary/multinomial
            from sklearn.linear_model import LogisticRegression

            lr_params = {
                "C": hyperparameters.get("C", 1.0),
                "penalty": hyperparameters.get("penalty", "l2"),
                "solver": hyperparameters.get("solver", "lbfgs"),
                "max_iter": hyperparameters.get("max_iter", 1000),
                "random_state": 42,
                "multi_class": "multinomial",
            }
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(**lr_params)),
                ]
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def train(
        self,
        data_path: str,
        hyperparameters: Dict[str, Any] = None,
        validation_split: float = 0.2,
        target_column: str = "score",
    ) -> Dict[str, float]:
        """Train multi-class model."""
        ml_logger.info(f"Training {self.model_type} model...")
        hyperparameters = hyperparameters or {}
        try:
            df = pd.read_csv(data_path)
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")

            y = df[target_column].astype(str)
            X = df.drop(columns=[target_column])
            self.feature_columns = X.columns.tolist()

            # One-hot encode categorical
            categorical_cols = X.select_dtypes(include=["object", "bool"]).columns
            if len(categorical_cols) > 0:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                self.feature_columns = X.columns.tolist()

            # Label encode target
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_encoded,
                test_size=validation_split,
                random_state=42,
                stratify=y_encoded,
            )

            self.pipeline = self._create_pipeline(hyperparameters)

            # Hyperparameter tuning
            if hyperparameters.get("tune_hyperparameters", False):
                if self.model_type == "xgboost":
                    param_grid = {
                        "classifier__n_estimators": [100, 200],
                        "classifier__max_depth": [4, 6],
                        "classifier__learning_rate": [0.05, 0.1],
                    }
                else:
                    param_grid = {
                        "classifier__C": [0.1, 1.0, 10.0],
                        "classifier__penalty": ["l2"],
                        "classifier__solver": ["lbfgs"],
                    }
                grid = GridSearchCV(
                    self.pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1
                )
                grid.fit(X_train, y_train)
                self.pipeline = grid.best_estimator_
                ml_logger.info(f"Best params: {grid.best_params_}")
            else:
                self.pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = self.pipeline.predict(X_val)
            y_proba = self.pipeline.predict_proba(X_val)
            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(
                    precision_score(y_val, y_pred, average="macro", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_val, y_pred, average="macro", zero_division=0)
                ),
                "f1_score": float(
                    f1_score(y_val, y_pred, average="macro", zero_division=0)
                ),
            }

            # Save model
            model_data = {
                "pipeline": self.pipeline,
                "label_encoder": le,
                "feature_columns": self.feature_columns,
            }
            metadata = {
                "model_type": f"score_predictor_{self.model_type}",
                "algorithm": (
                    "XGBoost" if self.model_type == "xgboost" else "LogisticRegression"
                ),
                "metrics": metrics,
                "classes": le.classes_.tolist(),
                "num_features": len(self.feature_columns),
            }
            model_key = "lead_scoring"  # Use same key as train_and_save_lead_score
            model_registry.save_model(
                model_data, model_name=model_key, metadata=metadata
            )
            ml_logger.info(
                f"Model trained and saved: {model_key} | F1: {metrics['f1_score']:.4f}"
            )
            return metrics

        except Exception as e:
            ml_logger.error(f"Training failed: {e}")
            raise

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict lead category with confidence."""
        self._load_model()
        if not self.pipeline or self.label_encoder is None:
            raise ValueError("Model or label encoder not loaded.")

        try:
            df = self._prepare_features(features)
            proba = self.pipeline.predict_proba(df)[0]
            pred_idx = int(np.argmax(proba))
            pred_class = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(proba[pred_idx])

            result = {
                "score": confidence,
                "category": pred_class,
                "confidence": confidence,
                "probabilities": {
                    "cold": float(proba[0]),
                    "warm": float(proba[1]),
                    "hot": float(proba[2]),
                },
            }
            ml_logger.debug(f"Prediction: {result}")
            return result
        except Exception as e:
            ml_logger.error(f"Prediction failed: {e}")
            raise

    async def batch_predict(
        self, features_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch prediction."""
        self._load_model()
        if not self.pipeline or self.label_encoder is None:
            raise ValueError("Model or label encoder not loaded.")

        try:
            df = pd.DataFrame(features_list)
            df = self._prepare_features(df)
            probabilities = self.pipeline.predict_proba(df)
            predictions = self.pipeline.predict(df)

            results = []
            for i, (proba, pred_class) in enumerate(zip(probabilities, predictions)):
                proba_dict = {
                    self.label_encoder.classes_[j]: float(p)
                    for j, p in enumerate(proba)
                }
                confidence = float(proba.max())
                result = {
                    "features": features_list[i],
                    "score": confidence,
                    "category": pred_class,
                    "confidence": confidence,
                    "probabilities": proba_dict,
                }
                results.append(result)
            return results
        except Exception as e:
            ml_logger.error(f"Batch prediction failed: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Return sorted feature importance."""
        if not self.pipeline or not self.feature_columns:
            raise ValueError("Model not loaded")
        try:
            clf = self.pipeline.named_steps["classifier"]
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_).mean(axis=0)
            else:
                raise ValueError("No feature importance")
            importance_dict = dict(zip(self.feature_columns, importances))
            return dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        except Exception as e:
            ml_logger.error(f"Feature importance failed: {e}")
            raise
