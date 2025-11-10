from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from app.core.logging import ml_logger
from app.core.model_registry import model_registry


DEFAULT_DATASET_PATH = "C:\\Users\\rmeri\\ml-service\\data\\whatsapp_leads.csv"


# -----------------------------------------------------------------------------
# Utility: Load CSV
# -----------------------------------------------------------------------------
def load_init_csv(path: Optional[str] = None) -> pd.DataFrame:
    csv_path = Path(path or DEFAULT_DATASET_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    ml_logger.info(f"Loaded lead scoring dataset: {len(df)} rows")
    return df


# -----------------------------------------------------------------------------
# Utility: 80/20 Split
# -----------------------------------------------------------------------------
def split_80_20(df: pd.DataFrame, *, label_col: str = "score", random_state: int = 42):
    strat = df[label_col] if label_col in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=strat
    )
    ml_logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


# -----------------------------------------------------------------------------
# Utility: Dump metrics JSON
# -----------------------------------------------------------------------------
def dump_metrics_json(
    metrics: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    out = Path(output_path or "models/lead_score/metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        import numpy as np

        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return list(o)
        return str(o)

    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=_default)

    ml_logger.info(f"Saved metrics JSON to {out}")
    return str(out)


# -----------------------------------------------------------------------------
# Utility: Evaluate
# -----------------------------------------------------------------------------
def evaluate(pipeline: Pipeline, X, y_true, *, output_path: Optional[str] = None):
    y_pred = pipeline.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    out_path = dump_metrics_json(metrics, output_path)
    ml_logger.info(f"Evaluation done. Metrics saved to {out_path}")
    return metrics, out_path


# -----------------------------------------------------------------------------
# ✅ MAIN FUNCTION — Train Lead Scoring Model
# -----------------------------------------------------------------------------
def train_and_save_lead_score(
    data_path: Optional[str] = None,
    *,
    label_col: str = "score",
    text_col: str = "user_msg",
    metrics_output_path: Optional[str] = None,
    cv: int = 3,
    model_name: str = "lead_scoring",
    save_to_registry: bool = True,
) -> Dict[str, Any]:
    """Train and save a TF-IDF + XGBoost model for lead scoring."""

    # 1️⃣ Load dataset
    df = load_init_csv(data_path or DEFAULT_DATASET_PATH)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found.")

    # 2️⃣ Feature columns
    numeric_features = [
        "messages_in_session",
        "conversation_duration_minutes",
        "user_response_time_avg_seconds",
    ]
    categorical_features = [
        "user_initiated_conversation",
        "is_returning_customer",
        "time_of_day",
    ]

    for col in numeric_features + categorical_features:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")

    # 3️⃣ Split data
    train_df, val_df = split_80_20(df, label_col=label_col)
    X_train, X_val = train_df, val_df

    # 4️⃣ Label encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(df[label_col].astype(str))
    y_train = label_encoder.transform(train_df[label_col].astype(str))
    y_val = label_encoder.transform(val_df[label_col].astype(str))

    # 5️⃣ Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(stop_words="english"), text_col),
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 6️⃣ Model definition
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",  # MULTI-CLASS
        num_class=3,
        eval_metric="mlogloss",  # Use multi-class log loss
        use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    param_grid = {
        "features__text__ngram_range": [(1, 1), (1, 2)],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [4, 6],
        "clf__n_estimators": [200, 300],
    }

    # 7️⃣ Build pipeline
    pipeline = Pipeline(
        [
            ("features", preprocessor),
            ("clf", model),
        ]
    )

    # 8️⃣ Grid search
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_
    best_params = grid.best_params_

    ml_logger.info(f"Best params: {best_params}")

    # 9️⃣ Evaluate
    metrics, metrics_path = evaluate(
        best_pipeline, X_val, y_val, output_path=metrics_output_path
    )

    # 🔟 Save metadata + model
    metadata = {
        "model_type": "lead_scoring_xgboost",
        "algorithm": "TF-IDF + XGBoost",
        "best_params": best_params,
        "metrics": metrics,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "label_col": label_col,
        "text_col": text_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": label_encoder.classes_.tolist(),
        "cv": cv,
        "scoring": "f1",
        "dataset_path": str(data_path or DEFAULT_DATASET_PATH),
    }

    model_artifact = {"pipeline": best_pipeline, "label_encoder": label_encoder}
    model_key = None

    if save_to_registry:
        model_key = model_registry.save_model(
            model=model_artifact,
            model_name=model_name,
            metadata=metadata,
            serializer="joblib",
        )

    result = {
        "best_params": best_params,
        "metrics": metrics,
        "metrics_path": metrics_path,
        "model_key": model_key,
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
    }

    ml_logger.info(f"✅ Lead scoring training complete. Model saved: {model_key}")
    return result
