import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from app.core.logging import ml_logger
from app.core.model_registry import model_registry

DEFAULT_DATASET_PATH = (
    "data/intent/Bitext-customer-support-llm-chatbot-training-dataset/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def load_init_csv(path: Optional[str] = None) -> pd.DataFrame:
    """Load the Bitext sample customer support dataset and return it.

    Args:
        path: Optional override path to the CSV. If not provided, uses
              the default Bitext dataset path under `data/intent/...`.

    Returns:
        A pandas DataFrame containing the dataset.
    """
    csv_path = Path(path or DEFAULT_DATASET_PATH)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {csv_path}. Provide a valid path."
        )

    df = pd.read_csv(csv_path)
    ml_logger.info(f"Loaded intent dataset: {len(df)} rows from {csv_path}")
    return df


def split_80_20(
    df: pd.DataFrame,
    *,
    label_col: Optional[str] = "intent",
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into 80/20 train/validation sets.

    Args:
        df: Input DataFrame to split.
        label_col: Column to use for stratification if present.
        random_state: Random state for reproducibility.
        stratify: Whether to stratify by the label column (if available).

    Returns:
        Tuple of (train_df, val_df).
    """
    do_stratify = stratify and label_col and label_col in df.columns
    strat = df[label_col] if do_stratify else None

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=strat
    )
    ml_logger.info(
        f"Performed 80/20 split: train={len(train_df)}, val={len(val_df)}, "
        f"stratified={'yes' if do_stratify else 'no'}"
    )
    return train_df, val_df


def grid_search_c_ngram(
    X: List[str],
    y: List[Any],
    *,
    c_values: Optional[List[float]] = None,
    ngram_ranges: Optional[List[Tuple[int, int]]] = None,
    cv: int = 3,
    n_jobs: int = -1,
    scoring: str = "f1_macro",
    stop_words: Optional[str] = "english",
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Grid-search SVC(C) and TF-IDF ngram-range (with probabilities).

    Returns the best fitted pipeline, best params, and cv_results_.
    """
    c_values = c_values or [0.1, 1.0, 10.0]
    ngram_ranges = ngram_ranges or [(1, 1), (1, 2)]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words=stop_words)),
            ("clf", SVC(probability=True, kernel="linear")),
        ]
    )

    param_grid = {
        "tfidf__ngram_range": ngram_ranges,
        "clf__C": c_values,
    }

    ml_logger.info(
        f"Starting grid search over C={c_values} and ngram_range={ngram_ranges}"
    )
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        scoring=scoring,
        refit=True,
        verbose=0,
    )
    grid.fit(X, y)

    ml_logger.info(
        f"Best params: {grid.best_params_}; best score: {grid.best_score_:.4f}"
    )
    return grid.best_estimator_, grid.best_params_, grid.cv_results_


def dump_metrics_json(
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
    *,
    indent: int = 2,
) -> str:
    """Dump metrics to a JSON file and return the path.

    Args:
        metrics: Dictionary of metrics to serialize.
        output_path: Where to write the JSON. Defaults to
            "models/intent/metrics.json".
        indent: JSON indentation.

    Returns:
        The string path to the written JSON file.
    """
    out = Path(output_path or "models/intent/metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        try:
            import numpy as np  # local import to avoid hard dependency at import time

            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, set):
            return list(o)
        # Fallback string conversion
        return str(o)

    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=indent, default=_default)

    ml_logger.info(f"Wrote metrics JSON to {out}")
    return str(out)


def evaluate(
    pipeline: Pipeline,
    X: List[str],
    y_true: List[Any],
    *,
    output_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Evaluate a trained pipeline and dump metrics JSON.

    Computes accuracy, macro precision/recall/F1, classification report, and
    confusion matrix. Writes metrics to JSON via dump_metrics_json.

    Returns (metrics_dict, written_json_path).
    """
    y_pred = pipeline.predict(X)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    out_path = dump_metrics_json(metrics, output_path)
    ml_logger.info(f"Evaluation completed. Metrics saved to {out_path}")
    return metrics, out_path


def train_and_save_intent(
    data_path: Optional[str] = None,
    *,
    label_col: str = "intent",
    text_col: Optional[str] = None,
    c_values: Optional[List[float]] = None,
    ngram_ranges: Optional[List[Tuple[int, int]]] = None,
    metrics_output_path: Optional[str] = None,
    cv: int = 3,
    scoring: str = "f1_macro",
    model_name: str = "intent",
    save_to_registry: bool = True,
) -> Dict[str, Any]:
    """Orchestrate training: split → grid search → evaluate → save model.

    - Loads CSV (defaults to Bitext sample dataset) and adapts columns.
    - Performs 80/20 stratified split by `label_col`.
    - Grid-searches SVC C and TF-IDF ngram_range (probability-enabled).
    - Evaluates on validation set and writes metrics JSON.
    - Saves the fitted pipeline to the model registry (S3/MinIO).

    Returns a summary dict with best_params, metrics, metrics_path, and model_key.
    """
    # 1) Load dataset
    csv_path = data_path or DEFAULT_DATASET_PATH
    df = load_init_csv(csv_path)

    # Infer or map text column
    if text_col is None:
        if "text" in df.columns:
            text_col = "text"
        elif "instruction" in df.columns:
            # Map instruction → text for our pipeline
            df = df.rename(columns={"instruction": "text"})
            text_col = "text"
        else:
            raise ValueError(
                "Could not infer text column. Expected 'text' or 'instruction'."
            )

    # Validate label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    # 2) Split 80/20
    train_df, val_df = split_80_20(df, label_col=label_col)

    X_train = train_df[text_col].astype(str).tolist()
    X_val = val_df[text_col].astype(str).tolist()

    # Fit label encoder on all labels for consistent class index mapping
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    label_encoder.fit(df[label_col].astype(str).tolist())
    y_train = label_encoder.transform(train_df[label_col].astype(str).tolist())
    y_val = label_encoder.transform(val_df[label_col].astype(str).tolist())

    # 3) Grid search
    best_pipeline, best_params, cv_results = grid_search_c_ngram(
        X_train,
        y_train,
        c_values=c_values,
        ngram_ranges=ngram_ranges,
        cv=cv,
        scoring=scoring,
    )

    # 4) Evaluate
    metrics, metrics_path = evaluate(
        best_pipeline, X_val, y_val, output_path=metrics_output_path
    )

    # 5) Save model to registry (MinIO/S3) if enabled
    metadata: Dict[str, Any] = {
        "model_type": "intent_classifier",
        "algorithm": "TF-IDF + SVC",
        "best_params": best_params,
        "metrics": metrics,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "label_col": label_col,
        "text_col": text_col,
        "classes": label_encoder.classes_.tolist(),
        "dataset_path": str(csv_path),
        "cv": cv,
        "scoring": scoring,
    }

    # Save both pipeline and label encoder for compatibility with IntentPredictor
    model_artifact = {
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
    }

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
    if model_key:
        ml_logger.info(
            "Training complete: model saved to %s; metrics at %s",
            model_key,
            metrics_path,
        )
    else:
        ml_logger.info(
            "Training complete: metrics at %s (registry save skipped)", metrics_path
        )
    return result
