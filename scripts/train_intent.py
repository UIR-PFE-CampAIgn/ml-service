import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is on sys.path so `app` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from app.ml.train_intent import train_and_save_intent, DEFAULT_DATASET_PATH
from app.core.logging import setup_logging


def _parse_c_values(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_ngram_ranges(s: str) -> List[Tuple[int, int]]:
    # format: "1,1;1,2"
    parts = [p.strip() for p in s.split(";") if p.strip()]
    ranges: List[Tuple[int, int]] = []
    for p in parts:
        a, b = p.split(",")
        ranges.append((int(a), int(b)))
    return ranges


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Train intent classifier and save to registry")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to CSV dataset (default: Bitext sample CSV)",
    )
    parser.add_argument("--label-col", default="intent", help="Label column name")
    parser.add_argument(
        "--text-col",
        default=None,
        help="Text column name. If omitted, tries 'text' then maps 'instruction'→'text'",
    )
    parser.add_argument(
        "--c-values",
        default="0.1,1.0,10.0",
        help="Comma-separated C values for grid search",
    )
    parser.add_argument(
        "--ngram-ranges",
        default="1,1;1,2",
        help="Semicolon-separated ngram ranges, e.g. '1,1;1,2'",
    )
    parser.add_argument(
        "--metrics-path",
        default="models/intent/metrics.json",
        help="Where to write metrics JSON locally",
    )
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    parser.add_argument(
        "--scoring", default="f1_macro", help="Scikit-learn scoring metric"
    )
    parser.add_argument(
        "--model-name", default="intent", help="Model name key in registry"
    )
    parser.add_argument(
        "--no-registry-save",
        action="store_true",
        help="Skip saving the trained model to the registry (MinIO/S3)",
    )

    args = parser.parse_args()

    result = train_and_save_intent(
        data_path=args.data_path,
        label_col=args.label_col,
        text_col=args.text_col,
        c_values=_parse_c_values(args.c_values),
        ngram_ranges=_parse_ngram_ranges(args.ngram_ranges),
        metrics_output_path=args.metrics_path,
        cv=args.cv,
        scoring=args.scoring,
        model_name=args.model_name,
        save_to_registry=not args.no_registry_save,
    )

    print("Best params:", result["best_params"])
    if result["model_key"]:
        print("Model saved to:", result["model_key"])
    else:
        print("Model not saved to registry (skipped)")
    print("Metrics JSON:", result["metrics_path"])


if __name__ == "__main__":
    main()
