import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so `app` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=ROOT / ".env", override=False)

import asyncio

from app.core.logging import setup_logging
from app.ml.score import ScorePredictor


def _read_json_file(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_features(features_json: Optional[str], features_file: Optional[str]) -> Dict[str, Any]:
    if features_file:
        data = _read_json_file(features_file)
    elif features_json:
        data = json.loads(features_json)
    else:
        raise ValueError("Provide --features-json or --features-file")

    if not isinstance(data, dict):
        raise ValueError("Features must be a JSON object mapping feature names to values")
    return data


def _parse_batch(batch_file: Optional[str]) -> List[Dict[str, Any]]:
    if not batch_file:
        return []
    data = _read_json_file(batch_file)
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise ValueError("Batch file must be a JSON array of feature objects")
    return data


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Load latest score model and predict from JSON features"
    )
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "xgboost"],
        help="Which score model variant to use",
    )
    parser.add_argument(
        "--features-json",
        help="Inline JSON for a single example, e.g. '{"feature_a": 1, "feature_b": 0.2}'",
    )
    parser.add_argument(
        "--features-file",
        help="Path to a JSON file containing a single feature object",
    )
    parser.add_argument(
        "--batch-file",
        help="Path to a JSON file containing an array of feature objects for batch prediction",
    )
    parser.add_argument(
        "--show-trained-features",
        action="store_true",
        help="After loading, print the trained feature column order and exit",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the prediction JSON",
    )

    args = parser.parse_args()

    predictor = ScorePredictor(model_type=args.model_type)

    # Optionally show expected features and exit
    if args.show_trained_features:
        cols = predictor.feature_columns or []
        out = {"model_type": args.model_type, "feature_columns": cols}
        print(json.dumps(out, indent=2) if args.pretty else json.dumps(out))
        return

    # Batch mode
    if args.batch_file:
        batch = _parse_batch(args.batch_file)
        results = asyncio.run(predictor.batch_predict(batch))
        print(json.dumps(results, indent=2) if args.pretty else json.dumps(results))
        return

    # Single example mode
    features = _parse_features(args.features_json, args.features_file)
    result = asyncio.run(predictor.predict(features))
    print(json.dumps(result, indent=2) if args.pretty else json.dumps(result))


if __name__ == "__main__":
    main()

