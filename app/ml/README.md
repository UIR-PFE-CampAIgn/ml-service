| Step                    | What to add                                                                                                                 | Glance at the code                                                                      |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **1. Dataset spec**     | `data/intent.csv` with columns `text,intent` (UTF-8, one row per message).                                                  | `"Combien ça coûte ?",pricing`                                                          |
| **2. Training script**  | `scripts/train_intent.py` → loads CSV, does 80/20 split, grid-searches C & ngram-range, dumps metrics JSON + model Pickle.  | `LinearSVC()` inside a `Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())])` |
| **3. FastAPI router**   | `app/api/v1/intent.py` exposes `GET /predict_intent?text=...`—loads Pickle once at startup.                                 | `python intent = model.predict([text])[0]`                                              |
| **4. Unit tests**       | `tests/unit/test_intent.py` – smoke-test prediction, assert latency < 10 ms.                                                |                                                                                         |
| **5. CI step**          | Extend `ci.yml`: `python scripts/train_intent.py --eval-only && pytest`. Fail build if F-1 < 0.85.                          |                                                                                         |
| **6. Storage**          | Use the existing `model_registry.save(model,"intent")` so the Pickle lands in MinIO → versioned.                            |                                                                                         |
| **7. Gateway contract** | `campaign-gateway-api` now calls `GET /ml/v1/predict_intent` and adds `intent_pricing`, `intent_support`, … flags to Mongo. |                                                                                         |

## Dataset - Intent

HuggingFace: bitext/Bitext-customer-support-llm-chatbot-training-dataset

#### Stratification purpose

When stratify is given (i.e. not None), the function ensures that the proportion of each class (in the label) in the train and val splits matches (approximately) the proportion in the original data. This prevents class imbalance issues in one split.

#### Grid search

Use different model settings over chunks of data and evaluate each model's score to pick the best.

#### Dumps metrics JSON

Once we find the best model from the previous step:

1. Freeze that model
2. Run it on your 200 reserved test messages.
3. Compute these metrics:
   - Accuracy (percentage correct)
   - Precision / recall / F1 for each intent
   - Maybe confusion matrix
4. Write those results into a JSON file for reference purpose

### Best score

{'clf**C': 10.0, 'tfidf**ngram_range': (1, 1)}; best score: 0.9793
