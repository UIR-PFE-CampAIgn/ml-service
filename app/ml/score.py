import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import optuna
from optuna.pruners import MedianPruner
import json

from app.core.config import settings
from app.core.model_registry import model_registry
from app.core.logging import ml_logger


# ============================================
# FEATURE EXTRACTORS (MODEL-BASED)
# ============================================

class WhatsAppFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract behavioral and linguistic features using semantic embeddings."""
    
    # Reference sentences for semantic similarity
    BUYING_INTENT_REFERENCES = [
        "I want to buy this product",
        "How much does this cost?",
        "I'm interested in purchasing",
        "Can I place an order?",
        "What's the price for this?",
        "I need to make a payment"
    ]
    
    URGENCY_REFERENCES = [
        "I need this urgently",
        "Can I get this today?",
        "This is time-sensitive",
        "I need this as soon as possible",
        "When is the earliest delivery?"
    ]
    
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.5):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.embedding_model = None
        self.buying_embeddings = None
        self.urgency_embeddings = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """Initialize the embedding model and compute reference embeddings."""
        ml_logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        
        # Pre-compute reference embeddings
        self.buying_embeddings = self.embedding_model.encode(
            self.BUYING_INTENT_REFERENCES,
            convert_to_numpy=True
        )
        self.urgency_embeddings = self.embedding_model.encode(
            self.URGENCY_REFERENCES,
            convert_to_numpy=True
        )
        ml_logger.info("Reference embeddings computed")
        return self
    
    def _compute_semantic_similarity(self, text: str, reference_embeddings: np.ndarray) -> float:
        """Compute maximum cosine similarity between text and reference embeddings."""
        if not text or not text.strip():
            return 0.0
        
        text_embedding = self.embedding_model.encode([text], convert_to_numpy=True)
        similarities = cosine_similarity(text_embedding, reference_embeddings)[0]
        return float(np.max(similarities))
    
    def _extract_semantic_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features using embedding similarity."""
        buying_similarity = self._compute_semantic_similarity(text, self.buying_embeddings)
        urgency_similarity = self._compute_semantic_similarity(text, self.urgency_embeddings)
        
        return {
            'buying_intent_score': buying_similarity,
            'has_buying_intent': float(buying_similarity >= self.similarity_threshold),
            'urgency_score': urgency_similarity,
            'has_urgency': float(urgency_similarity >= self.similarity_threshold),
            'combined_intent_score': (buying_similarity + urgency_similarity) / 2
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract basic linguistic features."""
        text_lower = text.lower()
        words = text.split()
        
        return {
            'msg_length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'has_question': float('?' in text),
            'question_density': text.count('?') / max(len(words), 1),
            'has_exclamation': float('!' in text),
            'has_numbers': float(bool(re.search(r'\d', text))),
            'has_currency': float(bool(re.search(r'[$€£¥₹]|\bdh\b|\bmad\b', text_lower))),
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }
    
    def _extract_behavioral_features(self, item: Dict) -> Dict[str, float]:
        """Extract behavioral features from conversation metadata."""
        msg_count = item.get('messages_in_session', 0)
        conv_duration = item.get('conversation_duration_minutes', 0)
        response_time = item.get('user_response_time_avg_seconds', 0)
        
        msg_per_minute = msg_count / max(conv_duration, 1)
        response_speed_score = 1 / (1 + response_time / 60)
        
        return {
            'messages_in_session': msg_count,
            'message_intensity': np.tanh(msg_count / 10),
            'conversation_duration_minutes': conv_duration,
            'conv_engagement': np.log1p(conv_duration) / 3,
            'user_response_time_avg_seconds': response_time,
            'response_speed_score': response_speed_score,
            'messages_per_minute': msg_per_minute,
            'user_initiated': float(item.get('user_initiated_conversation', False)),
            'returning_customer': float(item.get('is_returning_customer', False)),
        }
    
    def _extract_temporal_features(self, item: Dict) -> Dict[str, float]:
        """Extract time-based features."""
        time_of_day = item.get('time_of_day', 'unknown')
        return {
            'time_business_hours': float(time_of_day == 'business_hours'),
            'time_extended_hours': float(time_of_day == 'extended_hours'),
            'time_off_hours': float(time_of_day == 'off_hours'),
        }
    
    def _extract_composite_features(self, features: Dict) -> Dict[str, float]:
        """Create interaction features."""
        intent_behavior_score = (
            features.get('buying_intent_score', 0) * 0.4 +
            features.get('urgency_score', 0) * 0.3 +
            features.get('message_intensity', 0) * 0.15 +
            features.get('response_speed_score', 0) * 0.15
        )
        
        engagement_quality = (
            features.get('conv_engagement', 0) * 0.5 +
            features.get('messages_per_minute', 0) * 0.3 +
            features.get('returning_customer', 0) * 0.2
        )
        
        return {
            'intent_behavior_score': intent_behavior_score,
            'engagement_quality': engagement_quality,
            'high_value_signal': float(
                intent_behavior_score > 0.6 and engagement_quality > 0.5
            )
        }
    
    def transform(self, X):
        """Transform input data into feature vectors."""
        if self.embedding_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Normalize input
        if isinstance(X, list) and len(X) > 0 and not isinstance(X[0], dict):
            X = X[0] if len(X) == 1 else X
        data_list = [X] if isinstance(X, dict) else X
        
        feature_rows = []
        for item in data_list:
            all_features = {}
            
            text = item.get('user_msg', '')
            all_features.update(self._extract_semantic_features(text))
            all_features.update(self._extract_linguistic_features(text))
            all_features.update(self._extract_behavioral_features(item))
            all_features.update(self._extract_temporal_features(item))
            all_features.update(self._extract_composite_features(all_features))
            
            feature_rows.append(list(all_features.values()))
        
        if self.feature_names_ is None:
            self.feature_names_ = list(all_features.keys())
        
        return np.array(feature_rows)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_ or []


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract text field for TF-IDF."""
    
    def __init__(self, feature_name='user_msg'):
        self.feature_name = feature_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            return [item.get(self.feature_name, '') for item in X]
        elif isinstance(X, dict):
            return [X.get(self.feature_name, '')]
        return X


# ============================================
# HYPERPARAMETER TUNER
# ============================================

class XGBoostHyperparameterTuner:
    """Intelligent hyperparameter tuning using Optuna."""
    
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.best_params = None
        self.study = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'tree_method': 'hist',
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        return roc_auc_score(self.y_val, y_pred_proba)
    
    def tune(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            study_name='xgboost_whatsapp_optimization'
        )
        
        ml_logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        ml_logger.info(f"Best ROC-AUC: {self.study.best_value:.4f}")
        ml_logger.info(f"Best params: {self.best_params}")
        
        return self.best_params


# ============================================
# SCORE PREDICTOR 
# ============================================

class ScorePredictor:
    """Score prediction using LogisticRegression or XGBoost pipeline."""
    ## xgboost -> multi class (many segmentation)
    ## decision_tree_classifier -> multi classification
    ## logistic_regression -> performant when it comes to 2 classes
    def __init__(self, model_type: str = "logistic_regression", model_path: Optional[str] = None):
        self.model_type = model_type  # "logistic_regression" or "xgboost"
        self.model_path = model_path or settings.score_model_path
        self.pipeline = None
        self.best_params = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model from registry."""
        try:
            model_data = model_registry.load_model("score_xgboost", version="latest")
            if model_data:
                self.pipeline = model_data.get("pipeline")
                self.best_params = model_data.get("best_params")
                ml_logger.info("Score model loaded successfully")
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
        xgb_params = xgb_params or {}
        default_params = {
            'n_estimators': 150,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'min_child_weight': 3,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        default_params.update(xgb_params)
        
        feature_union = FeatureUnion([
            ('whatsapp_features', Pipeline([
                ('extract', WhatsAppFeatureExtractor()),
                ('scale', StandardScaler())
            ])),
            ('text_tfidf', Pipeline([
                ('extract', TextFeatureExtractor('user_msg')),
                ('tfidf', TfidfVectorizer(
                    max_features=50,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.85
                ))
            ]))
        ])
        
        return Pipeline([
            ('features', feature_union),
            ('classifier', xgb.XGBClassifier(**xgb_params))
        ])
    
    async def train(
        self,
        data_path: str,
        tune_hyperparameters: bool = True,
        tuning_trials: int = 50,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the XGBoost model with optional hyperparameter tuning.
        
        Args:
            data_path: Path to CSV with training data
            tune_hyperparameters: Whether to run Optuna optimization
            tuning_trials: Number of Optuna trials
            validation_split: Validation set size
        """
        ml_logger.info("="*60)
        ml_logger.info("Starting XGBoost training")
        ml_logger.info("="*60)
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            required_cols = self.REQUIRED_FEATURES + ['score']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            y = df['score'].astype(int)
            if not set(y.unique()).issubset({0, 1}):
                raise ValueError("Score must be 0 or 1")
            
            X = df[self.REQUIRED_FEATURES].to_dict('records')
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            ml_logger.info(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
            
            # Extract features first
            ml_logger.info("Extracting features...")
            feature_pipeline = FeatureUnion([
                ('whatsapp_features', Pipeline([
                    ('extract', WhatsAppFeatureExtractor()),
                    ('scale', StandardScaler())
                ])),
                ('text_tfidf', Pipeline([
                    ('extract', TextFeatureExtractor('user_msg')),
                    ('tfidf', TfidfVectorizer(max_features=50, ngram_range=(1, 2)))
                ]))
            ])
            
            feature_pipeline.fit(X_train)
            X_train_features = feature_pipeline.transform(X_train)
            X_val_features = feature_pipeline.transform(X_val)
            
            # Hyperparameter tuning
            if tune_hyperparameters:
                ml_logger.info(f"Running hyperparameter optimization ({tuning_trials} trials)...")
                tuner = XGBoostHyperparameterTuner(
                    X_train_features, y_train,
                    X_val_features, y_val
                )
                self.best_params = tuner.tune(n_trials=tuning_trials)
            
            # Train final model
            ml_logger.info("Training final model...")
            self.pipeline = self._create_pipeline(self.best_params)
            self.pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.pipeline.predict(X_val)
            y_pred_proba = self.pipeline.predict_proba(X_val)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1_score': f1_score(y_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            # Save model
            model_data = {
                'pipeline': self.pipeline,
                'best_params': self.best_params
            }
            metadata = {
                'model_type': 'whatsapp_lead_score_xgboost_optimized',
                'metrics': metrics,
                'training_samples': len(X_train),
                'features': self.REQUIRED_FEATURES,
                'hyperparameters': self.best_params
            }
            
            model_registry.save_model(
                model=model_data,
                model_name="score_xgboost",
                metadata=metadata
            )
            
            ml_logger.info("="*60)
            ml_logger.info("✅ TRAINING COMPLETE!")
            ml_logger.info(f"Accuracy:  {metrics['accuracy']:.1%}")
            ml_logger.info(f"Precision: {metrics['precision']:.1%}")
            ml_logger.info(f"Recall:    {metrics['recall']:.1%}")
            ml_logger.info(f"F1 Score:  {metrics['f1_score']:.1%}")
            ml_logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            ml_logger.info("="*60)
            
            return metrics
            
        except Exception as e:
            ml_logger.error(f"Training failed: {e}")
            raise
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict lead score."""
        if not self.pipeline:
            raise ValueError("Model not loaded. Train first.")
        
        try:
            missing = [f for f in self.REQUIRED_FEATURES if f not in features]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            
            probability = self.pipeline.predict_proba([features])[0][1]
            score = round(float(probability), 3)
            
            if score >= 0.75:
                category = "hot"
            elif score >= 0.50:
                category = "warm"
            else:
                category = "cold"
            
            confidence = round(abs(score - 0.5) * 2, 3)
            
            return {
                'score': score,
                'category': category,
                'confidence': confidence
            }
            
        except Exception as e:
            ml_logger.error(f"Prediction failed: {e}")
            raise


# Singleton instance
score_predictor = ScorePredictor()