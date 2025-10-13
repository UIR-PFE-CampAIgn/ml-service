"""
WhatsApp Lead Score Predictor - XGBoost Implementation
=======================================================
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb

from app.core.config import settings
from app.core.model_registry import model_registry
from app.core.logging import ml_logger


# ============================================
# FEATURE EXTRACTORS
# ============================================

class WhatsAppFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract behavioral and linguistic features from WhatsApp conversations."""
    
    BUYING_KEYWORDS = [
        'buy', 'purchase', 'price', 'cost', 'how much', 'payment', 'pay',
        'order', 'delivery', 'invoice', 'quote', 'interested', 'want', 'need',
        'combien', 'prix', 'acheter'  # French for Morocco
    ]
    
    URGENCY_KEYWORDS = [
        'now', 'today', 'urgent', 'asap', 'quickly', 'immediately', 'soon',
        'maintenant', 'urgent', 'vite'
    ]
    
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        return self
    
    def _extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from message text."""
        text_lower = text.lower()
        
        features = {
            'msg_length': len(text),
            'word_count': len(text.split()),
            'has_question': float('?' in text),
            'has_exclamation': float('!' in text),
            'has_numbers': float(bool(re.search(r'\d', text))),
        }
        
        buying_count = sum(1 for kw in self.BUYING_KEYWORDS if kw in text_lower)
        features['buying_intent_count'] = buying_count
        features['has_buying_intent'] = float(buying_count > 0)
        
        urgency_count = sum(1 for kw in self.URGENCY_KEYWORDS if kw in text_lower)
        features['urgency_count'] = urgency_count
        features['has_urgency'] = float(urgency_count > 0)
        
        return features
    
    def _extract_behavioral_features(self, item: Dict) -> Dict[str, float]:
        """Extract behavioral features from conversation metadata."""
        features = {}
        
        msg_count = item.get('messages_in_session', 0)
        features['messages_in_session'] = msg_count
        features['high_message_count'] = float(msg_count >= 8)
        features['message_intensity'] = min(msg_count / 15.0, 1.0)
        
        conv_duration = item.get('conversation_duration_minutes', 0)
        features['conversation_duration_minutes'] = conv_duration
        features['long_conversation'] = float(conv_duration > 15)
        features['short_conversation'] = float(conv_duration < 5)
        
        response_time = item.get('user_response_time_avg_seconds', 0)
        features['user_response_time_avg_seconds'] = response_time
        features['fast_responder'] = float(0 < response_time < 60)
        features['slow_responder'] = float(response_time > 120)
        
        features['user_initiated'] = float(item.get('user_initiated_conversation', False))
        features['returning_customer'] = float(item.get('is_returning_customer', False))
        
        return features
    
    def _extract_temporal_features(self, item: Dict) -> Dict[str, float]:
        """Extract time-of-day features."""
        time_of_day = item.get('time_of_day', 'unknown')
        
        return {
            'time_business_hours': float(time_of_day == 'business_hours'),
            'time_extended_hours': float(time_of_day == 'extended_hours'),
            'time_off_hours': float(time_of_day == 'off_hours'),
        }
    
    def _extract_composite_features(self, features: Dict) -> Dict[str, float]:
        """Create composite/interaction features."""
        return {
            'strong_signal': (
                features['has_buying_intent'] * 3 +
                features['has_urgency'] * 2 +
                features['user_initiated'] * 1 +
                features['returning_customer'] * 2
            ),
            'engagement_score': (
                features['long_conversation'] * 2 +
                features['high_message_count'] * 2 +
                features['fast_responder'] * 1
            )
        }
    
    def transform(self, X):
        """Transform input data into feature vectors."""
        if isinstance(X, list) and len(X) > 0 and not isinstance(X[0], dict):
            X = X[0] if len(X) == 1 else X
        
        data_list = [X] if isinstance(X, dict) else X
        
        feature_rows = []
        for item in data_list:
            all_features = {}
            
            text = item.get('user_msg', '')
            all_features.update(self._extract_text_features(text))
            all_features.update(self._extract_behavioral_features(item))
            all_features.update(self._extract_temporal_features(item))
            all_features.update(self._extract_composite_features(all_features))
            
            feature_rows.append(list(all_features.values()))
        
        if self.feature_names_ is None:
            self.feature_names_ = list(all_features.keys())
        
        return np.array(feature_rows)
    
    def get_feature_names(self):
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
# SCORE PREDICTOR
# ============================================

class ScorePredictor:
    """
    WhatsApp Lead Score Predictor using XGBoost.
    
    Input: 7 features (messages_in_session, user_msg, conversation_duration_minutes,
                      user_response_time_avg_seconds, user_initiated_conversation,
                      is_returning_customer, time_of_day)
    Output: {score: float, category: str, confidence: float}
    """
    
    REQUIRED_FEATURES = [
        'messages_in_session', 'user_msg', 'conversation_duration_minutes',
        'user_response_time_avg_seconds', 'user_initiated_conversation',
        'is_returning_customer', 'time_of_day'
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.score_model_path
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model from registry."""
        try:
            model_data = model_registry.load_model("score_xgboost", version="latest")
            if model_data:
                self.pipeline = model_data.get("pipeline")
                ml_logger.info("Score model loaded successfully")
        except Exception as e:
            ml_logger.warning(f"Could not load score model: {e}")
            self.pipeline = None
    
    def _create_pipeline(self, xgb_params: dict = None) -> Pipeline:
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
            ('classifier', xgb.XGBClassifier(**default_params))
        ])
    
    async def train(
        self,
        data_path: str,
        hyperparameters: Dict[str, Any] = None,
        validation_split: float = 0.2,
        grid_search: bool = False
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            data_path: Path to CSV with training data
            hyperparameters: Optional model hyperparameters
            validation_split: Validation set size
            grid_search: Whether to perform grid search
        
        Returns:
            Dictionary with training metrics
        """
        ml_logger.info("Starting XGBoost training")
        
        try:
            # Load and validate data
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
            
            # Train with or without grid search
            if grid_search:
                ml_logger.info("Running grid search...")
                pipeline = self._create_pipeline()
                
                param_grid = {
                    'classifier__n_estimators': [100, 150, 200],
                    'classifier__max_depth': [4, 5, 6],
                    'classifier__learning_rate': [0.03, 0.05, 0.1]
                }
                
                grid = GridSearchCV(
                    pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
                )
                grid.fit(X_train, y_train)
                self.pipeline = grid.best_estimator_
                
                ml_logger.info(f"Best params: {grid.best_params_}")
            else:
                self.pipeline = self._create_pipeline(hyperparameters)
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
            model_data = {'pipeline': self.pipeline}
            metadata = {
                'model_type': 'whatsapp_lead_score_xgboost',
                'metrics': metrics,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'features': self.REQUIRED_FEATURES
            }
            
            model_registry.save_model(
                model=model_data,
                model_name="score_xgboost",
                metadata=metadata
            )
            
            ml_logger.info(f"Training complete! ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"\n✅ Training Complete!")
            print(f"Accuracy:  {metrics['accuracy']:.1%}")
            print(f"Precision: {metrics['precision']:.1%}")
            print(f"Recall:    {metrics['recall']:.1%}")
            print(f"F1 Score:  {metrics['f1_score']:.1%}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            ml_logger.error(f"Training failed: {e}")
            raise
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict lead score.
        
        Returns: {score: float, category: str, confidence: float}
        """
        if not self.pipeline:
            raise ValueError("Model not loaded. Train first.")
        
        try:
            # Validate input
            missing = [f for f in self.REQUIRED_FEATURES if f not in features]
            if missing:
                raise ValueError(f"Missing required features: {missing}")
            
            # Predict probability
            probability = self.pipeline.predict_proba([features])[0][1]
            score = round(float(probability), 3)
            
            # Categorize
            if score >= 0.75:
                category = "hot"
            elif score >= 0.50:
                category = "warm"
            else:
                category = "cold"
            
            # Confidence (distance from decision boundary)
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