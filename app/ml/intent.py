import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import asyncio
from pathlib import Path

from app.core.config import settings
from app.core.model_registry import model_registry
from app.core.logging import ml_logger


class IntentPredictor:
    """Intent classification using TF-IDF + SVM pipeline."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.intent_model_path
        self.pipeline = None
        self.label_encoder = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from registry."""
        try:
            model_data = model_registry.load_model("intent", version="latest")
            if model_data:
                self.pipeline = model_data.get("pipeline")
                self.label_encoder = model_data.get("label_encoder")
                ml_logger.info("Intent model loaded successfully")
        except Exception as e:
            ml_logger.warning(f"Could not load intent model: {e}")
            self.pipeline = None
            self.label_encoder = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for intent classification."""
        # Basic preprocessing
        text = text.lower().strip()
        # Add more preprocessing as needed (remove special chars, etc.)
        return text
    
    def _create_pipeline(self, hyperparameters: Dict[str, Any]) -> Pipeline:
        """Create TF-IDF + SVM pipeline."""
        
        # TF-IDF parameters
        tfidf_params = {
            'max_features': hyperparameters.get('max_features', 5000),
            'ngram_range': hyperparameters.get('ngram_range', (1, 2)),
            'min_df': hyperparameters.get('min_df', 2),
            'max_df': hyperparameters.get('max_df', 0.95),
            'stop_words': hyperparameters.get('stop_words', 'english')
        }
        
        # SVM parameters
        svm_params = {
            'kernel': hyperparameters.get('kernel', 'rbf'),
            'C': hyperparameters.get('C', 1.0),
            'gamma': hyperparameters.get('gamma', 'scale'),
            'probability': True  # Enable probability estimates
        }
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('svm', SVC(**svm_params))
        ])
        
        return pipeline
    
    async def train(
        self, 
        data_path: str, 
        hyperparameters: Dict[str, Any] = None,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the intent classification model.
        
        Args:
            data_path: Path to training data CSV file
            hyperparameters: Model hyperparameters
            validation_split: Fraction for validation data
            
        Returns:
            Training metrics dictionary
        """
        ml_logger.info("Starting intent model training")
        
        hyperparameters = hyperparameters or {}
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            if 'text' not in df.columns or 'intent' not in df.columns:
                raise ValueError("Data must have 'text' and 'intent' columns")
            
            # Preprocess
            df['text'] = df['text'].apply(self._preprocess_text)
            
            # Encode labels
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(df['intent'])
            X = df['text'].values
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Create and train pipeline
            self.pipeline = self._create_pipeline(hyperparameters)
            
            # Hyperparameter tuning if enabled
            if hyperparameters.get('tune_hyperparameters', False):
                param_grid = {
                    'tfidf__max_features': [3000, 5000, 10000],
                    'svm__C': [0.1, 1.0, 10.0],
                    'svm__kernel': ['rbf', 'linear']
                }
                
                grid_search = GridSearchCV(
                    self.pipeline, 
                    param_grid, 
                    cv=3, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                self.pipeline = grid_search.best_estimator_
                ml_logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                self.pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Get per-class metrics
            report = classification_report(y_val, y_pred, output_dict=True)
            
            metrics = {
                'accuracy': accuracy,
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1_score': report['macro avg']['f1-score']
            }
            
            # Save model
            model_data = {
                'pipeline': self.pipeline,
                'label_encoder': self.label_encoder
            }
            
            metadata = {
                'model_type': 'intent_classifier',
                'algorithm': 'TF-IDF + SVM',
                'metrics': metrics,
                'hyperparameters': hyperparameters,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'num_classes': len(self.label_encoder.classes_),
                'classes': self.label_encoder.classes_.tolist()
            }
            
            model_registry.save_model(
                model=model_data,
                model_name="intent",
                metadata=metadata
            )
            
            ml_logger.info(f"Intent model training completed with accuracy: {accuracy:.4f}")
            return metrics
            
        except Exception as e:
            ml_logger.error(f"Intent model training failed: {e}")
            raise
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent for input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predicted intent and confidence
        """
        if not self.pipeline or not self.label_encoder:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Get prediction and probabilities
            prediction = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            
            # Decode label
            intent = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:3]  # Top 3
            top_predictions = []
            
            for idx in top_indices:
                label = self.label_encoder.inverse_transform([idx])[0]
                prob = float(probabilities[idx])
                top_predictions.append({'intent': label, 'confidence': prob})
            
            result = {
                'intent': intent,
                'confidence': confidence,
                'top_predictions': top_predictions
            }
            
            ml_logger.debug(f"Intent prediction: {result}")
            return result
            
        except Exception as e:
            ml_logger.error(f"Intent prediction failed: {e}")
            raise
    
    async def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intents for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        if not self.pipeline or not self.label_encoder:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Get predictions and probabilities
            predictions = self.pipeline.predict(processed_texts)
            probabilities = self.pipeline.predict_proba(processed_texts)
            
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                # Decode label
                intent = self.label_encoder.inverse_transform([pred])[0]
                confidence = float(max(probs))
                
                # Get top predictions
                top_indices = np.argsort(probs)[::-1][:3]
                top_predictions = []
                
                for idx in top_indices:
                    label = self.label_encoder.inverse_transform([idx])[0]
                    prob = float(probs[idx])
                    top_predictions.append({'intent': label, 'confidence': prob})
                
                results.append({
                    'text': texts[i],
                    'intent': intent,
                    'confidence': confidence,
                    'top_predictions': top_predictions
                })
            
            return results
            
        except Exception as e:
            ml_logger.error(f"Batch intent prediction failed: {e}")
            raise