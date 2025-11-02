# st.set_page_config(
#     page_title="Auto Claim AI",
#     page_icon="🏥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import uuid
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize claim classifier
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.feature_names = []
        
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with default model
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize with a simple rule-based classifier"""
        self.model = self._create_rule_based_classifier()
    
    def _create_rule_based_classifier(self):
        """Create a rule-based classifier for demonstration"""
        class RuleBasedClassifier:
            def predict_proba(self, X):
                # Simple rule-based logic
                predictions = []
                for features in X:
                    # Extract features
                    amount = features.get('amount', 0)
                    diagnosis = features.get('diagnosis', '').lower()
                    has_name = features.get('has_name', False)
                    has_date = features.get('has_date', False)
                    
                    # Rule-based scoring
                    score = 0.5  # Base score
                    
                    # Amount-based rules
                    if amount > 0:
                        if amount < 1000:
                            score += 0.3  # Low amount = higher approval
                        elif amount < 5000:
                            score += 0.1  # Medium amount
                        else:
                            score -= 0.2  # High amount = lower approval
                    
                    # Diagnosis-based rules
                    if 'fracture' in diagnosis or 'broken' in diagnosis:
                        score += 0.2
                    elif 'surgery' in diagnosis:
                        score += 0.1
                    elif 'routine' in diagnosis or 'checkup' in diagnosis:
                        score -= 0.1
                    
                    # Completeness rules
                    if has_name and has_date:
                        score += 0.2
                    elif has_name or has_date:
                        score += 0.1
                    
                    # Clamp score between 0 and 1
                    score = max(0.0, min(1.0, score))
                    
                    # Return probability distribution
                    predictions.append([1 - score, score])  # [reject_prob, approve_prob]
                
                return np.array(predictions)
            
            def predict(self, X):
                probas = self.predict_proba(X)
                return np.argmax(probas, axis=1)
        
        return RuleBasedClassifier()
    
    def extract_features(self, extracted_fields: Dict[str, str], raw_text: str = "") -> Dict[str, float]:
        """
        Extract features for classification
        
        Args:
            extracted_fields: Extracted claim fields
            raw_text: Raw text from OCR
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Amount features
        amount = extracted_fields.get('amount', '0')
        try:
            amount_val = float(amount.replace(',', ''))
            features['amount'] = amount_val
            features['amount_log'] = np.log(amount_val + 1)
        except (ValueError, AttributeError):
            features['amount'] = 0.0
            features['amount_log'] = 0.0
        
        # Text features
        diagnosis = extracted_fields.get('diagnosis', '').lower()
        features['diagnosis_length'] = len(diagnosis)
        features['has_fracture'] = 1.0 if 'fracture' in diagnosis or 'broken' in diagnosis else 0.0
        features['has_surgery'] = 1.0 if 'surgery' in diagnosis else 0.0
        features['has_emergency'] = 1.0 if 'emergency' in diagnosis else 0.0
        features['has_routine'] = 1.0 if 'routine' in diagnosis or 'checkup' in diagnosis else 0.0
        
        # Completeness features
        features['has_name'] = 1.0 if extracted_fields.get('name') else 0.0
        features['has_date'] = 1.0 if extracted_fields.get('date') else 0.0
        features['has_provider'] = 1.0 if extracted_fields.get('provider') else 0.0
        features['has_policy'] = 1.0 if extracted_fields.get('policy_number') else 0.0
        
        # Count total fields extracted
        features['field_completeness'] = len(extracted_fields) / 8.0  # Assuming 8 possible fields
        
        # Text complexity features
        if raw_text:
            features['text_length'] = len(raw_text)
            features['word_count'] = len(raw_text.split())
            features['avg_word_length'] = np.mean([len(word) for word in raw_text.split()]) if raw_text.split() else 0
        
        return features
    
    def classify_claim(self, extracted_fields: Dict[str, str], raw_text: str = "", settings: dict = None) -> Dict[str, any]:
        """
        Classify a claim and return results
        
        Args:
            extracted_fields: Extracted claim fields
            raw_text: Raw text from OCR
            settings: Settings dictionary from the database
        
        Returns:
            Classification results
        """
        try:
            # Extract features
            features = self.extract_features(extracted_fields, raw_text)


            # Use settings or defaults
            confidence_threshold = float(settings.get('confidence_threshold', 75)) if settings else 75
            risk_threshold = float(settings.get('risk_threshold', 60)) if settings else 60
            max_auto_approval_amount = float(settings.get('max_auto_approval_amount', 5000)) if settings else 5000
            auto_approval_enabled = settings.get('auto_approval_enabled', 'true').lower() == 'true' if settings else True

            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # Get probability predictions
                X = [features]  # Wrap in list for batch prediction
                probabilities = self.model.predict_proba(X)[0]

                # Determine class
                if len(probabilities) == 2:
                    reject_prob, approve_prob = probabilities
                    prediction = "Auto-Approve" if approve_prob > reject_prob else "Manual Review"
                    confidence = max(approve_prob, reject_prob)
                else:
                    prediction = "Manual Review"
                    confidence = 0.5
            else:
                # Fallback for rule-based classifier
                prediction = "Auto-Approve" if features.get('amount', 0) < max_auto_approval_amount else "Manual Review"
                confidence = 0.8 if prediction == "Auto-Approve" else 0.6

            # Calculate risk score
            risk_score = self._calculate_risk_score(features)

            # Apply settings to override prediction if needed
            if not auto_approval_enabled:
                prediction = "Manual Review"
            elif prediction == "Auto-Approve":
                if confidence * 100 < confidence_threshold or risk_score * 100 > risk_threshold:
                    prediction = "Manual Review"

            # Determine recommended action
            recommended_action = self._get_recommended_action(prediction, confidence * 100, risk_score * 100)

            return {
                'prediction': prediction,
                'confidence': round(confidence * 100, 1),
                'risk_score': round(risk_score * 100, 1),
                'recommended_action': recommended_action,
                'features_used': list(features.keys()),
                'extracted_fields': extracted_fields
            }
        
        except Exception as e:
            logger.error(f"Error classifying claim: {e}")
            return {
                'prediction': "Manual Review",
                'confidence': 0.0,
                'risk_score': 100.0,
                'recommended_action': "Manual review required due to processing error",
                'features_used': [],
                'extracted_fields': extracted_fields
            }
    
    def _calculate_risk_score(self, features: Dict[str, float]) -> float:
        """
        Calculate risk score based on features
        
        Args:
            features: Extracted features
            
        Returns:
            Risk score (0-1)
        """
        risk_score = 0.0
        
        # Amount-based risk
        amount = features.get('amount', 0)
        if amount > 10000:
            risk_score += 0.4
        elif amount > 5000:
            risk_score += 0.2
        elif amount > 1000:
            risk_score += 0.1
        
        # Completeness risk
        completeness = features.get('field_completeness', 0)
        if completeness < 0.5:
            risk_score += 0.3
        elif completeness < 0.7:
            risk_score += 0.1
        
        # Diagnosis-based risk
        if features.get('has_routine', 0):
            risk_score += 0.1  # Routine checkups might be lower priority
        
        # Missing critical fields
        if not features.get('has_name', 0):
            risk_score += 0.2
        if not features.get('has_date', 0):
            risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def _get_recommended_action(self, prediction: str, confidence: float, risk_score: float) -> str:
        """
        Get recommended action based on classification results
        
        Args:
            prediction: Classification prediction
            confidence: Confidence score
            risk_score: Risk score
            
        Returns:
            Recommended action string
        """
        if prediction == "Auto-Approve":
            if confidence > 90:
                return "Auto-approve claim"
            elif confidence > 75:
                return "Auto-approve with flag for review"
            else:
                return "Quick manual review recommended"
        else:
            if risk_score > 70:
                return "Detailed manual review required"
            elif risk_score > 50:
                return "Standard manual review"
            else:
                return "Quick manual review"
    
    def train_model(self, training_data: List[Dict], labels: List[str], save_path: str = None):
        """
        Train the classifier with new data
        
        Args:
            training_data: List of feature dictionaries
            labels: List of labels (approve/reject)
            save_path: Path to save the trained model
        """
        try:
            # Convert to DataFrame
            X = pd.DataFrame(training_data)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy:.3f}")
            
            # Save model
            if save_path:
                self.save_model(save_path)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def save_model(self, model_path: str):
        """
        Save the trained model
        
        Args:
            model_path: Path to save the model
        """
        try:
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """
        Load a trained model
        
        Args:
            model_path: Path to the saved model
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.vectorizer = model_data.get('vectorizer')
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', [])
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise 