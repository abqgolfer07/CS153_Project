from transformers import pipeline
from typing import Dict, Union
import numpy as np
from sqlalchemy.orm import Session
from models import MessageSentiment

class SentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis pipeline using a pre-trained model
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None
        )
        
        # Emotion mapping for aggregation
        self.plutchik_mapping = {
            'joy': ['joy', 'excitement', 'love'],
            'trust': ['admiration', 'approval', 'gratitude'],
            'fear': ['fear', 'nervousness', 'worry'],
            'surprise': ['surprise', 'confusion', 'amazement'],
            'sadness': ['sadness', 'disappointment', 'grief'],
            'disgust': ['disgust', 'disapproval', 'annoyance'],
            'anger': ['anger', 'rage', 'hate'],
            'anticipation': ['curiosity', 'interest', 'anticipation']
        }

    def _normalize_score(self, score: float) -> float:
        """Normalize scores to range [-1, 1]"""
        return max(min(score, 1.0), -1.0)

    def _aggregate_emotions(self, emotions: list) -> Dict[str, float]:
        """Aggregate fine-grained emotions into Plutchik's basic emotions"""
        plutchik_scores = {emotion: 0.0 for emotion in self.plutchik_mapping.keys()}
        
        for emotion in emotions:
            label = emotion['label']
            score = emotion['score']
            
            # Map the emotion to Plutchik's wheel
            for plutchik_emotion, related_emotions in self.plutchik_mapping.items():
                if label in related_emotions:
                    plutchik_scores[plutchik_emotion] = max(plutchik_scores[plutchik_emotion], score)
        
        return plutchik_scores

    def analyze(self, text: str) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Perform comprehensive sentiment analysis on the text
        Returns a dictionary with emotion scores and derived metrics
        """
        # Get raw emotion predictions
        emotions = self.sentiment_pipeline(text)[0]
        
        # Aggregate emotions into Plutchik's basic emotions
        plutchik_scores = self._aggregate_emotions(emotions)
        
        # Calculate derived metrics
        intensity = np.mean(list(plutchik_scores.values()))
        
        # Calculate compound score (weighted average of positive and negative emotions)
        positive_emotions = ['joy', 'trust', 'anticipation']
        negative_emotions = ['fear', 'sadness', 'disgust', 'anger']
        
        positive_score = np.mean([plutchik_scores[emotion] for emotion in positive_emotions])
        negative_score = np.mean([plutchik_scores[emotion] for emotion in negative_emotions])
        compound_score = self._normalize_score(positive_score - negative_score)
        
        # Calculate confidence based on the strength of the strongest emotions
        confidence = max(emotions, key=lambda x: x['score'])['score']
        
        return {
            'emotions': plutchik_scores,
            'compound_score': compound_score,
            'intensity': intensity,
            'confidence': confidence
        }

    def create_sentiment_record(self, db_session: Session, chat_log_id: int, text: str) -> MessageSentiment:
        """
        Analyze text and create a MessageSentiment record in the database
        """
        analysis = self.analyze(text)
        
        sentiment = MessageSentiment(
            chat_log_id=chat_log_id,
            joy=analysis['emotions']['joy'],
            trust=analysis['emotions']['trust'],
            fear=analysis['emotions']['fear'],
            surprise=analysis['emotions']['surprise'],
            sadness=analysis['emotions']['sadness'],
            disgust=analysis['emotions']['disgust'],
            anger=analysis['emotions']['anger'],
            anticipation=analysis['emotions']['anticipation'],
            confidence=analysis['confidence'],
            intensity=analysis['intensity'],
            compound_score=analysis['compound_score']
        )
        
        db_session.add(sentiment)
        return sentiment 