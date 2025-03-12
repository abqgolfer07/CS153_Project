from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class ChatLog(Base):
    __tablename__ = 'chat_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    username = Column(String(100), nullable=False)
    message_content = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with sentiment analysis
    sentiment = relationship("MessageSentiment", back_populates="chat_log", uselist=False)

class MessageSentiment(Base):
    __tablename__ = 'message_sentiments'
    
    id = Column(Integer, primary_key=True)
    chat_log_id = Column(Integer, ForeignKey('chat_logs.id'))
    
    # Core emotions (based on Plutchik's wheel of emotions)
    joy = Column(Float)
    trust = Column(Float)
    fear = Column(Float)
    surprise = Column(Float)
    sadness = Column(Float)
    disgust = Column(Float)
    anger = Column(Float)
    anticipation = Column(Float)
    
    # Secondary metrics
    confidence = Column(Float)
    intensity = Column(Float)
    
    # Overall sentiment score (-1 to 1)
    compound_score = Column(Float)
    
    # Relationship with chat log
    chat_log = relationship("ChatLog", back_populates="sentiment")

class FutureMessage(Base):
    """Store messages intended for future reading"""
    __tablename__ = 'future_messages'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    username = Column(String(100), nullable=False)
    original_message = Column(Text, nullable=False)
    contextualized_message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sentiment_id = Column(Integer, ForeignKey('message_sentiments.id'))
    
    # Relationship with sentiment analysis
    sentiment = relationship("MessageSentiment")

class Feedback(Base):
    """Store user feedback about the time capsule experience"""
    __tablename__ = 'user_feedback'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    username = Column(String(100), nullable=False)
    feedback_text = Column(Text, nullable=False)
    rating = Column(Integer)  # Optional numerical rating
    created_at = Column(DateTime, default=datetime.utcnow)
    sentiment_id = Column(Integer, ForeignKey('message_sentiments.id'))
    
    # Relationship with sentiment analysis
    sentiment = relationship("MessageSentiment")

# Create database engine and tables
engine = create_engine('sqlite:///chat_logs.db')
Base.metadata.create_all(engine) 