from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, UTC

Base = declarative_base()

class ChatLog(Base):
    __tablename__ = 'chat_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False)
    username = Column(String(100), nullable=False)
    message_content = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    
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
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
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
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    sentiment_id = Column(Integer, ForeignKey('message_sentiments.id'))
    
    # Relationship with sentiment analysis
    sentiment = relationship("MessageSentiment")

class UserProfile(Base):
    """Store user engagement metrics and statistics"""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), nullable=False, unique=True)
    username = Column(String(100), nullable=False)
    total_entries = Column(Integer, default=0)
    total_words = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    streak_days = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_entry_date = Column(DateTime)
    reflection_score = Column(Float, default=0.0)  # Score based on entry depth/quality
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))
    
    # Relationship with achievements
    achievements = relationship("UserAchievement", back_populates="user_profile")

class Achievement(Base):
    """Define available achievements and their criteria"""
    __tablename__ = 'achievements'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    criteria_type = Column(String(50), nullable=False)  # e.g., 'streak', 'total_entries', 'sentiment'
    criteria_value = Column(Integer, nullable=False)  # Value needed to earn achievement
    badge_icon = Column(String(100), nullable=False)  # Icon/emoji representing the achievement
    tier = Column(Integer, default=1)  # Achievement tier (1=bronze, 2=silver, 3=gold)
    
    # Relationship with user achievements
    user_achievements = relationship("UserAchievement", back_populates="achievement")

class UserAchievement(Base):
    """Track which achievements each user has earned"""
    __tablename__ = 'user_achievements'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100), ForeignKey('user_profiles.user_id'), nullable=False)
    achievement_id = Column(Integer, ForeignKey('achievements.id'), nullable=False)
    earned_at = Column(DateTime, default=lambda: datetime.now(UTC))
    progress = Column(Float, default=0.0)  # Progress towards achievement (0-100%)
    
    # Relationships
    user_profile = relationship("UserProfile", back_populates="achievements")
    achievement = relationship("Achievement", back_populates="user_achievements")

# Create database engine and tables
engine = create_engine('sqlite:///chat_logs.db')
Base.metadata.create_all(engine) 