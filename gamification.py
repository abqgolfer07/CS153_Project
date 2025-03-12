from datetime import datetime, timedelta, UTC
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import UserProfile, Achievement, UserAchievement, ChatLog, MessageSentiment
import logging

logger = logging.getLogger(__name__)

class GamificationManager:
    """Manage user achievements, profiles, and gamification features"""
    
    def __init__(self, session: Session):
        self.session = session
        self._init_achievements()

    def _init_achievements(self):
        """Initialize default achievements if they don't exist"""
        default_achievements = [
            # Journaling Frequency Achievements
            {
                'name': 'First Step',
                'description': 'Write your first journal entry',
                'criteria_type': 'total_entries',
                'criteria_value': 1,
                'badge_icon': '🌱',
                'tier': 1
            },
            {
                'name': 'Consistent Journaler',
                'description': 'Maintain a 7-day journaling streak',
                'criteria_type': 'streak',
                'criteria_value': 7,
                'badge_icon': '📝',
                'tier': 1
            },
            {
                'name': 'Dedicated Diarist',
                'description': 'Maintain a 30-day journaling streak',
                'criteria_type': 'streak',
                'criteria_value': 30,
                'badge_icon': '✍️',
                'tier': 2
            },
            
            # Entry Quality Achievements
            {
                'name': 'Deep Thinker',
                'description': 'Write a highly reflective entry',
                'criteria_type': 'reflection_score',
                'criteria_value': 90,
                'badge_icon': '🤔',
                'tier': 2
            },
            {
                'name': 'Wordsmith',
                'description': 'Write entries totaling over 10,000 words',
                'criteria_type': 'total_words',
                'criteria_value': 10000,
                'badge_icon': '📚',
                'tier': 2
            },
            
            # Emotional Growth Achievements
            {
                'name': 'Emotional Explorer',
                'description': 'Experience a wide range of emotions in your entries',
                'criteria_type': 'emotion_variety',
                'criteria_value': 6,
                'badge_icon': '🎭',
                'tier': 2
            },
            {
                'name': 'Growth Mindset',
                'description': 'Show consistent improvement in emotional well-being',
                'criteria_type': 'sentiment_trend',
                'criteria_value': 30,
                'badge_icon': '🌟',
                'tier': 3
            },
            
            # Milestone Achievements
            {
                'name': 'Century Club',
                'description': 'Write 100 journal entries',
                'criteria_type': 'total_entries',
                'criteria_value': 100,
                'badge_icon': '💯',
                'tier': 3
            }
        ]
        
        # Add achievements if they don't exist
        for achievement in default_achievements:
            existing = self.session.query(Achievement).filter_by(name=achievement['name']).first()
            if not existing:
                new_achievement = Achievement(**achievement)
                self.session.add(new_achievement)
        
        try:
            self.session.commit()
        except Exception as e:
            logger.error(f"Error initializing achievements: {str(e)}")
            self.session.rollback()

    async def get_or_create_profile(self, user_id: str, username: str) -> UserProfile:
        """Get existing user profile or create a new one"""
        profile = self.session.query(UserProfile).filter_by(user_id=user_id).first()
        
        if not profile:
            profile = UserProfile(
                user_id=user_id,
                username=username,
                created_at=datetime.now(UTC)
            )
            self.session.add(profile)
            try:
                self.session.commit()
            except Exception as e:
                logger.error(f"Error creating user profile: {str(e)}")
                self.session.rollback()
                raise
        
        return profile

    async def update_profile_stats(self, user_id: str, username: str):
        """Update user profile statistics based on their journal entries"""
        profile = await self.get_or_create_profile(user_id, username)
        
        try:
            # Get total entries and words
            entries = self.session.query(ChatLog).filter_by(user_id=user_id).all()
            total_entries = len(entries)
            total_words = sum(len(entry.message_content.split()) for entry in entries)
            
            # Calculate average sentiment
            sentiments = [entry.sentiment.compound_score for entry in entries if entry.sentiment]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
            
            # Calculate streak
            if entries:
                dates = sorted([entry.timestamp.date() for entry in entries])
                current_streak = 1
                max_streak = 1
                
                for i in range(1, len(dates)):
                    if (dates[i] - dates[i-1]).days == 1:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 1
                
                # Check if streak is still active
                if (datetime.now(UTC).date() - dates[-1]).days > 1:
                    current_streak = 0
            else:
                current_streak = 0
                max_streak = 0
            
            # Update profile
            profile.total_entries = total_entries
            profile.total_words = total_words
            profile.avg_sentiment = avg_sentiment
            profile.streak_days = current_streak
            profile.longest_streak = max(max_streak, profile.longest_streak)
            profile.last_entry_date = entries[-1].timestamp if entries else None
            
            # Calculate reflection score based on entry quality metrics
            if entries:
                latest_entry = entries[-1]
                words = len(latest_entry.message_content.split())
                sentiment_range = max(sentiments) - min(sentiments) if sentiments else 0
                
                # Score based on length, emotional range, and sentiment intensity
                reflection_score = min(100, (
                    (words / 200) * 40 +  # Length component (40%)
                    (sentiment_range * 30) +  # Emotional range (30%)
                    (abs(latest_entry.sentiment.compound_score) * 30)  # Intensity (30%)
                )) if latest_entry.sentiment else 0
                
                profile.reflection_score = reflection_score
            
            self.session.commit()
            
            # Check for achievements
            await self.check_achievements(profile)
            
        except Exception as e:
            logger.error(f"Error updating profile stats: {str(e)}")
            self.session.rollback()
            raise

    async def check_achievements(self, profile: UserProfile) -> list:
        """Check and award any newly earned achievements"""
        new_achievements = []
        
        try:
            all_achievements = self.session.query(Achievement).all()
            
            for achievement in all_achievements:
                # Skip if already earned
                if any(ua.achievement_id == achievement.id for ua in profile.achievements):
                    continue
                
                # Check if achievement criteria are met
                earned = False
                progress = 0.0
                
                if achievement.criteria_type == 'total_entries':
                    progress = (profile.total_entries / achievement.criteria_value) * 100
                    earned = profile.total_entries >= achievement.criteria_value
                
                elif achievement.criteria_type == 'streak':
                    progress = (profile.streak_days / achievement.criteria_value) * 100
                    earned = profile.streak_days >= achievement.criteria_value
                
                elif achievement.criteria_type == 'reflection_score':
                    progress = (profile.reflection_score / achievement.criteria_value) * 100
                    earned = profile.reflection_score >= achievement.criteria_value
                
                elif achievement.criteria_type == 'total_words':
                    progress = (profile.total_words / achievement.criteria_value) * 100
                    earned = profile.total_words >= achievement.criteria_value
                
                elif achievement.criteria_type == 'emotion_variety':
                    # Count unique dominant emotions in recent entries
                    recent_entries = (
                        self.session.query(ChatLog)
                        .filter_by(user_id=profile.user_id)
                        .order_by(desc(ChatLog.timestamp))
                        .limit(10)
                        .all()
                    )
                    
                    emotions = set()
                    for entry in recent_entries:
                        if entry.sentiment:
                            emotions.add(max(
                                ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation'],
                                key=lambda e: getattr(entry.sentiment, e)
                            ))
                    
                    progress = (len(emotions) / achievement.criteria_value) * 100
                    earned = len(emotions) >= achievement.criteria_value
                
                elif achievement.criteria_type == 'sentiment_trend':
                    # Check for consistent sentiment improvement over 30 days
                    month_ago = datetime.now(UTC) - timedelta(days=30)
                    entries = (
                        self.session.query(ChatLog)
                        .filter(
                            ChatLog.user_id == profile.user_id,
                            ChatLog.timestamp >= month_ago
                        )
                        .order_by(ChatLog.timestamp)
                        .all()
                    )
                    
                    if len(entries) >= 10:  # Need enough entries to establish a trend
                        sentiments = [e.sentiment.compound_score for e in entries if e.sentiment]
                        if sentiments:
                            # Calculate trend using simple linear regression
                            x = list(range(len(sentiments)))
                            y = sentiments
                            n = len(x)
                            
                            if n > 1:  # Need at least 2 points for a trend
                                slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / \
                                       (n * sum(x[i] * x[i] for i in range(n)) - sum(x) * sum(x))
                                
                                # Progress based on positive trend strength
                                progress = max(0, min(100, slope * 1000))  # Scale slope to 0-100
                                earned = slope > 0 and progress >= 50  # Significant positive trend
                
                # Update progress or award achievement
                if earned:
                    new_achievement = UserAchievement(
                        user_id=profile.user_id,
                        achievement_id=achievement.id,
                        progress=100.0
                    )
                    self.session.add(new_achievement)
                    new_achievements.append(achievement)
                else:
                    # Store progress
                    existing_progress = (
                        self.session.query(UserAchievement)
                        .filter_by(
                            user_id=profile.user_id,
                            achievement_id=achievement.id
                        )
                        .first()
                    )
                    
                    if existing_progress:
                        existing_progress.progress = progress
                    else:
                        new_progress = UserAchievement(
                            user_id=profile.user_id,
                            achievement_id=achievement.id,
                            progress=progress
                        )
                        self.session.add(new_progress)
            
            self.session.commit()
            return new_achievements
            
        except Exception as e:
            logger.error(f"Error checking achievements: {str(e)}")
            self.session.rollback()
            return []

    async def get_profile_data(self, user_id: str) -> dict:
        """Get formatted profile data for display"""
        profile = self.session.query(UserProfile).filter_by(user_id=user_id).first()
        if not profile:
            return None
        
        # Get earned achievements
        earned_achievements = (
            self.session.query(Achievement, UserAchievement)
            .join(UserAchievement)
            .filter(UserAchievement.user_id == user_id)
            .all()
        )
        
        # Get in-progress achievements
        in_progress = (
            self.session.query(Achievement, UserAchievement)
            .join(UserAchievement)
            .filter(
                UserAchievement.user_id == user_id,
                UserAchievement.progress < 100
            )
            .all()
        )
        
        return {
            "username": profile.username,
            "total_entries": profile.total_entries,
            "total_words": profile.total_words,
            "avg_sentiment": profile.avg_sentiment,
            "current_streak": profile.streak_days,
            "longest_streak": profile.longest_streak,
            "last_entry": profile.last_entry_date,
            "reflection_score": profile.reflection_score,
            "earned_achievements": [
                {
                    "name": ach.name,
                    "description": ach.description,
                    "icon": ach.badge_icon,
                    "earned_at": ua.earned_at
                }
                for ach, ua in earned_achievements
            ],
            "in_progress": [
                {
                    "name": ach.name,
                    "description": ach.description,
                    "icon": ach.badge_icon,
                    "progress": ua.progress
                }
                for ach, ua in in_progress
            ]
        }

    async def get_leaderboard(self, category: str = "total_entries", limit: int = 10) -> list:
        """Get leaderboard data for a specific category"""
        valid_categories = {
            "total_entries": UserProfile.total_entries,
            "streak": UserProfile.streak_days,
            "words": UserProfile.total_words,
            "reflection": UserProfile.reflection_score,
            "achievements": None  # Special case, handled separately
        }
        
        if category not in valid_categories:
            raise ValueError(f"Invalid leaderboard category: {category}")
        
        if category == "achievements":
            # Count achievements per user
            leaderboard = (
                self.session.query(
                    UserProfile.username,
                    func.count(UserAchievement.id).label('achievement_count')
                )
                .join(UserAchievement)
                .group_by(UserProfile.user_id)
                .order_by(desc('achievement_count'))
                .limit(limit)
                .all()
            )
            
            return [
                {
                    "username": entry[0],
                    "value": entry[1],
                    "label": "achievements earned"
                }
                for entry in leaderboard
            ]
        else:
            # Get leaderboard for other categories
            leaderboard = (
                self.session.query(UserProfile)
                .order_by(desc(valid_categories[category]))
                .limit(limit)
                .all()
            )
            
            labels = {
                "total_entries": "entries",
                "streak": "day streak",
                "words": "words written",
                "reflection": "reflection score"
            }
            
            return [
                {
                    "username": profile.username,
                    "value": getattr(profile, category),
                    "label": labels[category]
                }
                for profile in leaderboard
            ] 