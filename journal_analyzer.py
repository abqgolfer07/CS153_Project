from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Tuple, Any
from sqlalchemy.orm import Session
from models import ChatLog, MessageSentiment, FutureMessage, engine
from sentiment_analyzer import SentimentAnalyzer
from mistralai import Mistral
import os
import logging
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("discord")

class JournalAnalyzer:
    def __init__(self):
        """Initialize the journal analyzer with necessary components"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.Session = sessionmaker(bind=engine)
        
        # Define the prompt template for theme analysis
        self.theme_analysis_prompt = """You are a psychological analysis assistant. Analyze the following journal entry and provide a structured analysis.

Journal entry: {entry_text}

Respond ONLY with a JSON object in this exact format (no other text, no newlines at start):
{{"themes":["theme1","theme2","theme3"],"emotional_patterns":["pattern1","pattern2"],"recurring_ideas":["idea1","idea2"],"growth_indicators":["indicator1","indicator2"],"focus_areas":["area1","area2"]}}

Guidelines for analysis:
- Themes: Key topics and subjects discussed
- Emotional patterns: Recurring emotional states or responses
- Recurring ideas: Common thoughts or concerns expressed
- Growth indicators: Signs of personal development or learning
- Focus areas: Aspects that need attention or improvement

Be specific and insightful. Each list should contain 2-3 concrete items based on the entry."""

        # Add future message prompt template
        self.future_message_prompt = """As an empathetic AI assistant, analyze this message intended for the author's future self.
        Provide a thoughtful response that:
        1. Reflects on the current emotional state and context
        2. Highlights potential areas of growth or change
        3. Suggests what the author might reflect on when reading this in the future
        4. Preserves the original message while adding meaningful context
        
        Original message: {message}
        
        Respond in this format:
        Dear Future Self,
        
        [Original Message: {message}]
        
        Current Context & Reflection:
        [Your contextualized response here, incorporating emotional analysis and future reflection points]
        
        When you read this in the future, consider:
        - [3-4 specific reflection points]
        
        With care,
        Your Past Self (with AI assistance)"""

        # Add reflection prompt template
        self.reflection_prompt = '''You are an empathetic AI assistant analyzing a user's journal entries over time. 
Review these entries chronologically and create an insightful reflection that highlights personal growth and changes.

Past Entries (from oldest to newest):
{entries}

Analyze these entries and provide a structured reflection in the following format:

1. Emotional Journey:
[Analyze how emotions and sentiments have evolved over time]

2. Recurring Themes:
[Identify and discuss 2-3 main themes that appear across entries]

3. Personal Growth:
[Highlight specific areas where growth or change is evident]

4. Shifting Perspectives:
[Note how viewpoints or approaches have changed]

5. Forgotten Ideas & Insights:
[Surface valuable thoughts or plans that may have been forgotten]

6. Current Patterns:
[Identify current behavioral or thinking patterns based on recent entries]

7. Future Directions:
[Suggest 2-3 areas for continued growth based on the analysis]

Keep the tone empathetic and supportive while providing specific examples from the entries to support your analysis.'''

        # Add timeline analysis prompt template
        self.timeline_prompt = '''You are an empathetic AI assistant analyzing a user's complete journal timeline.
Review these entries chronologically and identify key milestones, emotional shifts, and personal growth moments.

Timeline Entries (from oldest to newest):
{entries}

Sentiment Trends:
{sentiment_trends}

Create a personalized letter that follows this format:

Dear [Current Self],

[A warm, personal opening reflecting on the journey documented in these entries]

Key Milestones:
[List 3-4 significant moments or shifts identified from the entries, with specific dates and emotional context]

Your Growth Journey:
[Describe the evolution of thoughts, feelings, and perspectives over time]

Lessons & Insights:
[Share 2-3 important lessons or realizations evident from the entries]

Looking Forward:
[Offer encouragement and insights for the future based on observed patterns and growth]

With care and reflection,
[Your Past Self]

Keep the tone deeply personal and empathetic, using specific examples from the entries to make the letter feel authentic and meaningful.'''

        # Add feedback analysis prompt template
        self.feedback_analysis_prompt = '''You are an AI assistant analyzing user feedback about a time capsule journaling experience.
Review this feedback and provide a structured analysis that will help improve the system.

User Feedback: {feedback_text}

Previous Feedback Themes (if any):
{previous_themes}

Analyze this feedback and provide insights in the following format:

1. Key Points:
[Extract 2-3 main points from the feedback]

2. Sentiment:
[Analyze the overall tone - positive, negative, or mixed]

3. Feature Feedback:
[Identify specific features mentioned and user's opinion]

4. Suggested Improvements:
[Extract or infer 1-2 concrete suggestions]

5. Common Themes:
[Note if this feedback aligns with previous themes]

Keep the analysis constructive and action-oriented.'''

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the journal entry text
        - Remove extra whitespace
        - Normalize line endings
        - Remove special characters while preserving essential punctuation
        """
        # Remove extra whitespace and normalize line endings
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters while preserving essential punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text

    def log_entry(self, user_id: str, entry_text: str) -> Tuple[ChatLog, MessageSentiment]:
        """
        Store a journal entry with its analysis in the database
        
        Args:
            user_id: The unique identifier for the user
            entry_text: The raw journal entry text
        
        Returns:
            Tuple containing the created ChatLog and MessageSentiment objects
        """
        # Preprocess the entry text
        processed_text = self.preprocess_text(entry_text)
        
        # Create database session
        db_session = self.Session()
        try:
            # Create chat log entry
            chat_log = ChatLog(
                user_id=user_id,
                username=user_id,  # Using user_id as username for journal entries
                message_content=processed_text,
                bot_response="Journal Entry Logged",  # Placeholder response
                timestamp=datetime.utcnow()
            )
            db_session.add(chat_log)
            db_session.flush()  # Get the chat_log.id
            
            # Create sentiment record
            sentiment = self.sentiment_analyzer.create_sentiment_record(
                db_session, 
                chat_log.id, 
                processed_text
            )
            
            # Commit the transaction
            db_session.commit()
            return chat_log, sentiment
            
        finally:
            db_session.close()

    async def analyze_sentiment(self, entry_text: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment and theme analysis on the journal entry
        
        Args:
            entry_text: The journal entry text to analyze
            
        Returns:
            Dictionary containing sentiment scores, emotions, and thematic analysis
        """
        # Get basic sentiment analysis
        sentiment_analysis = self.sentiment_analyzer.analyze(entry_text)
        
        # Perform theme analysis using Mistral
        theme_prompt = self.theme_analysis_prompt.format(entry_text=entry_text)
        
        try:
            theme_response = await self.mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "You are a psychological analysis assistant. Respond only with the exact JSON format requested, no additional text or formatting."},
                    {"role": "user", "content": theme_prompt}
                ]
            )
            
            # Clean and parse the response
            response_text = theme_response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw theme analysis response: {response_text}")
            
            # Remove any potential markdown code block formatting
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text)
            response_text = re.sub(r'^```\s*|\s*```$', '', response_text)
            
            # Ensure we have a valid JSON object
            if not response_text.startswith('{'):
                raise ValueError("Response does not start with '{'")
                
            # Parse the theme analysis JSON
            theme_analysis = json.loads(response_text)
            
            # Validate and ensure all required keys exist with valid values
            required_keys = ['themes', 'emotional_patterns', 'recurring_ideas', 'growth_indicators', 'focus_areas']
            for key in required_keys:
                if key not in theme_analysis:
                    theme_analysis[key] = []
                if not isinstance(theme_analysis[key], list) or not theme_analysis[key]:
                    theme_analysis[key] = [f"No {key.replace('_', ' ')} identified"]
                # Ensure we have at least 2 items
                while len(theme_analysis[key]) < 2:
                    theme_analysis[key].append(f"Additional {key.replace('_', ' ')}")
            
        except Exception as e:
            logger.error(f"Error in theme analysis: {str(e)}")
            logger.error(f"Response text: {response_text if 'response_text' in locals() else 'No response'}")
            
            # Provide meaningful fallback content
            theme_analysis = {
                "themes": ["Personal Experience", "Daily Activities", "Self-Reflection"],
                "emotional_patterns": ["Emotional Processing", "Adaptive Response"],
                "recurring_ideas": ["Personal Growth", "Life Experiences"],
                "growth_indicators": ["Learning Process", "Self-Awareness"],
                "focus_areas": ["Emotional Management", "Personal Development"]
            }
        
        # Combine sentiment and theme analysis
        return {
            "sentiment": {
                "compound_score": sentiment_analysis["compound_score"],
                "emotions": sentiment_analysis["emotions"],
                "intensity": sentiment_analysis["intensity"],
                "confidence": sentiment_analysis["confidence"]
            },
            "themes": theme_analysis
        }

    async def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the user's recent journal entries with their analysis
        
        Args:
            user_id: The user's unique identifier
            limit: Maximum number of entries to retrieve
            
        Returns:
            List of dictionaries containing entry text, sentiment, and analysis
        """
        db_session = self.Session()
        try:
            # Get recent entries for the user
            entries = (
                db_session.query(ChatLog)
                .filter(ChatLog.user_id == user_id)
                .order_by(ChatLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            
            history = []
            for entry in entries:
                if entry.sentiment:
                    history.append({
                        "timestamp": entry.timestamp.isoformat(),
                        "text": entry.message_content,
                        "sentiment": {
                            "compound_score": entry.sentiment.compound_score,
                            "dominant_emotion": max([
                                ("joy", entry.sentiment.joy),
                                ("trust", entry.sentiment.trust),
                                ("fear", entry.sentiment.fear),
                                ("surprise", entry.sentiment.surprise),
                                ("sadness", entry.sentiment.sadness),
                                ("disgust", entry.sentiment.disgust),
                                ("anger", entry.sentiment.anger),
                                ("anticipation", entry.sentiment.anticipation)
                            ], key=lambda x: x[1])[0],
                            "intensity": entry.sentiment.intensity,
                            "confidence": entry.sentiment.confidence
                        }
                    })
            
            return history
            
        finally:
            db_session.close()

    async def get_emotional_trends(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze emotional trends over time for a user
        
        Args:
            user_id: The user's unique identifier
            days: Number of days to analyze
            
        Returns:
            Dictionary containing emotional trends and patterns
        """
        db_session = self.Session()
        try:
            # Get entries within the specified time range
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            entries = (
                db_session.query(ChatLog)
                .filter(
                    ChatLog.user_id == user_id,
                    ChatLog.timestamp >= cutoff_date
                )
                .order_by(ChatLog.timestamp.asc())
                .all()
            )
            
            # Calculate emotional trends
            emotion_scores = {
                "joy": [],
                "trust": [],
                "fear": [],
                "surprise": [],
                "sadness": [],
                "disgust": [],
                "anger": [],
                "anticipation": []
            }
            
            compound_scores = []
            dates = []
            
            for entry in entries:
                if entry.sentiment:
                    dates.append(entry.timestamp.date().isoformat())
                    compound_scores.append(entry.sentiment.compound_score)
                    
                    for emotion in emotion_scores:
                        emotion_scores[emotion].append(
                            getattr(entry.sentiment, emotion)
                        )
            
            # Calculate trends and patterns
            return {
                "dates": dates,
                "compound_trend": compound_scores,
                "emotion_trends": emotion_scores,
                "dominant_emotions": [
                    max(
                        [(emotion, scores[i]) for emotion, scores in emotion_scores.items()],
                        key=lambda x: x[1]
                    )[0]
                    for i in range(len(dates))
                ] if dates else []  # Add check for empty dates list
            }
            
        finally:
            db_session.close()

    async def create_future_message(self, user_id: str, username: str, message: str) -> Tuple[Dict[str, Any], str]:
        """
        Create a contextualized future message
        
        Args:
            user_id: The user's ID
            username: The user's username
            message: The original message for their future self
            
        Returns:
            Tuple of (Future message dict with all attributes, contextualized message string)
        """
        db_session = self.Session()
        try:
            # First, perform sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.analyze(message)
            
            # Create sentiment record
            sentiment = MessageSentiment(
                joy=sentiment_analysis["emotions"]["joy"],
                trust=sentiment_analysis["emotions"]["trust"],
                fear=sentiment_analysis["emotions"]["fear"],
                surprise=sentiment_analysis["emotions"]["surprise"],
                sadness=sentiment_analysis["emotions"]["sadness"],
                disgust=sentiment_analysis["emotions"]["disgust"],
                anger=sentiment_analysis["emotions"]["anger"],
                anticipation=sentiment_analysis["emotions"]["anticipation"],
                confidence=sentiment_analysis["confidence"],
                intensity=sentiment_analysis["intensity"],
                compound_score=sentiment_analysis["compound_score"]
            )
            db_session.add(sentiment)
            db_session.flush()  # Get the sentiment ID
            
            # Get AI contextualization
            future_prompt = self.future_message_prompt.format(message=message)
            response = await self.mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": "You are an empathetic AI assistant helping to contextualize messages for future reflection."},
                    {"role": "user", "content": future_prompt}
                ]
            )
            
            contextualized_message = response.choices[0].message.content.strip()
            
            # Create future message record
            future_message = FutureMessage(
                user_id=user_id,
                username=username,
                original_message=message,
                contextualized_message=contextualized_message,
                sentiment_id=sentiment.id
            )
            db_session.add(future_message)
            
            # Commit the transaction
            db_session.commit()
            
            # Create a dictionary with all the necessary attributes before closing the session
            future_message_dict = {
                "id": future_message.id,
                "user_id": future_message.user_id,
                "username": future_message.username,
                "original_message": future_message.original_message,
                "contextualized_message": future_message.contextualized_message,
                "created_at": future_message.created_at.isoformat() if future_message.created_at else None,
                "sentiment_id": future_message.sentiment_id
            }
            
            return future_message_dict, contextualized_message
            
        except Exception as e:
            logger.error(f"Error creating future message: {str(e)}")
            db_session.rollback()
            raise
        finally:
            db_session.close()

    async def get_future_messages(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve a user's future messages
        
        Args:
            user_id: The user's ID
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of future messages with their context and sentiment
        """
        db_session = self.Session()
        try:
            messages = (
                db_session.query(FutureMessage)
                .filter(FutureMessage.user_id == user_id)
                .order_by(FutureMessage.created_at.desc())
                .limit(limit)
                .all()
            )
            
            return [{
                "created_at": msg.created_at.isoformat(),
                "original_message": msg.original_message,
                "contextualized_message": msg.contextualized_message,
                "sentiment": {
                    "compound_score": msg.sentiment.compound_score,
                    "dominant_emotion": max([
                        ("joy", msg.sentiment.joy),
                        ("trust", msg.sentiment.trust),
                        ("fear", msg.sentiment.fear),
                        ("surprise", msg.sentiment.surprise),
                        ("sadness", msg.sentiment.sadness),
                        ("disgust", msg.sentiment.disgust),
                        ("anger", msg.sentiment.anger),
                        ("anticipation", msg.sentiment.anticipation)
                    ], key=lambda x: x[1])[0]
                } if msg.sentiment else None
            } for msg in messages]
            
        finally:
            db_session.close()

    async def analyze_reflection(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze a user's past journal entries to generate a reflective analysis
        
        Args:
            user_id: The user's unique identifier
            days: Number of past days to analyze (default 30)
            
        Returns:
            Dictionary containing the reflection analysis and metadata
        """
        db_session = self.Session()
        try:
            # Get entries within the specified time range
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            entries = (
                db_session.query(ChatLog)
                .filter(
                    ChatLog.user_id == user_id,
                    ChatLog.timestamp >= cutoff_date
                )
                .order_by(ChatLog.timestamp.asc())  # Oldest to newest
                .all()
            )
            
            if not entries:
                return {
                    "success": False,
                    "message": "No journal entries found for the specified time period."
                }
            
            # Format entries for the prompt
            formatted_entries = []
            for entry in entries:
                date_str = entry.timestamp.strftime("%Y-%m-%d")
                formatted_entries.append(f"Date: {date_str}\nEntry: {entry.message_content}\n")
            
            entries_text = "\n".join(formatted_entries)
            
            # Generate reflection using Mistral
            reflection_response = await self.mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic AI assistant providing insightful reflection analysis."
                    },
                    {
                        "role": "user",
                        "content": self.reflection_prompt.format(entries=entries_text)
                    }
                ]
            )
            
            reflection_text = reflection_response.choices[0].message.content.strip()
            
            # Calculate some metadata
            entry_count = len(entries)
            date_range = {
                "start": entries[0].timestamp.strftime("%Y-%m-%d"),
                "end": entries[-1].timestamp.strftime("%Y-%m-%d")
            }
            
            return {
                "success": True,
                "reflection": reflection_text,
                "metadata": {
                    "entry_count": entry_count,
                    "date_range": date_range,
                    "days_analyzed": days
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating reflection: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating reflection: {str(e)}"
            }
        finally:
            db_session.close()

    async def generate_timeline(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive timeline analysis of all user entries
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            Dictionary containing timeline data, milestones, and reflective letter
        """
        db_session = self.Session()
        try:
            # Get all entries for the user
            entries = (
                db_session.query(ChatLog)
                .filter(ChatLog.user_id == user_id)
                .order_by(ChatLog.timestamp.asc())  # Oldest to newest
                .all()
            )
            
            if not entries:
                return {
                    "success": False,
                    "message": "No journal entries found to create a timeline."
                }
            
            # Format entries and calculate sentiment trends
            formatted_entries = []
            sentiment_data = []
            milestones = []
            prev_sentiment = None
            
            for i, entry in enumerate(entries):
                # Format entry for display
                date_str = entry.timestamp.strftime("%Y-%m-%d")
                formatted_entries.append({
                    "date": date_str,
                    "content": entry.message_content,
                    "timestamp": entry.timestamp.isoformat(),
                    "sentiment": None if not entry.sentiment else {
                        "compound_score": entry.sentiment.compound_score,
                        "dominant_emotion": max([
                            ("joy", entry.sentiment.joy),
                            ("trust", entry.sentiment.trust),
                            ("fear", entry.sentiment.fear),
                            ("surprise", entry.sentiment.surprise),
                            ("sadness", entry.sentiment.sadness),
                            ("disgust", entry.sentiment.disgust),
                            ("anger", entry.sentiment.anger),
                            ("anticipation", entry.sentiment.anticipation)
                        ], key=lambda x: x[1])[0]
                    }
                })
                
                # Track sentiment changes for milestone detection
                if entry.sentiment:
                    current_sentiment = {
                        "score": entry.sentiment.compound_score,
                        "emotion": max([
                            ("joy", entry.sentiment.joy),
                            ("trust", entry.sentiment.trust),
                            ("fear", entry.sentiment.fear),
                            ("surprise", entry.sentiment.surprise),
                            ("sadness", entry.sentiment.sadness),
                            ("disgust", entry.sentiment.disgust),
                            ("anger", entry.sentiment.anger),
                            ("anticipation", entry.sentiment.anticipation)
                        ], key=lambda x: x[1])[0]
                    }
                    
                    sentiment_data.append(current_sentiment)
                    
                    # Detect significant sentiment shifts
                    if prev_sentiment and abs(current_sentiment["score"] - prev_sentiment["score"]) > 0.5:
                        milestones.append({
                            "date": date_str,
                            "type": "sentiment_shift",
                            "description": f"Significant emotional shift from {prev_sentiment['emotion']} to {current_sentiment['emotion']}",
                            "entry_index": i
                        })
                    
                    prev_sentiment = current_sentiment
            
            # Format entries for the AI prompt
            prompt_entries = []
            for entry in formatted_entries:
                prompt_entries.append(
                    f"Date: {entry['date']}\n"
                    f"Entry: {entry['content']}\n"
                    f"Emotion: {entry['sentiment']['dominant_emotion'] if entry['sentiment'] else 'Unknown'}\n"
                )
            
            # Calculate overall sentiment trends
            sentiment_trends = {
                "start_period": formatted_entries[0]["date"],
                "end_period": formatted_entries[-1]["date"],
                "overall_direction": "improving" if sentiment_data[-1]["score"] > sentiment_data[0]["score"] else "declining",
                "dominant_emotions": list(set(s["emotion"] for s in sentiment_data[:5]))  # Most recent emotions
            }
            
            # Generate reflective letter using Mistral
            letter_response = await self.mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic AI assistant creating a personalized reflective letter."
                    },
                    {
                        "role": "user",
                        "content": self.timeline_prompt.format(
                            entries="\n\n".join(prompt_entries),
                            sentiment_trends=json.dumps(sentiment_trends, indent=2)
                        )
                    }
                ]
            )
            
            reflective_letter = letter_response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "timeline": {
                    "entries": formatted_entries,
                    "milestones": milestones,
                    "sentiment_trends": sentiment_trends
                },
                "reflective_letter": reflective_letter,
                "metadata": {
                    "entry_count": len(entries),
                    "date_range": {
                        "start": formatted_entries[0]["date"],
                        "end": formatted_entries[-1]["date"]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating timeline: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating timeline: {str(e)}"
            }
        finally:
            db_session.close()

    async def store_feedback(self, user_id: str, username: str, feedback_text: str, rating: int = None) -> Dict[str, Any]:
        """
        Store and analyze user feedback about the time capsule experience
        
        Args:
            user_id: The user's ID
            username: The user's username
            feedback_text: The feedback text
            rating: Optional numerical rating
            
        Returns:
            Dictionary containing the feedback analysis and metadata
        """
        db_session = self.Session()
        try:
            # First, perform sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.analyze(feedback_text)
            
            # Create sentiment record
            sentiment = MessageSentiment(
                joy=sentiment_analysis["emotions"]["joy"],
                trust=sentiment_analysis["emotions"]["trust"],
                fear=sentiment_analysis["emotions"]["fear"],
                surprise=sentiment_analysis["emotions"]["surprise"],
                sadness=sentiment_analysis["emotions"]["sadness"],
                disgust=sentiment_analysis["emotions"]["disgust"],
                anger=sentiment_analysis["emotions"]["anger"],
                anticipation=sentiment_analysis["emotions"]["anticipation"],
                confidence=sentiment_analysis["confidence"],
                intensity=sentiment_analysis["intensity"],
                compound_score=sentiment_analysis["compound_score"]
            )
            db_session.add(sentiment)
            db_session.flush()  # Get the sentiment ID
            
            # Create feedback record
            from models import Feedback
            feedback = Feedback(
                user_id=user_id,
                username=username,
                feedback_text=feedback_text,
                rating=rating,
                sentiment_id=sentiment.id
            )
            db_session.add(feedback)
            
            # Get previous feedback themes for context
            previous_feedback = (
                db_session.query(Feedback)
                .order_by(Feedback.created_at.desc())
                .limit(5)
                .all()
            )
            
            previous_themes = "No previous feedback available."
            if previous_feedback:
                themes = [f"- {f.feedback_text[:100]}..." for f in previous_feedback]
                previous_themes = "\n".join(themes)
            
            # Generate feedback analysis using Mistral
            analysis_response = await self.mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant analyzing user feedback to improve the time capsule experience."
                    },
                    {
                        "role": "user",
                        "content": self.feedback_analysis_prompt.format(
                            feedback_text=feedback_text,
                            previous_themes=previous_themes
                        )
                    }
                ]
            )
            
            analysis_text = analysis_response.choices[0].message.content.strip()
            
            # Commit the transaction
            db_session.commit()
            
            return {
                "success": True,
                "feedback_id": feedback.id,
                "analysis": analysis_text,
                "sentiment": {
                    "compound_score": sentiment.compound_score,
                    "dominant_emotion": max([
                        ("joy", sentiment.joy),
                        ("trust", sentiment.trust),
                        ("fear", sentiment.fear),
                        ("surprise", sentiment.surprise),
                        ("sadness", sentiment.sadness),
                        ("disgust", sentiment.disgust),
                        ("anger", sentiment.anger),
                        ("anticipation", sentiment.anticipation)
                    ], key=lambda x: x[1])[0]
                }
            }
            
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            db_session.rollback()
            return {
                "success": False,
                "message": f"Error storing feedback: {str(e)}"
            }
        finally:
            db_session.close()

    async def analyze_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze trends and common themes in user feedback
        
        Returns:
            Dictionary containing feedback trends and suggested improvements
        """
        db_session = self.Session()
        try:
            # Get all feedback entries
            from models import Feedback
            feedback_entries = (
                db_session.query(Feedback)
                .order_by(Feedback.created_at.desc())
                .all()
            )
            
            if not feedback_entries:
                return {
                    "success": False,
                    "message": "No feedback entries found to analyze."
                }
            
            # Format feedback for analysis
            formatted_feedback = []
            for entry in feedback_entries:
                formatted_feedback.append(
                    f"Date: {entry.created_at.strftime('%Y-%m-%d')}\n"
                    f"Feedback: {entry.feedback_text}\n"
                    f"Rating: {entry.rating if entry.rating else 'Not provided'}\n"
                    f"Sentiment: {entry.sentiment.compound_score if entry.sentiment else 'Unknown'}\n"
                )
            
            # Generate trends analysis using Mistral
            trends_prompt = f"""Analyze these user feedback entries and identify overall trends, common themes, and actionable improvements.

Feedback Entries:
{formatted_feedback}

Provide analysis in the following format:
1. Common Themes
2. User Satisfaction Trends
3. Most Requested Features/Improvements
4. Areas of Concern
5. Recommended Actions"""
            
            trends_response = await self.mistral_client.chat.complete_async(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant analyzing user feedback trends to improve the time capsule experience."
                    },
                    {
                        "role": "user",
                        "content": trends_prompt
                    }
                ]
            )
            
            trends_analysis = trends_response.choices[0].message.content.strip()
            
            # Calculate some metadata
            feedback_count = len(feedback_entries)
            date_range = {
                "start": feedback_entries[-1].created_at.strftime("%Y-%m-%d"),
                "end": feedback_entries[0].created_at.strftime("%Y-%m-%d")
            }
            
            # Calculate average rating if available
            ratings = [f.rating for f in feedback_entries if f.rating is not None]
            avg_rating = sum(ratings) / len(ratings) if ratings else None
            
            # Calculate sentiment trends
            sentiment_scores = [f.sentiment.compound_score for f in feedback_entries if f.sentiment]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
            
            return {
                "success": True,
                "trends_analysis": trends_analysis,
                "metadata": {
                    "feedback_count": feedback_count,
                    "date_range": date_range,
                    "average_rating": avg_rating,
                    "average_sentiment": avg_sentiment
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feedback trends: {str(e)}")
            return {
                "success": False,
                "message": f"Error analyzing feedback trends: {str(e)}"
            }
        finally:
            db_session.close() 