from datetime import datetime, timedelta, UTC
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from models import ChatLog, MessageSentiment
import os
import logging

logger = logging.getLogger("discord")

class Dashboard:
    def __init__(self, session_maker):
        """Initialize the dashboard with database session maker"""
        self.Session = session_maker
        
        # Create a directory for storing temporary chart images
        self.chart_dir = "temp_charts"
        if not os.path.exists(self.chart_dir):
            os.makedirs(self.chart_dir)

    def _get_user_data(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """
        Retrieve user's journal entries and sentiment data
        
        Args:
            user_id: The user's ID
            days: Number of past days to analyze
            
        Returns:
            DataFrame containing journal entries and sentiment scores
        """
        db_session = self.Session()
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now(UTC) - timedelta(days=days)
            
            # Query entries with their sentiment scores
            entries = (
                db_session.query(
                    ChatLog.timestamp,
                    ChatLog.message_content,
                    MessageSentiment.compound_score,
                    MessageSentiment.joy,
                    MessageSentiment.trust,
                    MessageSentiment.fear,
                    MessageSentiment.surprise,
                    MessageSentiment.sadness,
                    MessageSentiment.disgust,
                    MessageSentiment.anger,
                    MessageSentiment.anticipation
                )
                .join(MessageSentiment)
                .filter(
                    ChatLog.user_id == user_id,
                    ChatLog.timestamp >= cutoff_date
                )
                .order_by(ChatLog.timestamp.asc())
                .all()
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(entries, columns=[
                'timestamp', 'content', 'compound_score',
                'joy', 'trust', 'fear', 'surprise',
                'sadness', 'disgust', 'anger', 'anticipation'
            ])
            
            return df
            
        finally:
            db_session.close()

    def generate_mood_trends(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Generate mood trend visualizations
        
        Args:
            user_id: The user's ID
            days: Number of past days to analyze
            
        Returns:
            Dictionary containing chart file paths and statistics
        """
        try:
            # Get user data
            df = self._get_user_data(user_id, days)
            
            if df.empty:
                return {
                    "success": False,
                    "message": "No journal entries found for the specified time period."
                }
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Overall Sentiment Over Time',
                    'Emotional Components Trends'
                ),
                vertical_spacing=0.2,
                specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
            )
            
            # Add overall sentiment trend
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['compound_score'],
                    mode='lines+markers',
                    name='Overall Sentiment',
                    line=dict(color='#2E86C1'),
                    hovertemplate=(
                        '<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br>'
                        '<b>Sentiment:</b> %{y:.2f}<br>'
                        '<extra></extra>'
                    )
                ),
                row=1, col=1
            )
            
            # Add emotional components
            emotions = ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']
            colors = ['#F4D03F', '#58D68D', '#EC7063', '#BB8FCE', '#5DADE2', '#F5B041', '#E74C3C', '#45B39D']
            
            for emotion, color in zip(emotions, colors):
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[emotion],
                        mode='lines',
                        name=emotion.title(),
                        line=dict(color=color),
                        hovertemplate=(
                            f'<b>{emotion.title()}:</b> %{{y:.2f}}<br>'
                            '<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br>'
                            '<extra></extra>'
                        )
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title_text="Your Emotional Journey",
                showlegend=True,
                height=1000,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Sentiment Score (-1 to 1)", row=1, col=1)
            fig.update_yaxes(title_text="Emotion Intensity", row=2, col=1)
            
            # Save the figure
            chart_path = os.path.join(self.chart_dir, f"mood_trends_{user_id}_{int(datetime.now(UTC).timestamp())}.png")
            fig.write_image(chart_path, scale=2)
            
            # Calculate statistics
            stats = {
                "total_entries": len(df),
                "avg_sentiment": df['compound_score'].mean(),
                "dominant_emotion": df[emotions].mean().idxmax(),
                "sentiment_trend": "improving" if df['compound_score'].iloc[-3:].mean() > df['compound_score'].iloc[:3].mean() else "declining",
                "date_range": {
                    "start": df['timestamp'].min().strftime("%Y-%m-%d"),
                    "end": df['timestamp'].max().strftime("%Y-%m-%d")
                }
            }
            
            return {
                "success": True,
                "chart_path": chart_path,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error generating mood trends: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating mood trends: {str(e)}"
            }

    def cleanup_old_charts(self, max_age_hours: int = 24):
        """
        Clean up old chart files
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        try:
            cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
            
            for filename in os.listdir(self.chart_dir):
                file_path = os.path.join(self.chart_dir, filename)
                file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_modified < cutoff_time:
                    os.remove(file_path)
                    
        except Exception as e:
            logger.error(f"Error cleaning up old charts: {str(e)}") 