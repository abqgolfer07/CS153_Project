import os
import discord
import logging
from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent
from sqlalchemy.orm import sessionmaker
from models import engine, ChatLog
from sentiment_analyzer import SentimentAnalyzer
from journal_analyzer import JournalAnalyzer
from datetime import datetime

PREFIX = "!"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Initialize components
agent = MistralAgent()
sentiment_analyzer = SentimentAnalyzer()
journal_analyzer = JournalAnalyzer()
Session = sessionmaker(bind=engine)

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")

# List of authorized user IDs for admin commands (add your Discord user ID)
AUTHORIZED_USERS = [
    "1341570352840445983"  # Replace with your Discord user ID
]

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return

    # Create database session
    db_session = Session()
    try:
        # Process the message with the agent
        logger.info(f"Processing message from {message.author}: {message.content}")
        response = await agent.run(message)

        # Create chat log entry
        chat_log = ChatLog(
            user_id=str(message.author.id),
            username=message.author.name,
            message_content=message.content,
            bot_response=response
        )
        db_session.add(chat_log)
        db_session.flush()  # This will populate the chat_log.id

        # Perform sentiment analysis and create sentiment record
        sentiment_analyzer.create_sentiment_record(db_session, chat_log.id, message.content)
        
        # Commit the transaction
        db_session.commit()

        # Send the response back to the channel
        await message.reply(response)

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        db_session.rollback()
        await message.reply("I encountered an error while processing your message. Please try again later.")
    
    finally:
        db_session.close()


# Commands


# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")


@bot.command(name="sentiment", help="Get sentiment analysis for recent messages")
async def get_sentiment(ctx):
    """Command to get sentiment analysis statistics for recent messages"""
    db_session = Session()
    try:
        # Get the 10 most recent chat logs with their sentiment analysis
        recent_logs = db_session.query(ChatLog).order_by(ChatLog.timestamp.desc()).limit(10).all()
        
        if not recent_logs:
            await ctx.send("No recent messages found to analyze.")
            return

        # Calculate average sentiment scores
        sentiment_summary = "Recent Chat Sentiment Analysis:\n\n"
        
        for log in recent_logs:
            if log.sentiment:
                sentiment = log.sentiment
                dominant_emotion = max([
                    ('Joy', sentiment.joy),
                    ('Trust', sentiment.trust),
                    ('Fear', sentiment.fear),
                    ('Surprise', sentiment.surprise),
                    ('Sadness', sentiment.sadness),
                    ('Disgust', sentiment.disgust),
                    ('Anger', sentiment.anger),
                    ('Anticipation', sentiment.anticipation)
                ], key=lambda x: x[1])

                sentiment_summary += f"Message: '{log.message_content[:50]}...'\n"
                sentiment_summary += f"Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})\n"
                sentiment_summary += f"Overall Sentiment: {sentiment.compound_score:.2f}\n"
                sentiment_summary += f"Confidence: {sentiment.confidence:.2f}\n\n"

        await ctx.send(sentiment_summary)

    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {str(e)}")
        await ctx.send("Error retrieving sentiment analysis.")
    
    finally:
        db_session.close()


@bot.command(name="journal", help="Log a journal entry and get sentiment analysis")
async def journal_entry(ctx, *, entry_text: str):
    """Log a journal entry and provide sentiment analysis"""
    try:
        # Log the entry
        chat_log, sentiment = journal_analyzer.log_entry(str(ctx.author.id), entry_text)
        
        # Perform comprehensive analysis
        analysis = await journal_analyzer.analyze_sentiment(entry_text)
        
        # Create a detailed response
        response = "📝 **Journal Entry Analysis**\n\n"
        
        # Sentiment summary
        response += "**Emotional Analysis:**\n"
        dominant_emotion = max(analysis['sentiment']['emotions'].items(), key=lambda x: x[1])
        response += f"• Primary Emotion: {dominant_emotion[0].title()} ({dominant_emotion[1]:.2f})\n"
        response += f"• Overall Sentiment: {analysis['sentiment']['compound_score']:.2f}\n"
        response += f"• Emotional Intensity: {analysis['sentiment']['intensity']:.2f}\n\n"
        
        # Theme analysis
        response += "**Themes and Patterns:**\n"
        response += "• Main Themes: " + ", ".join(analysis['themes']['themes'][:3]) + "\n"
        response += "• Emotional Patterns: " + ", ".join(analysis['themes']['emotional_patterns'][:2]) + "\n"
        response += "• Growth Indicators: " + ", ".join(analysis['themes']['growth_indicators'][:2]) + "\n"
        
        # Send the analysis
        await ctx.send(response)
        
    except Exception as e:
        logger.error(f"Error processing journal entry: {str(e)}")
        await ctx.send("I encountered an error while processing your journal entry. Please try again later.")

@bot.command(name="history", help="View your recent journal entries and emotional trends")
async def view_history(ctx, days: int = 7):
    """View recent journal history and emotional trends"""
    try:
        # Get user's history
        history = await journal_analyzer.get_user_history(str(ctx.author.id), limit=days)
        
        if not history:
            await ctx.send("No journal entries found for the specified time period.")
            return
        
        # Get emotional trends
        trends = await journal_analyzer.get_emotional_trends(str(ctx.author.id), days=days)
        
        # Create a summary message
        response = f"📊 **Your Journal History (Last {days} entries)**\n\n"
        
        # Add trend analysis
        if trends['dates']:
            response += "**Emotional Journey:**\n"
            response += "• Dominant Emotions: " + " → ".join(trends['dominant_emotions']) + "\n"
            
            # Calculate overall trend
            avg_start = sum(trends['compound_trend'][:3]) / 3 if len(trends['compound_trend']) >= 3 else trends['compound_trend'][0]
            avg_end = sum(trends['compound_trend'][-3:]) / 3 if len(trends['compound_trend']) >= 3 else trends['compound_trend'][-1]
            trend_direction = "improving" if avg_end > avg_start else "declining" if avg_end < avg_start else "stable"
            
            response += f"• Overall Trend: Your emotional state appears to be {trend_direction}\n\n"
        
        # Add recent entries
        response += "**Recent Entries:**\n"
        for entry in history[:5]:  # Show last 5 entries
            response += f"• {entry['timestamp'][:10]}: "
            response += f"{entry['text'][:50]}... "
            response += f"[{entry['sentiment']['dominant_emotion']} | Score: {entry['sentiment']['compound_score']:.2f}]\n"
        
        await ctx.send(response)
        
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        await ctx.send("I encountered an error while retrieving your journal history. Please try again later.")

@bot.command(name="futureMessage", help="Save a message for your future self with AI-enhanced context")
async def future_message(ctx, *, message: str):
    """Create a message for your future self with AI-enhanced context"""
    try:
        # Create the future message with context
        future_msg_dict, contextualized_message = await journal_analyzer.create_future_message(
            str(ctx.author.id),
            ctx.author.name,
            message
        )
        
        # Send the full message, splitting if necessary
        intro = "✉️ **Message for Your Future Self**\n\n"
        await ctx.send(intro)
        
        # Split message into chunks of ~1900 characters (Discord limit is 2000)
        message_chunks = [contextualized_message[i:i+1900] 
                        for i in range(0, len(contextualized_message), 1900)]
        
        for chunk in message_chunks:
            await ctx.send(chunk)
        
    except Exception as e:
        logger.error(f"Error creating future message: {str(e)}")
        await ctx.send("I encountered an error while saving your message for the future. Please try again later.")

@bot.command(name="viewFutureMessages", help="View your saved messages for your future self")
async def view_future_messages(ctx, limit: int = 5):
    """View your saved messages for your future self"""
    try:
        # Get the user's future messages
        messages = await journal_analyzer.get_future_messages(str(ctx.author.id), limit)
        
        if not messages:
            await ctx.send("You haven't saved any messages for your future self yet.")
            return
            
        # Send header
        await ctx.send("📝 **Your Messages to Your Future Self**\n")
        
        # Send each message separately
        for msg in messages:
            # Send message header
            header = f"📅 **Date:** {msg['created_at'][:10]}\n"
            header += f"💭 **Feeling:** {msg['sentiment']['dominant_emotion'].title() if msg['sentiment'] else 'Unknown'}\n\n"
            await ctx.send(header)
            
            # Split and send the full contextualized message
            message_chunks = [msg['contextualized_message'][i:i+1900] 
                            for i in range(0, len(msg['contextualized_message']), 1900)]
            
            for chunk in message_chunks:
                await ctx.send(chunk)
            
            # Add a separator between messages
            await ctx.send("─" * 40 + "\n")
        
    except Exception as e:
        logger.error(f"Error retrieving future messages: {str(e)}")
        await ctx.send("I encountered an error while retrieving your future messages. Please try again later.")

@bot.command(name="reflect", help="Generate a reflection analysis of your past journal entries")
async def reflect(ctx, days: int = 30):
    """
    Generate a reflection analysis of past journal entries
    
    Args:
        days: Number of past days to analyze (default: 30)
    """
    try:
        # Send initial message to indicate processing
        await ctx.send("🤔 Analyzing your journal entries... This may take a moment.")
        
        # Get reflection analysis
        reflection = await journal_analyzer.analyze_reflection(str(ctx.author.id), days)
        
        if not reflection["success"]:
            await ctx.send(reflection["message"])
            return
            
        # Format the response
        metadata = reflection["metadata"]
        header = (
            f"📔 **Journal Reflection Analysis**\n"
            f"Analyzing {metadata['entry_count']} entries from "
            f"{metadata['date_range']['start']} to {metadata['date_range']['end']}\n\n"
        )
        
        await ctx.send(header)
        
        # Split reflection text into chunks (Discord has a 2000 character limit)
        reflection_text = reflection["reflection"]
        chunks = [reflection_text[i:i+1900] for i in range(0, len(reflection_text), 1900)]
        
        # Send each chunk separately
        for chunk in chunks:
            await ctx.send(chunk)
            
        # Add a footer with a suggestion
        footer = "\n💭 *Use `!reflect [days]` to analyze a different time period.*"
        await ctx.send(footer)
        
    except Exception as e:
        logger.error(f"Error in reflect command: {str(e)}")
        await ctx.send("I encountered an error while generating your reflection. Please try again later.")

@bot.command(name="timeline", help="View your complete journal timeline with milestones and a reflective letter")
async def timeline(ctx):
    """Generate and display an interactive timeline of all journal entries"""
    try:
        # Send initial message to indicate processing
        processing_msg = await ctx.send("📊 Creating your journal timeline... This may take a moment.")
        
        # Generate timeline analysis
        timeline_data = await journal_analyzer.generate_timeline(str(ctx.author.id))
        
        if not timeline_data["success"]:
            await ctx.send(timeline_data["message"])
            return
        
        # Delete processing message
        await processing_msg.delete()
        
        # Send timeline header with metadata
        metadata = timeline_data["metadata"]
        header = (
            f"📚 **Your Journal Timeline**\n"
            f"Analyzing {metadata['entry_count']} entries from "
            f"{metadata['date_range']['start']} to {metadata['date_range']['end']}\n\n"
        )
        await ctx.send(header)
        
        # Send sentiment trends
        trends = timeline_data["timeline"]["sentiment_trends"]
        trends_msg = (
            "**Overall Journey**\n"
            f"• Time Period: {trends['start_period']} → {trends['end_period']}\n"
            f"• Emotional Direction: Your sentiment has been {trends['overall_direction']}\n"
            f"• Recent Dominant Emotions: {', '.join(trends['dominant_emotions'])}\n\n"
        )
        await ctx.send(trends_msg)
        
        # Send milestones
        if timeline_data["timeline"]["milestones"]:
            milestones_msg = "**Key Milestones**\n"
            for milestone in timeline_data["timeline"]["milestones"]:
                milestones_msg += f"• {milestone['date']}: {milestone['description']}\n"
            await ctx.send(milestones_msg + "\n")
        
        # Create interactive timeline
        entries = timeline_data["timeline"]["entries"]
        current_month = None
        month_entries = []
        
        for entry in entries:
            entry_date = datetime.fromisoformat(entry["timestamp"])
            month_key = entry_date.strftime("%B %Y")
            
            # If we've moved to a new month, send the previous month's entries
            if month_key != current_month and current_month is not None:
                month_msg = f"📅 **{current_month}**\n"
                for e in month_entries:
                    e_date = datetime.fromisoformat(e["timestamp"]).strftime("%d %b")
                    sentiment = e["sentiment"]
                    emotion = sentiment["dominant_emotion"].title() if sentiment else "Unknown"
                    score = f" (Score: {sentiment['compound_score']:.2f})" if sentiment else ""
                    
                    # Format entry with expandable preview
                    preview = e["content"][:100] + "..." if len(e["content"]) > 100 else e["content"]
                    month_msg += f"\n• {e_date} - {emotion}{score}\n```{preview}```\n"
                
                await ctx.send(month_msg)
                month_entries = []
            
            current_month = month_key
            month_entries.append(entry)
        
        # Send the last month's entries
        if month_entries:
            month_msg = f"📅 **{current_month}**\n"
            for e in month_entries:
                e_date = datetime.fromisoformat(e["timestamp"]).strftime("%d %b")
                sentiment = e["sentiment"]
                emotion = sentiment["dominant_emotion"].title() if sentiment else "Unknown"
                score = f" (Score: {sentiment['compound_score']:.2f})" if sentiment else ""
                
                # Format entry with expandable preview
                preview = e["content"][:100] + "..." if len(e["content"]) > 100 else e["content"]
                month_msg += f"\n• {e_date} - {emotion}{score}\n```{preview}```\n"
            
            await ctx.send(month_msg)
        
        # Send the reflective letter
        await ctx.send("\n📝 **A Letter from Your Past Self**\n")
        
        # Split letter into chunks if needed
        letter_chunks = [timeline_data["reflective_letter"][i:i+1900] 
                        for i in range(0, len(timeline_data["reflective_letter"]), 1900)]
        
        for chunk in letter_chunks:
            await ctx.send(chunk)
        
        # Add footer with usage tips
        footer = (
            "\n💡 **Timeline Tips**:\n"
            "• Entries are grouped by month for easy navigation\n"
            "• Each entry shows the date, dominant emotion, and sentiment score\n"
            "• Use `!journal` to add new entries\n"
            "• Use `!reflect` for a focused analysis of recent entries"
        )
        await ctx.send(footer)
        
    except Exception as e:
        logger.error(f"Error generating timeline: {str(e)}")
        await ctx.send("I encountered an error while generating your timeline. Please try again later.")

@bot.command(name="feedback", help="Submit feedback about your time capsule experience. Optional: Add a rating (1-5)")
async def submit_feedback(ctx, rating: int = None, *, feedback_text: str):
    """
    Submit feedback about the time capsule experience
    
    Args:
        rating: Optional rating from 1-5
        feedback_text: The feedback text
    """
    try:
        # Validate rating if provided
        if rating is not None and not (1 <= rating <= 5):
            await ctx.send("⚠️ Rating must be between 1 and 5. Your feedback will be recorded without a rating.")
            rating = None
        
        # Send initial message
        await ctx.send("📝 Processing your feedback... Thank you for helping us improve!")
        
        # Store and analyze feedback
        result = await journal_analyzer.store_feedback(
            str(ctx.author.id),
            ctx.author.name,
            feedback_text,
            rating
        )
        
        if not result["success"]:
            await ctx.send("❌ I encountered an error while processing your feedback. Please try again later.")
            return
        
        # Format the response
        response = "✨ **Thank you for your feedback!**\n\n"
        
        # Add rating if provided
        if rating:
            response += f"Your Rating: {'⭐' * rating}\n\n"
        
        # Add sentiment analysis
        sentiment = result["sentiment"]
        response += f"Feedback Tone: {sentiment['dominant_emotion'].title()}\n"
        response += f"Overall Sentiment: {sentiment['compound_score']:.2f}\n\n"
        
        # Add AI analysis
        response += "**Analysis of Your Feedback:**\n"
        
        # Split analysis into chunks if needed
        analysis_chunks = [result["analysis"][i:i+1900] for i in range(0, len(result["analysis"]), 1900)]
        
        # Send the response and analysis
        await ctx.send(response)
        for chunk in analysis_chunks:
            await ctx.send(chunk)
        
    except Exception as e:
        logger.error(f"Error in feedback command: {str(e)}")
        await ctx.send("❌ I encountered an error while processing your feedback. Please try again later.")

@bot.command(name="viewFeedback", help="View analysis of all feedback (System designers only)")
async def view_feedback(ctx):
    """View analysis of all feedback and trends"""
    try:
        # Check if user is authorized
        if str(ctx.author.id) not in AUTHORIZED_USERS:
            await ctx.send("⚠️ This command is only available to system designers.")
            return
        
        # Send initial message
        await ctx.send("📊 Analyzing feedback trends... This may take a moment.")
        
        # Get feedback analysis
        analysis = await journal_analyzer.analyze_feedback_trends()
        
        if not analysis["success"]:
            await ctx.send(analysis["message"])
            return
        
        # Format the response
        metadata = analysis["metadata"]
        header = (
            f"📈 **Time Capsule Feedback Analysis**\n"
            f"Analyzing {metadata['feedback_count']} feedback entries from "
            f"{metadata['date_range']['start']} to {metadata['date_range']['end']}\n\n"
        )
        
        # Add rating and sentiment averages if available
        if metadata['average_rating']:
            header += f"Average Rating: {'⭐' * round(metadata['average_rating'])} ({metadata['average_rating']:.1f}/5)\n"
        if metadata['average_sentiment']:
            header += f"Average Sentiment: {metadata['average_sentiment']:.2f}\n"
        
        await ctx.send(header)
        
        # Split analysis into chunks and send
        analysis_chunks = [analysis["trends_analysis"][i:i+1900] for i in range(0, len(analysis["trends_analysis"]), 1900)]
        for chunk in analysis_chunks:
            await ctx.send(chunk)
        
        # Add a footer
        footer = "\n💡 *Use `!feedback [rating] [text]` to submit new feedback.*"
        await ctx.send(footer)
        
    except Exception as e:
        logger.error(f"Error in viewFeedback command: {str(e)}")
        await ctx.send("❌ I encountered an error while retrieving feedback analysis. Please try again later.")

# Start the bot, connecting it to the gateway
bot.run(token)
