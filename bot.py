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
from dashboard import Dashboard
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

# Initialize database session maker
Session = sessionmaker(bind=engine)

# Initialize components
agent = MistralAgent()
sentiment_analyzer = SentimentAnalyzer()
journal_analyzer = JournalAnalyzer()
dashboard = Dashboard(Session)

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

@bot.command(name="feedback", help="Submit feedback about your time capsule experience. Rating (1-5) and feedback text are required.")
async def submit_feedback(ctx, rating: int, *, feedback_text: str):
    """
    Submit feedback about the time capsule experience
    
    Args:
        rating: Rating from 1-5 (required)
        feedback_text: The feedback text (required)
    """
    try:
        # Validate rating
        if not (1 <= rating <= 5):
            await ctx.send("⚠️ Rating must be between 1 and 5. Please try again with a valid rating.")
            return
        
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
        
        # Add rating
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
        footer = "\n💡 *Use `!feedback <rating> <message>` to submit new feedback.*"
        await ctx.send(footer)
        
    except Exception as e:
        logger.error(f"Error in viewFeedback command: {str(e)}")
        await ctx.send("❌ I encountered an error while retrieving feedback analysis. Please try again later.")

@bot.command(name="dashboard", help="View your mood trends dashboard with interactive charts")
async def view_dashboard(ctx, days: int = 30):
    """
    Generate and display a dashboard of mood trends
    
    Args:
        days: Number of past days to analyze (default: 30)
    """
    try:
        # Send initial message
        await ctx.send("📊 Generating your mood trends dashboard... This may take a moment.")
        
        # Generate dashboard
        result = dashboard.generate_mood_trends(str(ctx.author.id), days)
        
        if not result["success"]:
            await ctx.send(result["message"])
            return
        
        # Format statistics and detailed analysis
        stats = result["stats"]
        
        # Create header with overview
        header_msg = (
            f"📈 **Mood Dashboard Analysis** (Last {days} days)\n\n"
            f"I've analyzed your journal entries from {stats['date_range']['start']} to {stats['date_range']['end']}, "
            f"processing {stats['total_entries']} entries to create a comprehensive emotional journey visualization.\n\n"
        )
        await ctx.send(header_msg)
        
        # Send the chart
        await ctx.send(file=discord.File(result["chart_path"]))
        
        # Detailed explanation of the charts
        chart_explanation = (
            "**📊 Understanding Your Dashboard**\n\n"
            "The dashboard shows two interconnected charts:\n\n"
            "**1. Overall Sentiment Trend (Top Chart)**\n"
            "• Blue line shows your overall emotional state over time\n"
            "• Range from -1 (very negative) to +1 (very positive)\n"
            "• Each point represents a journal entry\n"
            "• Hover over points to see exact dates and scores\n\n"
            "**2. Emotional Components (Bottom Chart)**\n"
            "• Shows the intensity of 8 core emotions:\n"
            "  - Joy (Yellow) 🌟\n"
            "  - Trust (Green) 🤝\n"
            "  - Fear (Red) 😨\n"
            "  - Surprise (Purple) 😮\n"
            "  - Sadness (Blue) 😢\n"
            "  - Disgust (Orange) 😖\n"
            "  - Anger (Dark Red) 😠\n"
            "  - Anticipation (Teal) 🎯\n\n"
            "• Each emotion is tracked on a scale of 0 to 1\n"
            "• Lines show how each emotion fluctuates over time\n"
        )
        await ctx.send(chart_explanation)
        
        # Detailed analysis of the user's emotional patterns
        analysis_msg = (
            "**🔍 Your Emotional Patterns**\n\n"
            f"• **Overall Mood**: Your average sentiment is {stats['avg_sentiment']:.2f}, "
            f"indicating an overall {_get_sentiment_description(stats['avg_sentiment'])} state\n\n"
            f"• **Dominant Emotion**: {stats['dominant_emotion'].title()} has been your most prominent emotion, "
            f"suggesting {_get_emotion_insight(stats['dominant_emotion'])}\n\n"
            f"• **Trend Analysis**: Your emotional state appears to be {stats['sentiment_trend']}, "
            f"{_get_trend_insight(stats['sentiment_trend'])}\n\n"
            "**💡 Insights**\n"
            "• Use the legend to toggle specific emotions\n"
            "• Look for patterns in how emotions cluster together\n"
            "• Notice which times of day/week show stronger positive or negative sentiments\n"
            "• Track how your dominant emotions shift over time\n\n"
            "**🎯 Suggestions**\n"
            "• Try journaling at different times to capture varied emotional states\n"
            "• Use `!reflect` to get deeper insights into specific time periods\n"
            "• Compare this dashboard with your `!timeline` to see the bigger picture\n"
        )
        await ctx.send(analysis_msg)
        
        # Clean up old charts
        dashboard.cleanup_old_charts()
        
    except Exception as e:
        logger.error(f"Error in dashboard command: {str(e)}")
        await ctx.send("❌ I encountered an error while generating your dashboard. Please try again later.")

@bot.command(name="lifeStory", help="Generate an interactive narrative of your journaling journey")
async def life_story(ctx):
    """Generate and display an interactive life story from journal entries"""
    try:
        # Send initial message
        processing_msg = await ctx.send("📖 Creating your life story... This may take a moment.")
        
        # Generate life story
        story = await journal_analyzer.generate_life_story(str(ctx.author.id))
        
        if not story["success"]:
            await ctx.send(story["message"])
            return
        
        # Delete processing message
        await processing_msg.delete()

        # Define number emojis first
        number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣"]

        # Helper function to split and send messages
        async def send_chunked_message(content, max_length=1000):
            """Split and send a message in chunks"""
            chunks = []
            current_chunk = ""
            
            for line in content.split('\n'):
                if len(current_chunk) + len(line) + 1 > max_length:
                    chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk += ('\n' + line if current_chunk else line)
            
            if current_chunk:
                chunks.append(current_chunk)
            
            for chunk in chunks:
                if chunk.strip():  # Only send non-empty chunks
                    await ctx.send(chunk)
        
        # Send metadata in smaller chunks
        metadata = story["metadata"]
        await send_chunked_message(
            "📚 **Your Life Story Through Journaling**\n\n"
            f"Based on {metadata['total_entries']} journal entries from "
            f"{metadata['date_range']['start']} to {metadata['date_range']['end']}\n"
            f"Featuring {metadata['significant_events']} significant moments."
        )
        
        await send_chunked_message(
            f"\nYour emotional journey has shown {metadata['emotional_journey']['major_shifts']} major shifts, "
            f"with an {metadata['emotional_journey']['overall_arc']} trajectory.\n\n"
            "Click on the chapter numbers below to explore your story."
        )
        
        await ctx.send("─" * 40)
        
        # Only proceed with chapters if there are story sections
        if story["story_sections"]:
            # Create and send table of contents with emojis
            toc = "**📑 Chapters**\n"
            toc += "React to the numbers below to read each chapter:\n\n"
            
            # Store story sections for reaction handling
            ctx.bot.story_sections = story["story_sections"]
            
            # Add chapters with their corresponding emojis
            for i, section in enumerate(story["story_sections"]):
                toc += f"{number_emojis[i]} {section['title']}\n"
            
            # Send table of contents and store the message for reactions
            toc_msg = await ctx.send(toc)
            
            # Add reactions to the table of contents message
            for i in range(len(story["story_sections"])):
                try:
                    await toc_msg.add_reaction(number_emojis[i])
                except Exception as e:
                    logger.error(f"Error adding reaction: {str(e)}")
            
            await ctx.send("─" * 40)
            
            # Send the prologue (first section) automatically
            first_section = story["story_sections"][0]
            await ctx.send(f"**{first_section['title']}**\n")
            
            # Split content into very small chunks
            content = "\n".join(first_section['content'])
            await send_chunked_message(content)
            await ctx.send("─" * 40)
        else:
            await ctx.send("No chapters available in your story yet. Try adding more journal entries!")
        
        # Send timeline of events in small chunks
        await ctx.send("**📅 Key Moments**\n")
        
        for event in story["events"]:
            emotion_indicator = "📈" if event.get("sentiment", 0) > 0 else "📉" if event.get("sentiment", 0) < 0 else "📊"
            event_text = (
                f"{emotion_indicator} **{event['date']}**\n"
                f"• {event['content'][:100]}...\n"
                f"• Feeling: {event.get('dominant_emotion', 'Mixed emotions').title()}\n"
            )
            await ctx.send(event_text)
        
        await ctx.send("─" * 40)
        
        # Send footer tips in chunks
        footer_tips = [
            "💡 **Story Navigation Tips:**",
            "• React to the number emojis above to read each chapter",
            "• Each chapter focuses on a distinct phase of your journey",
            "• The timeline shows key moments that shaped your story",
            "• Use `!reflect` to dive deeper into specific periods"
        ]
        
        for tip in footer_tips:
            await ctx.send(tip)
        
    except Exception as e:
        logger.error(f"Error generating life story: {str(e)}")
        await ctx.send("❌ I encountered an error while creating your life story. Please try again later.")

@bot.event
async def on_reaction_add(reaction, user):
    """Handle reactions for story navigation"""
    if user.bot:
        return
        
    # Check if this is a story navigation reaction
    if not hasattr(bot, 'story_sections'):
        return
        
    number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣"]
    if reaction.emoji in number_emojis:
        try:
            # Get the corresponding story section
            section_index = number_emojis.index(reaction.emoji)
            if section_index < len(bot.story_sections):
                section = bot.story_sections[section_index]
                
                # Send the chapter title
                await reaction.message.channel.send(f"**{section['title']}**\n")
                
                # Split content into very small chunks
                content = "\n".join(section['content'])
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                # Send each chunk separately
                for chunk in chunks:
                    if chunk.strip():  # Only send non-empty chunks
                        await reaction.message.channel.send(chunk)
                
                await reaction.message.channel.send("─" * 40)
                    
        except Exception as e:
            logger.error(f"Error handling story reaction: {str(e)}")
            await reaction.message.channel.send("❌ I encountered an error while navigating your story. Please try again.")

@bot.command(name="menu", help="Display all available commands and their usage")
async def menu(ctx):
    """Display all available commands and their usage"""
    menu_text = "🤖 **Available Commands**\n\n"

    # Journaling Commands
    menu_text += "📝 **Journaling**\n"
    menu_text += "`!journal <message>` - Log a journal entry and get sentiment analysis\n"
    menu_text += "`!history [days=7]` - View your recent journal entries and emotional trends\n"
    menu_text += "`!reflect [days=30]` - Generate a reflection analysis of your past entries\n\n"

    # Time Capsule Commands
    menu_text += "⏳ **Time Capsule**\n"
    menu_text += "`!futureMessage <message>` - Save a message for your future self\n"
    menu_text += "`!viewFutureMessages [limit=5]` - View your saved messages\n\n"

    # Analysis Commands
    menu_text += "📊 **Analysis & Visualization**\n"
    menu_text += "`!sentiment` - Get sentiment analysis for recent messages\n"
    menu_text += "`!dashboard [days=30]` - View your mood trends dashboard\n"
    menu_text += "`!timeline` - View your complete journal timeline\n"
    menu_text += "`!lifeStory` - Generate an interactive narrative of your journey\n\n"

    # Feedback Commands
    menu_text += "💭 **Feedback**\n"
    menu_text += "`!feedback <rating> <message>` - Submit feedback (rating: 1-5 required)\n"
    menu_text += "`!viewFeedback` - View analysis of all feedback (Admin only)\n\n"

    # Parameter Notation
    menu_text += "📌 **Parameter Notation**\n"
    menu_text += "• `<parameter>` - Required parameter\n"
    menu_text += "• `[parameter]` - Optional parameter\n"
    menu_text += "• `[parameter=default]` - Optional parameter with default value\n\n"

    # Tips
    menu_text += "💡 **Tips**\n"
    menu_text += "• Commands are not case-sensitive\n"
    menu_text += "• Use quotes for messages containing spaces\n"
    menu_text += "• Some commands may take a moment to process\n"

    await ctx.send(menu_text)

def _get_sentiment_description(score: float) -> str:
    """Get a description of the sentiment score"""
    if score >= 0.5:
        return "very positive"
    elif score >= 0.1:
        return "somewhat positive"
    elif score > -0.1:
        return "neutral"
    elif score > -0.5:
        return "somewhat negative"
    else:
        return "very negative"

def _get_emotion_insight(emotion: str) -> str:
    """Get insight about an emotion"""
    insights = {
        "joy": "you've experienced moments of happiness and satisfaction",
        "trust": "you've developed confidence and faith in your experiences",
        "fear": "you've faced some challenging or uncertain situations",
        "surprise": "you've encountered unexpected moments or revelations",
        "sadness": "you've processed some difficult emotions or experiences",
        "disgust": "you've encountered some frustrating or unpleasant situations",
        "anger": "you've dealt with some frustrating or unjust situations",
        "anticipation": "you've looked forward to future events or changes"
    }
    return insights.get(emotion.lower(), "you've experienced various emotional states")

def _get_trend_insight(trend: str) -> str:
    """Get insight about an emotional trend"""
    insights = {
        "improving": "suggesting positive growth and development in your emotional well-being",
        "declining": "indicating you might benefit from some self-care and reflection",
        "stable": "showing consistency in your emotional state",
        "fluctuating": "showing natural variations in your emotional journey"
    }
    return insights.get(trend.lower(), "showing the natural ebb and flow of emotions")

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        command_name = ctx.message.content.split()[0][1:]  # Remove the prefix
        response = (
            f"❓ Command `{command_name}` not found.\n\n"
            "💡 **Need help?**\n"
            "• Use `!menu` to see all available commands\n"
            "• Check your spelling and try again\n"
            "• Commands are case-sensitive\n"
            "• Make sure to include required parameters"
        )
        await ctx.send(response)
    elif isinstance(error, commands.MissingRequiredArgument):
        # Get command help text
        command = ctx.command
        cmd_name = command.name
        cmd_help = command.help or "No description available"
        
        # Create a helpful error message
        response = (
            f"⚠️ Missing required argument: `{error.param.name}`\n\n"
            f"**Command:** `!{cmd_name}`\n"
            f"**Description:** {cmd_help}\n\n"
        )
        
        # Add specific usage examples based on the command
        if cmd_name == "journal":
            response += (
                "**Example Usage:**\n"
                "`!journal Today was a great day! I accomplished...`\n"
                "Make sure to include your journal entry text after the command."
            )
        elif cmd_name == "futureMessage":
            response += (
                "**Example Usage:**\n"
                "`!futureMessage Dear future self, remember to...`\n"
                "Make sure to include your message text after the command."
            )
        elif cmd_name == "feedback":
            response += (
                "**Example Usage:**\n"
                "`!feedback 5 I really enjoyed using this bot because...`\n\n"
                "**Format:**\n"
                "`!feedback <rating> <message>`\n"
                "• Rating must be a number from 1 to 5\n"
                "• Message is your feedback text"
            )
        else:
            response += (
                "**Usage:**\n"
                f"`!{cmd_name} <{error.param.name}>`\n"
                f"Type `!menu` for a full list of commands and their usage."
            )
        
        await ctx.send(response)
    elif isinstance(error, commands.BadArgument):
        # Get the command name
        cmd_name = ctx.command.name if ctx.command else "unknown"
        
        # Special handling for feedback command rating errors
        if cmd_name == "feedback" and "Converting to \"int\" failed for parameter \"rating\"" in str(error):
            response = (
                "⚠️ **Invalid Rating Format**\n\n"
                "The feedback command requires a rating number (1-5) followed by your feedback message.\n\n"
                "**Correct Format:**\n"
                "`!feedback <rating> <message>`\n\n"
                "**Examples:**\n"
                "✅ `!feedback 5 This bot is amazing!`\n"
                "✅ `!feedback 3 It's good but could be better`\n"
                "❌ `!feedback Great job!` (missing rating)\n"
                "❌ `!feedback awesome 5` (rating should come first)\n\n"
                "Please try again with a rating number (1-5) followed by your message."
            )
        else:
            # Generic bad argument handling for other commands
            response = (
                "⚠️ Invalid argument type provided.\n\n"
                f"**Error:** {str(error)}\n\n"
                "Make sure you're providing the correct type of argument:\n"
                "• Numbers should be whole numbers (e.g., `30` not `30.5`)\n"
                "• Text should be provided after the command\n"
                "• For ratings, use numbers 1-5\n\n"
                "Type `!menu` for more information about command usage."
            )
        await ctx.send(response)
    else:
        # Log other errors and send a user-friendly message
        logger.error(f"Command error: {str(error)}")
        await ctx.send(
            "❌ An error occurred while processing your command.\n"
            "Please check `!menu` for correct command usage or try again later."
        )

# Start the bot, connecting it to the gateway
bot.run(token)
