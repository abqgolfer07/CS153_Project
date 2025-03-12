from datetime import datetime, timedelta, UTC
from sqlalchemy.orm import Session
from sqlalchemy import desc
from models import MemoryCapsule, CapsuleEntry, ChatLog
from typing import Dict, List, Any
import logging
from mistralai import Mistral
import os

logger = logging.getLogger(__name__)

class MemoryCapsuleManager:
    """Manage themed memory capsules and their entries"""
    
    def __init__(self, session: Session):
        self.session = session
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        
        # Define the prompt template for generating capsule narratives
        self.narrative_prompt = """You are an AI assistant creating a narrative summary for a themed memory capsule.
Theme: {theme}
Description: {description}

Journal Entries:
{entries}

Create an engaging narrative that:
1. Weaves together the entries into a cohesive story
2. Highlights emotional patterns and growth
3. Identifies key moments and milestones
4. Maintains the specific theme throughout
5. Provides insights and reflections

Format the response with clear sections and thoughtful transitions."""

    async def create_capsule(self, user_id: str, name: str, description: str = None) -> Dict[str, Any]:
        """Create a new themed memory capsule"""
        try:
            # Check if capsule with same name exists for user
            existing = (
                self.session.query(MemoryCapsule)
                .filter(
                    MemoryCapsule.user_id == user_id,
                    MemoryCapsule.name == name
                )
                .first()
            )
            
            if existing:
                return {
                    "success": False,
                    "message": f"A capsule named '{name}' already exists."
                }
            
            # Create new capsule
            capsule = MemoryCapsule(
                user_id=user_id,
                name=name,
                description=description
            )
            self.session.add(capsule)
            self.session.commit()
            
            return {
                "success": True,
                "capsule_id": capsule.id,
                "message": f"Created new memory capsule: {name}"
            }
            
        except Exception as e:
            logger.error(f"Error creating memory capsule: {str(e)}")
            self.session.rollback()
            return {
                "success": False,
                "message": f"Error creating capsule: {str(e)}"
            }

    async def add_entry(self, user_id: str, capsule_id: int, chat_log_id: int) -> Dict[str, Any]:
        """Add a journal entry to a memory capsule"""
        try:
            # Verify capsule belongs to user
            capsule = (
                self.session.query(MemoryCapsule)
                .filter(
                    MemoryCapsule.id == capsule_id,
                    MemoryCapsule.user_id == user_id
                )
                .first()
            )
            
            if not capsule:
                return {
                    "success": False,
                    "message": "Capsule not found or access denied."
                }
            
            # Verify chat log belongs to user
            chat_log = (
                self.session.query(ChatLog)
                .filter(
                    ChatLog.id == chat_log_id,
                    ChatLog.user_id == user_id
                )
                .first()
            )
            
            if not chat_log:
                return {
                    "success": False,
                    "message": "Journal entry not found or access denied."
                }
            
            # Check if entry is already in capsule
            existing = (
                self.session.query(CapsuleEntry)
                .filter(
                    CapsuleEntry.capsule_id == capsule_id,
                    CapsuleEntry.chat_log_id == chat_log_id
                )
                .first()
            )
            
            if existing:
                return {
                    "success": False,
                    "message": "This entry is already in the capsule."
                }
            
            # Add entry to capsule
            capsule_entry = CapsuleEntry(
                capsule_id=capsule_id,
                chat_log_id=chat_log_id
            )
            self.session.add(capsule_entry)
            self.session.commit()
            
            return {
                "success": True,
                "message": "Entry added to capsule successfully."
            }
            
        except Exception as e:
            logger.error(f"Error adding entry to capsule: {str(e)}")
            self.session.rollback()
            return {
                "success": False,
                "message": f"Error adding entry: {str(e)}"
            }

    async def get_capsule_contents(self, user_id: str, capsule_id: int) -> Dict[str, Any]:
        """Get the contents and narrative summary of a memory capsule"""
        try:
            # Get capsule with entries
            capsule = (
                self.session.query(MemoryCapsule)
                .filter(
                    MemoryCapsule.id == capsule_id,
                    MemoryCapsule.user_id == user_id
                )
                .first()
            )
            
            if not capsule:
                return {
                    "success": False,
                    "message": "Capsule not found or access denied."
                }
            
            # Get all entries in chronological order
            entries = (
                self.session.query(CapsuleEntry, ChatLog)
                .join(ChatLog)
                .filter(CapsuleEntry.capsule_id == capsule_id)
                .order_by(ChatLog.timestamp.asc())
                .all()
            )
            
            # Format entries for narrative generation
            formatted_entries = []
            for entry, chat_log in entries:
                date_str = chat_log.timestamp.strftime("%Y-%m-%d")
                formatted_entries.append(f"Date: {date_str}\nEntry: {chat_log.message_content}\n")
            
            entries_text = "\n".join(formatted_entries)
            
            # Generate narrative using Mistral
            if entries:
                narrative_response = await self.mistral_client.chat.complete_async(
                    model="mistral-large-latest",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an AI assistant creating themed memory narratives."
                        },
                        {
                            "role": "user",
                            "content": self.narrative_prompt.format(
                                theme=capsule.name,
                                description=capsule.description or "No description provided.",
                                entries=entries_text
                            )
                        }
                    ]
                )
                narrative = narrative_response.choices[0].message.content.strip()
            else:
                narrative = "No entries in this capsule yet."
            
            return {
                "success": True,
                "capsule": {
                    "name": capsule.name,
                    "description": capsule.description,
                    "created_at": capsule.created_at.isoformat(),
                    "entry_count": len(entries)
                },
                "entries": [
                    {
                        "id": chat_log.id,
                        "content": chat_log.message_content,
                        "timestamp": chat_log.timestamp.isoformat(),
                        "added_at": entry.added_at.isoformat()
                    }
                    for entry, chat_log in entries
                ],
                "narrative": narrative
            }
            
        except Exception as e:
            logger.error(f"Error retrieving capsule contents: {str(e)}")
            return {
                "success": False,
                "message": f"Error retrieving capsule: {str(e)}"
            }

    async def list_capsules(self, user_id: str) -> Dict[str, Any]:
        """List all memory capsules for a user"""
        try:
            capsules = (
                self.session.query(MemoryCapsule)
                .filter(MemoryCapsule.user_id == user_id)
                .order_by(MemoryCapsule.created_at.desc())
                .all()
            )
            
            return {
                "success": True,
                "capsules": [
                    {
                        "id": capsule.id,
                        "name": capsule.name,
                        "description": capsule.description,
                        "created_at": capsule.created_at.isoformat(),
                        "entry_count": len(capsule.entries)
                    }
                    for capsule in capsules
                ]
            }
            
        except Exception as e:
            logger.error(f"Error listing capsules: {str(e)}")
            return {
                "success": False,
                "message": f"Error listing capsules: {str(e)}"
            }

    async def delete_capsule(self, user_id: str, capsule_id: int) -> Dict[str, Any]:
        """Delete a memory capsule and its entries"""
        try:
            # Verify capsule belongs to user
            capsule = (
                self.session.query(MemoryCapsule)
                .filter(
                    MemoryCapsule.id == capsule_id,
                    MemoryCapsule.user_id == user_id
                )
                .first()
            )
            
            if not capsule:
                return {
                    "success": False,
                    "message": "Capsule not found or access denied."
                }
            
            # Delete capsule (cascade will handle entries)
            self.session.delete(capsule)
            self.session.commit()
            
            return {
                "success": True,
                "message": f"Deleted capsule: {capsule.name}"
            }
            
        except Exception as e:
            logger.error(f"Error deleting capsule: {str(e)}")
            self.session.rollback()
            return {
                "success": False,
                "message": f"Error deleting capsule: {str(e)}"
            } 