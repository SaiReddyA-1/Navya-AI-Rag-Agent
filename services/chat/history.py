"""
Chat History Manager — Persists conversations to JSON file.
Survives browser refresh, container restart, and terminal close.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger("chat_history")

HISTORY_DIR = Path("Data/chat_history")
HISTORY_FILE = HISTORY_DIR / "conversations.json"
MAX_CONVERSATIONS = 50


class ChatHistoryManager:
    """
    Persists chat conversations to a JSON file in Data/chat_history/.
    Data/ is Docker-volume-mounted, so history survives container restarts.
    """

    def __init__(self, history_file: Path = None):
        self.history_file = history_file or HISTORY_FILE
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def load_conversations(self) -> List[Dict[str, Any]]:
        if not self.history_file.exists():
            return []
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
            return data.get("conversations", [])
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load chat history: {e}")
            return []

    def save_conversation(
        self,
        messages: List[Dict[str, str]],
        conversation_id: Optional[str] = None,
    ) -> str:
        conversations = self.load_conversations()

        # Strip non-serializable data from messages
        clean_messages = []
        for m in messages:
            clean = {"role": m["role"], "content": m["content"]}
            if m.get("sources"):
                clean["sources"] = m["sources"]
            if m.get("confidence") is not None:
                clean["confidence"] = m["confidence"]
            clean_messages.append(clean)

        conv_id = conversation_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        # Find first user message for preview
        preview = ""
        for m in clean_messages:
            if m["role"] == "user":
                preview = m["content"][:100]
                break

        conv = {
            "id": conv_id,
            "updated": datetime.utcnow().isoformat() + "Z",
            "message_count": len(clean_messages),
            "preview": preview,
            "messages": clean_messages,
        }

        # Replace existing or append
        conversations = [c for c in conversations if c["id"] != conv_id]
        conversations.append(conv)

        # Keep only recent conversations
        conversations = conversations[-MAX_CONVERSATIONS:]

        with open(self.history_file, "w") as f:
            json.dump({"conversations": conversations}, f, indent=2)

        return conv_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        for c in self.load_conversations():
            if c["id"] == conversation_id:
                return c
        return None

    def delete_conversation(self, conversation_id: str):
        conversations = self.load_conversations()
        conversations = [c for c in conversations if c["id"] != conversation_id]
        with open(self.history_file, "w") as f:
            json.dump({"conversations": conversations}, f, indent=2)

    def clear_all(self):
        with open(self.history_file, "w") as f:
            json.dump({"conversations": []}, f, indent=2)
