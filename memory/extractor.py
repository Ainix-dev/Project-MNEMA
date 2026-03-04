# memory/extractor.py
import re
from config import cfg


class MemoryExtractor:
    PREFERENCE_PATTERNS = [
        r"\b(i prefer|i like|i love|i hate|i dislike|i always|i never|my favorite|i want)\b",
        r"\b(please (always|never|don't|do))\b",
        r"\b(i'm (a|an|the))\b",
    ]
    CORRECTION_PATTERNS = [
        r"\b(no,? (actually|that's wrong)|wrong|incorrect|you're mistaken|not right)\b",
        r"\b(i said|i meant|i told you)\b",
        r"\b(that's not what i)\b",
    ]
    FACT_PATTERNS = [
        r"\b(my name is|i am|i work|i live|i'm from|my (job|role|age|birthday))\b",
        r"\b(i have|i own|i use)\b",
    ]

    def extract(self, user_message: str, turn: int) -> list[dict]:
        memories = []
        msg_lower = user_message.lower()

        if self._matches(msg_lower, self.CORRECTION_PATTERNS):
            memories.append({
                "content": f"[USER] corrected the model: {user_message}",
                "type": "correction",
                "importance": cfg.importance_weights["correction"],
                "turn": turn,
            })
            return memories

        if self._matches(msg_lower, self.PREFERENCE_PATTERNS):
            memories.append({
                "content": f"[USER] stated: {user_message}",
                "type": "preference",
                "importance": cfg.importance_weights["preference"],
                "turn": turn,
            })

        if self._matches(msg_lower, self.FACT_PATTERNS):
            memories.append({
                "content": f"[USER] stated: {user_message}",
                "type": "fact",
                "importance": cfg.importance_weights["fact"],
                "turn": turn,
            })

        if not memories and len(user_message.split()) > 10:
            memories.append({
                "content": f"[USER] said: {user_message}",
                "type": "casual",
                "importance": cfg.importance_weights["casual"],
                "turn": turn,
            })

        return memories

    def _matches(self, text: str, patterns: list) -> bool:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
