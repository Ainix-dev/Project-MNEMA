# memory/extractor.py
import re
from config import cfg


# ── First-person → third-person rewrite map ───────────────────────────────────
# Applied before storing any memory so the model never confuses
# "my name is Ken" as potentially being about itself.

_REWRITES = [
    # Identity / name
    (r"\bmy name is\b",        "the user's name is"),
    (r"\bi am called\b",       "the user is called"),
    (r"\bpeople call me\b",    "people call the user"),

    # State / role
    (r"\bi am\b",              "the user is"),
    (r"\bi'm\b",               "the user is"),
    (r"\bi was\b",             "the user was"),
    (r"\bi work\b",            "the user works"),
    (r"\bi live\b",            "the user lives"),
    (r"\bi'm from\b",          "the user is from"),

    # Preferences
    (r"\bi prefer\b",          "the user prefers"),
    (r"\bi like\b",            "the user likes"),
    (r"\bi love\b",            "the user loves"),
    (r"\bi hate\b",            "the user hates"),
    (r"\bi dislike\b",         "the user dislikes"),
    (r"\bi always\b",          "the user always"),
    (r"\bi never\b",           "the user never"),
    (r"\bmy favorite\b",       "the user's favorite"),
    (r"\bi want\b",            "the user wants"),
    (r"\bi need\b",            "the user needs"),
    (r"\bi enjoy\b",           "the user enjoys"),

    # Possessives
    (r"\bmy\b",                "the user's"),

    # Have / own / use
    (r"\bi have\b",            "the user has"),
    (r"\bi own\b",             "the user owns"),
    (r"\bi use\b",             "the user uses"),

    # Please always / never
    (r"\bplease always\b",     "the user wants you to always"),
    (r"\bplease never\b",      "the user wants you to never"),
    (r"\bplease don't\b",      "the user wants you to not"),
    (r"\bplease do\b",         "the user wants you to"),
]


def _rewrite(text: str) -> str:
    """
    Rewrite first-person user statements into third-person attribution
    so stored memories are unambiguous when read by the model.

    "My name is Ken"     → "the user's name is Ken"
    "I like chicken"     → "the user likes chicken"
    "I'm a developer"    → "the user is a developer"
    """
    result = text
    for pattern, replacement in _REWRITES:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    # Capitalise first letter only
    if result:
        result = result[0].upper() + result[1:]
    return result


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
                "content": f"[USER] corrected the model: {_rewrite(user_message)}",
                "type": "correction",
                "importance": cfg.importance_weights["correction"],
                "turn": turn,
            })
            return memories

        if self._matches(msg_lower, self.PREFERENCE_PATTERNS):
            memories.append({
                "content": _rewrite(user_message),
                "type": "preference",
                "importance": cfg.importance_weights["preference"],
                "turn": turn,
            })

        if self._matches(msg_lower, self.FACT_PATTERNS):
            memories.append({
                "content": _rewrite(user_message),
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
