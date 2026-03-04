# memory/store.py
import sqlite3
import json
import time
import uuid
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from config import cfg


class MemoryStore:
    """
    Two-layer store:
    - ChromaDB: vector similarity search for retrieval
    - SQLite:   metadata (strength, timestamps, importance, type)
    """

    def __init__(self, db_path="./data/memory.db", chroma_path="./data/chroma"):
        # Vector store
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"}
        )

        # Metadata store
        self.db_path = db_path
        self._init_db()

        # Embedding model (small, fast, local)
        self.embedder = SentenceTransformer(cfg.embedding_model)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,       -- 'fact', 'preference', 'correction', 'casual'
                importance_score REAL NOT NULL,
                strength REAL NOT NULL,          -- current Ebbinghaus strength (0.0 - 1.0)
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,        -- unix timestamp
                last_accessed REAL NOT NULL,
                last_reinforced REAL NOT NULL,
                source_turn INTEGER,             -- which conversation turn it came from
                archived INTEGER DEFAULT 0       -- 1 = faded out, kept for audit
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_archived ON memories(archived)")
        conn.commit()
        conn.close()

    def add(self, content: str, memory_type: str, importance: float, turn: int = 0):
        """Store a new memory."""
        mem_id = str(uuid.uuid4())
        now = time.time()
        embedding = self.embedder.encode(content).tolist()

        # Chroma (for retrieval)
        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"type": memory_type, "strength": 1.0}]
        )

        # SQLite (for decay management)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO memories
            (id, content, memory_type, importance_score, strength,
             created_at, last_accessed, last_reinforced, source_turn)
            VALUES (?, ?, ?, ?, 1.0, ?, ?, ?, ?)
        """, (mem_id, content, memory_type, importance, now, now, now, turn))
        conn.commit()
        conn.close()
        return mem_id

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        top_k = top_k or cfg.top_k_memories
        query_embedding = self.embedder.encode(query).tolist()

        alive_ids = self._get_alive_ids()
        if not alive_ids:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(alive_ids)),
            ids=alive_ids,
            include=["documents", "metadatas", "distances"]
        )

        memories = []
        if results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                memories.append({
                    "id": mem_id,
                    "content": results["documents"][0][i],
                    "type": results["metadatas"][0][i].get("type"),
                    "similarity": 1 - results["distances"][0][i],
                })
                self._reinforce(mem_id)

        return memories

    def _reinforce(self, mem_id: str):
        """Boost strength of a memory that was just accessed (spaced repetition)."""
        now = time.time()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT strength FROM memories WHERE id=?", (mem_id,)
        ).fetchone()
        if row:
            new_strength = min(
                cfg.max_strength,
                row[0] + cfg.reinforcement_boost * (1.0 - row[0])  # diminishing returns
            )
            conn.execute("""
                UPDATE memories
                SET strength=?, last_reinforced=?, access_count=access_count+1, last_accessed=?
                WHERE id=?
            """, (new_strength, now, now, mem_id))
            conn.commit()
        conn.close()

    def _get_alive_ids(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id FROM memories WHERE archived=0 AND strength >= ?",
            (cfg.min_strength_threshold,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_consolidation_candidates(self) -> list[dict]:
        """Returns high-strength memories ready for LoRA consolidation."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, content, memory_type, strength, importance_score
            FROM memories
            WHERE archived=0 AND strength >= ? AND access_count >= 1
            ORDER BY strength DESC, importance_score DESC
        """, (cfg.consolidation_min_strength,)).fetchall()
        conn.close()
        return [
            {"id": r[0], "content": r[1], "type": r[2],
             "strength": r[3], "importance": r[4]}
            for r in rows
        ]

    def get_all_for_decay(self) -> list[dict]:
        """Returns all alive memories for the decay pass."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, strength, last_reinforced, importance_score
            FROM memories WHERE archived=0
        """).fetchall()
        conn.close()
        return [
            {"id": r[0], "strength": r[1],
             "last_reinforced": r[2], "importance": r[3]}
            for r in rows
        ]

    def update_strength(self, mem_id: str, new_strength: float):
        """Called by the decay engine."""
        conn = sqlite3.connect(self.db_path)
        if new_strength < cfg.min_strength_threshold:
            # Archive instead of delete — keep a record
            conn.execute(
                "UPDATE memories SET archived=1, strength=? WHERE id=?",
                (new_strength, mem_id)
            )
            # Also remove from Chroma so it won't be retrieved
            try:
                self.collection.delete(ids=[mem_id])
            except Exception:
                pass
        else:
            conn.execute(
                "UPDATE memories SET strength=? WHERE id=?",
                (new_strength, mem_id)
            )
        conn.commit()
        conn.close()
