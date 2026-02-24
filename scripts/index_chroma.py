# index_chroma.py
"""
Index candidate documents into a local ChromaDB collection.

Inputs:
- candidates_path = OUT / "candidates.jsonl" (from build_documents.py)

Outputs:
- Local persistent Chroma DB stored in ./local/chroma_db

Usage:
  python index_chroma.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dbx.paths import CHROMA, OUT


JSONL_PATH = OUT / "candidates.jsonl"
CHROMA_DIR = CHROMA
COLLECTION_NAME = "candidates_v1"

# Good multilingual default (works with German+English reasonably well).
# You can swap this later.
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 128


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Missing {JSONL_PATH}. Run build_documents.py first.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("🔧 Loading documents...")
    rows = load_jsonl(JSONL_PATH)

    ids = [r["cand_id"] for r in rows]
    docs = [r["text"] for r in rows]
    metas = [r.get("meta", {}) for r in rows]

    print(f"✅ Loaded {len(rows)} documents from {JSONL_PATH}")

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("🗃️ Initializing Chroma (persistent)...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Create or get collection. We set metadata so we can see which embed model was used.
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embed_model": EMBED_MODEL_NAME},
    )

    # Important: If you want "rebuild", delete existing items first.
    # For simplicity, we clear the collection every time we re-index.
    # This avoids stale docs if you changed build_documents.py output.
    try:
        existing_count = coll.count()
    except Exception:
        existing_count = 0

    if existing_count > 0:
        print(f"⚠️ Collection already has {existing_count} items. Clearing it for a clean rebuild...")
        # Chroma supports delete(where={}) in newer versions; delete all by ids is safest.
        # We'll fetch all ids via pagination not needed here since small, but count could be large later.
        # For v1, delete by ids loaded from JSONL is sufficient.
        coll.delete(ids=ids)
        # Note: if the collection had extra ids not present in JSONL, those would remain.
        # For a true full reset, delete and recreate the collection. We'll do that when needed.

    print("📦 Embedding + upserting batches...")
    total = len(docs)
    done = 0

    for batch_ids, batch_docs, batch_meta in zip(
        chunked(ids, BATCH_SIZE), chunked(docs, BATCH_SIZE), chunked(metas, BATCH_SIZE)
    ):
        # Compute embeddings locally
        emb = model.encode(batch_docs, show_progress_bar=False, normalize_embeddings=True)

        # Chroma expects list[float] lists
        coll.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=emb.tolist(),
        )
        done += len(batch_docs)
        print(f"  - upserted {done}/{total}")

    print("\n✅ Indexing complete.")
    print(f"Chroma dir: {CHROMA_DIR.resolve()}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Items now in collection: {coll.count()}")
    print("\nNext: run search_chroma.py with a job description.")


if __name__ == "__main__":
    main()