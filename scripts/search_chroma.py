# search_chroma.py
"""
Search the local ChromaDB collection for the most similar candidates.

Usage:
  python search_chroma.py "IAM Consultant One Identity Active Directory SailPoint"

If you run without args, it uses a default query.
"""

from __future__ import annotations
from itertools import islice
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dbx.paths import CHROMA, OUT


CHROMA_DIR = CHROMA
COLLECTION_NAME = "candidates_v1"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 500
TOP_K_PRINT = 20
OUT_IDS_PATH = OUT / "top500_ids.txt"


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "IAM Consultant One Identity Active Directory SailPoint"

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Missing {CHROMA_DIR}. Run index_chroma.py first.")

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("🗃️ Opening Chroma (persistent)...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_collection(COLLECTION_NAME)

    print(f"\n🔎 Query:\n{query}\n")
    q_emb = model.encode([query], normalize_embeddings=True)

    res = coll.query(
        query_embeddings=q_emb.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    print(f"✅ Top {TOP_K_PRINT} results:\n")
    for i, (cid, doc, meta, dist) in enumerate(islice(zip(ids, docs, metas, dists), TOP_K_PRINT), start=1,):
        # For cosine distance, smaller is better (if using normalized embeddings).
        # We'll print a similarity-ish score as (1 - distance).
        sim = 1.0 - float(dist)
        doc_short = doc[:220].replace("\n", " ") + ("…" if len(doc) > 220 else "")
        print(f"{i:02d}. {cid} | score≈{sim:.3f} | skills={meta.get('n_skills')} work={meta.get('has_work')}")
        print(f"    {doc_short}\n")

    print("Next step: take the top 500 IDs (instead of top 20) for Gemini batch ranking.")

    OUT.mkdir(exist_ok=True)

    with open(OUT_IDS_PATH, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(cid + "\n")

    print(f"✅ Wrote {len(ids)} IDs to {OUT_IDS_PATH}")


if __name__ == "__main__":
    main()