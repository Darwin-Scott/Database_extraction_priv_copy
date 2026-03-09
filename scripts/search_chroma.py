# scripts/search_chroma.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

# ... your existing imports ...
# e.g. import chromadb, sentence_transformers, sqlite3, etc.

_EMBEDDER = None
_CHROMA_COLLECTION = None


def get_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Process-level singleton. This is the key to 'no reload' in Streamlit (same process),
    and also useful for in-proc CLI runs.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        print("🧠 Loading embedding model...")
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER


def get_collection(persist_dir: Path, collection_name: str = "candidates_v1"):
    """
    Process-level singleton for Chroma collection.
    """
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        print("🗃️ Opening Chroma (persistent)...")
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        _CHROMA_COLLECTION = client.get_or_create_collection(name=collection_name)
    return _CHROMA_COLLECTION


def search_chroma_inproc(
    *,
    query: str,
    persist_dir: Path,
    top_k: int,
    print_k: int,
    out_path: Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    collection_name: str = "candidates_v1",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """
    Runs a semantic search against persistent Chroma, writes IDs to out_path,
    and logs a preview similarly to CLI.

    Returns a small dict with ids, distances, metadatas, etc.
    """
    embedder = get_embedder(model_name=model_name)
    col = get_collection(persist_dir=persist_dir, collection_name=collection_name)

    log("\n🔎 Query:")
    log(query)

    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    res = col.query(
        query_embeddings=q_emb,
        n_results=int(top_k),
        include=["metadatas", "distances", "documents"],
    )

    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    # Write IDs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    # Preview (approx score: 1 - distance if cosine distance; adjust if you use something else)
    k = min(int(print_k), len(ids))
    log(f"\n✅ Top {k} results:\n")
    for i in range(k):
        cid = ids[i]
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else None
        score = (1.0 - dist) if dist is not None else None
        m = metas[i] if i < len(metas) and metas[i] is not None else {}
        headline = (m.get("headline") or m.get("Headline") or "").strip()
        skills_n = m.get("skills_count") or m.get("skills") or 0
        work = m.get("work") if "work" in m else m.get("has_work", "")

        score_s = f"{score:.3f}" if score is not None else "—"
        log(f"{i+1:02d}. {cid} | score≈{score_s} | skills={skills_n} work={work}")
        if headline:
            log(f"    Headline: {headline}")

    log(f"\n✅ Wrote {len(ids)} IDs to {out_path}")
    return {"ids": ids, "distances": dists, "metadatas": metas}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--print-k", type=int, default=20)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--persist-dir", type=str, default="local/chroma_db")
    ap.add_argument("--collection", type=str, default="candidates_v1")
    ap.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("query", nargs="+")
    args = ap.parse_args()

    query = " ".join(args.query).strip()
    out_path = Path(args.out)
    persist_dir = Path(args.persist_dir)

    search_chroma_inproc(
        query=query,
        persist_dir=persist_dir,
        top_k=int(args.top_k),
        print_k=int(args.print_k),
        out_path=out_path,
        model_name=args.model,
        collection_name=args.collection,
        log=print,
    )


if __name__ == "__main__":
    main()