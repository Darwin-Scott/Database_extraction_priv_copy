from pathlib import Path
import chromadb

PERSIST_DIR = Path("local/chroma_db")

def main():
    print("persist_dir:", PERSIST_DIR.resolve())
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    # list collections
    cols = client.list_collections()
    print("collections:", [c.name for c in cols])

    for c in cols:
        col = client.get_collection(c.name)
        try:
            n = col.count()
        except Exception as e:
            n = f"ERR: {e}"
        print(f"- {c.name}: count={n}")

if __name__ == "__main__":
    main()