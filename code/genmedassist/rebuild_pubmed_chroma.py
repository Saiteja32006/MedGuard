import os
import json
from pathlib import Path

import chromadb


CHROMA_DIR = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "chromadb_data"
JSON_PATH = CHROMA_DIR / "pubmed_abstracts.json"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Could not find: {JSON_PATH}")

with JSON_PATH.open("r", encoding="utf-8") as f:
    abstracts = json.load(f)

# Deduplicate by PMID
seen = set()
unique = []

for a in abstracts:
    pmid = str(a.get("pmid", "")).strip()
    abstract = str(a.get("abstract", "")).strip()

    if not pmid or not abstract:
        continue

    if pmid not in seen:
        seen.add(pmid)
        unique.append({
            "pmid": pmid,
            "title": str(a.get("title", "")),
            "drug": str(a.get("drug", "")),
            "abstract": abstract,
        })

print(f"Deduped: {len(abstracts)} -> {len(unique)}")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    client.delete_collection("pubmed_evidence")
except Exception:
    pass

col = client.create_collection("pubmed_evidence")

for i in range(0, len(unique), 50):
    batch = unique[i:i + 50]

    col.add(
        documents=[a["abstract"] for a in batch],
        metadatas=[
            {
                "pmid": a["pmid"],
                "title": a["title"],
                "drug": a["drug"],
            }
            for a in batch
        ],
        ids=[f"pubmed_{a['pmid']}" for a in batch],
    )

    print(f"  Batch {i // 50 + 1}: {len(batch)} docs")

print(f"Done: {col.count()} documents in ChromaDB")

if col.count() > 0:
    r = col.query(
        query_texts=["metformin renal impairment"],
        n_results=min(2, col.count()),
    )

    for m in r["metadatas"][0]:
        print(f"  PMID {m['pmid']}: {m['title'][:60]}")
