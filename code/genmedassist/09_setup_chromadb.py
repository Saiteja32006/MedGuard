#!/usr/bin/env python3
"""
09_setup_chromadb.py — Build VectorRAG (ChromaDB + PubMed Abstracts)
=====================================================================

Fetches PubMed abstracts for drugs in MedConflict-Bench and indexes
them in ChromaDB for evidence retrieval ("Why is this flagged?").

This completes the system architecture: GraphRAG (Neo4j) + VectorRAG (ChromaDB).

RUN ON LAB MACHINE (has more storage):
  pip install chromadb biopython sentence-transformers
  python 09_setup_chromadb.py

Or on Colab:
  !pip install chromadb biopython sentence-transformers
  # Then run cells

PubMed Entrez API: free, no key needed for <3 requests/second.
Set NCBI_EMAIL for politeness (they may block without it).

Author: Ala Sai Teja, IIIT Sricity
"""

import json
import time
import os
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIG
# ============================================================================

# Lab machine:
BENCH_FILE = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "medconflict_bench_v2.jsonl"
CHROMA_DIR = Path(os.getenv("GENMEDASSIST_DIR", ".")) / "chromadb_data"
# Colab:
# BENCH_FILE = Path("/content/drive/MyDrive/NIPS_2026/MedConflict-Bench/medconflict_bench_v2.jsonl")
# CHROMA_DIR = Path("/content/drive/MyDrive/NIPS_2026/chromadb_data")

NCBI_EMAIL = os.getenv("NCBI_EMAIL", "your_email@example.com")  # Replace with your email
MAX_ABSTRACTS_PER_DRUG = 20
MAX_TOTAL_ABSTRACTS = 500  # Keep it manageable

CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STEP 1: Extract unique drugs from benchmark
# ============================================================================

def get_benchmark_drugs(bench_file):
    """Extract unique drug names from the benchmark."""
    drugs = set()
    examples = [json.loads(l) for l in open(bench_file, encoding="utf-8")]

    for ex in examples:
        drug = ex.get("drug", "")
        if drug:
            # Extract base drug name
            import re
            name = re.sub(r'\(.*?\)', '', drug).strip()
            name = name.split()[0].split('-')[0]
            if len(name) >= 3:
                drugs.add(name.lower())

        inter = ex.get("interacting_drug", "")
        if inter:
            name = re.sub(r'\(.*?\)', '', inter).strip()
            name = name.split()[0].split('-')[0]
            if len(name) >= 3:
                drugs.add(name.lower())

    return sorted(drugs)


# ============================================================================
# STEP 2: Fetch PubMed abstracts
# ============================================================================

def fetch_pubmed_abstracts(drug_name, max_results=20):
    """Fetch PubMed abstracts about drug safety/interactions."""
    from Bio import Entrez
    Entrez.email = NCBI_EMAIL

    # Search for drug + safety/interaction/contraindication
    query = f"{drug_name}[Title] AND (contraindication OR interaction OR adverse OR safety OR toxicity)"

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])

        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        abstracts = []
        for article in records.get("PubmedArticle", []):
            try:
                medline = article["MedlineCitation"]
                pmid = str(medline["PMID"])
                title = medline["Article"]["ArticleTitle"]
                abstract_parts = medline["Article"].get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(p) for p in abstract_parts)

                if abstract and len(abstract) > 100:
                    abstracts.append({
                        "pmid": pmid,
                        "title": str(title),
                        "abstract": abstract[:2000],  # cap length
                        "drug": drug_name,
                    })
            except (KeyError, IndexError):
                continue

        return abstracts

    except Exception as e:
        print(f"    Error fetching {drug_name}: {e}")
        return []


# ============================================================================
# STEP 3: Build ChromaDB collection
# ============================================================================

def build_chromadb(abstracts, chroma_dir):
    """Create ChromaDB collection from PubMed abstracts."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if any
    try:
        client.delete_collection("pubmed_evidence")
    except Exception:
        pass

    collection = client.create_collection(
        name="pubmed_evidence",
        metadata={"description": "PubMed abstracts for drug safety evidence retrieval"}
    )

    # Add documents in batches
    batch_size = 50
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i+batch_size]
        collection.add(
            documents=[a["abstract"] for a in batch],
            metadatas=[{"pmid": a["pmid"], "title": a["title"], "drug": a["drug"]} for a in batch],
            ids=[f"pubmed_{a['pmid']}" for a in batch],
        )
        print(f"    Added batch {i//batch_size + 1}: {len(batch)} documents")

    return collection


def query_evidence(collection, drug_name, condition, n_results=3):
    """Query ChromaDB for relevant evidence."""
    query = f"{drug_name} {condition} contraindication safety"
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("VectorRAG Setup — ChromaDB + PubMed Abstracts")
    print("=" * 60)

    abstracts_path = CHROMA_DIR / "pubmed_abstracts.json"

    # Check if abstracts already saved from previous run
    if abstracts_path.exists():
        print(f"\n[1/2] Loading saved abstracts from {abstracts_path}...")
        with open(abstracts_path, "r", encoding="utf-8") as f:
            all_abstracts = json.load(f)
        print(f"  Loaded {len(all_abstracts)} abstracts (skipping PubMed fetch)")
    else:
        # Step 1: Get drugs from benchmark
        print("\n[1/3] Extracting drugs from benchmark...")
        drugs = get_benchmark_drugs(BENCH_FILE)
        print(f"  Found {len(drugs)} unique drugs: {drugs[:10]}...")

        # Step 2: Fetch PubMed abstracts
        print(f"\n[2/3] Fetching PubMed abstracts ({MAX_ABSTRACTS_PER_DRUG} per drug)...")
        all_abstracts = []
        for i, drug in enumerate(drugs):
            if len(all_abstracts) >= MAX_TOTAL_ABSTRACTS:
                print(f"  Reached max {MAX_TOTAL_ABSTRACTS} abstracts, stopping")
                break

            remaining = MAX_TOTAL_ABSTRACTS - len(all_abstracts)
            n_fetch = min(MAX_ABSTRACTS_PER_DRUG, remaining)

            try:
                abstracts = fetch_pubmed_abstracts(drug, max_results=n_fetch)
                all_abstracts.extend(abstracts)
                print(f"  [{i+1}/{len(drugs)}] {drug}: {len(abstracts)} abstracts (total: {len(all_abstracts)})")
            except Exception as e:
                print(f"  [{i+1}/{len(drugs)}] {drug}: ERROR - {e}")

            time.sleep(0.4)

        print(f"\n  Total abstracts fetched: {len(all_abstracts)}")
        with open(abstracts_path, "w", encoding="utf-8") as f:
            json.dump(all_abstracts, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {abstracts_path}")

    # Build ChromaDB
    print(f"\n[2/2] Building ChromaDB collection...")
    collection = build_chromadb(all_abstracts, CHROMA_DIR)
    print(f"  Collection created: {collection.count()} documents")

    # Test query
    print(f"\n  Test query: 'metformin renal impairment'")
    results = query_evidence(collection, "metformin", "renal impairment", n_results=2)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"    [{i+1}] PMID: {meta['pmid']} | {meta['title'][:60]}...")
        print(f"        {doc[:150]}...")

    print(f"\n{'='*60}")
    print(f"CHROMADB SETUP COMPLETE")
    print(f"{'='*60}")
    print(f"  Collection: pubmed_evidence ({collection.count()} documents)")
    print(f"  Storage: {CHROMA_DIR}")
    print(f"\n  This completes the system architecture:")
    print(f"    GraphRAG (Neo4j) ✅")
    print(f"    VectorRAG (ChromaDB) ✅")


if __name__ == "__main__":
    main()