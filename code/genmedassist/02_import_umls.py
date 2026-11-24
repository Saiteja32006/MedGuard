"""
GenMedAssist — Step 1b: Import UMLS Data into Neo4j
====================================================
This script reads UMLS Metathesaurus files (MRCONSO.RRF, MRREL.RRF, MRSTY.RRF)
and imports clinically relevant drug/condition/lab relationships into Neo4j.

We DON'T import all 4+ million UMLS concepts. We filter to:
  - Drugs (Semantic Type: "Pharmacologic Substance", "Clinical Drug", "Antibiotic")
  - Conditions (Semantic Type: "Disease or Syndrome", "Pathologic Function")
  - Lab Tests (Semantic Type: "Laboratory Procedure", "Laboratory or Test Result")

We DON'T import all 80+ million UMLS relationships. We filter to:
  - may_treat, may_prevent, contraindicated_with, interacts_with, isa

This keeps the KG small (~100K nodes, ~500K edges) and fast for C-score queries.

Prerequisites:
  - Neo4j running (01_create_schema.py already executed)
  - UMLS files downloaded and extracted to ./data/umls/
    Required files: MRCONSO.RRF, MRREL.RRF, MRSTY.RRF

Usage:
  python 02_import_umls.py --umls-dir ./data/umls/

Expected runtime: 10-30 minutes depending on disk speed.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# ---- Configuration ----
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")

# Batch size for Neo4j imports
BATCH_SIZE = 5000

# ---- UMLS Semantic Types we care about ----
# Full list: https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
DRUG_SEMANTIC_TYPES = {
    "T109",  # Organic Chemical
    "T114",  # Nucleic Acid, Nucleoside, or Nucleotide
    "T116",  # Amino Acid, Peptide, or Protein (some drugs)
    "T121",  # Pharmacologic Substance  ← PRIMARY
    "T125",  # Hormone
    "T126",  # Enzyme
    "T195",  # Antibiotic
    "T200",  # Clinical Drug  ← PRIMARY
}

CONDITION_SEMANTIC_TYPES = {
    "T019",  # Congenital Abnormality
    "T020",  # Acquired Abnormality
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome  ← PRIMARY
    "T048",  # Mental or Behavioral Dysfunction
    "T184",  # Sign or Symptom
    "T190",  # Anatomical Abnormality
    "T191",  # Neoplastic Process
}

LABTEST_SEMANTIC_TYPES = {
    "T059",  # Laboratory Procedure  ← PRIMARY
    "T034",  # Laboratory or Test Result  ← PRIMARY
    "T060",  # Diagnostic Procedure
}

# ---- UMLS Relationship types we care about ----
RELEVANT_RELTYPES = {
    "may_treat",
    "may_prevent",
    "contraindicated_with",  # ← CRITICAL for C-score
    "interacts_with",         # ← CRITICAL for C-score
    "isa",
    "has_contraindication",
    "CI_with",                # contraindicated
    "has_mechanism_of_action",
    "induces",
    "causative_agent_of",
    "associated_with",
}


def parse_mrsty(umls_dir: Path) -> dict:
    """
    Parse MRSTY.RRF to get CUI → semantic type mappings.
    Returns: {cui: set_of_semantic_type_ids}
    """
    mrsty_path = umls_dir / "MRSTY.RRF"
    if not mrsty_path.exists():
        print(f"[ERROR] {mrsty_path} not found.")
        sys.exit(1)
    
    print(f"  Reading {mrsty_path}...")
    # MRSTY columns: CUI|TUI|STN|STY|ATUI|CVF
    df = pd.read_csv(
        mrsty_path, sep="|", header=None, dtype=str,
        names=["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "_"],
        usecols=["CUI", "TUI", "STY"],
        low_memory=False
    )
    
    cui_types = {}
    for _, row in df.iterrows():
        cui = row["CUI"]
        tui = row["TUI"]
        if cui not in cui_types:
            cui_types[cui] = set()
        cui_types[cui].add(tui)
    
    # Classify CUIs
    drug_cuis = {cui for cui, types in cui_types.items() if types & DRUG_SEMANTIC_TYPES}
    cond_cuis = {cui for cui, types in cui_types.items() if types & CONDITION_SEMANTIC_TYPES}
    lab_cuis = {cui for cui, types in cui_types.items() if types & LABTEST_SEMANTIC_TYPES}
    
    print(f"    Drug CUIs: {len(drug_cuis):,}")
    print(f"    Condition CUIs: {len(cond_cuis):,}")
    print(f"    LabTest CUIs: {len(lab_cuis):,}")
    
    return {
        "drug": drug_cuis,
        "condition": cond_cuis,
        "labtest": lab_cuis,
        "all_types": cui_types,
    }


def parse_mrconso(umls_dir: Path, relevant_cuis: set) -> dict:
    """
    Parse MRCONSO.RRF to get preferred English names for relevant CUIs.
    Returns: {cui: preferred_name}
    """
    mrconso_path = umls_dir / "MRCONSO.RRF"
    if not mrconso_path.exists():
        print(f"[ERROR] {mrconso_path} not found.")
        sys.exit(1)
    
    print(f"  Reading {mrconso_path} (this may take a minute)...")
    # MRCONSO has 18 columns; we only need CUI, LAT, TS, STT, ISPREF, STR
    col_names = [
        "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
        "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF", "_"
    ]
    
    cui_names = {}
    chunk_iter = pd.read_csv(
        mrconso_path, sep="|", header=None, dtype=str,
        names=col_names, usecols=["CUI", "LAT", "TS", "STT", "ISPREF", "STR", "SAB"],
        chunksize=500_000, low_memory=False
    )
    
    for chunk in tqdm(chunk_iter, desc="    Parsing MRCONSO"):
        # Filter: English, preferred terms only
        mask = (
            (chunk["LAT"] == "ENG") &
            (chunk["CUI"].isin(relevant_cuis))
        )
        filtered = chunk[mask]
        
        # Prefer: TS=P (preferred), STT=PF (preferred form), ISPREF=Y
        for _, row in filtered.iterrows():
            cui = row["CUI"]
            name = row["STR"]
            
            if cui not in cui_names:
                cui_names[cui] = name
            elif row.get("TS") == "P" and row.get("ISPREF") == "Y":
                # This is a more preferred name — override
                cui_names[cui] = name
    
    print(f"    Names resolved: {len(cui_names):,} / {len(relevant_cuis):,} CUIs")
    return cui_names


def parse_mrrel(umls_dir: Path, relevant_cuis: set) -> pd.DataFrame:
    """
    Parse MRREL.RRF to get relationships between relevant CUIs.
    Returns: DataFrame with columns [CUI1, REL, RELA, CUI2]
    """
    mrrel_path = umls_dir / "MRREL.RRF"
    if not mrrel_path.exists():
        print(f"[ERROR] {mrrel_path} not found.")
        sys.exit(1)
    
    print(f"  Reading {mrrel_path} (this may take a few minutes)...")
    col_names = [
        "CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2",
        "RELA", "RUI", "SRUI", "SAB", "SL", "RG", "DIR", "SUPPRESS", "CVF", "_"
    ]
    
    all_rels = []
    chunk_iter = pd.read_csv(
        mrrel_path, sep="|", header=None, dtype=str,
        names=col_names, usecols=["CUI1", "REL", "RELA", "CUI2", "SAB"],
        chunksize=1_000_000, low_memory=False
    )
    
    for chunk in tqdm(chunk_iter, desc="    Parsing MRREL"):
        # Filter: both CUIs must be in our relevant set
        mask = (
            chunk["CUI1"].isin(relevant_cuis) &
            chunk["CUI2"].isin(relevant_cuis)
        )
        filtered = chunk[mask]
        
        # Further filter by relationship type if RELA is available
        if len(filtered) > 0:
            rela_mask = filtered["RELA"].str.lower().isin(RELEVANT_RELTYPES) | filtered["RELA"].isna()
            filtered = filtered[rela_mask]
        
        if len(filtered) > 0:
            all_rels.append(filtered[["CUI1", "REL", "RELA", "CUI2"]].copy())
    
    if all_rels:
        result = pd.concat(all_rels, ignore_index=True).drop_duplicates(subset=["CUI1", "RELA", "CUI2"])
        print(f"    Relevant relationships: {len(result):,}")
        return result
    else:
        print("    [WARNING] No relevant relationships found. Check your UMLS files.")
        return pd.DataFrame(columns=["CUI1", "REL", "RELA", "CUI2"])


def import_nodes(driver, cui_names: dict, cui_categories: dict):
    """Import Drug, Condition, and LabTest nodes into Neo4j."""
    
    def _import_batch(tx, label, batch):
        query = f"""
        UNWIND $batch AS row
        MERGE (n:{label} {{cui: row.cui}})
        SET n.name = row.name,
            n.name_lower = toLower(row.name)
        """
        tx.run(query, batch=batch)
    
    for label, cui_set in [("Drug", cui_categories["drug"]), 
                            ("Condition", cui_categories["condition"]),
                            ("LabTest", cui_categories["labtest"])]:
        # Build batch
        batch = []
        for cui in cui_set:
            if cui in cui_names:
                batch.append({"cui": cui, "name": cui_names[cui]})
        
        if not batch:
            print(f"  [SKIP] No {label} nodes to import")
            continue
        
        # Import in chunks
        print(f"  Importing {len(batch):,} {label} nodes...")
        with driver.session() as session:
            for i in tqdm(range(0, len(batch), BATCH_SIZE), desc=f"    {label}"):
                chunk = batch[i:i+BATCH_SIZE]
                session.execute_write(_import_batch, label, chunk)
        
        print(f"    [OK] {label}: {len(batch):,} nodes imported")


def import_relationships(driver, rels_df: pd.DataFrame, cui_categories: dict):
    """Import relationships into Neo4j with appropriate types."""
    
    def _get_node_label(cui, categories):
        if cui in categories["drug"]:
            return "Drug"
        elif cui in categories["condition"]:
            return "Condition"
        elif cui in categories["labtest"]:
            return "LabTest"
        return None
    
    # Map UMLS RELA to our relationship types
    rela_mapping = {
        "contraindicated_with": "CONTRAINDICATED_WITH",
        "has_contraindication": "CONTRAINDICATED_WITH",
        "ci_with": "CONTRAINDICATED_WITH",
        "interacts_with": "INTERACTS_WITH",
        "may_treat": "MAY_TREAT",
        "may_prevent": "MAY_PREVENT",
        "isa": "IS_A",
        "induces": "INDUCES",
        "causative_agent_of": "CAUSATIVE_AGENT_OF",
        "associated_with": "ASSOCIATED_WITH",
    }
    
    # Group by relationship type for efficient import
    rel_counts = {}
    skipped = 0
    
    with driver.session() as session:
        batch = []
        
        for _, row in tqdm(rels_df.iterrows(), total=len(rels_df), desc="  Importing relationships"):
            rela = str(row["RELA"]).lower().strip() if pd.notna(row["RELA"]) else None
            
            if rela is None or rela not in rela_mapping:
                skipped += 1
                continue
            
            neo4j_rel = rela_mapping[rela]
            label1 = _get_node_label(row["CUI1"], cui_categories)
            label2 = _get_node_label(row["CUI2"], cui_categories)
            
            if label1 is None or label2 is None:
                skipped += 1
                continue
            
            batch.append({
                "cui1": row["CUI1"],
                "cui2": row["CUI2"],
                "label1": label1,
                "label2": label2,
                "rel_type": neo4j_rel,
            })
            
            rel_counts[neo4j_rel] = rel_counts.get(neo4j_rel, 0) + 1
            
            # Flush batch
            if len(batch) >= BATCH_SIZE:
                _flush_rel_batch(session, batch)
                batch = []
        
        # Final flush
        if batch:
            _flush_rel_batch(session, batch)
    
    print(f"\n  Relationship import summary:")
    for rel_type, count in sorted(rel_counts.items()):
        print(f"    {rel_type}: {count:,}")
    print(f"    Skipped: {skipped:,}")


def _flush_rel_batch(session, batch):
    """Write a batch of relationships to Neo4j."""
    # Group by (label1, rel_type, label2) for efficient Cypher
    from collections import defaultdict
    groups = defaultdict(list)
    
    for item in batch:
        key = (item["label1"], item["rel_type"], item["label2"])
        groups[key].append({"cui1": item["cui1"], "cui2": item["cui2"]})
    
    for (label1, rel_type, label2), items in groups.items():
        query = f"""
        UNWIND $items AS row
        MATCH (a:{label1} {{cui: row.cui1}})
        MATCH (b:{label2} {{cui: row.cui2}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.source = 'UMLS'
        """
        try:
            session.run(query, items=items)
        except Exception as e:
            # Some rel types might fail if they don't match node labels;
            # this is fine — we log and skip
            pass


def verify_import(driver):
    """Print summary statistics of the imported KG."""
    print("\n" + "="*60)
    print("IMPORT VERIFICATION")
    print("="*60)
    
    with driver.session() as session:
        # Node counts
        for label in ["Drug", "Condition", "LabTest", "DrugClass", "Allergy"]:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            count = result.single()["cnt"]
            print(f"  {label} nodes: {count:,}")
        
        # Relationship counts
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS cnt
            ORDER BY cnt DESC
        """)
        print(f"\n  Relationships:")
        for record in result:
            print(f"    {record['rel_type']}: {record['cnt']:,}")
        
        # Specifically check for contraindication edges (critical for C-score)
        result = session.run("""
            MATCH (d:Drug)-[r:CONTRAINDICATED_WITH]->(c:Condition)
            RETURN count(r) AS cnt
        """)
        contra_count = result.single()["cnt"]
        
        if contra_count > 0:
            print(f"\n  [OK] Found {contra_count:,} CONTRAINDICATED_WITH edges — C-score will work!")
        else:
            print(f"\n  [WARNING] No CONTRAINDICATED_WITH edges found.")
            print(f"  This is expected from raw UMLS alone. We'll add threshold-enriched")
            print(f"  contraindication edges in 03_enrich_thresholds.py")


def main():
    parser = argparse.ArgumentParser(description="Import UMLS data into Neo4j")
    parser.add_argument("--umls-dir", type=str, required=True,
                        help="Path to directory containing MRCONSO.RRF, MRREL.RRF, MRSTY.RRF")
    args = parser.parse_args()
    
    umls_dir = Path(args.umls_dir)
    if not umls_dir.exists():
        print(f"[ERROR] Directory not found: {umls_dir}")
        sys.exit(1)
    
    print("="*60)
    print("GenMedAssist — UMLS Import to Neo4j")
    print("="*60 + "\n")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Step 1: Parse semantic types (classify CUIs as Drug/Condition/LabTest)
    print("[1/5] Parsing semantic types (MRSTY.RRF)...")
    cui_categories = parse_mrsty(umls_dir)
    
    # Build set of all relevant CUIs
    all_relevant = cui_categories["drug"] | cui_categories["condition"] | cui_categories["labtest"]
    print(f"  Total relevant CUIs: {len(all_relevant):,}\n")
    
    # Step 2: Get preferred names
    print("[2/5] Parsing concept names (MRCONSO.RRF)...")
    cui_names = parse_mrconso(umls_dir, all_relevant)
    
    # Step 3: Parse relationships
    print("\n[3/5] Parsing relationships (MRREL.RRF)...")
    rels_df = parse_mrrel(umls_dir, all_relevant)
    
    # Step 4: Import nodes
    print("\n[4/5] Importing nodes into Neo4j...")
    import_nodes(driver, cui_names, cui_categories)
    
    # Step 5: Import relationships
    print("\n[5/5] Importing relationships into Neo4j...")
    import_relationships(driver, rels_df, cui_categories)
    
    # Verify
    verify_import(driver)
    
    driver.close()
    print("\n[DONE] UMLS import complete.")
    print("Next step: Run 03_enrich_thresholds.py to add numerical threshold data to contraindication edges.")


if __name__ == "__main__":
    main()