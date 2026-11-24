import os
"""
GenMedAssist — Step 1a: Create Neo4j Schema
=============================================
Run this AFTER docker-compose up -d and Neo4j is healthy.

This creates:
  - Node labels and constraints (uniqueness on CUI/SNOMED IDs)
  - Indexes for fast lookup during C-score computation
  - The relationship types the KG will use

Usage:
  python 01_create_schema.py
"""

from neo4j import GraphDatabase
import sys

# ---- Configuration ----
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")


def create_schema(driver):
    """Create all constraints and indexes for the GenMedAssist KG."""
    
    with driver.session() as session:
        
        # ============================================
        # 1. UNIQUENESS CONSTRAINTS
        #    These also create indexes automatically.
        # ============================================
        
        constraints = [
            # Drug nodes — unique by UMLS CUI
            ("drug_cui", 
             "CREATE CONSTRAINT drug_cui IF NOT EXISTS "
             "FOR (d:Drug) REQUIRE d.cui IS UNIQUE"),
            
            # Condition nodes — unique by UMLS CUI
            ("condition_cui", 
             "CREATE CONSTRAINT condition_cui IF NOT EXISTS "
             "FOR (c:Condition) REQUIRE c.cui IS UNIQUE"),
            
            # DrugClass nodes — unique by UMLS CUI
            ("drugclass_cui", 
             "CREATE CONSTRAINT drugclass_cui IF NOT EXISTS "
             "FOR (dc:DrugClass) REQUIRE dc.cui IS UNIQUE"),
            
            # LabTest nodes — unique by LOINC code
            ("labtest_loinc", 
             "CREATE CONSTRAINT labtest_loinc IF NOT EXISTS "
             "FOR (lt:LabTest) REQUIRE lt.loinc IS UNIQUE"),
            
            # Allergy nodes — unique by CUI
            ("allergy_cui", 
             "CREATE CONSTRAINT allergy_cui IF NOT EXISTS "
             "FOR (a:Allergy) REQUIRE a.cui IS UNIQUE"),
        ]
        
        for name, query in constraints:
            try:
                session.run(query)
                print(f"  [OK] Constraint: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  [SKIP] Constraint {name} already exists")
                else:
                    print(f"  [ERROR] Constraint {name}: {e}")

        # ============================================
        # 2. ADDITIONAL INDEXES
        #    For fast text-based lookups during entity matching.
        # ============================================
        
        indexes = [
            # Drug name lookup (case-insensitive search during entity extraction)
            ("drug_name_idx",
             "CREATE INDEX drug_name_idx IF NOT EXISTS "
             "FOR (d:Drug) ON (d.name_lower)"),
            
            # Condition name lookup
            ("condition_name_idx",
             "CREATE INDEX condition_name_idx IF NOT EXISTS "
             "FOR (c:Condition) ON (c.name_lower)"),
            
            # LabTest name lookup (for matching "eGFR", "potassium", etc.)
            ("labtest_name_idx",
             "CREATE INDEX labtest_name_idx IF NOT EXISTS "
             "FOR (lt:LabTest) ON (lt.name_lower)"),
            
            # SNOMED ID index (for cross-referencing)
            ("drug_snomed_idx",
             "CREATE INDEX drug_snomed_idx IF NOT EXISTS "
             "FOR (d:Drug) ON (d.snomed_id)"),
            
            ("condition_snomed_idx",
             "CREATE INDEX condition_snomed_idx IF NOT EXISTS "
             "FOR (c:Condition) ON (c.snomed_id)"),
        ]
        
        for name, query in indexes:
            try:
                session.run(query)
                print(f"  [OK] Index: {name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  [SKIP] Index {name} already exists")
                else:
                    print(f"  [ERROR] Index {name}: {e}")

        # ============================================
        # 3. VERIFY SCHEMA
        # ============================================
        result = session.run("SHOW CONSTRAINTS")
        constraints_list = [r["name"] for r in result]
        print(f"\n  Active constraints: {len(constraints_list)}")
        
        result = session.run("SHOW INDEXES")
        indexes_list = [r["name"] for r in result]
        print(f"  Active indexes: {len(indexes_list)}")


def verify_connection(driver):
    """Check that Neo4j is reachable."""
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            value = result.single()["test"]
            assert value == 1
            print("[OK] Connected to Neo4j successfully.\n")
            return True
    except Exception as e:
        print(f"[ERROR] Cannot connect to Neo4j: {e}")
        print("  Make sure Neo4j is running: docker-compose up -d")
        print("  Wait 30-60 seconds after starting for Neo4j to initialize.")
        return False


def print_relationship_guide():
    """Print the relationship types we'll create during data import."""
    print("\n" + "="*60)
    print("RELATIONSHIP TYPES (created during data import)")
    print("="*60)
    
    rels = [
        ("CONTRAINDICATED_WITH", "Drug → Condition",
         "threshold_lab, threshold_op, threshold_val, source",
         "Metformin -[:CONTRAINDICATED_WITH {threshold_lab:'eGFR', threshold_op:'<', threshold_val:30}]-> Renal Impairment"),
        
        ("INTERACTS_WITH", "Drug → Drug",
         "severity, mechanism, source",
         "Fluconazole -[:INTERACTS_WITH {severity:'major', mechanism:'CYP2C9'}]-> Warfarin"),
        
        ("IS_A", "Drug → DrugClass",
         "(none)",
         "Metformin -[:IS_A]-> Biguanides"),
        
        ("HAS_LAB_MARKER", "Condition → LabTest",
         "normal_low, normal_high, unit",
         "Renal Impairment -[:HAS_LAB_MARKER {normal_low:60, unit:'mL/min'}]-> eGFR"),
        
        ("REQUIRES_MONITORING", "Drug → LabTest",
         "frequency, threshold_low, threshold_high",
         "Warfarin -[:REQUIRES_MONITORING]-> INR"),
        
        ("ALLERGIC_CROSS_REACTS", "DrugClass → DrugClass",
         "probability",
         "Penicillins -[:ALLERGIC_CROSS_REACTS {probability:0.10}]-> Cephalosporins"),
    ]
    
    for rel_type, direction, properties, example in rels:
        print(f"\n  :{rel_type}")
        print(f"    Direction:  {direction}")
        print(f"    Properties: {properties}")
        print(f"    Example:    {example}")
    
    print("\n" + "="*60)
    print("CRITICAL FOR C-SCORE:")
    print("  CONTRAINDICATED_WITH edges carry NUMERICAL THRESHOLDS.")
    print("  This is the core mechanism that makes our system work.")
    print("  During C-score computation, we compare:")
    print("    patient_lab_value vs. edge.threshold_val")
    print("  If patient_val violates the threshold → PathViol increases.")
    print("="*60)


def main():
    print("="*60)
    print("GenMedAssist — Neo4j Schema Setup")
    print("="*60 + "\n")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    if not verify_connection(driver):
        sys.exit(1)
    
    print("Creating schema...")
    create_schema(driver)
    
    print_relationship_guide()
    
    driver.close()
    print("\n[DONE] Schema created. Next step: download UMLS/SNOMED data and run 02_import_umls.py")


if __name__ == "__main__":
    main()
