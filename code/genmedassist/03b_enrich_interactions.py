#!/usr/bin/env python3
import os
"""
03b_enrich_interactions.py — Add Drug-Drug Interaction Edges to Neo4j
=====================================================================

Your KG has CONTRAINDICATED_WITH and REQUIRES_MONITORING edges from
03_enrich_thresholds.py, but is MISSING the INTERACTS_WITH edges
that 04_cscore.py queries for multi-hop detection.

This script adds curated DDI edges matching the MedConflict-Bench
Stratum 3 (multi-hop causal) rules.

Run on LAB MACHINE where Neo4j is running:
  cd C:\\Users\\AIML\\Sai_Teja _Honors\\genmedassist
  python 03b_enrich_interactions.py

Prerequisites:
  pip install neo4j
  Docker container genmedassist-neo4j must be running

Author: Ala Sai Teja, IIIT Sricity
"""

from neo4j import GraphDatabase

# Neo4j connection (from your docker-compose.yml)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "changeme")

# Drug-drug interactions matching MedConflict-Bench Stratum 3 rules
DDI_EDGES = [
    {
        "drug1": "Warfarin",
        "drug2": "Fluconazole",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Fluconazole inhibits CYP2C9, decreasing warfarin metabolism, leading to elevated INR and bleeding risk",
        "severity": "major",
        "affected_lab": "INR",
        "effect": "increased",
        "clinical_action": "monitor_closely",
        "evidence": "FDA label; multiple RCTs demonstrate 2-3x INR elevation with co-administration"
    },
    {
        "drug1": "Warfarin",
        "drug2": "Metronidazole",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Metronidazole inhibits CYP2C9, decreasing warfarin metabolism, leading to elevated INR",
        "severity": "major",
        "affected_lab": "INR",
        "effect": "increased",
        "clinical_action": "monitor_closely",
        "evidence": "FDA label; well-documented CYP2C9 inhibition"
    },
    {
        "drug1": "Warfarin",
        "drug2": "Amiodarone",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Amiodarone inhibits CYP2C9 and CYP3A4, decreasing warfarin metabolism, leading to elevated INR",
        "severity": "major",
        "affected_lab": "INR",
        "effect": "increased",
        "clinical_action": "dose_reduction",
        "evidence": "FDA label; warfarin dose typically reduced by 33-50% when amiodarone is added"
    },
    {
        "drug1": "Digoxin",
        "drug2": "Furosemide",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Furosemide causes renal potassium wasting, leading to hypokalemia, which increases digoxin toxicity",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "decreased",
        "clinical_action": "monitor_potassium",
        "evidence": "Well-established pharmacodynamic interaction; monitor K+ and supplement as needed"
    },
    {
        "drug1": "Digoxin",
        "drug2": "Bumetanide",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Bumetanide causes renal potassium wasting, leading to hypokalemia, which increases digoxin toxicity",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "decreased",
        "clinical_action": "monitor_potassium",
        "evidence": "Loop diuretic class effect; same mechanism as furosemide"
    },
    {
        "drug1": "Digoxin",
        "drug2": "Torsemide",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Torsemide causes renal potassium wasting, leading to hypokalemia, which increases digoxin toxicity",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "decreased",
        "clinical_action": "monitor_potassium",
        "evidence": "Loop diuretic class effect"
    },
    {
        "drug1": "Lisinopril",
        "drug2": "Spironolactone",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Both ACE inhibitors and potassium-sparing diuretics increase serum potassium, risk of life-threatening hyperkalemia",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "increased",
        "clinical_action": "monitor_potassium",
        "evidence": "RALES trial demonstrated risk; FDA black box consideration"
    },
    {
        "drug1": "Enalapril",
        "drug2": "Spironolactone",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Both ACE inhibitors and potassium-sparing diuretics increase serum potassium",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "increased",
        "clinical_action": "monitor_potassium",
        "evidence": "ACE-I class effect with K-sparing diuretics"
    },
    {
        "drug1": "Ramipril",
        "drug2": "Spironolactone",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Both ACE inhibitors and potassium-sparing diuretics increase serum potassium",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "increased",
        "clinical_action": "monitor_potassium",
        "evidence": "ACE-I class effect with K-sparing diuretics"
    },
    {
        "drug1": "Lisinopril",
        "drug2": "Eplerenone",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Both ACE inhibitors and mineralocorticoid receptor antagonists increase serum potassium",
        "severity": "major",
        "affected_lab": "Potassium",
        "effect": "increased",
        "clinical_action": "monitor_potassium",
        "evidence": "EMPHASIS-HF; monitor K+ closely"
    },
    {
        "drug1": "Methotrexate",
        "drug2": "Ibuprofen",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "NSAIDs reduce renal clearance of methotrexate, leading to accumulation and bone marrow suppression",
        "severity": "major",
        "affected_lab": "WBC",
        "effect": "decreased",
        "clinical_action": "avoid_combination",
        "evidence": "FDA label; multiple case reports of fatal pancytopenia"
    },
    {
        "drug1": "Methotrexate",
        "drug2": "Naproxen",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "NSAIDs reduce renal clearance of methotrexate",
        "severity": "major",
        "affected_lab": "WBC",
        "effect": "decreased",
        "clinical_action": "avoid_combination",
        "evidence": "NSAID class effect on MTX clearance"
    },
    {
        "drug1": "Methotrexate",
        "drug2": "Ketorolac",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "NSAIDs reduce renal clearance of methotrexate",
        "severity": "major",
        "affected_lab": "WBC",
        "effect": "decreased",
        "clinical_action": "contraindicated",
        "evidence": "Ketorolac specifically contraindicated with MTX per FDA label"
    },
    {
        "drug1": "Lithium",
        "drug2": "Ibuprofen",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "NSAIDs reduce lithium renal clearance via prostaglandin inhibition, leading to lithium accumulation and nephrotoxicity",
        "severity": "major",
        "affected_lab": "Creatinine",
        "effect": "increased",
        "clinical_action": "monitor_lithium_levels",
        "evidence": "FDA label; lithium levels can increase 15-50% with NSAID co-administration"
    },
    {
        "drug1": "Lithium",
        "drug2": "Naproxen",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "NSAIDs reduce lithium renal clearance",
        "severity": "major",
        "affected_lab": "Creatinine",
        "effect": "increased",
        "clinical_action": "monitor_lithium_levels",
        "evidence": "NSAID class effect on lithium clearance"
    },
    {
        "drug1": "Simvastatin",
        "drug2": "Clarithromycin",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Clarithromycin inhibits CYP3A4, causing statin accumulation, rhabdomyolysis risk",
        "severity": "major",
        "affected_lab": "ALT",
        "effect": "increased",
        "clinical_action": "contraindicated",
        "evidence": "FDA label; simvastatin contraindicated with strong CYP3A4 inhibitors"
    },
    {
        "drug1": "Atorvastatin",
        "drug2": "Clarithromycin",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Clarithromycin inhibits CYP3A4, causing statin accumulation",
        "severity": "major",
        "affected_lab": "ALT",
        "effect": "increased",
        "clinical_action": "avoid_combination",
        "evidence": "FDA label; use lowest atorvastatin dose or avoid"
    },
    {
        "drug1": "Simvastatin",
        "drug2": "Erythromycin",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Erythromycin inhibits CYP3A4, causing statin accumulation",
        "severity": "major",
        "affected_lab": "ALT",
        "effect": "increased",
        "clinical_action": "avoid_combination",
        "evidence": "Macrolide class CYP3A4 inhibition effect"
    },
    {
        "drug1": "Lovastatin",
        "drug2": "Clarithromycin",
        "rel_type": "INTERACTS_WITH",
        "mechanism": "Clarithromycin inhibits CYP3A4, causing statin accumulation",
        "severity": "major",
        "affected_lab": "ALT",
        "effect": "increased",
        "clinical_action": "contraindicated",
        "evidence": "FDA label; lovastatin contraindicated with strong CYP3A4 inhibitors"
    },
]


def main():
    print("=" * 60)
    print("Adding Drug-Drug Interaction Edges to Neo4j")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        # Check connection
        result = session.run("MATCH (n:Drug) RETURN count(n) AS cnt")
        drug_count = result.single()["cnt"]
        print(f"\nConnected to Neo4j: {drug_count:,} Drug nodes found")

        # Create INTERACTS_WITH edges
        created = 0
        skipped = 0

        for ddi in DDI_EDGES:
            # Check if edge already exists
            check = session.run("""
                MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
                WHERE toLower(d1.name) CONTAINS toLower($drug1)
                  AND toLower(d2.name) CONTAINS toLower($drug2)
                RETURN count(r) AS cnt
            """, drug1=ddi["drug1"], drug2=ddi["drug2"])

            if check.single()["cnt"] > 0:
                print(f"  ⏭ {ddi['drug1']} → {ddi['drug2']}: already exists")
                skipped += 1
                continue

            # Find the drug nodes (try exact match first, then contains)
            result = session.run("""
                MATCH (d1:Drug), (d2:Drug)
                WHERE toLower(d1.name) CONTAINS toLower($drug1)
                  AND toLower(d2.name) CONTAINS toLower($drug2)
                RETURN d1.name AS n1, d2.name AS n2
                LIMIT 1
            """, drug1=ddi["drug1"], drug2=ddi["drug2"])

            record = result.single()
            if record is None:
                # Drugs might not exist as separate nodes — create them
                print(f"  ⚠ Creating nodes for {ddi['drug1']} ↔ {ddi['drug2']}")
                session.run("""
                    MERGE (d1:Drug {name: $drug1})
                    MERGE (d2:Drug {name: $drug2})
                    CREATE (d1)-[r:INTERACTS_WITH {
                        mechanism: $mechanism,
                        severity: $severity,
                        affected_lab: $affected_lab,
                        effect: $effect,
                        clinical_action: $clinical_action,
                        evidence: $evidence,
                        source: 'curated_DDI_v2'
                    }]->(d2)
                    CREATE (d2)-[r2:INTERACTS_WITH {
                        mechanism: $mechanism,
                        severity: $severity,
                        affected_lab: $affected_lab,
                        effect: $effect,
                        clinical_action: $clinical_action,
                        evidence: $evidence,
                        source: 'curated_DDI_v2'
                    }]->(d1)
                """, **ddi)
            else:
                # Nodes exist, create edges both directions
                session.run("""
                    MATCH (d1:Drug), (d2:Drug)
                    WHERE toLower(d1.name) CONTAINS toLower($drug1)
                      AND toLower(d2.name) CONTAINS toLower($drug2)
                    WITH d1, d2 LIMIT 1
                    CREATE (d1)-[r:INTERACTS_WITH {
                        mechanism: $mechanism,
                        severity: $severity,
                        affected_lab: $affected_lab,
                        effect: $effect,
                        clinical_action: $clinical_action,
                        evidence: $evidence,
                        source: 'curated_DDI_v2'
                    }]->(d2)
                    CREATE (d2)-[r2:INTERACTS_WITH {
                        mechanism: $mechanism,
                        severity: $severity,
                        affected_lab: $affected_lab,
                        effect: $effect,
                        clinical_action: $clinical_action,
                        evidence: $evidence,
                        source: 'curated_DDI_v2'
                    }]->(d1)
                """, **ddi)

            print(f"  ✅ {ddi['drug1']} ↔ {ddi['drug2']} ({ddi['severity']}, {ddi['mechanism'][:50]}...)")
            created += 1

        # Verify
        result = session.run("MATCH ()-[r:INTERACTS_WITH]->() RETURN count(r) AS cnt")
        total_edges = result.single()["cnt"]

        print(f"\n{'='*60}")
        print(f"DONE: {created} new edges, {skipped} already existed")
        print(f"Total INTERACTS_WITH edges: {total_edges}")
        print(f"{'='*60}")

    driver.close()


if __name__ == "__main__":
    main()
