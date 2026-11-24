#!/usr/bin/env python3
"""
22_coverage_diagnostics.py — three follow-up checks on the coverage result
===========================================================================

CHECK 1  WBC itemid
    Methotrexate came back 14/5,496 evaluable (0.25%). Suspected cause:
    LAB_ITEMIDS uses 51300 (`WBC Count`) where the routine CBC value is
    51301 (`White Blood Cells`). Prints d_labitems labels and per-itemid
    volumes so you can confirm and fix.

CHECK 2  Categorical-rule evaluability convention
    `required_lab` returns on the first matching rule_id, and CAT-* sorts
    before MH-*/NUM-*. So metformin, warfarin and glipizide count as
    evaluable on their categorical rule without their numerical lab ever
    being required. This recomputes B and C under the STRICT convention
    (numerical lab required whenever a numerical rule exists) so you can
    report both and pick one knowingly.

CHECK 3  Heparin flushes
    Heparin is 16.7% of all in-scope volume. A 10-unit flush carries no
    thrombocytopenia risk, so if flushes dominate, metric A is inflated by
    a non-therapeutic entity. Prints the heparin drug-string distribution.

Needs 20_coverage_denominator.py in the same folder, and the cached
parquet files written by the full run.

Usage (Colab):
  python 22_coverage_diagnostics.py \
    --mimic-dir "/content/drive/MyDrive/NIPS_2026/Dataset/mimic-iv-hosp" \
    --bench-dir "/content/drive/MyDrive/NIPS_2026/MedConflict-Bench" \
    --cov-dir   "/content/drive/MyDrive/NIPS_2026/Rebuttal_Coverage/full"

  Add --checks 1     to run only check 1, etc. Default runs all three.
"""

import argparse
import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
_dep = _here / "20_coverage_denominator.py"
if not _dep.exists():
    sys.exit(f"FATAL: {_dep} not found (needed for the shared rule logic).")
_spec = importlib.util.spec_from_file_location("cov20", _dep)
cov20 = importlib.util.module_from_spec(_spec)
sys.modules["cov20"] = cov20
_spec.loader.exec_module(cov20)


# ============================================================================
def check1_wbc(mimic, sample_rows):
    print("=" * 74)
    print("CHECK 1 — WBC itemid")
    print("=" * 74)

    dli = pd.read_csv(mimic / "d_labitems.csv.gz", compression="gzip")
    hit = dli[dli["label"].astype(str).str.contains("WBC|White Blood",
                                                    case=False, na=False)]
    print("  d_labitems entries matching WBC / White Blood:")
    print(hit[["itemid", "label", "fluid", "category"]].to_string(index=False))
    print()

    cand = set(hit["itemid"].tolist()) | {51300, 51301}
    print(f"  counting volumes for {len(cand)} candidate itemids "
          f"(sampling first {sample_rows:,} lab rows)...")

    c = collections.Counter()
    scanned = 0
    for ch in pd.read_csv(mimic / "labevents.csv.gz", usecols=["itemid"],
                          chunksize=2_000_000, compression="gzip"):
        v = ch["itemid"].values
        c.update(v[np.isin(v, list(cand))].tolist())
        scanned += len(ch)
        if scanned >= sample_rows:
            break

    print(f"  scanned {scanned:,} lab rows")
    rows = []
    for iid in sorted(cand):
        lbl = dli.loc[dli["itemid"] == iid, "label"]
        rows.append((iid, lbl.iloc[0] if len(lbl) else "?", c.get(iid, 0)))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'itemid':>8}  {'count':>12}  label")
    for iid, lbl, n in rows:
        mark = "  <-- currently used" if iid == cov20.LAB_ITEMIDS.get("wbc") else ""
        print(f"  {iid:>8}  {n:>12,}  {lbl}{mark}")
    print()

    used = cov20.LAB_ITEMIDS.get("wbc")
    best = rows[0][0] if rows else None
    if best is not None and best != used:
        print(f"  >>> ACTION: WBC is currently {used}; highest-volume is {best}.")
        print(f"      Set LAB_ITEMIDS['wbc'] = {best} (or include both) and")
        print(f"      re-run stage 2 to refresh labs.parquet.")
    else:
        print("  >>> No change indicated.")
    print()


# ============================================================================
def check2_convention(mimic, bench, cov):
    print("=" * 74)
    print("CHECK 2 — categorical-rule evaluability convention")
    print("=" * 74)

    presc_p = cov / "presc_scoped.parquet"
    labs_p = cov / "labs.parquet"
    for p in (presc_p, labs_p):
        if not p.exists():
            print(f"  SKIP: {p} not found. Run the full coverage job first.")
            return

    cov20.BENCH = bench
    cov20.MIMIC = mimic
    rules, _ = cov20.stage0_rules(cov, verbose=False)
    rule_by_id = {r["rule_id"]: r for r in rules}

    presc = pd.read_parquet(presc_p)
    labs = pd.read_parquet(labs_p)
    labs = cov20.add_egfr(labs, cov)

    presc = presc.dropna(subset=["hadm_id", "starttime"]).copy()
    presc["hadm_id"] = presc["hadm_id"].astype("int64")
    labs = labs.dropna(subset=["hadm_id", "charttime", "valuenum"]).copy()
    labs["hadm_id"] = labs["hadm_id"].astype("int64")

    def lab_for(rid):
        r = rule_by_id.get(rid)
        if not r:
            return None
        raw = (r.get("lab_name") or "").strip()
        if not raw:
            return "__CATEGORICAL__"
        s = " ".join(cov20._punct.sub(" ", cov20._paren.sub(" ", raw.lower())).split())
        return "egfr" if s.startswith("egfr") else cov20.canon_lab(raw)

    def strict_lab(rule_ids):
        """Prefer a NUMERICAL requirement over a categorical one."""
        cat = False
        for rid in (rule_ids or "").split("|"):
            v = lab_for(rid)
            if v == "__CATEGORICAL__":
                cat = True
            elif v:
                return v
        return "__CATEGORICAL__" if cat else None

    scoped = presc[presc["ingredient"].astype(str) != ""].copy()
    scoped["req_lab"] = scoped["rule_ids"].astype(str).map(strict_lab)

    win = pd.Timedelta(hours=cov20.LAB_WINDOW_HOURS)

    def asof_hit(left, right, by_cols, ltime, rtime):
        if left.empty or right.empty:
            return np.zeros(len(left), dtype=bool)
        L = left.sort_values(ltime, kind="mergesort").copy()
        R = right.sort_values(rtime, kind="mergesort").copy()
        for cc in by_cols:
            if L[cc].dtype != R[cc].dtype:
                L[cc] = L[cc].astype(str)
                R[cc] = R[cc].astype(str)
        hit = np.zeros(len(L), dtype=bool)
        for direction in ("backward", "forward"):
            m = pd.merge_asof(L, R, left_on=ltime, right_on=rtime,
                              by=by_cols, direction=direction, tolerance=win)
            hit |= m[rtime].notna().to_numpy()
        return pd.Series(hit, index=L.index).reindex(left.index).to_numpy()

    def lenient_lab(rule_ids):
        for rid in (rule_ids or "").split("|"):
            v = lab_for(rid)
            if v:
                return v
        return None

    R = labs[["hadm_id", "lab", "charttime"]].rename(columns={"lab": "req_lab"})
    out = {}
    for tag, fn in (("LENIENT", lenient_lab), ("STRICT", strict_lab)):
        req = scoped["rule_ids"].astype(str).map(fn)
        cat_mask = req.eq("__CATEGORICAL__").to_numpy()
        need_mask = req.notna().to_numpy() & ~cat_mask
        ev = cat_mask.copy()
        if need_mask.any():
            L = scoped.loc[need_mask, ["hadm_id", "starttime"]].copy()
            L["req_lab"] = req[need_mask].values
            ev[need_mask] = asof_hit(L, R, ["hadm_id", "req_lab"],
                                     "starttime", "charttime")
        out[tag] = int(ev.sum())

    n_all, n_scope = len(presc), len(scoped)
    print(f"  orders analysed        : {n_all:,}")
    print(f"  in scope               : {n_scope:,}")
    print()
    for tag in ("LENIENT", "STRICT"):
        n_ev = out[tag]
        print(f"  {tag:<8} B = {n_ev / max(n_scope,1):.2%}   "
              f"C = {1.0 - n_ev / max(n_all,1):.2%}   evaluable={n_ev:,}")
    print(f"    LENIENT = first matching rule (CAT-* sorts first) — as reported")
    print(f"    STRICT  = numerical lab required whenever a numerical rule exists")
    print()
    print(f"  orders that flip to unevaluable under STRICT: "
          f"{out['LENIENT'] - out['STRICT']:,}")
    print("  >>> Report one convention and state which. The lenient reading is")
    print("      defensible (a categorical check IS evaluable); the strict one")
    print("      is the conservative number. Do not mix them.")
    print()


# ============================================================================
def check3_heparin(mimic):
    print("=" * 74)
    print("CHECK 3 — heparin flushes")
    print("=" * 74)

    c = collections.Counter()
    total = 0
    for ch in pd.read_csv(mimic / "prescriptions.csv.gz", usecols=["drug"],
                          chunksize=1_000_000, compression="gzip"):
        s = ch["drug"].dropna().astype(str)
        total += len(ch)
        c.update(s[s.str.contains("heparin", case=False, na=False)].tolist())

    n_hep = sum(c.values())
    print(f"  scanned {total:,} orders; {n_hep:,} contain 'heparin'")
    print()
    print(f"  {'count':>10}  drug string")
    for k, v in c.most_common(20):
        print(f"  {v:>10,}  {k}")
    print()

    flushish = sum(v for k, v in c.items()
                   if "flush" in k.lower() or "lock" in k.lower())
    print(f"  flush/lock entries: {flushish:,} "
          f"({flushish / max(n_hep,1):.1%} of heparin orders)")
    print(f"  therapeutic remainder: {n_hep - flushish:,}")
    print()
    print("  >>> If flushes are a large share, report metric A both with and")
    print("      without them. A 10-unit flush carries no thrombocytopenia")
    print("      risk, so counting it as in-scope inflates applicability.")
    print()


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--cov-dir", required=True,
                    help="folder with presc_scoped.parquet / labs.parquet "
                         "(usually Rebuttal_Coverage/full)")
    ap.add_argument("--checks", default="1,2,3")
    ap.add_argument("--sample-rows", type=int, default=30_000_000)
    args = ap.parse_args()

    mimic, bench, cov = (Path(args.mimic_dir), Path(args.bench_dir),
                         Path(args.cov_dir))
    want = {s.strip() for s in args.checks.split(",")}

    if "1" in want:
        check1_wbc(mimic, args.sample_rows)
    if "2" in want:
        check2_convention(mimic, bench, cov)
    if "3" in want:
        check3_heparin(mimic)


if __name__ == "__main__":
    main()
