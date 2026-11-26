#!/usr/bin/env python3
"""
00_inspect_schema.py  —  RUN THIS FIRST

Discovers the structure of your trajectory / benchmark JSONL files so that
01_tier0_analysis.py can be configured correctly.

Usage (from anywhere — paths resolve to the project root):
    python scripts/00_inspect_schema.py

Writes:
    out/schema_report.txt   <- paste this into chat if fields don't auto-resolve
"""

import argparse
import json
import os
import glob
import gzip
from pathlib import Path
from collections import Counter, defaultdict

# Project root = parent of scripts/ — so this works from any CWD
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "out"

MAX_SAMPLE = 400          # records to sample for key frequency
PREVIEW_CHARS = 90


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_jsonl(path, limit=None):
    """Yield parsed records from a .jsonl (or .json containing a list)."""
    with open_maybe_gzip(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":                       # whole-file JSON array
            try:
                data = json.load(f)
            except Exception as e:
                print(f"    !! could not parse as JSON array: {e}")
                return
            for i, rec in enumerate(data):
                if limit and i >= limit:
                    return
                yield rec
            return
        for i, line in enumerate(f):           # JSONL
            if limit and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def describe(value, depth=0, maxdepth=3):
    """One-line type + sample description of a value."""
    pad = "  " * depth
    if isinstance(value, dict):
        if depth >= maxdepth:
            return f"dict({len(value)} keys)"
        inner = []
        for k, v in list(value.items())[:12]:
            inner.append(f"{pad}    {k}: {describe(v, depth + 1, maxdepth)}")
        return "dict {\n" + "\n".join(inner) + f"\n{pad}  }}"
    if isinstance(value, list):
        if not value:
            return "list(empty)"
        if depth >= maxdepth:
            return f"list({len(value)})"
        return f"list({len(value)}) of " + describe(value[0], depth + 1, maxdepth)
    if isinstance(value, bool):
        return f"bool = {value}"
    if isinstance(value, (int, float)):
        return f"{type(value).__name__} = {value}"
    if value is None:
        return "None"
    s = str(value)
    if len(s) > PREVIEW_CHARS:
        s = s[:PREVIEW_CHARS] + "..."
    return f"str = {s!r}"


def flatten_keys(obj, prefix=""):
    """All dotted key paths in a nested dict."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.append(p)
            out.extend(flatten_keys(v, p))
    elif isinstance(obj, list) and obj:
        out.extend(flatten_keys(obj[0], prefix + "[]"))
    return out


def value_profile(records, keypath, top=12):
    """Distribution of values at a dotted keypath."""
    vals = []
    for r in records:
        v = r
        ok = True
        for part in keypath.split("."):
            if isinstance(v, dict) and part in v:
                v = v[part]
            else:
                ok = False
                break
        if ok and not isinstance(v, (dict, list)):
            vals.append(v)
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(top), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA),
                    help="folder containing your jsonl/json files")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "schema_report.txt")

    patterns = ["*.jsonl", "*.jsonl.gz", "*.json", "*.json.gz"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(args.data_dir, "**", p), recursive=True))
    files = sorted(set(files))

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("TIER 0 — SCHEMA INSPECTION REPORT")
    emit("=" * 78)
    emit(f"data dir : {os.path.abspath(args.data_dir)}")
    emit(f"files    : {len(files)}")
    emit()

    if not files:
        emit("!! No .jsonl/.json files found.")
        emit("   Put your trajectory + benchmark files under --data-dir and re-run.")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        return

    for path in files:
        size_mb = os.path.getsize(path) / 1e6
        emit("-" * 78)
        emit(f"FILE: {os.path.relpath(path, args.data_dir)}   ({size_mb:.2f} MB)")
        emit("-" * 78)

        sample = list(iter_jsonl(path, limit=MAX_SAMPLE))
        if not sample:
            emit("  (no parseable records)")
            emit()
            continue

        total = sum(1 for _ in iter_jsonl(path))
        emit(f"  records: {total}   (sampled {len(sample)})")
        emit()

        emit("  FIRST RECORD STRUCTURE:")
        emit("  " + describe(sample[0], depth=1))
        emit()

        keycount = Counter()
        for r in sample:
            for k in set(flatten_keys(r)):
                keycount[k] += 1
        emit("  KEY PATHS (name — present in N/%d sampled):" % len(sample))
        for k, n in sorted(keycount.items()):
            emit(f"    {k:<50} {n}")
        emit()

        # Profile any key that looks decision- / label- / score- / stratum-like
        interesting = [k for k in keycount
                       if any(t in k.lower() for t in
                              ("decis", "label", "strat", "score", "conflict",
                               "type", "status", "outcome", "pred", "gt",
                               "ground", "cui", "egfr", "unit", "rule", "path"))]
        if interesting:
            emit("  VALUE PROFILES (candidate fields):")
            for k in sorted(interesting):
                prof = value_profile(sample, k)
                if prof:
                    top, n = prof
                    shown = ", ".join(f"{v!r}×{c}" for v, c in top)
                    emit(f"    {k}  (n={n})")
                    emit(f"        {shown}")
            emit()

    emit("=" * 78)
    emit("NEXT STEP")
    emit("=" * 78)
    emit("1. Open out/schema_report.txt")
    emit("2. Open scripts/01_tier0_analysis.py and edit the FIELD_MAP block")
    emit("   so each logical field points at the real key path shown above.")
    emit("3. Run: python scripts/01_tier0_analysis.py")
    emit()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[saved] {report_path}")


if __name__ == "__main__":
    main()
