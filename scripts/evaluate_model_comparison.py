#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def model_slug(model: str) -> str:
    s = model.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "" or x == "None":
            return None
        return int(float(x))
    except Exception:
        return None


def extract_top_ids(parsed: Optional[dict], key: str = "top50") -> List[str]:
    if not isinstance(parsed, dict):
        return []
    arr = parsed.get(key)
    if not isinstance(arr, list):
        return []
    out: List[str] = []
    for item in arr:
        if isinstance(item, dict):
            cid = item.get("cand_id")
            if isinstance(cid, str) and cid.strip():
                out.append(cid.strip())
    # de-dup, keep order
    seen = set()
    dedup = []
    for cid in out:
        if cid not in seen:
            seen.add(cid)
            dedup.append(cid)
    return dedup


def jaccard(a: Iterable[str], b: Iterable[str]) -> Optional[float]:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return None
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_at_k(a: List[str], b: List[str], k: int) -> Optional[float]:
    if k <= 0:
        return None
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / float(k)


def spearman(a: List[str], b: List[str]) -> Optional[float]:
    """
    Proper Spearman over the intersection:
    - create paired ranks for items in intersection
    - compute Pearson correlation of rank vectors
    This keeps result in [-1,1].
    """
    ra = {cid: i + 1 for i, cid in enumerate(a)}
    rb = {cid: i + 1 for i, cid in enumerate(b)}
    common = [cid for cid in ra if cid in rb]
    n = len(common)
    if n < 2:
        return None

    xs = [float(ra[cid]) for cid in common]
    ys = [float(rb[cid]) for cid in common]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    deny = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


@dataclass
class Run:
    model: str
    slug: str
    status: str
    cost_usd: Optional[float]
    cost_basis: str
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    dir_path: Path
    json_path: Path
    raw_path: Path
    usage_path: Path
    top_ids: List[str]


def read_cost_report(cost_csv: Path) -> List[dict]:
    if not cost_csv.exists():
        raise SystemExit(f"Missing cost report: {cost_csv}")
    with cost_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_runs(base_dir: Path) -> List[Run]:
    rows = read_cost_report(base_dir / "cost_report.csv")

    runs: List[Run] = []
    for r in rows:
        model = (r.get("model") or "").strip()
        if not model:
            continue
        slug = model_slug(model)
        d = base_dir / slug

        json_path = Path((r.get("json") or "").strip()) if (r.get("json") or "").strip() else (d / f"gemini_ranked_{slug}.json")
        raw_path = Path((r.get("raw") or "").strip()) if (r.get("raw") or "").strip() else (d / f"gemini_ranked_{slug}_raw.txt")
        usage_path = Path((r.get("usage") or "").strip()) if (r.get("usage") or "").strip() else (d / f"usage_{slug}.json")

        # Make them relative if they are absolute but file doesn't exist (bundle portability)
        # Prefer the standard locations under base_dir.
        if not json_path.exists():
            json_path = d / f"gemini_ranked_{slug}.json"
        if not raw_path.exists():
            raw_path = d / f"gemini_ranked_{slug}_raw.txt"
        if not usage_path.exists():
            usage_path = d / f"usage_{slug}.json"

        parsed = read_json(json_path) if json_path.exists() else None
        top_ids = extract_top_ids(parsed, "top50")

        runs.append(
            Run(
                model=model,
                slug=slug,
                status=(r.get("status") or "unknown").strip(),
                cost_usd=safe_float(r.get("cost_usd")),
                cost_basis=(r.get("cost_basis") or "unknown").strip(),
                prompt_tokens=safe_int(r.get("prompt_tokens")),
                output_tokens=safe_int(r.get("output_tokens")),
                total_tokens=safe_int(r.get("total_tokens")),
                dir_path=d,
                json_path=json_path,
                raw_path=raw_path,
                usage_path=usage_path,
                top_ids=top_ids,
            )
        )

    # de-dup by model (your cost_report can contain duplicates if you ran twice)
    best: Dict[str, Run] = {}
    for run in runs:
        prev = best.get(run.model)
        if prev is None:
            best[run.model] = run
            continue
        # Prefer ok over error, and prefer having parsed ids
        score_prev = (1 if prev.status == "ok" else 0, len(prev.top_ids))
        score_new = (1 if run.status == "ok" else 0, len(run.top_ids))
        if score_new > score_prev:
            best[run.model] = run

    return list(best.values())


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="local/out/model_comparison")
    ap.add_argument("--bundle-root", type=str, default="local/out/model_comparison_bundle")
    ap.add_argument("--k", type=str, default="10,20,50")
    args = ap.parse_args()

    base_dir = Path(args.base)
    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    ks = sorted({int(x) for x in args.k.split(",") if x.strip().isdigit() and int(x) > 0}) or [10, 20, 50]
    k_consensus = max(ks)

    runs = load_runs(base_dir)
    if not runs:
        raise SystemExit("No runs found. Is cost_report.csv present?")

    # Pairwise
    pair_rows: List[dict] = []
    overlap_rows: List[dict] = []

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i], runs[j]
            jac = jaccard(a.top_ids, b.top_ids)
            spr = spearman(a.top_ids, b.top_ids)
            pair_rows.append(
                {
                    "model_a": a.model,
                    "model_b": b.model,
                    "jaccard_top50": f"{jac:.6f}" if jac is not None else "",
                    "spearman_r_intersection": f"{spr:.6f}" if spr is not None else "",
                    "common_count": str(len(set(a.top_ids) & set(b.top_ids))),
                    "a_top50_count": str(len(a.top_ids)),
                    "b_top50_count": str(len(b.top_ids)),
                }
            )
            ov = {"model_a": a.model, "model_b": b.model}
            for k in ks:
                v = overlap_at_k(a.top_ids, b.top_ids, k)
                ov[f"overlap@{k}"] = f"{v:.6f}" if v is not None else ""
            overlap_rows.append(ov)

    # Consensus similarity (avg overlap@k_consensus vs others)
    cons_sum: Dict[str, float] = {r.model: 0.0 for r in runs}
    cons_n: Dict[str, int] = {r.model: 0 for r in runs}
    for row in overlap_rows:
        a = row["model_a"]
        b = row["model_b"]
        v = safe_float(row.get(f"overlap@{k_consensus}"))
        if v is None:
            continue
        cons_sum[a] += v
        cons_sum[b] += v
        cons_n[a] += 1
        cons_n[b] += 1

    model_rows: List[dict] = []
    for r in runs:
        avg_sim = (cons_sum[r.model] / cons_n[r.model]) if cons_n[r.model] else None
        ce = (avg_sim / r.cost_usd) if (avg_sim is not None and r.cost_usd and r.cost_usd > 0) else None
        model_rows.append(
            {
                "model": r.model,
                "slug": r.slug,
                "status": r.status,
                "cost_usd": f"{r.cost_usd:.6f}" if r.cost_usd is not None else "",
                "cost_basis": r.cost_basis,
                "prompt_tokens": r.prompt_tokens or "",
                "output_tokens": r.output_tokens or "",
                "total_tokens": r.total_tokens or "",
                f"avg_overlap@{k_consensus}_vs_others": f"{avg_sim:.6f}" if avg_sim is not None else "",
                "cost_effectiveness_proxy": f"{ce:.6f}" if ce is not None else "",
                "top50_count": len(r.top_ids),
            }
        )

    # Bundle
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = Path(args.bundle_root) / ts
    reports_dir = bundle_root / "reports"
    models_dir = bundle_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Copy cost report
    shutil.copy2(base_dir / "cost_report.csv", reports_dir / "cost_report.csv")

    # Copy per-model artifacts
    for r in runs:
        d = models_dir / r.slug
        d.mkdir(parents=True, exist_ok=True)
        if r.json_path.exists():
            shutil.copy2(r.json_path, d / r.json_path.name)
        if r.raw_path.exists():
            shutil.copy2(r.raw_path, d / r.raw_path.name)
        if r.usage_path.exists():
            shutil.copy2(r.usage_path, d / r.usage_path.name)
        run_log = r.dir_path / "run_log.txt"
        if run_log.exists():
            shutil.copy2(run_log, d / "run_log.txt")

    # Write reports
    write_csv(
        reports_dir / "pairwise_similarity.csv",
        ["model_a", "model_b", "jaccard_top50", "spearman_r_intersection", "common_count", "a_top50_count", "b_top50_count"],
        pair_rows,
    )
    write_csv(
        reports_dir / "overlap_at_k.csv",
        ["model_a", "model_b"] + [f"overlap@{k}" for k in ks],
        overlap_rows,
    )
    write_csv(
        reports_dir / "model_table.csv",
        [
            "model", "slug", "status", "cost_usd", "cost_basis",
            "prompt_tokens", "output_tokens", "total_tokens",
            f"avg_overlap@{max(ks)}_vs_others", "cost_effectiveness_proxy", "top50_count"
        ],
        model_rows,
    )

    # Summary
    (reports_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Model comparison summary",
                "",
                f"- Created: {datetime.now().isoformat(timespec='seconds')}",
                f"- Models: {len(runs)}",
                f"- Similarity basis: cand_id overlap in top50",
                f"- Overlap Ks: {', '.join(map(str, ks))}",
                "",
                "Open `reports/model_table.csv` and sort by `cost_usd` and `avg_overlap@...`.",
            ]
        ),
        encoding="utf-8",
    )

    print("✅ Bundle created:", bundle_root)


if __name__ == "__main__":
    main()