#!/usr/bin/env python3
"""
Create a shareable recruiter packet for one model-comparison bundle run.

It collects:
- Prompt that was fed to models (local/out/gemini_batch.txt)
- "Starter database" extracted from the prompt (candidate blocks)
- Top-50 outputs (cand_id, score, reason) for a fixed list of models

Output folder structure:
  local/out/recruiter_packets/<bundle_timestamp>/
    00_README.md
    01_prompt/
      prompt.txt
    02_starter_database/
      candidates_extracted.txt
      candidates_extracted.csv
    03_model_outputs/
      <model_slug>__top50.txt
      <model_slug>__top50.csv
      <model_slug>__raw.txt
      <model_slug>__json.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---- Adjust if your project layout differs ----
ROOT = Path(__file__).resolve().parents[1]
LOCAL_OUT = ROOT / "local" / "out"
DEFAULT_PROMPT_PATH = LOCAL_OUT / "gemini_batch.txt"

DEFAULT_BUNDLE_ROOT = LOCAL_OUT / "model_comparison_bundle"


MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash-lite",
]


def model_slug(model: str) -> str:
    s = model.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(p: Path, rows: List[dict], fieldnames: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def extract_top50(parsed: Optional[dict]) -> List[dict]:
    """
    Returns list of {cand_id, score, reason}.
    If JSON is malformed or missing, returns [].
    """
    if not isinstance(parsed, dict):
        return []
    top = parsed.get("top50")
    if not isinstance(top, list):
        return []
    out: List[dict] = []
    for item in top:
        if not isinstance(item, dict):
            continue
        cid = item.get("cand_id")
        score = item.get("score")
        reason = item.get("reason")
        if isinstance(cid, str) and cid.strip():
            out.append(
                {
                    "cand_id": cid.strip(),
                    "score": score if score is not None else "",
                    "reason": reason if isinstance(reason, str) else "",
                }
            )
    return out


def extract_candidates_from_prompt(prompt: str) -> List[dict]:
    """
    Best-effort extraction of candidate blocks from the prompt.
    This is intentionally flexible because prompt formats vary.

    Heuristics:
    - Find occurrences like "CAND_0000123" and capture a surrounding block.
    - Use next candidate marker or end as boundary.

    Output rows:
      cand_id, block_text
    """
    # Find all candidate IDs with their positions
    matches = list(re.finditer(r"\bCAND_\d{7}\b", prompt))
    if not matches:
        return []

    rows: List[dict] = []
    for idx, m in enumerate(matches):
        cid = m.group(0)
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(prompt)

        # Expand slightly backwards to include header lines if present
        back = max(0, start - 200)
        block = prompt[back:end].strip()

        # Keep it readable: cut extremely large blocks
        if len(block) > 8000:
            block = block[:8000] + "\n...[truncated]..."

        rows.append({"cand_id": cid, "block_text": block})

    # De-dup by cand_id, keep first
    seen = set()
    dedup: List[dict] = []
    for r in rows:
        if r["cand_id"] in seen:
            continue
        seen.add(r["cand_id"])
        dedup.append(r)
    return dedup


def resolve_bundle_paths(bundle_ts: str) -> Tuple[Path, Path, Path]:
    """
    Returns:
      bundle_dir, bundle_models_dir, bundle_reports_dir
    """
    bundle_dir = DEFAULT_BUNDLE_ROOT / bundle_ts
    if not bundle_dir.exists():
        raise SystemExit(f"Bundle folder not found: {bundle_dir}")
    models_dir = bundle_dir / "models"
    reports_dir = bundle_dir / "reports"
    return bundle_dir, models_dir, reports_dir


def find_model_json_in_bundle(models_dir: Path, model: str) -> Optional[Path]:
    """
    In the bundle we copied per model under models/<slug>/gemini_ranked_<slug>.json
    """
    slug = model_slug(model)
    d = models_dir / slug
    p = d / f"gemini_ranked_{slug}.json"
    if p.exists():
        return p
    return None


def find_model_raw_in_bundle(models_dir: Path, model: str) -> Optional[Path]:
    slug = model_slug(model)
    d = models_dir / slug
    p = d / f"gemini_ranked_{slug}_raw.txt"
    if p.exists():
        return p
    # sometimes raw naming differs; also try any *_raw.txt
    for cand in d.glob("*raw*.txt"):
        return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle-ts",
        type=str,
        required=True,
        help="Bundle timestamp folder name, e.g. 20260225_170758",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(LOCAL_OUT / "recruiter_packets"),
        help="Root folder where recruiter packets will be written",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help="Path to prompt file (gemini_batch.txt)",
    )
    args = ap.parse_args()

    bundle_dir, bundle_models_dir, bundle_reports_dir = resolve_bundle_paths(args.bundle_ts)

    out_root = Path(args.out_dir) / args.bundle_ts
    prompt_out_dir = out_root / "01_prompt"
    db_out_dir = out_root / "02_starter_database"
    model_out_dir = out_root / "03_model_outputs"

    out_root.mkdir(parents=True, exist_ok=True)

    # --- Prompt ---
    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    prompt_text = read_text(prompt_path)
    write_text(prompt_out_dir / "prompt.txt", prompt_text)

    # --- Starter database extraction (best effort) ---
    candidates = extract_candidates_from_prompt(prompt_text)

    if candidates:
        # text version
        blocks = []
        for r in candidates:
            blocks.append(f"===== {r['cand_id']} =====\n{r['block_text']}\n")
        write_text(db_out_dir / "candidates_extracted.txt", "\n".join(blocks))

        # csv version (block_text is big but workable)
        write_csv(db_out_dir / "candidates_extracted.csv", candidates, ["cand_id", "block_text"])
    else:
        # fallback: at least include the full prompt
        write_text(
            db_out_dir / "candidates_extracted.txt",
            "Could not automatically extract candidate blocks from the prompt.\n"
            "Use 01_prompt/prompt.txt as the starter database reference.\n",
        )
        write_csv(db_out_dir / "candidates_extracted.csv", [], ["cand_id", "block_text"])

    # --- Model outputs ---
    missing_models: List[str] = []
    for model in MODELS:
        slug = model_slug(model)
        json_path = find_model_json_in_bundle(bundle_models_dir, model)
        raw_path = find_model_raw_in_bundle(bundle_models_dir, model)

        parsed = read_json(json_path) if json_path else None
        top50 = extract_top50(parsed)

        if not top50:
            missing_models.append(model)

        # Write normalized top50 outputs
        write_csv(model_out_dir / f"{slug}__top50.csv", top50, ["cand_id", "score", "reason"])
        lines = []
        lines.append(f"MODEL: {model}")
        lines.append("")
        if top50:
            for i, r in enumerate(top50[:50], start=1):
                reason = (r.get("reason") or "").strip()
                reason = reason.replace("\r\n", "\n").replace("\r", "\n")
                lines.append(f"{i:02d}. {r.get('cand_id','')} | score={r.get('score','')}")
                if reason:
                    lines.append(f"    reason: {reason}")
                lines.append("")
        else:
            lines.append("No valid top50 parsed from JSON (file missing, malformed, or top50 absent).")
            if json_path:
                lines.append(f"JSON path: {json_path}")
            if raw_path:
                lines.append(f"RAW path:  {raw_path}")

        write_text(model_out_dir / f"{slug}__top50.txt", "\n".join(lines))

        # Also copy raw/json for transparency if present
        if raw_path and raw_path.exists():
            write_text(model_out_dir / f"{slug}__raw.txt", read_text(raw_path))
        if json_path and json_path.exists():
            write_json(model_out_dir / f"{slug}__json.json", parsed if parsed is not None else {})

    # --- README ---
    readme = []
    readme.append("# Recruiter review packet")
    readme.append("")
    readme.append(f"- Bundle source: {bundle_dir}")
    readme.append(f"- Created at: {datetime.now().isoformat(timespec='seconds')}")
    readme.append("")
    readme.append("## What to do")
    readme.append("")
    readme.append("1) Open `01_prompt/prompt.txt` to see the exact prompt sent to the model.")
    readme.append("2) Use `02_starter_database/candidates_extracted.txt` (or the prompt) to create your own Top-50 list.")
    readme.append("3) Compare your Top-50 with the four model Top-50 lists in `03_model_outputs/`.")
    readme.append("")
    readme.append("## Included models")
    for m in MODELS:
        readme.append(f"- {m}")
    readme.append("")
    if missing_models:
        readme.append("## Note: Some model outputs were missing or malformed")
        for m in missing_models:
            readme.append(f"- {m} (top50 could not be parsed)")
        readme.append("")
        readme.append("In that case, check the corresponding `__raw.txt` for what the model returned.")
    write_text(out_root / "00_README.md", "\n".join(readme))

    print("✅ Recruiter packet created:")
    print(out_root)


if __name__ == "__main__":
    main()