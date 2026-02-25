# scripts/compare_gemini_models.py
"""
Run gemini_rank.py across multiple Gemini models and store outputs separately.

Outputs:
- local/out/model_comparison/<model_slug>/gemini_ranked.json
- local/out/model_comparison/<model_slug>/gemini_ranked_raw.txt
- local/out/model_comparison/<model_slug>/usage.json (if gemini_rank.py supports --usage-out)
- local/out/model_comparison/cost_report.csv
- local/out/model_comparison/cost_report.txt

Run:
  python scripts/compare_gemini_models.py
  python scripts/compare_gemini_models.py --models gemini-2.5-flash-lite,gemini-2.5-flash
  python scripts/compare_gemini_models.py --max-models 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOCAL = ROOT / "local"
OUT = LOCAL / "out"
IN_PROMPT = OUT / "gemini_batch.txt"

BASE_DIR = OUT / "model_comparison"

DEFAULT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-pro",
]


# ---- Pricing (USD per 1M tokens) from your pasted table ----
# We assume prompts <= 200k tokens for 3.x / 2.5-pro etc for this pipeline.
PRICES_USD_PER_1M = {
    # Gemini 3.1 Pro Preview + customtools
    "gemini-3.1-pro-preview": {"in": 2.00, "out": 12.00},
    "gemini-3.1-pro-preview-customtools": {"in": 2.00, "out": 12.00},
    # Gemini 3 Pro Preview
    "gemini-3-pro-preview": {"in": 2.00, "out": 12.00},
    # Gemini 3 Flash Preview
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},
    # Gemini 2.5 Pro (and 2.5-pro)
    "gemini-2.5-pro": {"in": 1.25, "out": 10.00},
    "gemini-2.5-pro-preview": {"in": 1.25, "out": 10.00},  # just in case
    # Gemini 2.5 Flash
    "gemini-2.5-flash": {"in": 0.30, "out": 2.50},
    # Gemini 2.5 Flash-Lite (+ preview)
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-flash-lite-preview-09-2025": {"in": 0.10, "out": 0.40},
    # Gemini 2.0 Flash
    "gemini-2.0-flash": {"in": 0.10, "out": 0.40},
    # Gemini 2.0 Flash-Lite
    "gemini-2.0-flash-lite": {"in": 0.075, "out": 0.30},
    # Gemini 2.0 Pro (not in your pasted pricing section explicitly; we leave unknown)
    "gemini-2.0-pro": {"in": None, "out": None},
}


def model_slug(model: str) -> str:
    # filesystem-safe name
    s = model.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def approx_tokens_from_chars(n_chars: int) -> int:
    return max(1, n_chars // 4)


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


def read_usage_json(path: Path) -> Usage:
    if not path.exists():
        return Usage()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        # Handle both our suggested schema and possible dict schemas
        pt = obj.get("prompt_token_count") if isinstance(obj, dict) else None
        ot = obj.get("candidates_token_count") if isinstance(obj, dict) else None
        tt = obj.get("total_token_count") if isinstance(obj, dict) else None
        # fallback keys
        if pt is None and isinstance(obj, dict):
            pt = obj.get("prompt_tokens") or obj.get("input_tokens")
        if ot is None and isinstance(obj, dict):
            ot = obj.get("output_tokens") or obj.get("completion_tokens")
        if tt is None and isinstance(obj, dict):
            tt = obj.get("total_tokens")
        return Usage(
            prompt_tokens=int(pt) if pt is not None else None,
            output_tokens=int(ot) if ot is not None else None,
            total_tokens=int(tt) if tt is not None else None,
        )
    except Exception:
        return Usage()


def calc_cost_usd(model: str, prompt_tokens: Optional[int], output_tokens: Optional[int], fallback_in_tokens: int, fallback_out_tokens: int) -> tuple[Optional[float], str]:
    pricing = PRICES_USD_PER_1M.get(model, None)
    if not pricing or pricing.get("in") is None or pricing.get("out") is None:
        return None, "unknown_pricing"

    in_rate = pricing["in"]
    out_rate = pricing["out"]

    pt = prompt_tokens if prompt_tokens is not None else fallback_in_tokens
    ot = output_tokens if output_tokens is not None else fallback_out_tokens

    cost = (pt / 1_000_000) * in_rate + (ot / 1_000_000) * out_rate
    basis = "usage_tokens" if (prompt_tokens is not None or output_tokens is not None) else "estimated_tokens"
    return float(cost), basis


def run_subprocess(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=300,  # 5 minutes max
        )
        out = (p.stdout or "") + ("\n" if p.stdout and p.stderr else "") + (p.stderr or "")
        return p.returncode, out
    except subprocess.TimeoutExpired as e:
        return 124, f"Timeout after 300 seconds\n{e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="", help="Comma-separated models (overrides default list).")
    ap.add_argument("--max-models", type=int, default=0, help="If >0, only run first N models.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_output_tokens", type=int, default=8192)
    ap.add_argument("--skip-existing", action="store_true", help="Skip a model if its JSON output already exists.")
    args = ap.parse_args()

    if not IN_PROMPT.exists():
        raise SystemExit(f"Missing input prompt: {IN_PROMPT}. Run prepare_gemini_batch.py first.")

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else list(DEFAULT_MODELS)
    if args.max_models and args.max_models > 0:
        models = models[: args.max_models]

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    prompt_text = IN_PROMPT.read_text(encoding="utf-8")
    in_chars = len(prompt_text)
    est_in_tokens = approx_tokens_from_chars(in_chars)

    py = sys.executable

    rows: list[dict[str, Any]] = []
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"== Model comparison starting: {started} ==")
    print(f"Prompt chars: {in_chars:,} | estimated input tokens ~{est_in_tokens:,}")
    print(f"Models: {len(models)}")

    for m in models:
        slug = model_slug(m)
        out_dir = BASE_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        out_json = out_dir / f"gemini_ranked_{slug}.json"
        out_raw = out_dir / f"gemini_ranked_{slug}_raw.txt"
        out_usage = out_dir / f"usage_{slug}.json"

        if args.skip_existing and out_json.exists():
            print(f"[skip] {m} because {out_json.name} exists")
            usage = read_usage_json(out_usage)
            cost, basis = calc_cost_usd(m, usage.prompt_tokens, usage.output_tokens, est_in_tokens, 2000)
            rows.append(
                {
                    "model": m,
                    "status": "skipped_existing",
                    "json": str(out_json),
                    "raw": str(out_raw),
                    "usage": str(out_usage),
                    "prompt_tokens": usage.prompt_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "est_prompt_tokens": est_in_tokens,
                    "est_output_tokens": 2000,
                    "cost_usd": cost,
                    "cost_basis": basis,
                }
            )
            continue

        cmd = [
            py,
            str(SCRIPTS / "gemini_rank.py"),
            "--model",
            m,
            "--in",
            str(IN_PROMPT),
            "--out",
            str(out_json),
            "--raw",
            str(out_raw),
            "--temperature",
            str(args.temperature),
            "--max_output_tokens",
            str(args.max_output_tokens),
        ]

        # If your gemini_rank.py includes the patch, this will work; otherwise it will error for unknown arg.
        # So we try with --usage-out first; if it fails due to arg parsing, rerun without it.
        cmd_with_usage = cmd + ["--usage-out", str(out_usage)]

        print(f"\n== Running: {m} ==")
        rc, out = run_subprocess(cmd_with_usage)

        if rc != 0 and ("--usage-out" in out or "unrecognized arguments: --usage-out" in out):
            print("Note: gemini_rank.py does not support --usage-out yet; rerunning without usage capture.")
            rc, out = run_subprocess(cmd)

        status = "ok" if rc == 0 else "error"

        # Try to read usage (may not exist)
        usage = read_usage_json(out_usage)

        # If no usage, we still estimate output tokens roughly from raw length
        fallback_out_tokens = approx_tokens_from_chars(out_raw.read_text(encoding="utf-8").__len__()) if out_raw.exists() else 2000

        cost, basis = calc_cost_usd(
            m,
            usage.prompt_tokens,
            usage.output_tokens,
            est_in_tokens,
            fallback_out_tokens,
        )

        # Save run log per model
        (out_dir / "run_log.txt").write_text(out, encoding="utf-8")

        rows.append(
            {
                "model": m,
                "status": status,
                "json": str(out_json),
                "raw": str(out_raw),
                "usage": str(out_usage),
                "prompt_tokens": usage.prompt_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "est_prompt_tokens": est_in_tokens,
                "est_output_tokens": fallback_out_tokens,
                "cost_usd": cost,
                "cost_basis": basis,
            }
        )

        print(f"Status: {status}")
        if cost is not None:
            print(f"Estimated cost (USD): {cost:.6f} ({basis})")
        else:
            print("Cost: unknown (no pricing mapping for this model)")

    # Write reports
    report_csv = BASE_DIR / "cost_report.csv"
    report_txt = BASE_DIR / "cost_report.txt"

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "status",
                "prompt_tokens",
                "output_tokens",
                "total_tokens",
                "est_prompt_tokens",
                "est_output_tokens",
                "cost_usd",
                "cost_basis",
                "json",
                "raw",
                "usage",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = []
    lines.append(f"Model comparison run: {started}")
    lines.append(f"Prompt chars: {in_chars:,} | est input tokens ~{est_in_tokens:,}")
    lines.append("")
    for r in rows:
        cost = r["cost_usd"]
        cost_s = f"{cost:.6f} USD" if isinstance(cost, (int, float)) else "unknown"
        lines.append(f"- {r['model']}: {r['status']}, cost={cost_s} ({r['cost_basis']})")
        lines.append(f"  json: {r['json']}")
        lines.append(f"  raw:  {r['raw']}")
        if r["prompt_tokens"] is not None or r["output_tokens"] is not None:
            lines.append(f"  tokens: prompt={r['prompt_tokens']} output={r['output_tokens']} total={r['total_tokens']}")
        else:
            lines.append(f"  tokens (est): prompt~{r['est_prompt_tokens']} output~{r['est_output_tokens']}")
        lines.append("")

    report_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n== Done ==")
    print(f"Wrote: {report_csv}")
    print(f"Wrote: {report_txt}")
    print(f"Outputs in: {BASE_DIR}")


if __name__ == "__main__":
    main()