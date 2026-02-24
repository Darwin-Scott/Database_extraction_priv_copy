# gemini_rank.py
"""
Call Gemini to rank candidates (Stage 2: deep matching).

Inputs:
- out/gemini_batch.txt  (from prepare_gemini_batch.py)

Outputs:
- out/gemini_ranked_raw.txt   (raw model output)
- out/gemini_ranked.json      (parsed JSON if possible)

Usage:
  python gemini_rank.py
  python gemini_rank.py --model gemini-2.0-flash
  python gemini_rank.py --in out/gemini_batch.txt --out out/gemini_ranked.json

Requirements:
  pip install google-genai

API Key:
  Set env var GEMINI_API_KEY (or GOOGLE_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


DEFAULT_IN = Path("out") / "gemini_batch.txt"
DEFAULT_OUT_JSON = Path("out") / "gemini_ranked.json"
DEFAULT_OUT_RAW = Path("out") / "gemini_ranked_raw.txt"

# Choose a sane default; you can override with --model
DEFAULT_MODEL = "gemini-2.0-flash"


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON strictly; if it fails, attempt to extract the first {...} block.
    """
    text = text.strip()

    # First try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find the first JSON object in the text
    # (If the model accidentally adds some extra characters)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="Path to gemini_batch.txt")
    parser.add_argument("--out", dest="out_json", default=str(DEFAULT_OUT_JSON), help="Path to output JSON file")
    parser.add_argument("--raw", dest="out_raw", default=str(DEFAULT_OUT_RAW), help="Path to output raw text file")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL, help="Gemini model name (e.g., gemini-2.0-flash)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_json = Path(args.out_json)
    out_raw = Path(args.out_raw)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}. Run prepare_gemini_batch.py first.")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    prompt = in_path.read_text(encoding="utf-8")

    # Create client. It will pick up GEMINI_API_KEY or GOOGLE_API_KEY from env automatically.
    # (You can also pass api_key=... explicitly.)
    client = genai.Client()

    # JSON mode: request JSON output
    config = types.GenerateContentConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
    )

    print("🚀 Calling Gemini...")
    print(f"   model: {args.model}")
    print(f"   input chars: {len(prompt):,}")

    try:
        resp = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        print("\n❌ Gemini request failed.")
        print("Most common causes:")
        print("- Prompt too large for model/request limits")
        print("- API key missing/invalid (set GEMINI_API_KEY)")
        print("- Network / transient API errors")
        print("\nRaw error:")
        raise

    # The SDK returns a response object; the text is usually in resp.text
    raw_text = (resp.text or "").strip()

    # Save raw output no matter what
    out_raw.write_text(raw_text, encoding="utf-8")
    print(f"✅ Wrote raw output: {out_raw}")

    parsed = extract_json_object(raw_text)
    if not parsed:
        print("⚠️ Could not parse JSON from model output.")
        print("Open the raw output file and inspect:")
        print(f"  {out_raw}")
        return

    out_json.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Parsed JSON saved: {out_json}")

    # Quick sanity checks
    top50 = parsed.get("top50")
    if isinstance(top50, list):
        print(f"✅ top50 length: {len(top50)}")
        if top50[:3]:
            print("Top 3 preview:")
            for i, item in enumerate(top50[:3], start=1):
                cid = item.get("cand_id")
                score = item.get("score")
                reason = item.get("reason", "")
                reason_short = (reason[:120] + "…") if isinstance(reason, str) and len(reason) > 120 else reason
                print(f"  {i}. {cid} score={score} reason={reason_short}")
    else:
        print("⚠️ JSON parsed, but missing expected key 'top50' as a list.")
        print("Check output JSON structure.")

    # Clean up HTTP resources
    try:
        client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()