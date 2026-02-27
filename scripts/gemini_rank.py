# scripts/gemini_rank.py
"""
Call Gemini to rank candidates (Stage 2: deep matching).

Inputs:
- local/out/gemini_batch.txt  (from prepare_gemini_batch.py)

Outputs:
- local/out/gemini_ranked_raw.txt   (raw model output)
- local/out/gemini_ranked.json      (parsed JSON if possible)

Usage:
  python scripts/gemini_rank.py
  python scripts/gemini_rank.py --model gemini-2.0-flash
  python scripts/gemini_rank.py --skip-if-exists
  python scripts/gemini_rank.py --dry-run
  python scripts/gemini_rank.py --ping
  python scripts/gemini_rank.py --mock local/out/mock_gemini_ranked.json

Optional (new):
  python scripts/gemini_rank.py --rank-n 100
  python scripts/gemini_rank.py --rank-key top100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types

from dbx.paths import OUT

DEFAULT_IN = OUT / "gemini_batch.txt"
DEFAULT_OUT_JSON = OUT / "gemini_ranked.json"
DEFAULT_OUT_RAW = OUT / "gemini_ranked_raw.txt"

DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-09-2025"  # cost-effective default


def approx_tokens_from_chars(n_chars: int) -> int:
    return max(1, n_chars // 4)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t.strip())

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    candidates: list[str] = []
    stack = 0
    start = None
    for i, ch in enumerate(t):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    candidates.append(t[start : i + 1])
                    start = None

    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def nice_quota_hint(err_text: str) -> str:
    if "free_tier" in err_text.lower():
        return (
            "\nLikely cause:\n"
            "- Your API key/project is currently treated as FREE TIER, and free-tier quota is 0.\n"
            "- Fix: ensure the key belongs to a Google Cloud project with BILLING ENABLED + Gemini API enabled.\n"
            "- Create a new API key in the billed project and use that.\n"
        )
    if "quota" in err_text.lower() or "RESOURCE_EXHAUSTED" in err_text:
        return (
            "\nLikely cause:\n"
            "- Quota / rate limit hit (requests per minute/day or input token per minute).\n"
            "- Try again later or reduce number of candidates / shorten prompt.\n"
        )
    return ""


def do_ping(client: genai.Client, model: str) -> None:
    print("🏓 Ping: sending tiny request to validate billing/quota...")
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=32,
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(
        model=model,
        contents='{"ping":true,"msg":"reply with {\\"ok\\":true}"}',
        config=cfg,
    )
    print("✅ Ping response text:", (resp.text or "").strip())


def infer_rank_key(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[list]]:
    """
    Find a key like 'top50', 'top100', ... whose value is a list.
    Prefer the largest topN list if multiple exist.
    """
    best_key = None
    best_list = None
    best_n = -1
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        m = re.fullmatch(r"top(\d+)", k.strip())
        if not m:
            continue
        if not isinstance(v, list):
            continue
        n = int(m.group(1))
        if n > best_n:
            best_n = n
            best_key = k
            best_list = v
    return best_key, best_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="Path to gemini_batch.txt")
    parser.add_argument("--out", dest="out_json", default=str(DEFAULT_OUT_JSON), help="Path to output JSON file")
    parser.add_argument("--raw", dest="out_raw", default=str(DEFAULT_OUT_RAW), help="Path to output raw text file")
    parser.add_argument("--model", dest="model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--usage-out", type=str, default="", help="Optional: write usage metadata JSON to this path.")

    parser.add_argument("--skip-if-exists", action="store_true", help="Skip API call if output JSON already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt size + rough token estimate; no API call.")
    parser.add_argument("--ping", action="store_true", help="Send a tiny request to validate billing/quota.")
    parser.add_argument("--mock", type=str, default="", help="Use a local mock JSON file instead of calling Gemini.")

    # NEW: optional expectations for output key
    parser.add_argument("--rank-n", type=int, default=0, help="Optional: expected rank size, e.g. 100 -> top100.")
    parser.add_argument("--rank-key", type=str, default="", help="Optional: expected key, e.g. top100.")

    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_json = Path(args.out_json)
    out_raw = Path(args.out_raw)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}. Run scripts/prepare_gemini_batch.py first.")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    # MOCK MODE
    if args.mock:
        mock_path = Path(args.mock)
        if not mock_path.exists():
            raise FileNotFoundError(f"Mock JSON not found: {mock_path}")
        shutil.copyfile(mock_path, out_json)
        out_raw.write_text(mock_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"🧪 MOCK mode: copied {mock_path} -> {out_json}")
        print(f"🧪 MOCK mode: wrote raw mirror -> {out_raw}")
        return

    # SKIP MODE
    if args.skip_if_exists and out_json.exists():
        print(f"⏭️  Skipping Gemini call because {out_json} already exists (--skip-if-exists).")
        return

    prompt = in_path.read_text(encoding="utf-8")
    n_chars = len(prompt)
    est_in_tokens = approx_tokens_from_chars(n_chars)

    print("🚀 Gemini Stage-2 ranking")
    print(f"   model: {args.model}")
    print(f"   input chars: {n_chars:,}")
    print(f"   rough input tokens: ~{est_in_tokens:,} (very rough)")

    if args.dry_run:
        print("✅ Dry-run only. No API call made.")
        return

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise SystemExit("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) env var.")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if args.ping:
        try:
            do_ping(client, args.model)
        except Exception as e:
            msg = str(e)
            print("\n❌ Ping failed.")
            print(nice_quota_hint(msg))
            raise
        finally:
            try:
                client.close()
            except Exception:
                pass
        return

    config = types.GenerateContentConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
    )

    try:
        resp = client.models.generate_content(
            model=args.model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        msg = str(e)
        print("\n❌ Gemini request failed.")
        print(nice_quota_hint(msg))
        raise
    finally:
        try:
            client.close()
        except Exception:
            pass

    usage_obj = None
    try:
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage_obj = {
                "prompt_token_count": getattr(um, "prompt_token_count", None),
                "candidates_token_count": getattr(um, "candidates_token_count", None),
                "total_token_count": getattr(um, "total_token_count", None),
            }
        else:
            u = getattr(resp, "usage", None)
            if isinstance(u, dict):
                usage_obj = u
    except Exception:
        usage_obj = None

    if args.usage_out and usage_obj is not None:
        Path(args.usage_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.usage_out).write_text(json.dumps(usage_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote usage: {args.usage_out}")

    # If a previous JSON exists, move it aside so we can't accidentally render stale results.
    if out_json.exists():
        bak = out_json.with_suffix(out_json.suffix + ".bak")
        out_json.replace(bak)
        print(f"🧹 Moved existing JSON aside: {out_json} -> {bak}")

    raw_text = (resp.text or "").strip()
    out_raw.write_text(raw_text, encoding="utf-8")
    print(f"✅ Wrote raw output: {out_raw}")

    parsed = extract_json_object(raw_text)
    if not parsed:
        print("⚠️ Could not parse JSON from model output. Inspect raw file:")
        print(f"   {out_raw}")

        # Make sure no stale JSON remains
        if out_json.exists():
            out_json.unlink()
            print(f"🧹 Deleted stale JSON: {out_json}")
        return

    out_json.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Parsed JSON saved: {out_json}")

    expected_key = ""
    if args.rank_key.strip():
        expected_key = args.rank_key.strip()
    elif args.rank_n and args.rank_n > 0:
        expected_key = f"top{int(args.rank_n)}"

    # Preview (auto-detect topN list)
    key, lst = infer_rank_key(parsed)
    if expected_key and expected_key != key:
        print(f"⚠️ Expected key '{expected_key}', but detected '{key}'. (Will still preview detected key.)")

    if isinstance(lst, list):
        print(f"✅ {key} length: {len(lst)}")
        for i, item in enumerate(lst[:3], start=1):
            if not isinstance(item, dict):
                continue
            cid = item.get("cand_id")
            score = item.get("score")
            reason = item.get("reason", "")
            reason_short = (reason[:120] + "…") if isinstance(reason, str) and len(reason) > 120 else reason
            print(f"  {i}. {cid} score={score} reason={reason_short}")
    else:
        print("⚠️ JSON parsed, but could not find any key like 'top50'/'top100' with a list value.")


if __name__ == "__main__":
    main()