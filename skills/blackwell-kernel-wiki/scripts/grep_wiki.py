#!/usr/bin/env python3
"""Text search across wiki bodies and source PR descriptions.

Usage:
    grep_wiki.py "tcgen05.fence"
    grep_wiki.py "2-CTA backward" --only wiki
    grep_wiki.py "ping-pong" --context 3
    grep_wiki.py "nvfp4 block_scale" --any     # match if ANY word appears

Returns matching lines with file path, line number, and N context lines.
"""

import argparse
import re
import sys
from pathlib import Path

WIKI_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if not (WIKI_ROOT / "data" / "tags.yaml").exists():
    import os
    WIKI_ROOT = Path(os.environ.get("BLACKWELL_WIKI_ROOT", WIKI_ROOT))


def iter_files(scope):
    """Yield markdown file paths under the given scope."""
    dirs = {
        "wiki": ["wiki"],
        "sources": ["sources"],
        "all": ["wiki", "sources"],
    }
    for sub in dirs.get(scope, ["wiki", "sources"]):
        base = WIKI_ROOT / sub
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            yield md


def grep_file(path, patterns, context, any_match):
    """Search a single file for the pattern(s). Returns list of (line_no, context_text) tuples."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    results = []
    for i, line in enumerate(lines):
        if any_match:
            matched = any(re.search(p, line, re.IGNORECASE) for p in patterns)
        else:
            matched = all(re.search(p, line, re.IGNORECASE) for p in patterns)
        if matched:
            start = max(0, i - context)
            end = min(len(lines), i + context + 1)
            snippet = "\n".join(
                f"{j+1}{'→' if j == i else ':'} {lines[j]}"
                for j in range(start, end)
            )
            results.append((i + 1, snippet))
    return results


def main():
    parser = argparse.ArgumentParser(description="Text search across Blackwell kernel wiki")
    parser.add_argument("patterns", nargs="+", help="Search pattern(s) — all must match a line unless --any is used")
    parser.add_argument("--only", choices=["wiki", "sources", "all"], default="all",
                        help="Restrict search scope (default: all)")
    parser.add_argument("--context", type=int, default=1, help="Context lines around each match (default 1)")
    parser.add_argument("--any", action="store_true", help="Match if ANY pattern matches a line (default: all must match)")
    parser.add_argument("--limit", type=int, default=20, help="Max files reported (default 20)")
    parser.add_argument("--files-only", action="store_true", help="Print only matching file paths")
    args = parser.parse_args()

    # Compile patterns (literal substrings OK; special chars escaped by re search since we use re.search with raw)
    patterns = args.patterns

    matched_files = []
    for path in iter_files(args.only):
        hits = grep_file(path, patterns, args.context, args.any)
        if hits:
            matched_files.append((path, hits))

    matched_files = matched_files[:args.limit]

    if args.files_only:
        for path, _ in matched_files:
            print(path.relative_to(WIKI_ROOT))
        return

    if not matched_files:
        print("No matches.")
        return

    print(f"# {len(matched_files)} file(s) match")
    for path, hits in matched_files:
        print()
        print(f"## {path.relative_to(WIKI_ROOT)}  ({len(hits)} match{'es' if len(hits) != 1 else ''})")
        for line_no, snippet in hits[:5]:  # Cap per-file to 5 hits
            print(f"```")
            print(snippet)
            print(f"```")


if __name__ == "__main__":
    main()
