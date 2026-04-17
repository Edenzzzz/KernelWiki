# blackwell-kernel-wiki — Claude Code Skill

A self-contained skill for querying the Blackwell GPU kernel optimization knowledge base.

## What This Skill Provides

- **SKILL.md**: When-to-use spec, frontmatter (name, description, argument-hint, allowed-tools)
- **scripts/query.py**: Unified search across 545 pages (keyword + tag + repo + language + architecture + symptom filters)
- **scripts/get_page.py**: Fetch any page by `id` or path; can follow `sources:` to cite
- **scripts/grep_wiki.py**: Regex text search across wiki bodies and source PR descriptions
- **references/schema.md**: Condensed schema reference (page types, required fields, controlled vocabulary)
- **references/examples.md**: 10 worked query patterns with commands and synthesis templates

## Installation

### Option 1: Run in place (no install)
Just invoke the scripts directly from the repo:
```bash
python3 skills/blackwell-kernel-wiki/scripts/query.py "tcgen05 mma"
```

### Option 2: Install as a Claude Code skill
Copy or symlink this directory into Claude's skills location:

```bash
# User-wide install
mkdir -p ~/.claude/skills/
cp -r skills/blackwell-kernel-wiki ~/.claude/skills/

# Or symlink (stays in sync with repo updates)
ln -s "$(pwd)/skills/blackwell-kernel-wiki" ~/.claude/skills/blackwell-kernel-wiki
```

The skill will then be auto-discovered by Claude Code and shown as `/blackwell-kernel-wiki` or invokable by the skill-resolver.

### Option 3: Use as project skill
Claude Code auto-detects skills in `.claude/skills/` or `skills/` at the project root. No manual install needed.

## Environment

Scripts auto-detect the wiki root as 3 levels up from `scripts/*.py` (i.e. the repo root). If invoked from elsewhere, set:
```bash
export BLACKWELL_WIKI_ROOT=/path/to/blackwell-kernel-wiki
```

## Dependencies

Pure Python 3 + PyYAML. Install via the repo's `requirements.txt`:
```bash
pip install -r ../../requirements.txt
```

## Quick Smoke Test

```bash
# Should print a list of ≈4 NVFP4 kernel pages
python3 scripts/query.py --tag nvfp4 --type kernel --compact

# Should return FlashAttention-4 frontmatter
python3 scripts/get_page.py kernel-flash-attention-4 --frontmatter-only

# Should find at least 2 files mentioning tcgen05.fence
python3 scripts/grep_wiki.py "tcgen05.fence" --files-only
```

## How Claude Should Use This Skill

1. Read `SKILL.md` to decide when to engage
2. When engaging, start with one of the query paths in `SKILL.md` Section "How To Query"
3. For natural-language user questions, prefer `query.py` with keywords
4. Read retrieved page contents via `get_page.py` or direct `Read` tool
5. Follow `sources:` IDs for primary evidence
6. Apply the synthesis pattern in `references/examples.md`
