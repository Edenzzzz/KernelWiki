# Blackwell Kernel Optimization Knowledge Base

A structured knowledge base of NVIDIA Blackwell (SM100, B200) and Hopper (SM90, H100) GPU kernel optimization techniques, designed for LLM agent retrieval.

## What's Here

- **460 PR references** from NVIDIA/cutlass, sgl-project/sglang, vllm-project/vllm, flashinfer-ai/flashinfer, pytorch/pytorch (Jan 2025 – Apr 2026)
- **48 synthesized wiki pages**: hardware features, techniques, kernel case studies, problem patterns, DSL guides, migration guides
- **20 community blog summaries**, **10 official doc summaries**, **7 competition pages** (GPU Mode NVFP4 hackathon, FlashInfer MLSys 2026)
- **6 auto-generated cross-reference indices**: by problem / technique / hardware feature / repo / kernel type / language
- **5 candidate ledgers** tracking 3,928 merged PRs with include/defer/exclude decisions

## For LLM Agents: Use the Skill

The canonical way to query this KB from Claude Code is via the bundled skill:

```
skills/blackwell-kernel-wiki/
├── SKILL.md              # When to engage + 10 navigation paths
├── scripts/
│   ├── query.py          # Unified search (keywords + filters)
│   ├── get_page.py       # Fetch any page by id or path
│   └── grep_wiki.py      # Regex text search
├── references/
│   ├── schema.md         # Condensed schema reference
│   └── examples.md       # 10 worked query patterns
└── README.md             # Install + quick smoke test
```

See [`skills/blackwell-kernel-wiki/README.md`](skills/blackwell-kernel-wiki/README.md) for installation.

Quick start (no install needed):
```bash
python3 skills/blackwell-kernel-wiki/scripts/query.py "ping-pong attention" --limit 5
python3 skills/blackwell-kernel-wiki/scripts/get_page.py kernel-flash-attention-4
python3 skills/blackwell-kernel-wiki/scripts/grep_wiki.py "tcgen05.fence" --only wiki
```

## For Human Readers

Start with [`index.md`](index.md) for curated top-level navigation.

Read [`CLAUDE.md`](CLAUDE.md) for schema conventions and the 8-step navigation flow.

## Architecture

Three layers (inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)):

1. **`sources/`** — Raw data. Immutable summaries of PRs, blogs, docs, contests.
2. **`wiki/`** — Synthesized knowledge pages. Cross-referenced by `id`. All have YAML frontmatter.
3. **`queries/`** — Auto-generated cross-reference indices. Do not edit manually; regenerate via `scripts/generate-indices.py`.

Supporting files:
- `data/schemas.yaml` — Required/optional fields per page type
- `data/tags.yaml` — Controlled vocabulary (80+ tags)
- `data/aliases.yaml` — Canonical → synonym mappings
- `candidates/` — Reviewed PR candidate ledgers (per repo)

## Tooling

| Script | Purpose |
|--------|---------|
| `scripts/validate.py` | Validate YAML frontmatter, enforce schema, check link integrity |
| `scripts/generate-indices.py` | Regenerate `queries/*.md` from frontmatter |
| `scripts/generate-pr-pages.py` | Batch-generate source PR pages from candidate ledgers |

Setup:
```bash
pip install -r requirements.txt
python3 scripts/validate.py         # should report 545 files, 0 errors
python3 scripts/generate-indices.py # regenerate query indices
```

## Quality Gates

As of 2026-04-17:
- 545 files, 497 source IDs, 0 validation errors
- 0 broken links across 1,642 internal references
- All `verified` wiki pages have official-doc + upstream-code evidence
- All technique/kernel/language pages have compilable code snippets
- All Hopper-inclusive pages explain their `blackwell_relevance`

## Scope Rules

- **Blackwell-first**: SM100 content is primary. SM90 requires explicit `blackwell_relevance` field.
- **Kernel-only**: No distributed-system topics (DeepEP, DualPipe, EPLB are out of scope).
- **English canonical**: All content in English.
- **First-class DSLs**: CuTe DSL, CUDA C++, PTX, Triton. TileLang/cuTile/JAX-Pallas mentioned but no dedicated guides.

## Repository Layout

```
├── README.md                    # This file
├── CLAUDE.md                    # LLM navigation + schema conventions
├── index.md                     # Human-facing curated top-level index
├── requirements.txt             # Python deps (PyYAML)
│
├── data/                        # Schema + vocabulary
│   ├── schemas.yaml
│   ├── tags.yaml
│   └── aliases.yaml
│
├── scripts/                     # Validation + generation tooling
│   ├── validate.py
│   ├── generate-indices.py
│   └── generate-pr-pages.py
│
├── skills/                      # Claude Code skill (self-contained)
│   └── blackwell-kernel-wiki/
│       ├── SKILL.md
│       ├── scripts/
│       │   ├── query.py
│       │   ├── get_page.py
│       │   └── grep_wiki.py
│       └── references/
│           ├── schema.md
│           └── examples.md
│
├── candidates/                  # Reviewed PR ledgers (source of truth for ingestion)
│   ├── cutlass.yaml
│   ├── sglang.yaml
│   ├── vllm.yaml
│   ├── flashinfer.yaml
│   └── pytorch.yaml
│
├── sources/                     # Layer 1: raw data
│   ├── prs/{repo}/PR-{N}.md
│   ├── contests/{contest}/
│   ├── docs/
│   └── blogs/
│
├── wiki/                        # Layer 2: synthesized knowledge
│   ├── hardware/
│   ├── techniques/
│   ├── kernels/
│   ├── patterns/
│   ├── languages/
│   └── migration/
│
└── queries/                     # Layer 3: auto-generated indices
    ├── by-problem.md
    ├── by-technique.md
    ├── by-hardware-feature.md
    ├── by-repo.md
    ├── by-kernel-type.md
    └── by-language.md
```

## License

Summaries and wiki syntheses in this repository are derivative works citing upstream PRs, blogs, and docs. The tooling (`scripts/`, `skills/`, `data/`) is MIT-style; see individual files for any exceptions.
