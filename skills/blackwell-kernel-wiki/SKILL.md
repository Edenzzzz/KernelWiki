---
name: blackwell-kernel-wiki
description: Query a curated knowledge base of NVIDIA Blackwell (SM100, B200) and Hopper (SM90, H100) GPU kernel optimization techniques. Covers 460+ PRs from CUTLASS/SGLang/vLLM/FlashInfer/PyTorch, 7 kernel competitions (GPU Mode NVFP4 hackathon, FlashInfer MLSys 2026), 48 wiki synthesis pages (hardware features like tcgen05/TMEM/CLC/NVFP4, techniques like warp specialization, kernel case studies like FlashAttention-4/DeepGEMM/FlashMLA). Use when the user asks about writing/optimizing Blackwell GPU kernels in CUDA/CuTe-DSL/PTX/Triton, or wants references for B200-specific performance patterns.
argument-hint: "<query-type> [args]"
allowed-tools: "Bash,Read,Grep,Glob"
---

# Blackwell Kernel Wiki

Query a structured, cross-referenced knowledge base of GPU kernel optimization for NVIDIA Blackwell (SM100) and Hopper (SM90) architectures.

## When To Use This Skill

Invoke this skill when the user asks about:

- **Blackwell/SM100 kernel programming**: tcgen05.mma, TMEM, CLC, 2-SM cooperative, NVFP4, FP8 block scaling
- **Kernel implementations**: FlashAttention-4, DeepGEMM, FlashMLA, NSA, GatedDeltaNet, NVFP4 GEMM/GEMV, fused MoE
- **Performance patterns**: low SM utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls
- **DSLs for Blackwell**: CuTe DSL, CUDA C++ with PTX, Triton on Blackwell
- **Hopper → Blackwell migration**: wgmma → tcgen05, register → TMEM accumulators
- **PR references**: "How did vLLM/SGLang/FlashInfer/CUTLASS/PyTorch implement X for SM100?"
- **Competition solutions**: GPU Mode NVFP4 hackathon top solutions, FlashInfer MLSys 2026 submissions

## Knowledge Base Contents

- **545 total markdown pages**: 460 PR references + 48 wiki synthesis + 20 blogs + 10 docs + 7 contests
- **5 candidate ledgers**: 3,928 merged PRs classified (include/defer/exclude) from Jan 2025 - Apr 2026
- **6 auto-generated query indices**: by-problem, by-technique, by-hardware-feature, by-repo, by-kernel-type, by-language
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`

## How To Query (Navigation)

The knowledge base lives at the repository root (one level up from `skills/`). Use these query paths in order of specificity:

### Path 1: Known topic → direct wiki page
```bash
# List all wiki pages
ls wiki/hardware/ wiki/techniques/ wiki/kernels/ wiki/patterns/ wiki/languages/ wiki/migration/

# Read a specific page
cat wiki/hardware/tcgen05-mma.md
cat wiki/kernels/flash-attention-4.md
```

### Path 2: By problem/symptom → pattern page
```bash
# Open the diagnostic index
cat queries/by-problem.md
# Shows: symptom → pattern page → candidate techniques
```

### Path 3: By technique name → all examples
```bash
cat queries/by-technique.md
# Lists 15 techniques with architectures, confidence, reproducibility, source count
```

### Path 4: By hardware feature → related pages
```bash
cat queries/by-hardware-feature.md
# Maps tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
```

### Path 5: By kernel type → case studies
```bash
cat queries/by-kernel-type.md
# Shows gemm/attention/moe/mla/gated-delta-net etc. → pages
```

### Path 6: By language/DSL → guide + usage examples
```bash
cat queries/by-language.md
# cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
```

### Path 7: By source repo → all PRs
```bash
cat queries/by-repo.md
# Lists all 460 PRs across cutlass/sglang/vllm/flashinfer/pytorch with titles, dates, techniques
```

### Path 8: Unified search tool (recommended for natural language queries)
```bash
python3 skills/blackwell-kernel-wiki/scripts/query.py "how to fuse gate-up dual GEMM on Blackwell"
python3 skills/blackwell-kernel-wiki/scripts/query.py --tag nvfp4 --type kernel
python3 skills/blackwell-kernel-wiki/scripts/query.py --repo cutlass --limit 20
```

### Path 9: Get a specific page by ID
```bash
python3 skills/blackwell-kernel-wiki/scripts/get_page.py kernel-flash-attention-4
python3 skills/blackwell-kernel-wiki/scripts/get_page.py pr-cutlass-2472
```

### Path 10: Text search across wiki bodies
```bash
python3 skills/blackwell-kernel-wiki/scripts/grep_wiki.py "tcgen05.fence::after_thread_sync"
python3 skills/blackwell-kernel-wiki/scripts/grep_wiki.py "2-CTA backward"
```

## Schema Reference

Each page has YAML frontmatter. Key fields by page type are summarized in `skills/blackwell-kernel-wiki/references/schema.md` (condensed from `CLAUDE.md`). Full schema at `data/schemas.yaml`.

## Worked Query Examples

See `skills/blackwell-kernel-wiki/references/examples.md` for 10+ concrete examples: translating a user question → navigation path → synthesis.

## Output Pattern

When answering questions from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/papers.
3. **Respect confidence levels**: `verified` > `source-reported` > `inferred` > `experimental`.
4. **Include code snippets** from wiki pages when they exist — they're guaranteed to be compilable snippets (validated).
5. **Cross-check performance claims**: every `performance_claims` entry has `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id` — report all six.

## Quality Guarantees (as of 2026-04-17)

- 545 files validated, 0 errors
- 0 broken internal links across 1,642 links
- Every `verified` page has official-doc + upstream-code evidence
- Every technique/kernel/language page has compilable code
- All Hopper-inclusive pages have explicit `blackwell_relevance` field
