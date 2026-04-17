# Phase 2 Completion Plan (Draft)

## Status: INCOMPLETE (per Codex review 2026-04-17)

Total files: 475 / target 540+. Remaining gap: ~65 files.

## Remaining Work

### [P0] PyTorch PR coverage (AC-2.5)
- **Current**: 12 pages
- **Target**: 50+ pages
- **Gap**: +38 pages minimum
- **Root cause**: candidate ledger only has 31 PRs because original keyword set was too narrow

**Plan**:
1. Re-collect PyTorch candidate ledger with expanded keywords:
   - Existing: blackwell, sm100, sm_100, nvfp4, B200, cuda+13
   - Add: tensor_core, wgmma, tcgen05, cute, CUTLASS, TMA, inductor, triton,
         flash_attention, sdpa, flashattention, quantization, fp8, fp4,
         scaled_mm, block_scale, compute_10, launch_bounds, vec128,
         mxfp4, "sm_100a", bf16, autocast, compile, gemm
   - Include Hopper keywords that often touch same kernel paths: wgmma, sm90, hopper
2. Update candidates/pytorch.yaml (should grow from 31 to 100+ PRs)
3. Run generate-pr-pages.py candidates/pytorch.yaml --max=60
4. Validate and commit

### [P0] Wiki expansion (AC-6)
- **Current**: 37 wiki pages (same as before Phase 2)
- **Target**: +10 new wiki pages supported by 3+ sources each
- **Gap**: +10 pages minimum

**Plan**: Create new wiki pages covering concepts now supported by 3+ new sources:

1. **wiki/techniques/ping-pong-scheduling.md** — FlashAttention-4, 3+ FlashInfer PRs
2. **wiki/techniques/software-exp.md** — Already exists, expand with FA4 blog + papers
3. **wiki/techniques/kernel-fusion.md** — dual GEMM, fused MoE, SwiGLU (3+ contest+kernel sources)
4. **wiki/techniques/chunk-parallelism.md** — GatedDeltaNet, TFLA, NSA (3+ sources)
5. **wiki/techniques/cache-policy.md** — NVFP4 GEMV (Yue, Amandeep, Simon blogs + PTX doc)
6. **wiki/techniques/register-budgeting.md** — NVFP4 GEMV (3 blog sources)
7. **wiki/techniques/ld-evict-policy.md** — L1::evict_last patterns (3+ PTX/GEMV sources)

8. **wiki/kernels/gated-dual-gemm.md** — GPU Mode Problem 3 + vLLM fused gate-up + SGLang dual gemm
9. **wiki/kernels/sparse-mla.md** — FlashMLA sparse, DeepSeek V3.2 + 3+ FlashInfer/SGLang/vLLM PRs
10. **wiki/kernels/fp8-block-scale-gemm.md** — CUTLASS SM100 + vLLM FP8 GEMM + DeepGEMM

11. **wiki/hardware/green-context.md** — CUTLASS changelog + cuTile + CLC docs
12. **wiki/hardware/mbarrier.md** — pervasive in all kernel pages, needs dedicated page

13. **wiki/patterns/pipeline-stalls.md** — compute-bound + 3+ PR fixes
14. **wiki/patterns/moe-load-imbalance.md** — grouped GEMM, EPLB analysis

15. **wiki/kernels/flash-attention-hopper.md** (migration reference) — FA3 → FA4 progression

### [P0] FlashInfer contest submissions (AC-4 completion)
- **Current**: 4 GPU Mode pages have submissions, 3 FlashInfer track pages do NOT
- **Gap**: 3 FlashInfer track pages need submissions field

**Plan**: Update 3 FlashInfer MLSys 2026 track pages with `submissions:` frontmatter:
- track-a-fused-moe.md: FlashInfer-Bench leaderboard entries (Gemini 2.5 Pro, GPT-5, Claude Opus 4.1)
- track-b-sparse-attention.md: Same leaderboard for sparse attention
- track-c-gated-delta-net.md: Already updated by earlier agent (verify)

### [P1] Link integrity check (AC-7)
- Run the link integrity check script from Phase 1
- Ensure 0 broken links across queries/, wiki/, sources/, index.md

**Plan**:
```python
# Link integrity check - already proven in Phase 1
import re
from pathlib import Path
# Scan all markdown links in queries/, wiki/, index.md
# Verify each relative link resolves to an existing file
```

### [P1] Total file target (540+)
- **Current**: 475
- **Target**: 540+
- **Gap**: +65 files

**Distribution of gap**:
- +38 PyTorch pages → 513
- +10 wiki pages → 523
- +3 FlashInfer contest updates (no new files, just edits)
- Need +17 more files from:
  - Bulk-generate more FlashInfer/vLLM/SGLang pages (raise --max limits)
  - Or: add candidate PRs that got deferred after file-based re-triage

### [P2] Plan internal consistency
Fix contradictions in plan-phase2.md:
- Line 23: PyTorch AC is "≥ 50 pages"
- Line 397: PyTorch target is "15-25"
- Line 553: PyTorch target is "70+"

Resolution: standardize on "50+" per AC-2.5 (already committed).

## Execution Order

### Round N (this round)
1. Re-collect PyTorch candidates with expanded keywords (P0)
2. Generate PyTorch PR pages to hit 50+ (P0)
3. Update 3 FlashInfer contest pages with submissions (P0)
4. Create 10 new wiki pages (P0)
5. Run link integrity check (P1)
6. Validate + regenerate indices
7. Total should be ≥540 files

### Completion Gate
- All 8 ACs met
- 540+ files total
- 0 validation errors
- 0 broken links
- Codex confirms STATUS: complete

## Estimated Effort
- PyTorch re-collect + generate: ~10 min
- Wiki pages (10): ~20 min (can use batch script to template)
- Contest submissions: ~5 min
- Link check + validation: ~5 min
- **Total**: ~40 min of focused work

All work in this round is `coding` tag — no Codex analysis needed until final audit.
