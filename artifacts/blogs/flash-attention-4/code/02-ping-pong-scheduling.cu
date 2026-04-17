// Extracted from sources/blogs/flash-attention-4.md by scripts/extract_blog_code.py
// Heading: ## Key Code > ### Ping-pong scheduling
// Original fence language: cuda
// See artifacts/blogs/flash-attention-4/PROVENANCE.yaml for origin + license metadata.

// Ping-pong two 128-token query tiles per CTA. While one tile is in the
// softmax/rescale stage, the other issues tcgen05.mma — the 2x tensor-core
// throughput on B200 gets fed while the SFU-bound softmax stays out of the
// critical path.
for (int tile = 0; tile < Q_tiles; tile += 2) {
    issue_mma(query_a, key_block);
    wait_mma();
    softmax_and_rescale(query_a);           // SFU + MUFU path
    issue_mma(query_b, key_block);
    wait_mma();
    softmax_and_rescale(query_b);
}
