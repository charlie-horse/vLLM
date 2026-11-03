# Postmortem: Flint emits gibberish past 64 tokens of context after max-num-seqs 2 → 20

Status: root cause identified; fix on `claude/flint-gpu-opt` (`7f12d21`, plus a
follow-up that latches the counter allocation per device and stream). Rows
marked TBD are filled in after deploy.

## Summary

On 2026-11-03 c4130's production Flint server was switched to a config with
`--max-num-seqs 20` (was 2). From then on, every request whose context
(prompt + generated tokens) passed 64 tokens produced word salad. That held
at batch 1 too. The post-switch check sent only short answers and passed.
The regression went unnoticed for about 7 hours, until a user reported it.

## Impact

- Duration: from gen 40 at 15:41 local (c4130) until rollback or fix (TBD).
- Scope: every Flint request whose context exceeded 64 tokens. In practice,
  all agent and coding traffic. Only very short chat answers were correct.
- Data: no corruption outside served responses. LMCache saw 0 hits. Whether
  it stored bad KV is TBD; if it did, flush it after the fix.

## Timeline (2026-11-03; times from Ada; c4130 local unless marked)

| Time | Event |
| --- | --- |
| Before | Gen 39: Flint at `--max-num-seqs 2`, correct |
| — | `maxNumSeqs = 20` added in `charlie-horse/flake` `hosts/c4130/vllm/default.nix` (commit `4155ac6`, "servers" change) |
| 15:41 | Switch to gen 40; vllm pid 46458 starts |
| ~15:4x | Post-switch check: 12 concurrent "17×23" requests, ~540 tok/s aggregate, 0 waiting. **All outputs were shorter than 64 tokens** |
| ~22:30 UTC | Gibberish reported through the orchestrator |
| 22:3x UTC | Gen 39 and 40 builds found byte-identical; the only delta is `--max-num-seqs` |
| ~23:00 UTC | Hand-run ablations at seqs 20 (see Root cause) |
| ~23:20 UTC | Root cause: fused decode counters created under capture; fix `7f12d21` (Tess); ablation table complete |
| TBD | GPU unit tests, Gate A, review items; merge; deploy; LMCache flush |

## Symptoms and repro

- Serve profile: TP2, FP16, 262K, `--max-num-seqs 20`, prefix caching,
  LMCacheMPConnector (recompute), TurboMind MoE, default cudagraph settings.
- Temperature 0:
    - A 3-token prompt is fine for 61 generated tokens (context 64), then
    turns to salad.
    - A 37-token prompt breaks at generated token 28 (context 65).
    - A prompt over 64 tokens is garbage from the first output token, which
    comes from prefill.
- Batch 1 breaks, and so do 4 and 12 concurrent requests.
- A re-sent prompt (prefix-cache hit) is bad from its first token: its
  uncached tail runs as a small-q prefill through the same decode op.
- 64 tokens = 4 KV blocks at block size 16. The failure starts at the first
  token stored in block index 4, in both prefill and decode.

## Root cause

The one-launch grouped sink decode
(`VLLM_SM70_FLINT_ATTN_FUSED_DECODE`, added in the GPU decode fusions) keeps
int32 arrival counters, one per (row, KV head). They must be zero before a
launch, and the kernel's last CTA resets them. They were created lazily with
`torch.zeros` when the decode workspace was first used or grew
(`flash_attn_interface.py`, `_get_decode_workspace_for_plan`).

- At `--max-num-seqs 2` the eager warm-up and profile runs allocated the
  workspace at full size before any CUDA graph was captured.
- At 20 the capture list is larger, and the workspace grew **during capture**.
  A `torch.zeros` under capture only records a memset into that one graph, so
  the buffer was never zeroed for the graphs captured later, including
  batch 1.
- On those graphs, the arrival count never matched, so no CTA combined the
  partials and the attention output stayed stale. That only happens when more
  than one 64-token chunk is active (`kGqaFusedMinChunk`), hence context > 64.
- The small-q prefill of a prefix-cached prompt's tail, replayed in a
  PIECEWISE graph, used the same op and counters. That made the first token
  of a re-sent long prompt bad as well.

Ablation (seqs 20, fresh / prefix-hit repeat): baseline ok / broken;
`QK_ROPE_KV=0` broken / broken; `ATTN_FUSED_DECODE=0` ok / ok;
FULL_DECODE_ONLY ok / ok; `SHARE_DECODE_WORKSPACE=0` ok / ok; seqs 2 ok / ok;
fix ok / ok (175 tok/s against ~179).

The bug was latent from the moment the fused decode shipped. It only needed
a capture list larger than the eager warm-up's shapes.

## Fix

Counters now live in a per-(device, stream) cache. They are allocated and
zeroed only outside capture, preallocated from the profile run. A capture
without counters takes the correct two-launch path and logs a warning. vLLM captures on a
side stream. The per-descriptor eager warm-up (`cudagraph_num_of_warmups=1`,
the default) runs on that stream and allocates the counters there. The
fallback warning appears 0 times in the startup logs on both ranks, so the
fused path holds in every graph. That still depends on the warm-up, so a
follow-up commit latches the allocation per (device, stream). 175 against
179 tok/s is within run-to-run noise.

## Why we did not catch it

- **The post-switch check could not see it.** "17×23 → 391" and similar
  prompts finish well inside 64 tokens of context, and no prompt was re-sent,
  so the prefix-cache path was never exercised. The check tested throughput
  and concurrency (the purpose of the change), not long-output correctness.
- **The GPU unit tests captured graphs differently from the server.** They
  called eagerly, then captured one size. The server captures many sizes in
  descending order while the workspace grows.
- **A "config-only" change was treated as low risk.** `max_num_seqs` changes
  CUDA graph capture sizes, memory profiling, KV pool size and workspace
  sizing. Several SM70 fast paths are admitted by batch size (sampler B ≤ 2;
  GEMV, router and MoE decode M ≤ 2), and their workspaces are sized at
  capture. None of that was validated beyond `max-num-seqs 2`, the only
  value the validation runbook and the playbook ever used.
- **No continuous quality signal.** Nothing in production samples output
  quality. The first signal was a person.

## What went well

- The gen 39/40 build diff ruled out the build in minutes, thanks to the
  reproducible Nix build.
- Rollback is one generation switch away.
- The repro was cut to batch 1, temperature 0, and an exact 64-token
  boundary quickly.

## Action items

| # | Action | Owner | Status |
| --- | --- | --- | --- |
| 1 | Fix the root cause, with a unit test for descending capture with growth | Ada / Tess | Done in `7f12d21`; GPU tests not yet run (no pytest on c4130) |
| 1b | Make the fused path independent of the warm-up count (latch per device and stream); test warm-ups 0 and 1, eager on the default stream with capture on a side stream; count fallbacks in `/v1/sm70/acceleration` | Dennis / Ada | Follow-up commit; fallback warning seen 0 times today |
| 2 | Regression test: greedy to 256 tokens with prompts of 3, 60, 100 and 1000 tokens, each sent twice (prefix-cache hit), at batch 1 and 8, on a server at max-num-seqs 20 with default graphs plus an `--enforce-eager` variant, compared with known-good output | Ada / Ken | Planned |
| 3 | **Mandatory long-generation smoke test in the switch runbook.** After every serving-config change, before the switch counts as done: (a) a ≥ 1,000-token prompt that needs a specific answer (needle or a code task with asserts), (b) one greedy generation to ≥ 256 tokens compared with a stored golden output, (c) one batch of concurrent requests at the new max-num-seqs. Fail means automatic rollback | Ken | In the playbook (Phase 5, switch smoke test); runbook TBD |
| 4 | Treat `max-num-seqs`, `max-num-batched-tokens`, graph sizes, `max-model-len`, KV dtype and attention backend as code changes: run the validation runbook's `smoke,quality` stages at the new value before deploying | Coordinator | In the playbook (Phase 4, serving-config sweep) |
| 5 | Add a max-num-seqs and capture-size sweep (2, 3, 4, 8, 20, plus the default capture list) to `tools/flint/sm70_validate.sh` and to the playbook's Phase 4 | Ken | Playbook done; `sm70_validate.sh` TBD |
| 6 | Production quality canary: every N minutes, one fixed greedy long prompt against the golden output; alert on mismatch beyond the noise floor | Ken | TBD |
| 7 | Audit every lazily allocated "zero once" or "fill once" buffer reachable under capture, and every batch-admitted SM70 path, for state sized at init or capture (workspaces, counters, CTA counts) against the live batch and capture size | Dennis / Hedy | TBD |
| 8 | Flush LMCache L1 after the fix, if it stored KV computed by the bad build | Coordinator | TBD |

## Lessons

- A check that cannot fail is not a check. Validation prompts must exercise
  long context and long outputs, not just "17×23".
- A config change that alters capture sizes or memory layout is a code
  change on this fork, because fast paths are selected from them.
- Never create state that is "zero once" or "fill once" lazily. Under CUDA
  graph capture, the initialisation is recorded, not executed.
- Unit tests must capture the way the server does: many sizes, largest first,
  on a side stream, with buffers growing mid-capture.
