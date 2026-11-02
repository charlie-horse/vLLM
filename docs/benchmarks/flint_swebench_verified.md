# Flint on SWE-bench Verified (2x V100, NVFP4 W4A16)

Status: DRAFT. Values in `{{...}}` are filled in from
`scripts/analyze.py` once the run finishes.

## Summary

- Flint, served by this vLLM fork on 2x V100 (SM70, TP2), NVFP4 W4A16
  weights, FP16 activations.
- SWE-bench Verified, 500 instances x 1 attempt, mini-swe-agent 2.4.6 with
  the official SWE-bench config, official swebench 5.0.2 evaluation.
- pass@1: **{{PASS_AT_1}}** ({{RESOLVED}} / {{SCORED}} scored instances),
  95% Wilson CI {{CI95_LO}} – {{CI95_HI}}.
- {{INFRA_EXCLUDED}} instances excluded as infra failures (not counted against
  the model; listed in Results).
- Model card claim: 74.2% (BF16). Difference: {{DELTA_PP}} pp.
  {{ONE_LINE_VERDICT}}

## Claim vs result

| | Model card | This run |
|---|---|---|
| Checkpoint | BF16 | NVFP4 W4A16, FP16 activations |
| Hardware | not stated | 2x V100 32 GB, TP2 |
| Agent | mini-swe-agent (version, config, step limit not stated) | mini-swe-agent 2.4.6, official `swebench.yaml`, step limit 250 |
| Attempts | average pass@1 over >= 5 attempts | 1 attempt per instance |
| Instances | 500 | 500 ({{SCORED}} scored, {{INFRA_EXCLUDED}} infra-excluded) |
| Internet in task env | none | none (`--network=none`) |
| Agent timeout | up to 5 h | 5 h container timeout |
| pass@1 | 74.2% | {{PASS_AT_1}} (95% CI {{CI95_LO}} – {{CI95_HI}}) |

Sources: huggingface.co/RecursiveSI/flint (model card) and the accompanying
blog post. The blog states NVFP4 is within 0.7 pts of 16-bit across their
benchmarks, so the expected NVFP4 result is roughly 73.5–74.2% before
sampling noise.

## Method

- Instances: all 500 of `SWE-bench/SWE-bench_Verified` (test split), run in
  a fixed shuffled order (`all_ids_shuffled.txt`). One attempt per instance.
- Why one attempt: a single attempt over all 500 instances is an unbiased
  estimate of mean pass@1. The CI is dominated by instance variance, not
  attempt variance: the pilot-based estimate is about ±4 pp for 500x1 vs
  about ±3.5 pp for 500x5, at 5x the cost. Approved by the user before the
  run.
- Sampling (client): temperature 1.0, top_p 0.95, max_tokens 65536 (the
  NVFP4 card's client budget; the card's minimum is 32768). top_k 20 comes
  from the server default. Context length 262144.
- Server: this fork, Nix flake input locked at `b7b1753` (= main `c52fd8d`
  plus `nix/` packaging), built for sm_70 / CUDA 12.8. 2x Tesla
  V100-PCIe-32GB, no NVLink, TP2, host c4130. Same config for the whole
  run (the Sat 12:44 EDT restart changed only the LMCache disk cap).
- Weights: RecursiveSI/flint NVFP4 GGUF, repacked losslessly to a ModelOpt
  NVFP4 W4A16 checkpoint (`vllm.models.flint.convert_gguf` at `f66728d`):
  E2M1 weights and E4M3 block scales unchanged, BF16 tensors cast to FP16
  (range-checked). FP16 overflow is avoided with exact power-of-two shifts
  from layer 46 (MoE intermediate /16, residual stream /8).
- Serve args: `--tensor-parallel-size 2 --dtype half --max-model-len 262144
  --max-num-seqs 2 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90 --enable-prefix-caching
  --enable-auto-tool-choice --kv-cache-dtype auto
  --kernel-config.sm70_skinny_moe false --enable-prompt-tokens-details
  --tool-call-parser glm47 --reasoning-parser glm47
  --exclude-tools-when-tool-choice-none --override-generation-config
  '{"temperature":1.0,"top_p":0.95,"top_k":20}' --served-model-name flint`,
  async scheduling on. KV offload via LMCache (MP connector, chunk 256,
  64 GB host RAM L1, disk L2 180 GB then 100 GB, recompute on load
  failure). Decode: 178 tok/s single stream at 4K context.
- Reasoning round-trip (card requirement: keep reasoning in the history).
  The server returns `message.reasoning` (not `reasoning_content`). Stock
  mini-swe-agent 2.4.6 + litellm 1.106.6 already preserve it. The adapter
  `FlintModel` (`adapter/flint_model.py`) pins this by copying it into
  `reasoning_content` on the assistant message, and the chat template
  renders both. Check: a two-turn probe grew from 23 to 131 prompt tokens
  when reasoning was carried over. Adapter unit tests: 6/6 pass.
- Per instance (`r440/scripts/run_one.sh`): rate-limited image pre-pull,
  agent run, official `run_evaluation` on the produced `preds.json`, image
  cleanup. Two instances run in parallel (`run_all.sh`, `xargs -P2`),
  matching the server's two sequences.
- Scoring (`scripts/analyze.py`): each instance gets exactly one category.
  pass@1 = resolved / (done - infra). Infra categories (runner, API, eval,
  agent sandbox) are excluded from the denominator and listed separately.
  By default an eval-side test timeout is also treated as infra; the
  alternative count with `--timeout-is-model` is reported below.
- Retries that are not counted as attempts:
  - API failures: each model call retries for about 20 min (tenacity, 25
    attempts); if the agent still exits on an API error, the instance is
    re-run from scratch, up to 3 tries total. The previous trajectory is moved
    to `agent_retired/`.
  - Memory watchdog: if host MemAvailable drops below 8 GB, the newest task
    container is killed and the instance is re-run in a later pass (up to 3
    passes).
  - A failed image pull backs off up to about 2 h before the instance is
    marked `pull_failed` (infra).

## Harness versions

| Component | Version / source |
|---|---|
| mini-swe-agent | 2.4.6, official `config/benchmarks/swebench.yaml` |
| Step limit | 250 (official config) |
| Cost limit | not effective (`cost_tracking: ignore_errors`) |
| litellm | 1.106.6 |
| swebench | 5.0.2, official `swebench.harness.run_evaluation` |
| Dataset | `SWE-bench/SWE-bench_Verified`, test split |
| Eval images | `docker.io/swebench/sweb.eval.x86_64.*:latest` |
| Python | 3.12 (uv venv) |
| Container runtime | rootless podman (nix) on r440, Docker API socket |
| Overlay config | `benchmarks/flint_swebench/r440/flint.yaml` |

`SWE-bench/SWE-bench_Verified` has the same 500 instance IDs as
`princeton-nlp/SWE-bench_Verified`; swebench 5.x needs its `image` column.

## Deviations from the reference setup

The card does not publish its mini-swe-agent config, so "reference" here
means the official mini-swe-agent SWE-bench config.

| Deviation | Reason | Expected effect |
|---|---|---|
| Task container `--network=none` | Card: no internet for SWE-bench | Matches the card |
| `container_timeout: 5h` | Card blog: agent timeouts up to 5 h | Matches the card |
| Rootless podman instead of Docker | No Docker daemon on r440 | None on results |
| overlay `ignore_chown_errors=true` | matplotlib images have files owned by UIDs above the subuid range; those files map to root | None: tests run as root |
| nice 19 / ionice idle, `--cpus=6` per task container | Shared host; 2 parallel tasks use at most 12 cores | Slower tool calls only |
| 2 parallel agents | Server `max-num-seqs 2` | None on results |
| NVFP4 W4A16 on V100 instead of BF16 | Hardware | Card blog: within 0.7 pts |
| 1 attempt instead of >= 5 | Cost; see Method | Wider CI, same expectation |

## Validity checks

Audit by Nadia:

- Gold-patch sanity eval through the r440 podman setup
  (`astropy__astropy-12907`, run id `gold_r440`): resolved.
- Negative control (patch that only edits the README): not resolved.
- Resolved reports were checked to have actually run their FAIL_TO_PASS
  and PASS_TO_PASS tests.
- Model patches touch source files only and differ from the gold patches.
- Leakage attempts seen in trajectories (`pip download` of newer releases,
  `git log --all`) were blocked: no network in the task container, and the
  repos in the images contain no fix commits.
- {{EXTRA_VALIDITY_NOTES}}

## Results

pass@1: {{PASS_AT_1}} ({{RESOLVED}} / {{SCORED}}), 95% Wilson CI
{{CI95_LO}} – {{CI95_HI}}.
With eval test timeouts counted as model failures (`--timeout-is-model`):
{{PASS_AT_1_TIMEOUT_IS_MODEL}}.

### Per repo

| Repo | Resolved | Scored | Rate | Infra-excluded |
|---|---|---|---|---|
| astropy/astropy | {{R_ASTROPY}} | {{N_ASTROPY}} | {{P_ASTROPY}} | {{I_ASTROPY}} |
| django/django | {{R_DJANGO}} | {{N_DJANGO}} | {{P_DJANGO}} | {{I_DJANGO}} |
| matplotlib/matplotlib | {{R_MPL}} | {{N_MPL}} | {{P_MPL}} | {{I_MPL}} |
| mwaskom/seaborn | {{R_SEABORN}} | {{N_SEABORN}} | {{P_SEABORN}} | {{I_SEABORN}} |
| pallets/flask | {{R_FLASK}} | {{N_FLASK}} | {{P_FLASK}} | {{I_FLASK}} |
| psf/requests | {{R_REQUESTS}} | {{N_REQUESTS}} | {{P_REQUESTS}} | {{I_REQUESTS}} |
| pydata/xarray | {{R_XARRAY}} | {{N_XARRAY}} | {{P_XARRAY}} | {{I_XARRAY}} |
| pylint-dev/pylint | {{R_PYLINT}} | {{N_PYLINT}} | {{P_PYLINT}} | {{I_PYLINT}} |
| pytest-dev/pytest | {{R_PYTEST}} | {{N_PYTEST}} | {{P_PYTEST}} | {{I_PYTEST}} |
| scikit-learn/scikit-learn | {{R_SKLEARN}} | {{N_SKLEARN}} | {{P_SKLEARN}} | {{I_SKLEARN}} |
| sphinx-doc/sphinx | {{R_SPHINX}} | {{N_SPHINX}} | {{P_SPHINX}} | {{I_SPHINX}} |
| sympy/sympy | {{R_SYMPY}} | {{N_SYMPY}} | {{P_SYMPY}} | {{I_SYMPY}} |
| **Total** | {{RESOLVED}} | {{SCORED}} | {{PASS_AT_1}} | {{INFRA_EXCLUDED}} |

### Failure breakdown

| Category | Count | Share of scored |
|---|---|---|
| Resolved | {{RESOLVED}} | {{PASS_AT_1}} |
| Wrong patch (eval ran, not resolved) | {{N_WRONG_PATCH}} | {{S_WRONG_PATCH}} |
| No patch or step/time limit | {{N_NO_PATCH}} | {{S_NO_PATCH}} |
| Context overflow | {{N_CONTEXT}} | {{S_CONTEXT}} |
| Format / tool errors | {{N_FORMAT}} | {{S_FORMAT}} |
| Other (inspected manually) | {{N_OTHER}} | {{S_OTHER}} |
| Infra, excluded | {{INFRA_EXCLUDED}} | n/a |

Infra-excluded instances and reasons: {{INFRA_LIST}}

Of the wrong patches, {{N_PATCH_APPLY_FAILED}} failed to apply.

## Cost

Per scored instance (mean / median / p90):

| Metric | Mean | Median | p90 |
|---|---|---|---|
| Agent turns | {{TURNS_MEAN}} | {{TURNS_MED}} | {{TURNS_P90}} |
| Prompt tokens (sum over calls) | {{PT_MEAN}} | {{PT_MED}} | {{PT_P90}} |
| Completion tokens | {{CT_MEAN}} | {{CT_MED}} | {{CT_P90}} |
| Completion tokens per call | {{CPT_MEAN}} | {{CPT_MED}} | {{CPT_P90}} |
| Peak context (prompt tokens) | {{PEAK_MEAN}} | {{PEAK_MED}} | {{PEAK_P90}} |
| Agent wall time | {{AGENT_S_MEAN}} | {{AGENT_S_MED}} | {{AGENT_S_P90}} |
| Eval wall time | {{EVAL_S_MEAN}} | {{EVAL_S_MED}} | {{EVAL_S_P90}} |

- Largest peak context in the run: {{PEAK_MAX}} of 262144.
- Total run wall time: {{RUN_WALL_TIME}} (start 2026-10-31 10:53 EDT, end
  {{RUN_END}}).
- Total tokens: {{TOTAL_PROMPT_TOKENS}} prompt, {{TOTAL_COMPLETION_TOKENS}}
  completion.

Pilot (cloud sandbox, 10 instances): about 100 tok/s per stream decode with
2 streams; model time was about 60% of agent wall time. 7 resolved, 3 not
scored (Docker Hub 429 on image pulls).

Agent wall time includes API retry waits during the incidents below.

## Infra incidents

None of these were counted against the model. Instances whose agent hit an
API error were re-run from scratch (see Method); instances that still failed
are in the infra-excluded list.

| When (EDT) | Event | Handling |
|---|---|---|
| Oct 31 ~10:38 | vLLM restart (switch window) | Ridden out by per-call retries (~20 min), then up to 2 re-runs |
| Oct 31 12:44–12:47 | vLLM restart (LMCache cap change) | Same |
| Oct 31 ~13:20–14:15 | k8s rollout on r440 (cri-o to containerd, firewall reloads) | No effect: rootless podman is independent of cri-o; 0 infra errors |
| Oct 31 (smoke run) | Watchdog inherited the driver's lock fd and blocked the next driver start | Fixed: fd 8 closed for the watchdog (`1107909`) |
| Oct 31 13:36 | matplotlib image pulls failed on high-UID files | Fixed: overlay `ignore_chown_errors=true` (`b20d887`) |
| Oct 31 17:03 | `pytest-dev__pytest-7432`: transient layer error matched as missing image | Fixed match (`c1bd2a1`); instance re-queued |
| Nov 1 01:03 | Image prune raced a concurrent pull (layer not found) | Fixed: cleanup holds the pull lock (`3eb821e`) |

Fix commits on `claude/flint-swebench`:

```
3eb821e flint_swebench: serialize image cleanup with pulls
c1bd2a1 flint_swebench: don't treat transient layer errors as missing images
b20d887 flint_swebench: tolerate high-UID files in rootless podman pulls
1107909 flint_swebench: keep watchdog from inheriting the driver lock
f27f57b flint_swebench: r440 podman runner and infra-aware analysis
fb1e808 Add SWE-bench Verified harness for Flint (mini-swe-agent)
```

## Reproduction

Files: `benchmarks/flint_swebench/` (r440 copies in `r440/`, installed on
r440 as `~/flint-swebench/{env.sh,flint.yaml,conf/,scripts/,adapter/}`).

1. Serve Flint (c4130, 2x V100) with the serve args under Method. The runner expects
   `$FLINT_API_BASE/models` to list `flint`.
2. Environment (r440):

   ```bash
   . ~/flint-swebench/env.sh
   cd $R
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python \
     mini-swe-agent==2.4.6 litellm==1.106.6 swebench==5.0.2
   ```

3. Adapter tests:

   ```bash
   cd $R && .venv/bin/python -m pytest adapter -q
   ```

4. Gold sanity eval:

   ```bash
   cd $R/gold && .venv/bin/python -m swebench.harness.run_evaluation \
     -d SWE-bench/SWE-bench_Verified -s test -p gold \
     -i astropy__astropy-12907 --max_workers 1 -id gold_r440
   ```

5. Full run (detaches; log in `runs/driver.log`; resumable):

   ```bash
   $R/scripts/run_all.sh $R/all_ids_shuffled.txt 1
   ```

   Single instance: `$R/scripts/run_one.sh <INSTANCE_ID> 1`. The agent
   command inside `run_one.sh` is:

   ```bash
   PYTHONPATH=$R/adapter timeout 6h nice -n 19 ionice -c3 \
     $R/.venv/bin/mini-extra swebench --subset verified --split test \
     --filter "^${ID}\$" -c swebench.yaml -c $R/flint.yaml \
     -c "model.model_kwargs.api_base=$FLINT_API_BASE" \
     -w 1 -o $R/runs/att1/agent/$ID
   ```

   and the evaluation:

   ```bash
   nice -n 19 ionice -c3 $R/.venv/bin/python -m swebench.harness.run_evaluation \
     -d SWE-bench/SWE-bench_Verified -s test -i $ID \
     -p $R/runs/att1/agent/$ID/preds.json --max_workers 1 -id att1_$ID \
     --timeout 1800
   ```

6. Analysis:

   ```bash
   $R/.venv/bin/python $R/scripts/analyze.py --runs $R/runs
   $R/.venv/bin/python $R/scripts/analyze.py --runs $R/runs \
     --timeout-is-model --out $R/runs/analysis_timeout_is_model
   ```

## Raw data

On r440 at `~/flint-swebench/runs` (not committed):

- `att1/agent/<ID>/`: trajectories, `preds.json`, exit statuses;
  `att1/agent_retired/`: trajectories replaced by API re-runs.
- `att1/eval/`: `done_<ID>` markers, swebench reports and logs.
- `att1/log/`: per-instance agent, eval and runner logs.
- `driver.log`, `events.log`: runner and incident logs.
- `analysis.md`, `analysis.json`: output of `analyze.py`.
- `~/flint-swebench/gold/`: gold sanity eval.
