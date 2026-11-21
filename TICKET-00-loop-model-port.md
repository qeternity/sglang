# TICKET-00 — Land the Inkling-2-Mini loop model (port PR #51926)

- **Branch:** `zuma-inkling` (cut from `zuma-production` @ `3d5250f8a4`)
- **Model:** `thinkingmachines/Inkling-2-Mini` (`architectures: ["Inkling2LoopForConditionalGeneration"]`)
- **Hardware target:** 2× RTX 3090 (SM86, 24 GB each, PCIe x16 Gen4, no NVLink; operator runs patched drivers with P2P over PCIe)
- **Dependencies:** none (prerequisite for all other tickets)
- **Estimate:** 1–2 wks (cherry-pick applies cleanly) / +2 wks (reimplementation)
- **Status:** ✅ done (code port, 2026-11-21) — PR #51926 cherry-picked onto `zuma-inkling` as `bfc79a91de` ([Feat] inkling2: Support Inkling-2-Mini) + `19471d3355` ([Fix] doc/test). All inkling files byte-identical to the PR head; the 3 three-way-merged infra files (`cuda_graph_hook.py`, `model_config.py`, `logits_processor.py`) contain every inkling hunk — the only deltas vs the PR head are unrelated upstream drift (HYV4Flash arch, #51909 logprob fields, `flashinfer_megamoe` a2a entry). `py_compile` clean on all changed files; CPU-side unit tests green (test_inkling2.py 50 passed / 6 CUDA-skipped, test_model_overrides.py 144 passed). No separate "reference box" gate: the operator has no SM90/SM100 machine — the only GPU hardware is sauron (2× 3090 Ti). Numerical correctness is therefore gated in TICKET-06 on sauron itself: gsm8k vs the accuracy the PR authors published (94.01% GB300 / 93.93% H200), plus self-consistency (TP1-vs-TP2, determinism).

## Why this ticket exists

The Inkling-2-Mini checkpoint **cannot load at all** in this tree. Verified absent repo-wide
(`*.py`, `*.rs`, `*.cu`):

- No `Inkling2LoopForConditionalGeneration` class
- No `inkling2_loop` / `inkling2_loop_text` config classes (config registry in
  `python/sglang/srt/hf_transformers/common.py` only registers `Inkling2MMConfig` /
  `Inkling2ModelConfig` for model_type `inkling2`)
- No `loop_config` handling anywhere (the only `core_start`/`num_passes` hits are
  unrelated: `python/sglang/srt/lora/utils.py`, `python/sglang/multimodal_gen/.../flux3_video_vae.py`)

The cookbook page itself (`docs/cookbook/autoregressive/ThinkingMachines/Inkling-2-Mini.mdx`)
says: *"SGLang does not yet ship Inkling-2-Mini in a release — install it from pull request
**sgl-project/sglang#51926**"*. The flat Inkling-2 model (`Inkling2ForConditionalGeneration`,
model_type `inkling2`) **is** fully present (`python/sglang/srt/models/inkling2.py` +
`python/sglang/srt/models/inkling_common/` + `python/sglang/kernels/ops/*/inkling2_*`) — the
loop model is a distinct, later architecture.

The model now exists in this tree (this ticket). It cannot *run* on the 3090 until T1–T4
remove the SM90/SM100-only kernel gates and T5 wires the recipe; first full-model moment is
the T5→T6 deploy on sauron.

## The model (from the actual HF `config.json`)

| Field | Value |
|---|---|
| architectures | `Inkling2LoopForConditionalGeneration` |
| model_type | `inkling2_loop` (text: `inkling2_loop_text`) |
| hidden_size | 2304 |
| num_hidden_layers | 24 (physical) |
| vocab_size | 201024 |
| attention heads | 24 Q / 8 KV × 128 dim |
| d_rel / rel_extent | 16 / 1024 (relative attention) |
| sliding_window_size | 1024 (20 local layers + 4 full layers) |
| sconv_kernel_size | 4 (four short convs per layer: qkv/mlp sconv) |
| MoE | 128 routed experts, top-8, intermediate 1152, no shared experts |
| max_position_embeddings | 1048576 |
| **loop_config** | `{core_start: 6, core_end: 18, num_passes: 3, gate_rank: 64, is_varloop: false}` |
| mtp_config | 3 layers, hidden 2304, intermediate 9216, `target_layer_ids: [5, 23, 41, 48]`, `method: "dspark"`, `share_weights: true` |

### The 48-visit unroll (what the loop model must implement)

24 physical layers are expanded into **48 layer-visits per token**:

```
visits  0-5    = layers 0-5            (pre-loop)
visits  6-41   = layers 6-17 × 3 passes (core, 36 visits: 6-17, 18-29, 30-41)
visits 42-47   = layers 18-23          (post-loop)
visit 48       = final sentinel (post final-norm)
```

Sanity check against `mtp_config.target_layer_ids = [5, 23, 41, 48]`:
- 5 = last pre-loop layer (layer 5)
- 23 = mid-core (layer 11 on pass 2)
- 41 = layer 17 on pass 3 (last core pass)
- 48 = final sentinel

The MTP head's shared weights confirm the mapping: its checkpoint entries are
`model.layers.{0,1,5,17,23}.input_layernorm.weight` — i.e. it reuses target-layer state,
with `share_weights: true`.

## What the loop model must provide

1. **Config classes** — `inkling2_loop` / `inkling2_loop_text` registered in
   `python/sglang/srt/hf_transformers/common.py`; `loop_config` parsed
   (core_start, core_end, num_passes, gate_rank, is_varloop).
2. **Model class** — `Inkling2LoopForConditionalGeneration` (new file, e.g.
   `python/sglang/srt/models/inkling2_loop.py`), reusing the shared Inkling-2 blocks from
   `python/sglang/srt/models/inkling_common/` (attention, MoE, sconv, attn-res) exactly as
   the flat model does, but driving the 48-visit schedule.
3. **Loop state semantics** — per-pass vs shared state for the core layers' KV cache, sconv
   state (`Inkling2SconvState`), and attention-residual state (`Inkling2AttnRes`, expanded to
   the 48-visit space with `n_layers_per_attn_res_block` dividing 48). Server args referenced
   by the cookbook recipe (must exist): `--inkling2-loop-share-cache` (default on per recipe),
   `--inkling2-loop-num-passes` (3).
4. **MTP feature taps** at visits [5, 23, 41, 48] for the DSpark/MTP head
   (`python/sglang/srt/speculative/inkling2_mtp_worker.py` expects `Inkling2MTPConfig` —
   `dspark_components/dspark_config.py:244` already imports `Inkling2MMConfig,
   Inkling2MTPConfig` from `configs.inkling2`; the loop config must plug into the same
   `is_dspark()` selection at `srt/speculative/spec_info.py:341`).
5. **Radix-cache / prefix-cache interaction** — loop state keyed by token position (per-pass
   pools when `--inkling2-loop-share-cache` is off, shared pool when on); prefix cache must
   invalidate/reuse loop state consistently.
6. **Model registry** — `Inkling2LoopForConditionalGeneration` mapped in
   `srt/models.py` / `srt/model_loader` so AutoModel resolution finds it.

## Steps

1. Obtain the PR diff: `git fetch https://github.com/sgl-project/sglang.git pull/51926/head`
   (or download the diff from GitHub). Inspect commit list and touched files.
2. Assess conflict surface against `zuma-inkling`. Known risk: the kernel tree was reorganized
   into `python/sglang/kernels/ops/{gemm,moe,attention,layernorm,communication,speculative}/`;
   if the PR predates/straddles that reorg, expect import-path fixes on top of the content.
3. Apply (merge or manual port). Fix imports to the current `kernels.ops.*` layout.
4. Register config + model + server args (`--inkling2-loop-*`).
5. Add/extend unit tests mirroring `test/registered/unit/models/test_inkling2.py`
   (structure test with a Mini-shaped config: 24 layers, loop_config, 128 experts, mtp
   `target_layer_ids [5,23,41,48]`).
6. **Gate on SM90/SM100 hardware:** load `thinkingmachines/Inkling-2-Mini` with the existing
   FA4 + native-kernel path and produce 1-token lossless generation vs a reference
   (the PR's own tests are the reference; if absent, 1-token logits vs an independent
   implementation or the model authors' reference server).

## Acceptance criteria

- `sglang serve --model-path thinkingmachines/Inkling-2-Mini` starts on an SM90 or SM100 box
  with zero Ampere-specific changes.
- 1-token lossless generation matches reference logits (bf16 noise).
- `--inkling2-loop-share-cache` on/off both run; `--inkling2-loop-num-passes 3` honored.
- Structure/weight-load unit test passes (all 24 physical layers loaded, loop schedule
  verified at 48 visits, MTP head shares target weights).

## Risks

- PR cherry-pick conflicts with the `kernels/ops/` reorg → time risk; fallback is a
  reimplementation (~2 extra weeks) using the flat `inkling2.py` as the structural template.
- Loop-state semantics (shared vs per-pass KV, sconv state) are the subtle part; the
  radix-cache interaction is the likely source of correctness bugs — T6's prefix-hit test
  depends on this being right.
- The cookbook's ROCm (MI35x) recipe implies a parallel ROCm implementation may exist in the
  radixark fork; that code is a second reference to cross-check against if cherry-pick is messy.
