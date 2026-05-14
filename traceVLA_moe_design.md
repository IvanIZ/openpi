# TraceVLA Both-MoE Variant — Architecture & Size-Reduction Plan

This document specifies a proposed third variant of TraceVLA in which **both**
the action expert and the trace expert use the K=5 hard-routed MoE FFN, while
the trace MoE is shrunk along its non-constrained dimensions so that the
combined parameter count stays manageable.

Variants in scope:

- `trace_vla_lora` (current) — action expert = dense `gemma_300m_lora`, trace
  expert = MoE `trace_moe_gemma_300m` (5 experts).
- `trace_vla_actionmoe_lora` (current) — action expert = MoE
  `trace_moe_gemma_300m` (5 experts), trace expert = dense `gemma_300m`.
- **`trace_vla_bothmoe_lora` (proposed)** — action expert = MoE
  `trace_moe_gemma_300m` (5 experts, unchanged from `actionmoe`), trace expert
  = **smaller** K=5 MoE (new variant defined here).

The number of MoE experts is **fixed at K=5** for both streams — it
corresponds to the five LIBERO skill atoms (PICKUP\_FROM / PLACE\_ON /
PLACE\_IN / OPEN / TURN\_ON), each routed by a one-hot
`hard_combine_weights[..., K]` tensor. K is not a knob.

---

## 1. Architecture recap (so the reduction plan makes sense)

### 1.1 The three streams and joint attention

`Pi0TraceVLAActionMoe` (and the proposed `bothmoe` extension) is a three-stream
Gemma trunk: a single `TraceModule`
([`gemmoe_trace.py`](src/openpi/models/gemmoe_trace.py)) whose `nn.scan`
stamps out **N identical layers**, each layer being a `TraceBlock` over all
three streams in lockstep.

```
stream 0 : PaliGemma 2B VLM     (dense FFN, width=2048, mlp_dim=16384)
stream 1 : action expert        (5-expert MoE, width=1024, mlp_dim=4096)  [unchanged]
stream 2 : trace expert         (5-expert MoE, width=?,    mlp_dim=?)     [SHRINK THIS]
```

At every layer the three streams co-attend in **one shared joint-attention
call** — each stream computes its own q/k/v with its own projection weights,
then all three streams' q/k/v are concatenated along the sequence axis and
fed through one shared softmax. After attention, each stream re-projects back
to its own width and runs its own per-stream FFN (or MoE).

This coupling has two structural consequences:

| Property | Shared across streams? |
|---|---|
| Number of layers (`depth`) | **YES** — `assert all(c.depth == c0.depth)` in [`gemmoe_trace.py:336`](src/openpi/models/gemmoe_trace.py#L336) |
| `head_dim`, `num_heads`, `num_kv_heads` | **YES** — `assert` in [`gemmoe_trace.py:136-138`](src/openpi/models/gemmoe_trace.py#L136-L138) |
| Attention softmax operation | **YES** — one joint softmax per layer |
| Attention projection weights (q / kv / out) | **NO** — per stream, sized to that stream's `width` |
| Pre-attn / pre-FFW RMSNorm | **NO** — per stream |
| FFN (or MoE) weights | **NO** — per stream |
| Per-stream I/O projections (`trace_in_proj`, `trace_out_proj`) | **NO** — per stream |

So the trace stream is free along `width` and `mlp_dim`, but locked to
`depth=18` and to the shared head shape.

### 1.2 What `HardMoeBlock` looks like inside a layer

`HardMoeBlock` ([`gemmoe_trace.py:220-248`](src/openpi/models/gemmoe_trace.py#L220-L248))
holds K independent SwiGLU expert FFNs, named `expert_0 … expert_{K-1}`, with
no shared expert. Each expert is a `GemmoeBlockSparseTop2MLP`:

```
input  (dim = width)
  │
  ├─ w1: width → mlp_dim     ──┐
  │                            │  SwiGLU: silu(w1 x) * (w3 x)
  ├─ w3: width → mlp_dim     ──┘
  │
  ▼
hidden (dim = mlp_dim)
  │
  ▼  w2: mlp_dim → width
output (dim = width)
```

Per-expert parameter count: **3 × width × mlp\_dim** (w1, w3, w2 are all the
same shape `width × mlp_dim`).

A `HardMoeBlock` with K experts therefore costs **K × 3 × width × mlp\_dim**
parameters per layer. The router is parameter-free (the one-hot
`hard_combine_weights` is supplied by the caller from the skill id).

---

## 2. Where the size lives (current trace MoE, `trace_moe_gemma_300m`)

Trace stream config today:

```
width = 1024, depth = 18, mlp_dim = 4096,
num_heads = 8, num_kv_heads = 1, head_dim = 256,
num_local_experts = K = 5
```

Parameter accounting **per trace-stream layer**:

| Component | Formula | Count |
|---|---|---:|
| q\_einsum | `num_heads × width × head_dim` | 2.10 M |
| kv\_einsum | `2 × num_kv_heads × width × head_dim` | 0.52 M |
| attn\_vec\_einsum (out proj) | `num_heads × head_dim × width` | 2.10 M |
| **Attention subtotal / layer** | | **≈ 4.72 M** |
| One expert FFN (SwiGLU) | `3 × width × mlp_dim` | 12.58 M |
| **MoE subtotal / layer (K=5)** | `K × 3 × width × mlp_dim` | **≈ 62.92 M** |
| RMSNorms (pre-attn + pre-FFW + AdaRMS modulation) | small, scales with `width` | ≪ 1 M |
| **Per-layer trace stream** | | **≈ 67.6 M** |
| **Whole trace stream (× depth=18)** | | **≈ 1.22 B** |

The MoE FFN block alone accounts for **~93% of the trace-stream parameters**
(≈ 1.13 B of 1.22 B). Attention is a rounding error in comparison.

For context the same-shape action MoE in `trace_vla_actionmoe_lora` is also
≈ 1.22 B; PaliGemma-2B (stream 0) is ≈ 2.0 B. If we used
`trace_moe_gemma_300m` for both streams in the proposed `bothmoe` variant, the
non-VLM total would be ≈ 2.44 B on top of the 2.0 B VLM — far heavier than
either current variant. **The trace MoE has to shrink.**

---

## 3. What we can and cannot change

Constants (cannot change, structural):

- **`depth = 18`** — locked by joint-attention to PaliGemma's 18 layers.
- **`num_heads = 8, num_kv_heads = 1, head_dim = 256`** — locked by joint
  attention (all streams' q/k/v share a softmax, so head shape must match).
- **`K = 5`** — locked by the skill-routing semantics. The hard-routing
  signal is a one-hot of the 5 LIBERO skill atoms; reducing K would change
  the model's skill specialization story and require a new routing scheme.

Free knobs on the trace stream (independent of the action stream and the VLM):

- **`width`** — the trace stream's residual / token-embedding dim. Affects
  attention projection sizes **linearly** and the FFN matrices
  **linearly** in one of two dims. The trace stream has its own
  `trace_in_proj` (2 → width) and `trace_out_proj` (width → 2) at
  [`pi0_trace_vla_actionmoe.py:180-183`](src/openpi/models/pi0_trace_vla_actionmoe.py#L180-L183),
  so changing `width` is a pure local change.
- **`mlp_dim`** — the FFN hidden dim, i.e. the "fat middle" of each expert.
  Affects per-expert FFN size **linearly**. Has no cross-stream effect.

Master formula for the trace-stream MoE FFN parameter count:

```
trace_MoE_FFN_params = depth × K × 3 × width × mlp_dim
                     = 18 × 5 × 3 × width × mlp_dim
                     = 270 × width × mlp_dim
```

Current value: 270 × 1024 × 4096 ≈ **1.13 B** (matches the table above, up to
rounding for attention + norms).

Trace-stream attention params:

```
attn_params = depth × (q + kv + out)
            = 18 × (heads × width × head_dim
                     + 2 × kv_heads × width × head_dim
                     + heads × head_dim × width)
            = 18 × width × head_dim × (2 × heads + 2 × kv_heads)
            = 18 × width × 256 × 18
            = 82944 × width
```

So attention is ~85 M when width = 1024 and shrinks linearly with width.

---

## 4. Proposed reductions (K = 5 fixed)

K stays at 5 throughout. The recipes below shrink only `width` and/or
`mlp_dim`. All counts include attention; norms are negligible (≪ 1 M total).

| Recipe | width | mlp_dim | Per-expert FFN | Per-layer (FFN + attn) | Trace stream total (× 18) | Reduction vs current |
|---|---:|---:|---:|---:|---:|---:|
| **Current** (`trace_moe_gemma_300m`) | 1024 | 4096 | 12.58 M | 67.6 M | **≈ 1.22 B** | — |
| **A. Half mlp_dim** | 1024 | 2048 | 6.29 M | 36.2 M | **≈ 651 M** | −47% |
| **B. Quarter mlp_dim** | 1024 | 1024 | 3.14 M | 20.4 M | **≈ 367 M** | −70% |
| **C. Half width, half mlp_dim** ✅ | 512 | 2048 | 3.15 M | 18.1 M | **≈ 325 M** | −73% |
| **D. Half width, quarter mlp_dim** | 512 | 1024 | 1.57 M | 10.2 M | **≈ 184 M** | −85% |
| **E. Half width, eighth mlp_dim** | 512 | 512 | 0.79 M | 6.3 M | **≈ 113 M** | −91% |

Detailed per-layer breakdowns:

**Recipe A** — `width=1024, mlp_dim=2048, K=5`
- q + kv + out: 2.10 M + 0.52 M + 2.10 M = 4.72 M
- One expert: 3 × 1024 × 2048 = 6.29 M; × 5 experts = 31.46 M
- Per layer: 36.2 M → ×18 = **651 M**

**Recipe B** — `width=1024, mlp_dim=1024, K=5`
- Attn: 4.72 M (unchanged)
- One expert: 3 × 1024 × 1024 = 3.14 M; × 5 = 15.73 M
- Per layer: 20.4 M → ×18 = **367 M**

**Recipe C (recommended) — `width=512, mlp_dim=2048, K=5`**
- q: 8 × 512 × 256 = 1.05 M; kv: 2 × 1 × 512 × 256 = 0.26 M; out: 8 × 256 × 512 = 1.05 M → attn total 2.36 M
- One expert: 3 × 512 × 2048 = 3.15 M; × 5 = 15.73 M
- Per layer: 18.1 M → ×18 = **325 M**

**Recipe D** — `width=512, mlp_dim=1024, K=5`
- Attn: 2.36 M (unchanged from C)
- One expert: 3 × 512 × 1024 = 1.57 M; × 5 = 7.86 M
- Per layer: 10.2 M → ×18 = **184 M**

**Recipe E** — `width=512, mlp_dim=512, K=5`
- Attn: 2.36 M (unchanged)
- One expert: 3 × 512 × 512 = 0.79 M; × 5 = 3.93 M
- Per layer: 6.3 M → ×18 = **113 M**

### 4.1 Why aggressive shrinking is plausible

The trace head's prediction target is much smaller than the action head's:

| Stream | What it predicts | Output size |
|---|---|---:|
| Action | full action chunk (10 steps × 32 dims) | 320 numbers |
| Trace | future waypoints (20 × 2 pixel coords) | 40 numbers |

The trace stream also gets dense, low-noise conditioning:

- AdaRMS conditioning on the Fourier-encoded semantic target point.
- Hard inpainting clamp on row 0 to the current EE pixel.
- Optional second clamp on an appended row to the target point itself
  (`append_target_anchor=True`).

Given the trace task is geometrically simpler and is given strong anchors at
both endpoints, the trace MoE should not need anywhere near the same capacity
as the action MoE.

### 4.2 Recommended starting point: Recipe C

Recipe C (width = 512, mlp_dim = 2048, K = 5) reduces the trace MoE from
≈ 1.22 B to **≈ 325 M** — roughly the size of a single dense `gemma_300m`
(the original action expert in `trace_vla_lora`). This is the "natural" size
for an expert stream in this codebase and the same size as the previous
trace expert in the original `trace_vla_lora` variant before the MoE was
introduced.

Trade-off summary for Recipe C:

- **Size:** −73% on the trace MoE; total non-VLM params with action MoE
  unchanged: ≈ 1.22 B (action) + ≈ 0.33 B (trace) = **≈ 1.55 B**, versus
  ≈ 2.44 B if both streams used `trace_moe_gemma_300m`.
- **Capacity per expert:** matches a single SwiGLU FFN at half-width and
  half-hidden — small but well-shaped to the trace task's 40-number output.
- **Routing & semantics:** unchanged (still 5 hard-routed skill experts).
- **Joint attention:** unchanged (head shape and depth preserved).

If Recipe C trains stably and matches `trace_vla_actionmoe_lora` on Spatial
while recovering the trace-MoE benefit on Goal, the next move is Recipe D
(≈ 184 M, −85%) to push the efficiency frontier further. If it does not
match, Recipe B (full width, quarter mlp_dim, ≈ 367 M) is the safer fallback
because it leaves the residual-stream dim untouched.

---

## 5. Implementation plan

### 5.1 New `Config` variant in `gemmoe_trace.py`

Add a new variant to the `Variant` literal and `get_trace_config` switch in
[`gemmoe_trace.py`](src/openpi/models/gemmoe_trace.py):

```python
Variant = Literal[
    "trace_moe_dummy",
    "trace_moe_gemma_300m",
    "trace_moe_gemma_300m_lora",
    "trace_moe_small",      # <-- new (Recipe C)
]

def get_trace_config(variant: Variant) -> Config:
    ...
    if variant == "trace_moe_small":
        return Config(
            width=512,
            depth=18,
            mlp_dim=2048,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=5,
            num_experts_per_tok=1,
        )
```

Notes:

- `depth=18`, `num_heads=8`, `num_kv_heads=1`, `head_dim=256` are **not
  optional** — they must match the VLM and action stream to satisfy the
  joint-attention asserts.
- `num_local_experts=5` is fixed by the skill semantics.
- `num_experts_per_tok` is unused by `HardMoeBlock` (routing is external) but
  kept for `Config` shape compatibility.

If we later want to add an even smaller variant (Recipe D / E), define
`trace_moe_xsmall` etc. analogously.

### 5.2 New model: `Pi0TraceVLABothMoe`

The simplest path is a new top-level model that mirrors
`Pi0TraceVLAActionMoe` but accepts an MoE config for **both** streams 1 and
2. The dispatch inside `TraceBlock` ([`gemmoe_trace.py:295-308`](src/openpi/models/gemmoe_trace.py#L295-L308))
already picks `HardMoeBlock` for any stream whose `num_local_experts > 1`,
so the per-layer FFN selection is automatic — no change needed in the trunk.

Required code additions (new files, no edits to existing model code beyond
adding the new variant):

1. **`src/openpi/models/pi0_trace_vla_bothmoe_config.py`** — `dataclass`
   mirroring `Pi0TraceVLAActionMoeConfig` but with:
   - `action_expert_variant: _gemmoe_trace.Variant = "trace_moe_gemma_300m"`
   - `trace_expert_variant: _gemmoe_trace.Variant = "trace_moe_small"`
   - `num_action_experts: int = 5` and `num_trace_experts: int = 5`.
2. **`src/openpi/models/pi0_trace_vla_bothmoe.py`** — mirror of
   `pi0_trace_vla_actionmoe.py` with:
   - `trace_expert_config = _gemma_trace.get_trace_config(config.trace_expert_variant)`
     (instead of `_gemma.get_config`).
   - `_forward_planning` and `_forward_execution` pass the **same**
     `hard_combine_weights` tensor (one-hot over the 5 skill atoms) to both
     streams — since K=5 on both, a single per-token one-hot works for
     both dispatches.
3. **`src/openpi/training/config.py`** — new `LeRobotTraceVLABothMoeDataConfig`
   (a 3-line clone of `LeRobotTraceVLAActionMoeDataConfig` that type-checks
   for `Pi0TraceVLABothMoeConfig`), plus two new `TrainConfig` entries:
   `trace_vla_bothmoe` (full FT) and `trace_vla_bothmoe_lora` (paligemma
   LoRA, both experts full FT). The freeze filter is identical to
   `actionmoe`'s (stream-1 and stream-2 subtrees are excluded from the
   paligemma freeze).

### 5.3 Inference path

No changes needed in [`libero_traceVLA_test.py`](../../py_script/libero_traceVLA_test.py):
the inference script dispatches on `--model-name` and uses the generic
`TraceVLAPolicy` wrapper, which only requires `sample_actions_and_completion`,
`sample_trace`, `predict_completion` — all three are present in
`Pi0TraceVLAActionMoe` and will be carried over verbatim into
`Pi0TraceVLABothMoe`. To evaluate, just pass
`--model-name trace_vla_bothmoe_lora --checkpoint-dir <new ckpt>`.

### 5.4 Training cost (rough)

Forward-pass FLOPs scale ~linearly with the trace-stream FFN params. Recipe C
reduces trace-MoE FFN params by 73%, so per-step training cost should drop
modestly versus a hypothetical "both at full size" baseline. Compared to the
current `trace_vla_actionmoe_lora` (which has a dense ~300 M trace stream),
Recipe C's trace MoE is approximately the same FFN cost (the 5 experts add
work, but each expert is 4× smaller than the dense version), so training
throughput should be in the same ballpark as `actionmoe_lora` rather than
strictly slower.

---

## 6. Summary

- Trace MoE size is dominated by **`depth × K × 3 × width × mlp_dim`**.
- `depth`, head shape, and `K=5` are fixed by the architecture and the
  skill-routing design.
- The only available knobs are **`width`** and **`mlp_dim`** on the trace
  stream — both are free because the trace stream has its own I/O projections
  and its FFN is independent of the other streams.
- **Recipe C** (width = 512, mlp_dim = 2048, K = 5) is the recommended
  starting point: reduces the trace MoE from **≈ 1.22 B → ≈ 325 M (−73%)**
  while preserving all routing semantics and joint-attention constraints.
- Recipe B (≈ 367 M, full width) is the safer fallback if changing residual
  width hurts trace prediction; Recipe D (≈ 184 M) is the more aggressive
  next step if Recipe C trains well.
