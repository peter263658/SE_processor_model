# Low-Power Phase Status — 2026-05-19 → 2026-05-20 (FINAL)

Final status across the full QAT cycle. All Phase A/B/C/D/F numbers are in;
production stack identified and composite-verified in §13/§13a. Earlier
sections (§§1–9) preserve the chronological narrative including the in-flight
snapshots and the packet local_idx bug that was fixed mid-cycle; later
sections (§§9a–9d, §13, §13a) carry the post-fix and final numbers.

Base checkpoint: `Checkpoints/p4_qk_m4_qat_best.ckpt` → symlinks to
`13_aggressive_algo_qat/outputs/p4_lmul/lmul_qk_m4_2026-05-18_14-53-28/.../epoch=2-step=2904.ckpt`
(only 3 epochs of P4 QAT — a **weak warm-start**, not a hard upper bound: Phase A
already shows QAT can exceed the base ckpt's metric, e.g. a2 0.8012 → 0.8062;
Phase D d1 30-epoch QAT exceeded by +0.043 at sanity scale).

Gates exercised:
- 20-pair VCTK paper-style sanity: **MBSTOI ≥ 0.8031** (coarse smell test).
- Full VCTK paper-style: **MBSTOI ≥ 0.8302** (spec gate, decisive).
- Full TIMIT paper-style: **MBSTOI ≥ 0.7869** (spec gate, decisive).

The 20-pair sanity gate is about 0.027 looser than the full VCTK spec gate, so
"sanity pass" does not imply "spec pass" — see §8b and §9b for the empirical
gap. All "pass/fail" verdicts below should be read against the specific gate
listed for that row.

---

## 1. Code review (Task 1)

Inspected diffs against HEAD:

| File | Lines changed | Summary |
|---|---|---|
| `config/config.yaml` | +31 | adds `attention_datapath.lmul_bit_exact`, `attention_datapath.prob_packet.*`, and `low_power.{prelu_pow2, bn_affine, mha_projection}` blocks |
| `DCNN/models/model.py` | +90 / −13 | three `_low_power_*` config readers; `low_power` plumbed through `DCNN → Encoder/Decoder/MultiAttnBlock`; `nn.Linear` for input/output proj swapped for `GroupedLinear`; `ComplexMultiheadAttention(mha_projection=…)` |
| `DCNN/utils/complexPyTorch/complexLayers.py` | +341 / −16 | `_pow2_quantize_scale`/`_quantize_bn_scale`; `ComplexPReLU(pow2_slope, min_exp, max_exp)`; `_linear_group_mask`/`_mha_in_proj_group_mask`/`GroupedLinear`; `NaiveComplexBatchNorm{1,2}d(scale_mode, tie_real_imag_scale, …)`; `_packetize_q08_attention_probs`/`_q08_packet_fixed_matmul`; `_lmul_multiply_bit_exact`; `_mha_attention_forward(mha_projection=…)` |
| `DCNN/utils/mixed_precision_qat.py` | +1 | OR `projection_grouped_active` into `m.use_custom_forward` trigger |

No blockers. Minor observations:

1. Grouped MHA mask zeroes connectivity but does **not** shrink parameter count. Fine for accuracy screening; the MAC savings live in `analyze_low_power_arch.py`.
2. `_round_ste_or_plain` uses `globals().get("_ste_round")` lookup at forward time — works (module is fully loaded by the first call), but uglier than a direct reference.
3. `b0_prob_q08_grid` is a single override (`prob_quant=q0.8`), not an actual grid. Name is misleading.
4. `_lmul_multiply_bit_exact` is new and untested at scale; c0–c3 are the proving ground.
5. Base ckpt is only `epoch=2-step=2904` of P4 QAT — a weak warm-start that depresses sanity numbers for *unrecovered* variants. Not a ceiling: QAT can recover above it (confirmed by Phase A a2).

---

## 2. Smoke checks (Task 2)

```
py_compile  complexLayers.py model.py mixed_precision_qat.py analyze_low_power_arch.py  → OK
bash -n     run_low_power_phase_{a, bc_packet_lmul, d_mha_grouping, f_analysis}.sh        → OK
```

---

## 3. Patches applied during runs

`TRAIN_BATCH_SIZE` env var added to all three phase scripts (default 8). Root cause: original `training.batch_size=32` from `config/config.yaml` + L-Mul `_lmul_matmul`'s `(B, H, T, T, D)` materialization → CUDA OOM at the first training step on GPU 1 (81 GB available, allocation > 4.86 GiB on top of pre-existing usage). Eval at B=8 is fine; only training-time activations + gradients tip it over.

Diff (representative — same edit on `run_low_power_phase_{a, bc_packet_lmul, d_mha_grouping}.sh`):

```bash
BATCH_SIZE="${BATCH_SIZE:-8}"
+TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
LIMIT_PAIRS="${LIMIT_PAIRS:-20}"
...
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n bcctn python -u train.py \
    training.train_checkpoint_path="$BASE_CKPT" \
    training.n_epochs="$EPOCHS" \
    training.learning_rate="$LR" \
+   training.batch_size="$TRAIN_BATCH_SIZE" \
    "${common_overrides[@]}" \
    ...
```

Wall-time impact: B=32 → B=8 quadruples step count per epoch (968 → 3872 steps). Each variant takes ~2h50m instead of ~45 min if B=32 had fit. B=16 likely also fits and would halve the runtime — worth testing in a future pass.

---

## 4. Phase A sanity (Task 3 — TRAIN=0, base ckpt only)

20-pair VCTK paper-style averages.

| Variant | MBSTOI | SegSNR Δ | ILD (dB) | IPD (deg) | Gate |
|---|---|---|---|---|---|
| a0_prelu_pow2 | **0.8104** | +2.47 | 3.33 | 17.28 | ✅ pass |
| a2_bn_pow2 (untied, pow2 scale) | 0.8012 | +0.92 | 3.57 | 17.73 | ❌ −0.002 below |
| a3_bn_tie_pow2 (tied R/I, pow2 scale) | **0.8078** | −0.52 | 3.65 | 18.21 | ✅ MBSTOI only |
| noisy baseline | 0.7743 | — | 3.66 | 16.57 | — |

**Caveat**: only the 20-pair MBSTOI sanity gate is exercised here. a3's SegSNR
improvement is **negative** (−0.52 dB) and IPD error worsens (16.57 → 18.21 deg),
i.e. spatial cues degrade noticeably. Calling a3 a "pass" applies *only* to the
MBSTOI metric.

---

## 5. Phase A QAT (Task 4 — 10 epoch warmstart from base, B=8, LR=2e-4)

Wall time: 8h 47m for three variants on a single GPU. Picked best ckpt by val loss.

| Variant | Base | Finetuned | Δ | Best ckpt |
|---|---|---|---|---|
| a1_prelu_pow2_qat | 0.8104 | **0.8092** | −0.001 | epoch=5-step=23232 |
| a2_bn_pow2 | 0.8012 | **0.8062** | **+0.005** | epoch=4-step=19360 |
| a3_bn_tie_pow2 | 0.8078 | **0.8070** | −0.001 | epoch=9-step=38720 |

**Architecture verdict — scoped to 20-pair MBSTOI sanity only**:
- BN-affine pow2 quantization is approximate but MBSTOI-recoverable with short QAT (a2 evidence).
- Tied R/I scaling preserves MBSTOI; **full-gate ILD/IPD/SegSNR validation is still pending** before declaring it "free".
- PReLU pow2 does not need QAT for MBSTOI. Spatial-cue impact at full eval is not yet confirmed.

---

## 6. Phase D sanity (Task 7 — TRAIN=0, 20-pair VCTK)

Five grouped MHA projection variants, all evaluated against the **base ckpt** with no further training.

| Variant | MBSTOI | Δ vs 0.8104 | ILD (dB) | IPD (deg) | Total MACs | vs dense |
|---|---|---|---|---|---|---|
| d0_qkv_g2 | **0.7951** | −0.015 | 3.77 | 17.59 | 1.20 M | −14% |
| d1_qkv_g4 | 0.7619 | −0.049 | 4.97 | 18.35 | 1.10 M | −21% |
| d2_qkv_out_g4 | 0.6992 | −0.111 | 6.55 | 19.89 | 1.00 M | −28% |
| d3_io_g4 | 0.6927 | −0.118 | 6.87 | 20.30 | 1.00 M | −28% |
| d4_all_g4 | 0.6819 | −0.129 | 6.92 | 20.91 | 0.61 M | **−57%** |

Reading: zeroing block-diagonal connectivity without retraining collapses MBSTOI by 0.015 (best case, g=2 on QKV only) up to 0.13 (all four projections g=4). All variants are **below** the 0.8031 gate. d0 is closest to recoverable; d4 is far.

---

## 7. Phase F lane-sweep analysis (Task 9 — CPU only, runs in seconds)

`scripts/analyze_low_power_arch.py` accounting under 5 ms hop (= 80 samples @ 16 kHz, default hop 80), assuming MAC utilization 0.85. "Min lanes meeting 5 ms" = smallest lane count from `{64, 96, 128, 256}` whose duty cycle ≤ 1.0 at the listed clock.

| Config | Total MACs | At 5 MHz | At 10 MHz | At 20 MHz | JSON |
|---|---|---|---|---|---|
| f0 dense | 1.39 M | **96 lanes**, duty 0.68 (64 lanes overflows @ duty 1.02) | 64 lanes, duty 0.51 | 64 lanes, duty 0.26 | `f0_dense_64_128_256.json` |
| f3 qkv g=4 | 1.10 M | 64 lanes, duty **0.81** | 64 lanes, duty 0.40 | 64 lanes, duty 0.20 | `f3_qkv_g4_64_128_256.json` |
| f3 all-attn g=4 | 0.61 M | 64 lanes, duty **0.45** | 64 lanes, duty 0.22 | 64 lanes, duty 0.11 | `f3_all_attn_g4_64_128_256.json` |

Sweet spot depends on which Phase D variant survives QAT:
- if **d0 (qkv g=2)** is the only recoverable one → 64 lanes @ 10 MHz, ~−14% MAC, modest power win
- if **d2/d3 (qkv+out or input+output g=4)** also recover → 64 lanes @ 5 MHz, ~−28% MAC, halved clock
- if **d4 (all g=4)** recovers (sanity says unlikely) → 64 lanes @ 5 MHz, **−57% MAC**, 45% duty — best power, biggest accuracy risk

---

## 8. Phase D QAT (Task 8 — COMPLETE)

Both variants passed the 20-pair MBSTOI gate after 10-epoch QAT:

| Variant | Base | Finetuned | Δ | ILD (dB) | IPD (deg) | SegSNR Δ |
|---|---|---|---|---|---|---|
| d0_qkv_g2 | 0.7951 | **0.8050** | +0.010 | 3.37 | 17.22 | +2.48 |
| d1_qkv_g4 | 0.7619 | **0.8045** | **+0.043** | 3.52 | 17.20 | +2.48 |

Strongly validates the "weak warm-start, not ceiling" framing from §0: d1 closed
a 0.043 MBSTOI gap in 10 epoch. Both variants land within the dense-baseline
ILD/IPD envelope (a0: 3.33 / 17.28). Spatial cues hold at the 20-pair sanity
scale; full-set ILD/IPD/SegSNR validation still pending.

Result dirs:
- `results/low_power_phase_d_qat/d0_qkv_g2_finetuned_w20/`
- `results/low_power_phase_d_qat/d1_qkv_g4_finetuned_w20/`

Implication for operating point (cross-ref §7):
- d0 (qkv g=2) — 1.20 M MAC (−14% vs dense) — 64 lanes @ 10 MHz, duty 51%
- d1 (qkv g=4) — 1.10 M MAC (−21% vs dense) — 64 lanes @ 5 MHz, duty 81% — preferred if duty headroom is tight
- d2 (qkv+out g=4) not yet recovered (sanity gap 0.111); could revisit at 10 or 30 epoch given d1 closed 0.043

---

## 8b. Phase D full VCTK eval (Task 11 — COMPLETE)

20-pair sanity is too loose; ran full VCTK paper-style on the d0/d1 finetuned
ckpts to test against the spec gate (MBSTOI ≥ 0.8302):

| Variant | MBSTOI | vs 0.8302 gate | SegSNR Δ | ILD (dB) | IPD (deg) |
|---|---|---|---|---|---|
| d0_qkv_g2 (epoch=2-step=11616) | **0.8295** | **−0.0007 FAIL** | +2.50 | 3.33 | 16.05 |
| d1_qkv_g4 (epoch=5-step=23232) | **0.8300** | **−0.0002 FAIL** | +2.48 | 3.41 | 16.07 |
| noisy baseline (full set) | 0.7861 | — | — | 3.71 | 15.43 |

**Both variants miss the full-VCTK gate by ≤ 0.001** — essentially within metric
noise of the gate, but technically below. Notable:

- d1 (g=4) actually beats d0 (g=2) by 0.0005 on MBSTOI. The −21% MAC reduction
  from g=4 carries no measurable quality penalty over g=2. Strong signal that
  the MAC-efficient choice is the right hardware pick.
- ILD/IPD are tight against the dense baseline; spatial cues hold at full scale.
- SegSNR improvement ~+2.5 dB on both, healthy.

**Confirms the sanity-vs-full divergence**: 20-pair sanity reported +0.0014 to
+0.0019 margin over its 0.8031 gate, but full eval is −0.0002 to −0.0007 under
its 0.8302 gate. The sanity screen is roughly 0.027 MBSTOI looser than the spec
gate — keep this offset in mind for all future "passes sanity" claims.

Result dirs:
- `results/low_power_phase_d_full/d0_qkv_g2_full_vctk/`
- `results/low_power_phase_d_full/d1_qkv_g4_full_vctk/`

Logs: `/tmp/d0_full_vctk.log`, `/tmp/d1_full_vctk.log`.

Recommended next step: per TODO fallback ("if g=4 fails, extend g=2 to 30 epoch"),
extend d0 (or d1, since it's the better-performing) to 30-epoch QAT to close
the ≤0.001 gap. Wall time ≈ 9h at B=8 on one GPU.

---

## 9. Phase B/C sanity (Task 5 — COMPLETE; packet variants have a real bug)

8 variants, TRAIN=0, 20-pair VCTK paper-style:

| Variant | MBSTOI | ILD (dB) | IPD (deg) | Status |
|---|---|---|---|---|
| b0_prob_q08_grid | 0.8104 | 3.29 | 17.11 | ✅ clean |
| b1_packet_dequant | 0.8105 | 3.29 | 17.14 | ⚠️ biased (file failures) |
| b2_packet_fixed | 0.8105 | 3.29 | 17.14 | ⚠️ biased (file failures) |
| b3_packet_lmul | 0.8105 | 3.29 | 17.14 | ⚠️ biased (file failures) |
| c0_lmul_qk_bitexact | 0.8105 | 3.29 | 17.14 | ✅ clean |
| c1_lmul_scorev_bitexact | 0.8099 | 3.29 | 17.13 | ✅ clean |
| c2_lmul_attn_bitexact | 0.8101 | 3.29 | 17.13 | ✅ clean |
| c3_packet_lmul_bitexact | — | — | — | ❌ **total failure** |

**Clean results (no packet)**: b0 and c0/c1/c2 all sit within 0.0005 of the
dense baseline (0.8104). **Bit-exact L-Mul (`lmul_bit_exact=true`) passes the
20-pair sanity scope cleanly** at MBSTOI 0.8099–0.8105, validating the new
`_lmul_multiply_bit_exact` simulator for qk/score_v individually and combined.

**Packet bug**: b1/b2/b3/c3 hit
`q0.8 probability packet local_idx overflow; expected causal window <= 32 keys`
in `_packetize_q08_attention_probs` (complexLayers.py). 1280 file failures
total in the sanity matrix; c3 lost every file, b1/b2/b3 lost a fraction and
the reported MBSTOI 0.8105 is over the *surviving* files only — biased.

Root cause: the packet code does `torch.topk(probs.clamp(min=0), k=top_k)` over
the full key axis. Causal-mask zeros tie at 0.0; `torch.topk` may pick those
indices, which sit outside the local sliding window `[row − (2**local_idx_bits − 1), row]`.
The `local_idx = row − topk_idx` then violates the 5-bit field constraint
(`>= 32`) and the validation aborts the row. Triggered at early query rows
(row < 31) and whenever ≥1 zero-tied key beats real picks.

The fix must be window-aware (restrict `topk` to the causal sliding window),
not a config-level disable of `validate_local_idx`. A detailed prompt for the
external fixer (Codex) was drafted separately, covering: window-mask
construction with `2**local_idx_bits` width, `-inf` masking before `topk`,
boundary-row safe-index + `scatter_add` reconstruction, the bit-identity
check on c0/c1/c2 (must remain 0.8105 / 0.8099 / 0.8101 ± 1e-4 post-fix in
the c0, c1, c2 order), and a unit test for boundary rows.

Until that fix lands and b1/b2/b3/c3 re-run cleanly, **no Phase B/C QAT should
be launched** — biased base numbers would mislead the recovery decision.

Result dir: `results/low_power_phase_bc_sanity/` (each `*_base_w20/vctk/` is
present but b1/b2/b3/c3 entries are partial; c3's is essentially empty).
Log: `/tmp/phase_bc_sanity.log` (search for "Failed to process" to enumerate
the per-file failures).

### 9a. Phase B/C sanity v2 (post-fix, Codex 2026-05-20)

Codex landed a window-aware packet topk fix plus a windowed/chunked QK L-Mul
path. Scope of patch (complexLayers.py +515 lines; config.yaml +34 lines;
tests/test_low_power_packet.py NEW):

- `_packetize_q08_attention_probs`: causal local window restriction before topk;
  boundary slots use safe row-self index with zero contribution; dense
  reconstruction now uses `scatter_add`. `validate_local_idx=true` retained as
  internal invariant; failure message rewritten as "internal bug" wording.
- `_q08_packet_fixed_matmul`: rewritten to gather top-k V slots instead of
  materialising `(B, H, T, T, D)` dense products.
- `_lmul_matmul`: windowed/chunked path; opt-out via new
  `model.attention_datapath.lmul_qk_force_dense`. New
  `lmul_qk_chunk_size` for memory tuning.
- New config keys: `attention_datapath.lmul_qk_chunk_size`,
  `attention_datapath.lmul_qk_force_dense`, `attention_datapath.prob_packet.causal_window`.
  Patch decision: `causal_window=null` keeps the window coupled to
  `2**local_idx_bits`; explicit `causal_window` must fit the local-index field.
  `renormalize=true` uses the post-rounding packet code sum as denominator
  (matches RTL consumer).
- Unit test (`tests/test_low_power_packet.py`): boundary-row correctness,
  mid-row match against unmasked old path, packet fixed matmul, QK
  windowed-vs-dense `torch.equal=True`.

Verification results (post-fix, 20-pair VCTK paper-style, gate 0.8031):

| Variant | MBSTOI v1 (pre-fix) | MBSTOI v2 (post-fix) | Δ | Notes |
|---|---|---|---|---|
| b1_packet_dequant | 0.8105 (biased) | **0.8107** | +0.0002 | ✅ clean |
| b2_packet_fixed | 0.8105 (biased) | **0.8107** | +0.0002 | ✅ clean |
| b3_packet_lmul | 0.8105 (biased) | **0.8106** | +0.0001 | ✅ clean |
| c3_packet_lmul_bitexact | — (total failure) | **0.8102** | n/a | ✅ recovered |
| c0_lmul_qk_bitexact | 0.8105 | 0.8106 | +0.0001 | within noise |
| c1_lmul_scorev_bitexact | 0.8099 | 0.8106 | +0.0007 | tensor-level identity holds; drift is metric noise |
| c2_lmul_attn_bitexact | 0.8101 | 0.8100 | −0.0001 | within noise |

All 8 variants pass the 20-pair MBSTOI sanity gate post-fix. Zero
`Failed to process | local_idx overflow | Internal bug | Traceback` lines in
either v2 log.

c1's +0.0007 drift is the only non-noise movement. c1 targets `score_v` only
(it does NOT go through the new QK windowed/chunked path), so the drift can't
be attributed to QK chunk-vs-dense sum order. More likely sources: metric /
eval non-determinism (CUDA reduction order, file-selection or rng across
runs) or accumulation noise inside the score×V L-Mul path itself. Codex
confirmed the QK windowed vs dense forward tensors are `torch.equal=True` at
fixed seeds — that rules out a logical change on the QK side specifically.
Acceptable for the sanity scope; flag for closer inspection if full-VCTK eval
shows similar drift.

Logs: `/tmp/phase_bc_sanity_v2.log` (b1/b2/b3/c3),
`/tmp/phase_bc_pure_v2.log` (c0/c1/c2).
Result dirs: `results/low_power_phase_bc_sanity_v2/` and `_v2_pure/`.

Implication for next step: 20-pair sanity is now meaningful for all B/C
variants, but per §8b the sanity gate is 0.027 looser than the spec gate. Going
straight to Phase B/C QAT without first measuring full-VCTK on the most
informative variant (c3 = packet + bit-exact L-Mul, the production hardware
target) is the same trap that gave Phase D a sanity-pass / full-fail. Prefer:
short full-VCTK eval on c3 first, then decide QAT scope.

### 9b. Phase B/C full VCTK (post-fix, 2026-05-20 ~04:00)

All 8 variants evaluated on full VCTK paper-style against the **base ckpt** (no
extra QAT). Each took ~10 min on GPU 1 (serial pipeline).

| Variant | full-VCTK MBSTOI | vs gate 0.8302 | ILD (dB) | IPD (deg) |
|---|---|---|---|---|
| b0_prob_q08_grid | 0.8365 | **+0.0063** | 3.27 | 15.91 |
| b1_packet_dequant | 0.8365 | **+0.0063** | 3.27 | 15.91 |
| b2_packet_fixed | 0.8366 | **+0.0064** | 3.27 | 15.91 |
| b3_packet_lmul | 0.8361 | **+0.0059** | 3.29 | 15.92 |
| c0_lmul_qk_bitexact | 0.8365 | **+0.0063** | 3.27 | 15.92 |
| c1_lmul_scorev_bitexact | 0.8364 | **+0.0062** | 3.26 | 15.91 |
| c2_lmul_attn_bitexact | 0.8364 | **+0.0062** | 3.26 | 15.92 |
| **c3_packet_lmul_bitexact** | **0.8360** | **+0.0058** | 3.29 | 15.93 |

**All 8 variants clear the spec gate by +0.006 from the base ckpt — no QAT
needed.** The most aggressive variant (c3 = packet + bit-exact L-Mul, the
production hardware target) is 0.0006 below the cleanest variant — within
metric noise.

**Reading**: packet encoding, bit-exact L-Mul simulator, and q0.8 probability
quantization are effectively **lossless on top of the existing P4 QAT base**.
Unlike Phase D (which changes attention topology and needs recovery), Phase B/C
changes are numerical-only and live within the existing model's tolerance.

This invalidates the original "Phase B/C QAT (conditional)" plan: there is no
quality gap to recover. The c3 10-epoch QAT still running on GPU 7 is now a
control/comparison rather than a requirement; its result will tell us whether
QAT helps c3 at all (likely small or negative).

Result dir: `results/low_power_phase_bc_full/{variant}_full_vctk/`.
Log: `/tmp/bc_full_vctk_all.log`.

### 9c. Hardware implication — operating point unblocked

With Phase B/C lossless and Phase D's d1_qkv_g4 only 0.0002 below the full-VCTK
gate (and likely recoverable with the in-flight 30-epoch QAT), the operating
point picture from §7 sharpens:

- **Lossless B/C side**: deploy c3 (packet + bit-exact L-Mul + q0.8 + p4 QAT
  scope). Score×V uses packet gather (no `(B,H,T,T,D)` materialisation),
  QK uses windowed L-Mul. No accuracy penalty.
- **Phase D side**: choose between
  - d0_qkv_g2: −14% attention MAC, 64 lanes @ 10 MHz, duty 51% — safer
  - d1_qkv_g4: −21% attention MAC, 64 lanes @ 5 MHz, duty 81% — best power if
    the in-flight 30-epoch QAT closes the 0.0002 gap

Combined: full attention savings from grouping (d0 or d1) stack with the
Phase B/C path which costs nothing. Recommend d1 + c3 as the leading candidate
once d1_30ep numbers land (~11:00 today).

### 9d. Phase D 30-epoch QAT — full eval (2026-05-20)

| Variant | VCTK MBSTOI | vs 0.8302 | TIMIT MBSTOI | vs 0.7869 | ILD VCTK / TIMIT | IPD VCTK / TIMIT |
|---|---|---|---|---|---|---|
| d0_qkv_g2 10ep | 0.8295 | −0.0007 | 0.7861 | −0.0008 | 3.33 / 3.09 | 16.05 / 20.45 |
| d0_qkv_g2 **30ep** | **0.8326** | **+0.0024** ✅ | 0.7859 | −0.0010 ❌ | 3.30 / 3.07 | 15.83 / 20.22 |
| d1_qkv_g4 10ep | 0.8300 | −0.0002 | 0.7845 | −0.0024 | 3.41 / 3.21 | 16.07 / 20.55 |
| **d1_qkv_g4 30ep** | **0.8353** | **+0.0051** ✅ | **0.7881** | **+0.0012** ✅ | 3.30 / 3.06 | 15.87 / 20.32 |

**d1_qkv_g4 + 30 epoch QAT clears both spec gates.** Counter-intuitive but
confirmed: the more aggressive grouping (g=4, −21% attention MAC) outperforms
the safer one (g=2, −14% MAC) after long QAT. Likely explanation: g=4 forces
the model to use the available capacity more efficiently; the 30-epoch budget
is needed to find that solution, which is why the 10-epoch run looked worse.

d0 30ep clears VCTK but misses TIMIT by 0.0010. Not a deployable choice unless
TIMIT can be loosened or an additional 10-20 epoch is invested.

Result dirs:
- `results/low_power_phase_d_full/d0_qkv_g2_qat30ep_full/`
- `results/low_power_phase_d_full/d1_qkv_g4_qat30ep_full/`

Logs: `/tmp/d{0,1}_30ep_full_v2.log`.

---

## 13. Recommended operating point (FINAL)

Stack two orthogonal techniques:

1. **Phase D**: `model.low_power.mha_projection.qkv_groups=4` (d1 variant) — 30-epoch QAT on top of p4 base.
2. **Phase B/C**: `model.attention_datapath.{lmul_enabled=true, lmul_targets=[qk], lmul_mantissa_bits=4, lmul_bit_exact=true}` + `prob_packet.{enabled=true, score_v_mode=lmul, validate_local_idx=true}` (the c3 configuration) — no QAT needed; Score×V takes the packet consumer's L-Mul path, not the dense `lmul_targets` path.

| Metric | Value | Gate | Headroom |
|---|---|---|---|
| VCTK MBSTOI | 0.8353 (d1 30ep alone) | 0.8302 | +0.0051 |
| TIMIT MBSTOI | 0.7881 (d1 30ep alone) | 0.7869 | +0.0012 |
| ILD VCTK / TIMIT | 3.30 / 3.06 dB | — | comparable to dense baseline |
| IPD VCTK / TIMIT | 15.87° / 20.32° | — | comparable to dense baseline |
| Attention MAC | 1.10 M (−21% vs dense 1.39 M) | — | matches f3_qkv_g4 in §7 |
| Operating point | 64 lanes @ 5 MHz, duty 0.81 | — | meets 5 ms hop |

### 13a. Composite (d{0,1} 30ep ckpt + c3 attention datapath) — final integration

| Stack | VCTK | vs 0.8302 | TIMIT | vs 0.7869 | ILD V/T | IPD V/T |
|---|---|---|---|---|---|---|
| d0 30ep + c3 | 0.8325 | +0.0023 ✅ | **0.7854** | −0.0015 ❌ | 3.33 / 3.12 | 15.85 / 20.28 |
| **d1 30ep + c3** | **0.8354** | **+0.0052** ✅ | **0.7880** | **+0.0011** ✅ | 3.30 / 3.07 | 15.87 / 20.34 |

The d1+c3 stack is **bit-noise-equal to d1 alone** (VCTK 0.8354 vs 0.8353,
TIMIT 0.7880 vs 0.7881). Phase B/C numerical changes and Phase D structural
changes are truly orthogonal: stacking them doesn't compound the error budget.

**This is the production stack. No further training or tuning required for
the headline ASIC operating point.**

Result dirs: `results/low_power_composite/d{0,1}_qkv_g{2,4}_plus_c3_full/`.
Logs: `/tmp/composite_d{0,1}_c3.log`.

Open follow-ups (none on the critical path):
- B=16 OOM smoke test (Codex's earlier suggestion) — would halve future
  training wall time if anyone ever needs to re-train.
- d0 + 50-epoch QAT or TIMIT relaxation — only if a smaller-MAC operating
  point (g=2 instead of g=4) becomes preferred for some other reason.
- TIMIT eval on b0–b3, c0–c2 — purely informational. The c3 10-epoch control
  already passed TIMIT; the base-ckpt TIMIT runs for the other B/C variants are
  still unmeasured.

---

## 10. Original Phase D QAT launch context (historical)

Started in parallel ×2 on 2026-05-19 ≈ 20:47 local. Scope reduced from TODO's three variants (d0/d1/d2) to two:

| Variant | GPU | Bash PID | Python child PID | Log | Reason for skipping/keeping |
|---|---|---|---|---|---|
| d0_qkv_g2 | 1 | 1831208 | 1834637 | `/tmp/phase_d_qat_d0.log` | closest to sanity gap (−0.015), most likely to recover |
| d1_qkv_g4 | 4 | 1831666 | 1834651 | `/tmp/phase_d_qat_d1.log` | mid-gap (−0.049); tests whether g=4 on QKV alone recovers in 10 epoch |
| d2_qkv_out_g4 | — | — | — | — | skipped: sanity gap is 0.111, unlikely to recover in 10 epoch; revisit if d1 succeeds |

ETA ≈ 3 hours from launch (so ≈ 23:45 local).

**Status interpretation**: the only on-disk ckpts at the time of writing are
intermediate `epoch={1,2,3}-step=*.ckpt` val-loss best-saves, **not** final
`epoch=9` ckpts. There are **no** `*_finetuned_w20` Phase D result folders yet —
the script writes those only after training+finetuned-sanity completes. If you
see only `*_base_w20` folders, training is still in progress (or has been
interrupted, especially if all PIDs below are gone and no new ckpts appear).

To verify live status (check both parent bash AND child python; bash parent
alone can be alive while the actual work is dead):

```bash
ps -p 1831208,1831666,1834637,1834651 -o pid,ppid,stat,etime,cmd
pgrep -af "run_low_power_phase_d|train\.py"
tail -f /tmp/phase_d_qat_d0.log /tmp/phase_d_qat_d1.log
find outputs/low_power_phase_d -name "*.ckpt" -printf "%T+ %p\n" | sort
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
```

If `ps` shows only bash parents (no python child) and ckpt mtimes haven't
advanced in >5 min: training crashed/was killed; check the log tail for the
real cause before relaunching.

Decision rule from TODO: if g=4 fails but g=2 passes, extend g=2 to 30 epochs.

Open questions — Codex review resolutions:

1. **B=8 wall-time penalty** — accepted on the main line. Future work: run a
   `TRAIN_BATCH_SIZE=16` smoke / OOM test for ~100 steps before committing to
   a full Phase B/C QAT retake. Do not re-run any of Phase A/D at B=16 just to
   recover wall time — the existing B=8 results stand.
2. **Phase D warm-start** — keep `p4_qk_m4_qat_best.ckpt` so all D variants share
   the same baseline and stay comparable with Phase A. Fallback only if d0 fails
   at 10 epoch: try a grouped-MHA-specific recovery from `p1_warmstart_g42.ckpt`
   for the g=2 case (not on main line).
3. **Analyzer JSON overwrite** — `scripts/analyze_low_power_arch.py` is purely a
   function of `(input_groups, qkv_groups, out_groups, output_groups)`, so the
   sanity-run and QAT-run JSONs are byte-identical for the same variant; today's
   overwrite loses nothing. Still cleaned up in the script: analyzer JSON now
   lands in `${RESULT_ROOT}/${variant}.json` instead of the shared
   `results/low_power_analysis/{variant}.json`, so Phase D sanity vs QAT outputs
   are no longer co-located. Phase F (`run_low_power_phase_f_analysis.sh`)
   continues to write its dense / g=4 sweeps under `results/low_power_analysis/`
   — those are the canonical operating-point references.

---

## 11. Remaining work — none on the critical path

The cycle is complete. The production stack `d1_qkv_g4 (30 epoch QAT) + c3
packet_lmul_bitexact + p4 base` is composite-verified in §13a against both
spec gates (VCTK +0.0052, TIMIT +0.0011). Items below are optional follow-ups
that do **not** block the hardware decision.

| Item | Status | Notes |
|---|---|---|
| Packet local_idx fix | ✅ done (Codex 2026-05-20, §9a) | window-aware topk, scatter_add, unit test |
| Phase B/C sanity v2 | ✅ done (§9a) | all 8 variants clean; zero file failures |
| Phase B/C full VCTK | ✅ done (§9b) | all 8 variants pass spec gate by +0.006 |
| Phase D 10-epoch QAT full eval | ✅ done (§8b) | d0/d1 both miss VCTK/TIMIT by ≤0.001 |
| Phase D 30-epoch QAT | ✅ done (§9d) | d0 misses TIMIT; d1 passes both gates |
| c3 10-epoch QAT (control) | ✅ done | full VCTK 0.8348 (−0.001 vs base 0.836), TIMIT 0.7903; QAT slightly hurts VCTK on a saturated path |
| d{0,1} + c3 composite final eval | ✅ done (§13a) | stack lossless, d1+c3 is production |
| B=16 OOM smoke test (Codex earlier suggestion) | optional | would halve future QAT wall time; not on critical path |
| d0 + 50-epoch QAT or TIMIT relaxation | optional | only if g=2 ever becomes preferred (lower MAC, lower duty) |
| TIMIT eval on b0–b3, c0–c2 | optional | base-ckpt TIMIT for these variants remains unmeasured; informational only because the final d1+c3 composite already passes TIMIT |

---

## 12. File pointers

- spec: `LOW_POWER_TODO_2026-05-19.md`
- scripts: `scripts/run_low_power_phase_{a, bc_packet_lmul, d_mha_grouping, f_analysis}.sh`
- analyzer: `scripts/analyze_low_power_arch.py`
- sanity results: `results/low_power_phase_a_sanity/`, `results/low_power_phase_d_sanity/`
- QAT outputs (current run): `outputs/low_power_phase_a/`, `outputs/low_power_phase_d/`
- Phase F JSON: `results/low_power_analysis/`
- Phase A QAT log: `/tmp/phase_a_qat_v2.log`
- Phase D sanity log: `/tmp/phase_d_sanity.log`
- Phase D QAT logs: `/tmp/phase_d_qat_d0.log`, `/tmp/phase_d_qat_d1.log`
- Phase D full VCTK logs: `/tmp/d0_full_vctk.log`, `/tmp/d1_full_vctk.log`
- Phase D full VCTK results: `results/low_power_phase_d_full/d{0,1}_*_full_vctk/`
- Phase B/C sanity log: `/tmp/phase_bc_sanity.log` (pre-fix; partial)
- Phase B/C sanity results: `results/low_power_phase_bc_sanity/` (pre-fix; b1/b2/b3 partial, c3 empty)
- Phase B/C sanity v2 logs (post-fix): `/tmp/phase_bc_sanity_v2.log`, `/tmp/phase_bc_pure_v2.log`
- Phase B/C sanity v2 results: `results/low_power_phase_bc_sanity_v2/`, `_v2_pure/`
- Phase B/C full VCTK results (8 variants): `results/low_power_phase_bc_full/{variant}_full_vctk/`
- Phase B/C full VCTK log: `/tmp/bc_full_vctk_all.log`
- c3 10-epoch QAT log + post-eval log: `/tmp/phase_bc_qat_c3.log`, `/tmp/c3_qat10ep_full.log`
- c3 10-epoch QAT outputs: `outputs/low_power_phase_bc/c3_packet_lmul_bitexact_*/`
- c3 QAT post-eval results: `results/low_power_phase_bc_full/c3_qat10ep_full/`
- Phase D 30-epoch QAT logs: `/tmp/phase_d_qat_d{0,1}_30ep.log`
- Phase D 30-epoch QAT outputs: `outputs/low_power_phase_d/d{0,1}_qkv_g{2,4}_2026-05-20_*/`
- Phase D 30-epoch full eval logs (v2, post-`GROUPS`-fix): `/tmp/d{0,1}_30ep_full_v2.log`
- Phase D 30-epoch full eval results: `results/low_power_phase_d_full/d{0,1}_qkv_g{2,4}_qat30ep_full/`
- d0/d1 10-epoch TIMIT logs (v2): `/tmp/d_10ep_timit_v2.log` (combined VCTK+TIMIT)
- **Composite eval logs (FINAL)**: `/tmp/composite_d{0,1}_c3.log`
- **Composite eval results (FINAL)**: `results/low_power_composite/d{0,1}_qkv_g{2,4}_plus_c3_full/`
- Composite eval driver script: `/tmp/composite_d_c3_eval.sh`
- Packet fix unit test: `tests/test_low_power_packet.py`
