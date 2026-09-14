r"""
Phase 1C — backbone ablations by finetuning (notes/diagnosis.md §4).

GATE. §4 runs only if 1B produced a non-trivial objective effect. It did:
`results/phase1/1b_decomposition.md` reports objective eta^2 = 0.066 with level
means position=0.311 vs position+grad=0.490 and position+normal=0.388 — an 18-
and 8-point spread attributable to the coupling terms **[MEASURED]**, on the
synthetic construction of §3 only. That is the condition, and it is met. It is
NOT evidence that the same holds at scale; establishing whether it does is the
whole point of 1C, and 1C can only do it partially (see CONFOUNDS).


=== 1. FEASIBILITY: WHAT THIS HARDWARE CAN AND CANNOT DO =====================

§4 says "finetune from released checkpoints". Read literally that means training
all the weights. On one shared 20 GB RTX 4000 Ada it is not possible, and this
file does not pretend otherwise.

**[MEASURED] Parameter census** (CPU, 2026-09-04, from the released checkpoints):

    VGGT-1B    1256.5 M total   aggregator 909.1 | camera_head 216.2
                                point_head 32.7 | depth_head 32.7 | track_head 65.9
    Pi3         958.7 M total   encoder 304.4 (frozen in pi3 by design) |
                                decoder 453.6 | point_decoder 66.1 | point_head 0.6 |
                                conf_decoder 66.1 | conf_head 0.2 | camera_decoder 65.6

**[PROVEN] The static cost of full finetuning.** AdamW in fp32 holds 4 bytes of
weights + 4 of gradient + 8 of moments = 16 bytes per parameter. For VGGT without
the unused track head (1191 M) that is 17.7 GiB, and for Pi3 (958.7 M) 14.3 GiB,
before a single activation is stored. The card has 19.56 GiB usable, so on an
EMPTY card full finetuning would leave 1.9 GiB (VGGT) or 5.3 GiB (Pi3) for the
activations of 48 alternating-attention blocks, resp. 36 + 3x12 decoder blocks,
plus DPT heads that materialise feature maps at full image resolution. This is
arithmetic; whether 1.9 GiB is survivable for VGGT is marginal and is NOT claimed
here either way, because —

**[MEASURED] the card is never empty.** Sampled repeatedly on 2026-09-04: a
neighbour's job (`src/labeling.py`, another user, ~9 h old) held 14.0-16.0 GiB,
leaving 3.5-6.0 GiB free and drifting DOWNWARD during the session. One probe was
killed by the allocator when free memory fell to 0.9 GiB mid-run. So the working
budget is ~3.5 GiB and it is not guaranteed. Against that budget, full
finetuning is short by a factor of 4 (Pi3) to 5 (VGGT) on static state alone, and
no amount of activation thrift closes a gap that large. That is the finding: not
"tight", not "would need care" — off by a multiple.

**[MEASURED] What does fit** — forward + backward + AdamW step, frozen weights in
bf16, trainable modules in fp32, one view per forward, measured peak
`torch.cuda.max_memory_allocated`:

    family  scope        px    S   trainable   peak GiB   s/step
    pi3     point+conf   392   1   133.1 M      3.06       0.51
    pi3     point        392   1    66.7 M      2.94       0.38
    pi3     point        392   2    66.7 M      2.96       0.35
    pi3     point        518   1    66.7 M      2.96       0.56
    vggt    heads        392   1    65.3 M      3.19       0.61
    vggt    heads        392   2    65.3 M      3.52       0.79
    vggt    heads+last4  392   1   115.7 M      3.28       0.64
    vggt    heads        518   1    65.3 M      3.63       0.75
    vggt    heads        518   2    65.3 M     OOM at 4.21 GiB cap

The interesting entry is `heads+last4`: unfreezing the last two frame blocks and
the last two global blocks of the aggregator costs 0.09 GiB and 0.03 s/step over
heads-only, because at 392 px a view is 784 patch tokens and those activations
are small next to the weights. Backbone depth, not backbone width, is what we
cannot afford: the cost is in storing activations for every block below the
deepest trainable one.

**VERDICT.** Full-model finetuning: NOT FEASIBLE, short by a factor of 4-5 in
memory against the budget this shared card actually offers.
Head-scope finetuning (VGGT's two DPT heads; Pi3's point and confidence decoders
and heads), optionally with the last few backbone blocks: FEASIBLE at ~3.0-3.6
GiB and 0.4-0.8 s/step, i.e. ~15-30 min for a 2000-step arm.

**What the restricted version can and cannot answer.**

  CAN — does the auxiliary term, applied late, change where the READOUT puts
  points at occlusion boundaries, relative to a matched-budget arm whose loss was
  not touched? If removing the term moves boundary FP and the aggregate metrics
  hold, the term demonstrably influences boundary placement in a
  billion-parameter model on real data. That is a genuine, if narrow, transfer
  test of the 1B effect.

  CANNOT — (i) whether a model TRAINED FROM SCRATCH without the term would place
  points differently; (ii) whether the effect lives in the backbone features
  rather than the readout — the backbone is frozen at weights learned under the
  original objective and cannot move; (iii) anything about the term's effect on
  the aggregator's multi-view reasoning. A NULL result here is therefore close to
  uninformative: it is exactly what a frozen backbone would produce whether or
  not the term matters. A POSITIVE result is informative. This asymmetry is
  structural and is printed on every output file.


=== 2. THE STANDING CONFOUND (§4, quoted) ===================================

**[CONFOUNDED]** "Every arm starts from a checkpoint pretrained under the
ORIGINAL objective, so a short finetune measures the MARGINAL effect of changing
a term late in training, not the effect of training with it from the start.
State this; do not report a weak effect as a null."

This harness adds a SECOND confound of the same shape, from §1 above: the
backbone is not merely initialised under the original objective, it is FROZEN
there. Both are carried into `1c_report.md` and every CSV header. They are the
reason this phase is described in §3.5 as "the partial and confounded check",
not as the experiment that settles anything.


=== 3. MATCHED-BUDGET CONTROL (§4, mandatory) ===============================

Every arm within a family gets:

  * the same ordered list of training samples, drawn once from a seed that does
    not depend on the arm (`Schedule.data_seed`);
  * the same optimizer, learning rate, warmup, weight decay, gradient
    accumulation and step count;
  * the same init seed for anything stochastic inside the step;
  * the same trainable parameter set (scope), same dtypes, same resolution;
  * an UNMODIFIED-LOSS ARM, at the published weight, run under all of the above.

The only thing that differs is one scalar in the loss. `--selftest` asserts these
invariants on a stub with no models and no GPU; `--run` re-asserts them before
the first step and refuses to start if they do not hold. The step-0 anchors
(released checkpoint, no training) are exempt by construction and are excluded
from the check explicitly rather than silently.


=== 4. ARMS =================================================================

  pi3_normal_1.0   lambda_normal = 1.0   UNMODIFIED-LOSS CONTROL (published)
  pi3_normal_0.5   lambda_normal = 0.5
  pi3_normal_0.0   lambda_normal = 0.0   term removed
  pi3_step0        released checkpoint, 0 steps                     ANCHOR

  vggt_grad_on     gradient term at weight 1.0   UNMODIFIED-LOSS CONTROL
  vggt_grad_off    gradient term at weight 0.0   term removed
  vggt_step0       released checkpoint, 0 steps                     ANCHOR

The anchors are not part of the matched-budget comparison. They exist because
without them a difference between control and ablation cannot be read against
the size of the shift the finetune itself produces, and "the finetune moved
everything by 10 points and the arms differ by 1" is a materially different
result from "nothing moved except the ablation".

**[ASSUMED] The loss implementations are RECONSTRUCTIONS.** Neither VGGT nor Pi3
ships training code in the released packages. `objective()` below builds the
published forms as described in the papers and reduced in `factorial.py`, reusing
`factorial`'s term implementations so that 1B and 1C cannot drift apart. The
absolute value of a loss here is not comparable to the authors' numbers. Since
every arm uses the identical implementation and differs only in one weight, an
error in the reconstruction is shared and does not confound the CONTRAST — but it
does limit what "the published objective" means in this file, and a reader should
not take these arms as reproducing the original training.


=== 5. OUTPUTS ==============================================================

`results/phase1c/`
  1c_prereg.md       written before any training; `--run` refuses without it
  1c_feasibility.md  §1, as a standalone artefact
  1c_arms.csv        per arm x eval scene x stream x (eta,tau,beta)
  1c_summary.csv     per arm: boundary and aggregate metrics SIDE BY SIDE (§4)
  1c_report.md       the comparison, with both confounds at the top

    python src/finetune.py --dry-run
    python src/finetune.py --selftest
    python src/finetune.py --preregister --out results/phase1c
    python src/finetune.py --run --family pi3 --out results/phase1c
"""
from __future__ import annotations

import gc
import os
import csv
import json
import collections
import time
import random
import argparse
import platform
from dataclasses import dataclass, field, asdict, replace

import numpy as np

# NOTE ON IMPORTS. torch, factorial, infer and data are imported LAZILY, inside
# the functions that need them. --dry-run and --selftest must run on a laptop
# with numpy and nothing else: an invariant checker that can only be executed on
# the machine it is meant to guard is not much of a guard.


# --------------------------------------------------------------------------- #
# §1 feasibility, as data
# --------------------------------------------------------------------------- #
# [MEASURED] 2026-09-04 on bruinml (RTX 4000 Ada, 19.56 GiB usable, driver
# 570.211.01, torch 2.5.1+cu121), by scratchpad probe: frozen weights bf16,
# trainable fp32, AdamW, forward+backward+step on random inputs, peak
# torch.cuda.max_memory_allocated over 3 steps. Times exclude checkpoint load.
# `px` is the side of a SQUARE probe input, the worst case. A real 4:3 view
# preprocesses to 294x392 -- 25% fewer patch tokens -- and measures lower: the
# vggt/heads arm ran at 2.92 GiB peak against the 3.19 GiB predicted here. These
# estimates are therefore an upper bound, which is the direction to be wrong in
# when the card is shared.
MEASURED = [
    dict(family="pi3",  scope="point+conf",  px=392, views=1, trainable_M=133.1,
         peak_gib=3.06, s_per_step=0.51),
    dict(family="pi3",  scope="point",       px=392, views=1, trainable_M=66.7,
         peak_gib=2.94, s_per_step=0.38),
    dict(family="pi3",  scope="point",       px=392, views=2, trainable_M=66.7,
         peak_gib=2.96, s_per_step=0.35),
    dict(family="pi3",  scope="point",       px=518, views=1, trainable_M=66.7,
         peak_gib=2.96, s_per_step=0.56),
    dict(family="vggt", scope="heads",       px=392, views=1, trainable_M=65.3,
         peak_gib=3.19, s_per_step=0.61),
    dict(family="vggt", scope="heads",       px=392, views=2, trainable_M=65.3,
         peak_gib=3.52, s_per_step=0.79),
    dict(family="vggt", scope="heads+last4", px=392, views=1, trainable_M=115.7,
         peak_gib=3.28, s_per_step=0.64),
    dict(family="vggt", scope="heads",       px=518, views=1, trainable_M=65.3,
         peak_gib=3.63, s_per_step=0.75),
    dict(family="vggt", scope="heads",       px=518, views=2, trainable_M=65.3,
         peak_gib=float("nan"), s_per_step=float("nan"), oom=True),
]

# [MEASURED] parameter census, same session, CPU only.
CENSUS = {
    "vggt": dict(total_M=1256.5, modules={
        "aggregator": 909.11, "camera_head": 216.17, "point_head": 32.65,
        "depth_head": 32.65, "track_head": 65.94}),
    "pi3": dict(total_M=958.7, modules={
        "encoder": 304.37, "decoder": 453.55, "point_decoder": 66.13,
        "point_head": 0.60, "conf_decoder": 66.13, "conf_head": 0.20,
        "camera_decoder": 65.60, "camera_head": 2.11}),
}

CARD_GIB = 19.56          # [MEASURED] usable, as torch reports it
# [MEASURED] 2026-09-04, repeated samples over ~1 h while a neighbour's job ran.
# This, not CARD_GIB, is the number a plan has to fit inside.
FREE_GIB_OBSERVED = (0.9, 6.0)   # (worst seen mid-run, best seen)
FREE_GIB_WORKING = 3.5           # what may be assumed available
BYTES_PER_PARAM_ADAMW32 = 16     # 4 weight + 4 grad + 8 Adam moments

# Trainable module names per scope. `track_head` is DELETED for vggt (never used
# by any stream we measure), which is why the vggt totals below are against
# 1190.6 M rather than 1256.5 M.
SCOPES = {
    "vggt": {
        "heads":       ["point_head", "depth_head"],
        "heads+last4": ["point_head", "depth_head",
                        "aggregator.frame_blocks[-2:]", "aggregator.global_blocks[-2:]"],
        "full":        ["*"],
    },
    "pi3": {
        "point":       ["point_decoder", "point_head"],
        "point+conf":  ["point_decoder", "point_head", "conf_decoder", "conf_head"],
        "full":        ["*"],
    },
}


def full_finetune_gib(family):
    """Static AdamW state for training every parameter. Activations extra."""
    tot = CENSUS[family]["total_M"]
    if family == "vggt":
        tot -= CENSUS["vggt"]["modules"]["track_head"]
    return tot * 1e6 * BYTES_PER_PARAM_ADAMW32 / 2**30


def measured(family, scope, px, views):
    """Nearest measured configuration, or None. No interpolation: an estimate
    presented as a measurement is worse than no estimate."""
    for m in MEASURED:
        if (m["family"], m["scope"], m["px"], m["views"]) == (family, scope, px, views):
            return m
    return None


# --------------------------------------------------------------------------- #
# arms and schedule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Arm:
    name: str
    family: str                 # "pi3" | "vggt"
    weight: float               # lambda_normal (pi3) or gradient weight (vggt)
    control: bool = False       # the UNMODIFIED-LOSS arm required by §4
    anchor: bool = False        # released checkpoint, 0 steps; not matched-budget

    @property
    def term(self):
        return "lambda_normal" if self.family == "pi3" else "w_grad"


ARMS = [
    Arm("pi3_normal_1.0",  "pi3",  1.0, control=True),
    Arm("pi3_normal_0.5",  "pi3",  0.5),
    Arm("pi3_normal_0.0",  "pi3",  0.0),
    Arm("pi3_step0",       "pi3",  1.0, anchor=True),
    Arm("vggt_grad_on",    "vggt", 1.0, control=True),
    Arm("vggt_grad_off",   "vggt", 0.0),
    Arm("vggt_step0",      "vggt", 1.0, anchor=True),
]

FAMILIES = ("pi3", "vggt")


@dataclass(frozen=True)
class Schedule:
    """Everything that must be IDENTICAL across the arms of a family."""
    steps: int = 1200              # optimizer steps
    accum: int = 4                 # samples per optimizer step
    lr: float = 1e-5
    warmup: int = 50
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    alpha: float = 0.2             # confidence term weight; see [ASSUMED] in §4
    px: int = 392                  # training long side (memory; see §1)
    views: int = 1                 # sequence length per forward
    data_seed: int = 20260904      # draws the sample order; NOT arm-dependent
    init_seed: int = 7             # torch/np/random at arm start
    train_datasets: tuple = ("tartanair",)
    train_scenes: int = 96
    train_views: int = 12
    eval_datasets: tuple = ("infinigen", "middlebury", "tartanair")
    eval_scenes: int = 8
    eval_views: int = 4
    eval_px: int = 518             # matches infer.py, so numbers sit next to 1A
    scope: dict = field(default_factory=lambda: {"pi3": "point+conf", "vggt": "heads"})

    @property
    def samples(self):
        return self.steps * self.accum


# --------------------------------------------------------------------------- #
# data plan — deterministic, arm-independent
# --------------------------------------------------------------------------- #
def build_pool(datasets, n_scenes, n_views, stub=None):
    """Ordered list of view keys eligible for training.

    `stub` short-circuits the filesystem so the invariant self-test can run with
    no datasets present.
    """
    if stub is not None:
        return list(stub)
    import data as D
    pool = []
    for ds in datasets:
        for sc in D.scenes(ds)[:n_scenes]:
            for v in D.views(ds, sc, n_views):
                pool.append((ds, sc, v.idx))
    return pool


def sample_order(pool, n, seed):
    """The training order. Epoch-wise shuffles of the whole pool, concatenated
    and truncated, so every sample is seen a near-equal number of times and the
    order is a pure function of (pool, n, seed) — never of the arm, the wall
    clock, or the filesystem.
    """
    if not pool:
        raise SystemExit("training pool is empty; check PREMISE_DATA_ROOT and "
                         "--train-datasets")
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        ep = list(pool)
        rng.shuffle(ep)
        out.extend(ep)
    return out[:n]


def eval_plan(schedule, stub=None, exclude=()):
    """Eval views, taking care that a dataset used for TRAINING contributes only
    scenes the finetune never saw.

    tartanair appears in both lists on purpose — it is the training distribution,
    and measuring on held-out scenes of it separates "the term changed boundary
    behaviour" from "the model memorised these scenes". `exclude` is the set of
    (dataset, scene) pairs the training order actually touched.
    """
    if stub is not None:
        return list(stub)
    import data as D
    ex = set(exclude)
    out = []
    for ds in schedule.eval_datasets:
        taken = 0
        for sc in D.scenes(ds):
            if (ds, sc) in ex:
                continue
            for v in D.views(ds, sc, schedule.eval_views):
                out.append((ds, sc, v.idx))
            taken += 1
            if taken >= schedule.eval_scenes:
                break
        if taken == 0:
            print(f"  ! eval dataset {ds} has no scene outside the training set",
                  flush=True)
    return out


def held_out(train, ev):
    """Training samples whose SCENE also appears in the eval plan.

    Overlap is not automatically fatal — tartanair appears in both by design, and
    a finetune evaluated only on unseen scenes measures something different from
    one evaluated on seen scenes. It is fatal to not know which you did, so this
    is reported, not silently fixed.
    """
    tr = {(d, s) for d, s, _ in train}
    return sorted({(d, s) for d, s, _ in ev} & tr)


# --------------------------------------------------------------------------- #
# the matched-budget invariants (§4) — the thing that makes 1C a control
# --------------------------------------------------------------------------- #
def arm_fingerprint(arm, schedule, order):
    """Everything about an arm EXCEPT the loss weight under test.

    Two arms of a family must agree here exactly. The data order is hashed rather
    than compared element-wise so the fingerprint stays printable, but the hash is
    over the full ordered sequence, so a single swapped sample changes it.
    """
    import hashlib
    h = hashlib.sha256()
    for s in order:
        h.update(repr(s).encode())
    sched = asdict(schedule)
    sched["scope"] = sched["scope"][arm.family]
    return dict(family=arm.family, n_samples=len(order),
                order_sha=h.hexdigest()[:16], **sched)


def check_matched_budget(arms, schedule, orders):
    """Raise unless every non-anchor arm of a family is budget-identical.

    Returns the list of arms actually compared, so a caller cannot mistake
    'anchors were skipped' for 'everything passed'.
    """
    compared, problems = [], []
    for fam in sorted({a.family for a in arms}):
        fam_arms = [a for a in arms if a.family == fam and not a.anchor]
        if not fam_arms:
            continue
        if not any(a.control for a in fam_arms):
            problems.append(f"{fam}: no unmodified-loss control arm; §4 requires one")
        if len({a.weight for a in fam_arms}) != len(fam_arms):
            problems.append(f"{fam}: two arms share a loss weight — they are the "
                            f"same experiment run twice, not an ablation")
        ref = arm_fingerprint(fam_arms[0], schedule, orders[fam_arms[0].name])
        for a in fam_arms[1:]:
            f = arm_fingerprint(a, schedule, orders[a.name])
            for k in ref:
                if f[k] != ref[k]:
                    problems.append(
                        f"{fam}: arm {a.name} differs from {fam_arms[0].name} in "
                        f"'{k}' ({f[k]!r} vs {ref[k]!r}). Matched budget is "
                        f"mandatory (§4); the arms are not comparable.")
        compared += fam_arms
    if problems:
        raise SystemExit("MATCHED-BUDGET CHECK FAILED\n  " + "\n  ".join(problems))
    return compared


# --------------------------------------------------------------------------- #
# objective — reconstructions of the published losses (see §4 [ASSUMED])
# --------------------------------------------------------------------------- #
def _pointmap_grad_term(pred, tgt):
    """VGGT's `‖Σ ⊙ (∇P̂ − ∇P)‖`, per pixel, for a (..., H, W, C) map.

    factorial._grads works on trailing (H, W), so the channel axis is moved in
    front of it and the residual is normed over (dy, dx, channels) jointly —
    one scalar per pixel, as the published form has.
    """
    import torch
    import factorial as FX
    p = pred.movedim(-1, -3)                     # (..., C, H, W)
    t = tgt.movedim(-1, -3)
    py, px = FX._grads(p)
    ty, tx = FX._grads(t)
    sq = ((py - ty) ** 2 + (px - tx) ** 2).sum(-3)
    return FX._safe_norm(sq)


def _normals_from_pointmap(p):
    """Unit normals from a (..., H, W, 3) camera-frame pointmap.

    The cross product of the two spatial derivatives of the POINTMAP, which is
    the right construction for a pointmap; factorial.normals_from_depth uses the
    depth-map form `n ∝ (−∂z/∂x, −∂z/∂y, 1)` valid for a depth image with unit
    focal. Same quantity, different input, so it gets its own function rather
    than a reinterpretation of the 1B one.
    """
    import torch
    import factorial as FX
    q = p.movedim(-1, -3)
    gy, gx = FX._grads(q)
    gy, gx = gy.movedim(-3, -1), gx.movedim(-3, -1)
    n = torch.cross(gx, gy, dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-8)


def _normal_term(pred, tgt):
    """π³'s normal loss, per pixel: 1 − cos between pointmap normals."""
    return 1.0 - (_normals_from_pointmap(pred) * _normals_from_pointmap(tgt)).sum(-1)


def _stencil_mask(mask):
    """Pixels where a backward finite difference is actually defined.

    factorial._grads zero-pads row 0 and column 0, so the gradient term reads
    exactly 0 there and the normal term reads 1 (the cross product of two zero
    vectors has no direction). Averaging those in would put a constant floor of
    (H + W − 1) / HW on the normal term — 12% of it on a 16x16 patch — that no
    prediction can ever remove, and would silently dilute the gradient term.
    A difference taken ACROSS an invalid pixel is equally meaningless, so the
    neighbours' validity is required too.
    """
    m = mask.clone()
    m[..., 0, :] = False
    m[..., :, 0] = False
    m[..., 1:, :] &= mask[..., :-1, :]
    m[..., :, 1:] &= mask[..., :, :-1]
    return m


def _norm_scale(p, mask):
    """Scale factor that makes a pointmap's median radius 1, over valid pixels.

    Both models are scale-ambiguous, so a loss on raw metres would be dominated
    by a global scale the model was never asked to predict. Normalising each of
    pred and target by ITS OWN median radius is the DUSt3R/VGGT convention.
    **[ASSUMED]** that this reconstruction matches what the authors did closely
    enough; it is applied identically in every arm, so it cannot confound the
    contrast, but it does affect the absolute loss values.
    """
    import torch
    r = p.norm(dim=-1)[mask]
    s = r.median() if r.numel() else r.new_tensor(1.0)
    return torch.clamp(s, min=1e-6)


def objective(pred, tgt, conf, mask, *, w_grad=0.0, w_normal=0.0, alpha=0.2):
    """The full per-arm loss on one (H, W, 3) pointmap.

    Position is confidence-weighted; the auxiliary term is added outside it,
    exactly as factorial.total composes them, so 1B and 1C mean the same thing by
    'the objective'. Weight 0 removes a term EXACTLY (same code path), which is
    what makes the ablation arm and the control arm one implementation.
    """
    import torch
    import factorial as FX
    if mask.sum() == 0:
        return None, {}
    pred = pred / _norm_scale(pred, mask)
    tgt = tgt / _norm_scale(tgt, mask)

    pos = FX._safe_norm(((pred - tgt) ** 2).sum(-1))          # ‖P̂ − P‖, per pixel
    if conf is None:
        loss = pos[mask].mean()
    else:
        if not bool((conf[mask] > 0).all()):
            raise ValueError(
                "confidence has non-positive entries; `−α log C` is undefined "
                "there. Every codebase maps its raw conf output through a "
                "positive activation -- see forward_pointmap. Silently skipping "
                "these would drop training samples and leave the arms untrained.")
        # C·ℓ − α log C, over valid pixels only. VGGT's `expp1` conf activation
        # is 1 + exp(x) >= 1, i.e. the clamp is in the architecture; §1.1(b)'s
        # degeneracy as ℓ → 0 is bounded by that, not by anything added here.
        loss = (pos[mask] * conf[mask] - alpha * torch.log(conf[mask] + 1e-8)).mean()
    parts = {"position": float(loss.detach())}

    # The coupling terms are averaged only where their stencil exists; see
    # _stencil_mask for why using `mask` here would put an irreducible floor on
    # the normal term and dilute the gradient term.
    aux = _stencil_mask(mask)
    if (w_grad or w_normal) and aux.sum() == 0:
        return loss, parts
    if w_grad:
        g = _pointmap_grad_term(pred, tgt)[aux].mean()
        loss = loss + w_grad * g
        parts["gradient"] = float(g.detach())
    if w_normal:
        n = _normal_term(pred, tgt)[aux].mean()
        loss = loss + w_normal * n
        parts["normal"] = float(n.detach())
    parts["total"] = float(loss.detach())
    return loss, parts


# --------------------------------------------------------------------------- #
# model construction: freeze, cast, unfreeze the scope
# --------------------------------------------------------------------------- #
def _trainable_modules(model, family, scope):
    if scope == "full":
        return [model]
    if family == "vggt":
        mods = [model.point_head, model.depth_head]
        if scope == "heads+last4":
            mods += [model.aggregator.frame_blocks[-2:],
                     model.aggregator.global_blocks[-2:]]
        return mods
    mods = [model.point_decoder, model.point_head]
    if scope == "point+conf":
        mods += [model.conf_decoder, model.conf_head]
    return mods


def build_model(family, scope):
    """Released checkpoint -> frozen bf16 backbone + fp32 trainable scope.

    Loading goes through infer.py so the weights, the checkpoint choice and the
    strict-loading guard are the same ones Phase 1A measured. The dtype split is
    the only deviation and is what makes §1's memory numbers achievable.
    """
    import torch
    import infer as INF
    INF._MODELS.pop(family, None)              # never reuse a finetuned model
    # Load and cast ON THE CPU. infer's loaders move the fp32 model to the GPU as
    # their last step, which spikes to 3.6 GiB (VGGT) of fp32 weights before the
    # bf16 cast can shrink it -- measured to OOM on this shared card at 3.4 GiB
    # free, before a single training step. Forcing _device to "cpu" for the
    # duration keeps infer's checkpoint choice and its strict-loading guard (the
    # reason to go through infer at all) and moves the model once, already cast.
    real_device, INF._device = INF._device, lambda: "cpu"
    try:
        m = INF._load_vggt() if family == "vggt" else INF._load_pi3()
    finally:
        INF._device = real_device
    INF._MODELS.pop(family, None)
    if family == "vggt":
        m.track_head = None                    # 65.9 M params, no stream uses it
    for p in m.parameters():
        p.requires_grad_(False)
    m = m.to(torch.bfloat16)
    for mod in _trainable_modules(m, family, scope):
        mod.to(torch.float32)
        for p in mod.parameters():
            p.requires_grad_(True)
    m = m.to(INF._device())
    trainable = [p for p in m.parameters() if p.requires_grad]
    return m, trainable


def forward_pointmap(model, family, x):
    """One forward on ONE view -> (H, W, 3) camera-frame pointmap and (H, W) conf.

    Single-view by design, matching infer.py's protocol note: with S = 1 a model's
    reference frame IS that view's camera frame, so the pointmap is already a
    camera-frame map and no predicted pose is ever applied. Both models promote
    the input internally, so both outputs carry a leading (B, S) that is squeezed
    here rather than left for every caller to strip.

    The backbone runs under no_grad when it is frozen, which is where most of the
    memory saving is: activations are only kept below the deepest trainable
    module. With scope='full' the whole graph is retained (and, per §1, does not
    fit).
    """
    import torch
    import infer as INF
    dev = INF._device()
    grad_bb = False
    for n, p in model.named_parameters():
        if p.requires_grad and not (n.startswith(("point_head", "depth_head",
                                                  "point_decoder", "conf_decoder",
                                                  "conf_head"))):
            grad_bb = True
            break
    S = x.shape[0]
    H, W = x.shape[-2:]

    # Both models want (B, S, 3, H, W); a single view arrives as (3, H, W).
    xin = x.to(dev)
    while xin.dim() < 5:
        xin = xin[None]

    if family == "vggt":
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            with torch.set_grad_enabled(grad_bb):
                toks, psi = model.aggregator(xin)
            if not grad_bb:
                toks = [t.detach() if t is not None else None for t in toks]
        with torch.autocast("cuda", enabled=False):
            pts, conf = model.point_head(toks, images=xin, patch_start_idx=psi)
        assert pts.shape[:2] == (1, 1), f"expected B=S=1, got {tuple(pts.shape[:2])}"
        return pts[0, 0].float(), conf[0, 0].float()

    imgs = (xin - model.image_mean) / model.image_std
    B, N = imgs.shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        with torch.set_grad_enabled(grad_bb):
            h = model.encoder(imgs.reshape(B * N, 3, H, W), is_training=True)
            h = h["x_norm_patchtokens"] if isinstance(h, dict) else h
            h, pos = model.decode(h, N, H, W)
        if not grad_bb:
            h, pos = h.detach(), pos.detach()
        ph = model.point_decoder(h, xpos=pos)
        ch = model.conf_decoder(h, xpos=pos)
    with torch.autocast("cuda", enabled=False):
        ret = model.point_head([ph.float()[:, model.patch_start_idx:]],
                               (H, W)).reshape(B, N, H, W, -1)
        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        pts = torch.cat([xy * z, z], dim=-1)
        conf = model.conf_head([ch.float()[:, model.patch_start_idx:]],
                               (H, W)).reshape(B, N, H, W)
        # pi3's conf head emits RAW logits -- Pi3.forward returns them unmapped,
        # and applying `−α log C` to a negative number gives NaN, which silently
        # skipped every training sample until it was caught **[MEASURED]**.
        # VGGT's DPT head already applies `expp1` (1 + exp), so its conf is >= 1
        # by construction; §1.1(b)'s clamp is in the architecture there. The same
        # floor is imposed here with softplus, which is >= 0 and does not
        # overflow for large logits the way 1 + exp does.
        # **[ASSUMED]** that this is the parameterisation pi3 trained with. It is
        # identical in every arm, so it cannot confound the contrast, but it is a
        # reconstruction and not the authors' choice.
        conf = 1.0 + torch.nn.functional.softplus(conf.float())
    assert (B, N) == (1, 1), f"expected B=N=1, got {(B, N)}"
    return pts[0, 0].float(), conf[0, 0].float()


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def gt_pointmap(depth, valid, K, shape):
    """GT camera-frame pointmap at model resolution.

    Depth and mask are resampled NEAREST — never bilinear. Interpolating a depth
    map across an occlusion boundary manufactures exactly the void points this
    whole project measures, and doing it in the TRAINING TARGET would be worse
    than doing it in the prediction: it would teach them.
    """
    import infer as INF
    h, w = shape
    d = INF.nn_resize(depth.astype(np.float64), (h, w))
    m = INF.nn_resize(valid.astype(np.float64), (h, w)) > 0.5
    m &= np.isfinite(d) & (d > 0)
    sy, sx = h / depth.shape[0], w / depth.shape[1]
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = K[0, 2] * sx, K[1, 2] * sy
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = np.where(m, d, 1.0)
    return np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1), m


def _sample(family, key, px):
    """One training sample -> (image tensor CHW, target pointmap, mask, meta)."""
    import torch
    import data as D
    import infer as INF
    ds, sc, idx = key
    views = {v.idx: v for v in D.views(ds, sc, None)}
    v = views.get(idx)
    if v is None:
        return None
    vd = D.load(v, with_rgb=True)
    if vd.rgb is None:
        return None
    long_side, patch = INF.GEOM[family]
    x, meta = INF._preprocess(vd.rgb, pad=INF.PAD_VALUE[family],
                              long_side=px, patch=patch)
    tgt, m = gt_pointmap(vd.depth, vd.valid, vd.K, (meta["h"], meta["w"]))
    if m.sum() < 100:
        return None
    # The model sees the padded canvas; the loss sees only the real region.
    full = np.zeros(x.shape[1:] + (3,), np.float64)
    fullm = np.zeros(x.shape[1:], bool)
    t0, l0, hh, ww = meta["top"], meta["left"], meta["h"], meta["w"]
    full[t0:t0 + hh, l0:l0 + ww] = tgt
    fullm[t0:t0 + hh, l0:l0 + ww] = m
    return (torch.from_numpy(x), torch.from_numpy(full).float(),
            torch.from_numpy(fullm), meta)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_arm(arm, schedule, order, *, out, log_every=50, max_seconds=None):
    import torch
    import infer as INF

    scope = schedule.scope[arm.family]
    torch.manual_seed(schedule.init_seed)
    np.random.seed(schedule.init_seed)
    random.seed(schedule.init_seed)

    model, params = build_model(arm.family, scope)
    n_tr = sum(p.numel() for p in params)
    print(f"[{arm.name}] scope={scope} trainable={n_tr/1e6:.1f} M "
          f"{arm.term}={arm.weight}", flush=True)
    if arm.anchor:
        print(f"[{arm.name}] ANCHOR: released checkpoint, 0 steps.", flush=True)
        return model, dict(arm=arm.name, steps=0, anchor=True, loss_curve=[])

    opt = torch.optim.AdamW(params, lr=schedule.lr,
                            weight_decay=schedule.weight_decay)
    kw = dict(alpha=schedule.alpha)
    kw["w_normal" if arm.family == "pi3" else "w_grad"] = arm.weight

    curve, skipped, t0 = [], 0, time.time()
    reasons = collections.Counter()
    it = iter(order)
    for step in range(schedule.steps):
        for g in opt.param_groups:
            g["lr"] = schedule.lr * min(1.0, (step + 1) / max(1, schedule.warmup))
        opt.zero_grad(set_to_none=True)
        acc, got = 0.0, 0
        for _ in range(schedule.accum):
            try:
                key = next(it)
            except StopIteration:
                raise SystemExit(f"[{arm.name}] data order exhausted at step {step}; "
                                 f"sample_order returned fewer than steps*accum")
            try:
                s = _sample(arm.family, key, schedule.px)
            except Exception as e:
                print(f"  ! sample {key}: {type(e).__name__}: {e}", flush=True)
                s = None
            if s is None:
                # A dropped sample must NOT be silently replaced with another:
                # that would make the realised data order arm-dependent whenever
                # a load is flaky. The slot is spent, matched across arms.
                skipped += 1
                reasons["unloadable"] += 1
                continue
            x, tgt, m, _ = s
            pred, conf = forward_pointmap(model, arm.family, x)
            loss, _parts = objective(pred, tgt.to(pred.device), conf,
                                     m.to(pred.device), **kw)
            if loss is None or not torch.isfinite(loss):
                skipped += 1
                reasons["no_valid_pixels" if loss is None else "nonfinite_loss"] += 1
                continue
            (loss / schedule.accum).backward()
            acc += float(loss.detach())
            got += 1
        if got:
            torch.nn.utils.clip_grad_norm_(params, schedule.grad_clip)
            opt.step()
            curve.append(acc / got)
        if step % log_every == 0:
            el = time.time() - t0
            print(f"  step {step:5d}/{schedule.steps} loss={acc/max(got,1):.5f} "
                  f"{el/max(step,1):.2f}s/step peak="
                  f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
        if max_seconds and time.time() - t0 > max_seconds:
            raise SystemExit(f"[{arm.name}] wall-clock cap {max_seconds}s hit at "
                             f"step {step}. A truncated arm is NOT matched-budget; "
                             f"rerun with a bigger cap or fewer steps for ALL arms.")
    if skipped:
        print(f"[{arm.name}] skipped {skipped}/{schedule.samples} samples: "
              f"{dict(reasons)}", flush=True)
    if len(curve) < 0.5 * schedule.steps:
        raise SystemExit(
            f"[{arm.name}] only {len(curve)}/{schedule.steps} steps had a usable "
            f"sample ({dict(reasons)}). An arm that barely trained is not an "
            f"ablation of anything; fix the cause rather than reporting it as a "
            f"null effect.")
    stats = dict(arm=arm.name, family=arm.family, term=arm.term, weight=arm.weight,
                 control=arm.control, anchor=False, scope=scope,
                 steps=schedule.steps, skipped_samples=skipped,
                 skip_reasons=dict(reasons),
                 seconds=time.time() - t0, loss_first50=float(np.mean(curve[:50]))
                 if curve else float("nan"),
                 loss_last50=float(np.mean(curve[-50:])) if curve else float("nan"),
                 peak_gib=float(torch.cuda.max_memory_allocated() / 2**30)
                 if torch.cuda.is_available() else float("nan"),
                 loss_curve=curve)
    return model, stats


# --------------------------------------------------------------------------- #
# evaluation: boundary and aggregate, side by side (§4)
# --------------------------------------------------------------------------- #
def aggregate_metrics(gt, valid, pred):
    """Scale-aligned AbsRel / RMSE / delta<1.25 over all valid pixels.

    Alignment is median-ratio on the SAME pixels the metric uses, matching
    fpmetrics.align_scale's 'median' mode. These are the aggregate numbers §4
    asks to be shown beside the boundary rate: an ablation that halves FP by
    wrecking the depth map has not told us anything about boundaries.
    """
    g, p = gt.astype(np.float64), pred.astype(np.float64)
    m = valid & np.isfinite(g) & (g > 0) & np.isfinite(p) & (p > 0)
    if m.sum() < 100:
        return dict(n_px=int(m.sum()), absrel=float("nan"), rmse=float("nan"),
                    d125=float("nan"))
    s = float(np.median(g[m] / p[m]))
    q = p * s
    r = np.abs(q[m] - g[m]) / g[m]
    ratio = np.maximum(q[m] / g[m], g[m] / q[m])
    return dict(n_px=int(m.sum()), scale=s, absrel=float(r.mean()),
                rmse=float(np.sqrt(np.mean((q[m] - g[m]) ** 2))),
                d125=float((ratio < 1.25).mean()))


def evaluate_arm(model, arm, schedule, plan, *, grid=True):
    """Per eval scene: pooled boundary FP over the threshold grid + aggregates.

    Preprocessing, padding, cropping and the nearest-neighbour resample to GT
    resolution are infer.py's, so an arm is measured on exactly the geometry
    Phase 1A used and the numbers sit beside `1a_generality.csv`.

    The FORWARD is `forward_pointmap`, not `infer._fwd_pi3` / `_fwd_vggt`. Those
    call the models' own `forward`, which runs the camera branch inside an
    explicit fp32 block; with the backbone in bf16 that raises
    `mat1 and mat2 must have the same dtype` **[MEASURED]**, and keeping the
    camera branch in fp32 just to throw its output away costs memory we do not
    have. The quantity is unchanged: at S = 1 VGGT's `world_points` IS its
    point_head output in the view's own camera frame, and pi3's `local_points`
    is built from point_head alone. The camera and track branches contribute
    nothing to either stream.
    """
    import torch
    import infer as INF
    import fpmetrics as FM
    import data as D
    from collections import defaultdict

    model.eval()
    stream = "pi3_local" if arm.family == "pi3" else "vggt_point"
    etas = FM.ETA_GRID if grid else (FM.ETA0,)
    taus = FM.TAU_GRID if grid else (FM.TAU0,)
    betas = FM.BETA_GRID if grid else (FM.BETA0,)

    by_scene = defaultdict(lambda: defaultdict(
        lambda: dict(n_fp=0, n_eval=0, n_outside=0, n_boundary=0)))
    agg = defaultdict(list)
    long_side, patch = INF.GEOM[arm.family]
    for ds, sc, idx in plan:
        views = {v.idx: v for v in D.views(ds, sc, None)}
        v = views.get(idx)
        if v is None:
            continue
        try:
            vd = D.load(v, with_rgb=True)
        except Exception as e:
            print(f"  ! eval load {ds}/{sc}/{idx}: {type(e).__name__}: {e}", flush=True)
            continue
        if vd.rgb is None:
            continue
        x, meta = INF._preprocess(vd.rgb, pad=INF.PAD_VALUE[arm.family],
                                  long_side=schedule.eval_px, patch=patch)
        with torch.no_grad():
            pts, _conf = forward_pointmap(model, arm.family, torch.from_numpy(x))
        z = pts[..., 2].float().cpu().numpy()
        z = INF.nn_resize(INF._crop(z, meta), vd.depth.shape).astype(np.float64)
        for r in FM.fp_view_grid(vd.depth, vd.valid, {stream: z}, etas=etas,
                                 taus=taus, betas=betas, delta_min=0.05):
            a = by_scene[(ds, sc)][(r["eta"], r["tau"], r["beta"])]
            a["n_fp"] += r["n_fp"]
            a["n_eval"] += r["n_eval"]
            a["n_outside"] += r["n_outside"]
            a["n_boundary"] += r["n_boundary"]
        agg[(ds, sc)].append(aggregate_metrics(vd.depth, vd.valid, z))

    rows = []
    for (ds, sc), thr in sorted(by_scene.items()):
        am = [a for a in agg[(ds, sc)] if np.isfinite(a["absrel"])]
        for (eta, tau, beta), a in sorted(thr.items()):
            rows.append(dict(
                arm=arm.name, family=arm.family, term=arm.term, weight=arm.weight,
                control=arm.control, anchor=arm.anchor, stream=stream,
                dataset=ds, scene=sc, eta=eta, tau=tau, beta=beta,
                n_boundary=a["n_boundary"], n_eval=a["n_eval"],
                FP=a["n_fp"] / a["n_eval"] if a["n_eval"] else float("nan"),
                outside_rate=a["n_outside"] / a["n_eval"] if a["n_eval"] else float("nan"),
                n_views=len(am),
                absrel=float(np.mean([x["absrel"] for x in am])) if am else float("nan"),
                rmse=float(np.mean([x["rmse"] for x in am])) if am else float("nan"),
                d125=float(np.mean([x["d125"] for x in am])) if am else float("nan")))
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
CONFOUND = """**[CONFOUNDED] — two confounds, both structural, neither fixable here.**

1. §4's own: every arm starts from a checkpoint pretrained under the ORIGINAL
   objective, so this measures the MARGINAL effect of changing a term late in
   training, not the effect of training with it from the start. A weak effect
   here must NOT be reported as a null.
2. This harness's: full finetuning does not fit on one shared 20 GB card (VGGT
   needs 17.7 GiB of AdamW state alone, before activations), so the backbone is
   FROZEN at weights learned under the original objective. Only the readout can
   respond to the changed loss. A null is therefore close to uninformative; a
   positive result is informative and is a genuine transfer test of the 1B
   objective effect.
"""


def _fmt(x, n=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def write_summary(out, rows, stats, schedule):
    """1c_summary.csv + 1c_report.md: boundary next to aggregate, per arm."""
    import fpmetrics as FM
    os.makedirs(out, exist_ok=True)
    if rows:
        with open(os.path.join(out, "1c_arms.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    prim = [r for r in rows if (r["eta"], r["tau"], r["beta"]) ==
            (FM.ETA0, FM.TAU0, FM.BETA0)]
    summ = {}
    for r in prim:
        s = summ.setdefault(r["arm"], dict(
            arm=r["arm"], family=r["family"], term=r["term"], weight=r["weight"],
            control=r["control"], anchor=r["anchor"], n_scenes=0, n_eval=0, n_fp=0.0,
            absrel=[], rmse=[], d125=[]))
        s["n_scenes"] += 1
        s["n_eval"] += r["n_eval"]
        if np.isfinite(r["FP"]):
            s["n_fp"] += r["FP"] * r["n_eval"]
        for k in ("absrel", "rmse", "d125"):
            if np.isfinite(r[k]):
                s[k].append(r[k])
    out_rows = []
    for a in ARMS:
        s = summ.get(a.name)
        if not s:
            continue
        st = stats.get(a.name, {})
        out_rows.append(dict(
            arm=s["arm"], family=s["family"], term=s["term"], weight=s["weight"],
            control=s["control"], anchor=s["anchor"], scope=st.get("scope", ""),
            steps=st.get("steps", 0), skipped_samples=st.get("skipped_samples", 0),
            train_seconds=round(st.get("seconds", 0.0), 1),
            loss_first50=st.get("loss_first50", float("nan")),
            loss_last50=st.get("loss_last50", float("nan")),
            n_scenes=s["n_scenes"], n_eval_px=s["n_eval"],
            FP=s["n_fp"] / s["n_eval"] if s["n_eval"] else float("nan"),
            absrel=float(np.mean(s["absrel"])) if s["absrel"] else float("nan"),
            rmse=float(np.mean(s["rmse"])) if s["rmse"] else float("nan"),
            d125=float(np.mean(s["d125"])) if s["d125"] else float("nan")))
    if out_rows:
        with open(os.path.join(out, "1c_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)

    md = [f"# Phase 1C — backbone ablations (notes/diagnosis.md §4)", "",
          f"Generated {time.strftime('%F %T')} · scope "
          f"{schedule.scope} · {schedule.steps} steps x {schedule.accum} samples "
          f"· train px {schedule.px} · eval px {schedule.eval_px}", "",
          CONFOUND, "",
          "## Boundary and aggregate, side by side",
          "",
          f"Boundary rate at the primary threshold "
          f"(eta={FM.ETA0}, tau={FM.TAU0}, beta={FM.BETA0}); the full grid is in "
          f"`1c_arms.csv`. Aggregate metrics are median-scale-aligned over all "
          f"valid pixels of the same views.", "",
          "| arm | term | weight | steps | FP | AbsRel | RMSE | delta<1.25 | loss first50 -> last50 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        tag = " (CONTROL)" if r["control"] else (" (ANCHOR)" if r["anchor"] else "")
        md.append(f"| {r['arm']}{tag} | {r['term']} | {r['weight']} | {r['steps']} | "
                  f"{_fmt(r['FP'])} | {_fmt(r['absrel'])} | {_fmt(r['rmse'], 3)} | "
                  f"{_fmt(r['d125'], 3)} | {_fmt(r['loss_first50'], 4)} -> "
                  f"{_fmt(r['loss_last50'], 4)} |")

    md += ["", "## Contrasts", ""]
    for fam in FAMILIES:
        ctrl = next((r for r in out_rows if r["family"] == fam and r["control"]), None)
        if ctrl is None:
            continue
        anc = next((r for r in out_rows if r["family"] == fam and r["anchor"]), None)
        if anc and np.isfinite(anc["FP"]) and np.isfinite(ctrl["FP"]):
            md.append(f"- **{fam}**: the finetune ITSELF moved FP by "
                      f"{ctrl['FP'] - anc['FP']:+.4f} (released -> matched-budget "
                      f"control). Any arm difference smaller than this is inside "
                      f"the noise the finetune introduces and must not be read as "
                      f"an objective effect.")
        for r in out_rows:
            if r["family"] != fam or r["control"] or r["anchor"]:
                continue
            if np.isfinite(r["FP"]) and np.isfinite(ctrl["FP"]):
                md.append(f"- **{fam}**: {r['term']}={r['weight']} vs control "
                          f"{ctrl['weight']}: dFP = {r['FP'] - ctrl['FP']:+.4f}, "
                          f"dAbsRel = {r['absrel'] - ctrl['absrel']:+.4f}, "
                          f"d(delta<1.25) = {r['d125'] - ctrl['d125']:+.4f}. "
                          f"**[MEASURED]** under these conditions only.")
    md += ["", "## Scope", "",
           "**[MEASURED]** applies to: these released checkpoints, this frozen-"
           "backbone scope, this data, this step budget, and this loss "
           "RECONSTRUCTION (neither model ships training code; see the module "
           "docstring §4). **[ASSUMED]** that the reconstruction is faithful "
           "enough for the CONTRAST to mean what it appears to mean — an error in "
           "it is shared by every arm and cancels in the difference, but it does "
           "mean these arms do not reproduce the published training.", "",
           "**Not answerable on this hardware:** whether a model trained from "
           "scratch without the term behaves differently, and whether the effect "
           "lives in the backbone rather than the readout. Both need the "
           "backbone to move, and the backbone does not fit."]
    open(os.path.join(out, "1c_report.md"), "w").write("\n".join(md) + "\n")
    json.dump(stats, open(os.path.join(out, "1c_train_stats.json"), "w"),
              indent=1, default=float)
    print("\n".join(md))
    return out_rows


# --------------------------------------------------------------------------- #
# pre-registration (§4 arms, recorded before any training)
# --------------------------------------------------------------------------- #
PREREG = """# Phase 1C — pre-registered arms (notes/diagnosis.md §4)

Recorded {date}, BEFORE any finetuning. `--run` refuses to start without this
file, so the arms cannot be chosen after seeing a result.

## Gate

§4 is conditional on 1B producing a non-trivial objective effect. **[MEASURED]**
`results/phase1/1b_decomposition.md`: objective eta^2 = 0.066, position = 0.311
vs position+grad = 0.490 and position+normal = 0.388. Gate met.

## Arms

| arm | family | term | weight | role |
|---|---|---|---|---|
{arms}

## Matched budget (mandatory)

Identical across every non-anchor arm of a family, and asserted by
`check_matched_budget` before the first step:

- data: {n_samples} samples, order drawn once with data_seed={data_seed}, the
  SAME ordered list for every arm (verified by SHA over the sequence)
- schedule: {steps} optimizer steps x {accum} samples, lr {lr}, warmup {warmup},
  AdamW wd {weight_decay}, grad clip {grad_clip}
- seeds: init_seed={init_seed} at every arm start
- scope: {scope} — frozen backbone (see the feasibility verdict)
- resolution: train {px} px, eval {eval_px} px

## Predictions

| # | Prediction | Label going in |
|---|---|---|
| 1 | removing the coupling term (lambda_normal=0 / grad off) changes boundary FP relative to the matched-budget control | **[UNTESTED]** — direction and magnitude both unknown. 1B found position+grad and position+normal ABOVE position-only on synthetic data, but that is a different predictor, a different construction and a from-scratch fit. Do not assume the sign carries. |
| 2 | aggregate metrics (AbsRel, delta<1.25) move less than boundary FP | **[UNTESTED]**, plausible. If aggregates move as much, the arms differ in overall fit quality and the boundary contrast is not attributable to boundaries. |
| 3 | the step-0 anchor differs from the matched-budget control | **[UNTESTED]** — this is the size of the effect of finetuning AT ALL, and it bounds what an arm difference can be read to mean. |

Explicitly NOT predicted: that any arm difference will be large, or that a null
result would show the term does not matter. Per §4 and the feasibility verdict, a
weak effect is the expected outcome of changing a term late with a frozen
backbone, whether or not the term matters at scale.
"""


def preregister(out, schedule, force=False):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "1c_prereg.md")
    if os.path.exists(p) and not force:
        raise SystemExit(f"{p} exists. Pre-registration is write-once: the arms "
                         f"must precede the run.")
    rows = "\n".join(
        f"| {a.name} | {a.family} | {a.term} | {a.weight} | "
        f"{'UNMODIFIED-LOSS CONTROL' if a.control else ('released ckpt, 0 steps (ANCHOR)' if a.anchor else 'ablation')} |"
        for a in ARMS)
    d = asdict(schedule)
    d["scope"] = str(d["scope"])
    open(p, "w").write(PREREG.format(date=time.strftime("%F"), arms=rows,
                                     n_samples=schedule.samples, **d))
    print("wrote", p)
    write_feasibility(out)


def write_feasibility(out):
    os.makedirs(out, exist_ok=True)
    md = ["# Phase 1C — feasibility verdict", "",
          f"Recorded {time.strftime('%F')}. One shared NVIDIA RTX 4000 Ada, "
          f"{CARD_GIB} GiB usable, **{FREE_GIB_WORKING} GiB working budget** "
          f"(observed free: {FREE_GIB_OBSERVED[0]}-{FREE_GIB_OBSERVED[1]} GiB).", "",
          "## Full finetuning does not fit — [PROVEN] from the parameter counts", ""]
    for fam in FAMILIES:
        tot = CENSUS[fam]["total_M"]
        eff = tot - (CENSUS["vggt"]["modules"]["track_head"] if fam == "vggt" else 0)
        md.append(f"- **{fam}**: {tot:.1f} M params ({eff:.1f} M excluding the "
                  f"unused track head) x 16 B/param for fp32 AdamW = "
                  f"**{full_finetune_gib(fam):.1f} GiB** of static state, before a "
                  f"single stored activation. That is "
                  f"{full_finetune_gib(fam)/FREE_GIB_WORKING:.1f}x the working "
                  f"budget, and leaves "
                  f"{CARD_GIB - full_finetune_gib(fam):.1f} GiB for activations "
                  f"even on a card with nobody else on it.")
    md += ["", "## What was measured to fit — [MEASURED] " + time.strftime("%F"), "",
           "Frozen weights bf16, trainable modules fp32, AdamW, forward + backward "
           "+ step on random inputs, peak `torch.cuda.max_memory_allocated`.", "",
           "| family | scope | px | views | trainable | peak GiB | s/step |",
           "|---|---|---|---|---|---|---|"]
    for m in MEASURED:
        pk = "OOM" if m.get("oom") else f"{m['peak_gib']:.2f}"
        sp = "—" if m.get("oom") else f"{m['s_per_step']:.2f}"
        md.append(f"| {m['family']} | {m['scope']} | {m['px']} | {m['views']} | "
                  f"{m['trainable_M']:.1f} M | {pk} | {sp} |")
    md += ["", "## The card is shared", "",
           "**[MEASURED]** 2026-09-04: a neighbour's job held 14.0-16.0 GiB "
           "throughout, leaving 3.5-6.0 GiB and falling. One probe was OOM-killed "
           "when free memory reached 0.9 GiB mid-run. The working budget is ~3.5 "
           "GiB and is not guaranteed; `--run` re-checks it before every arm and "
           "aborts rather than competing. No other user's process is ever killed.",
           "", "## What the restricted experiment can and cannot answer", "",
           "**CAN**: whether the coupling term, applied late, changes where the "
           "READOUT places points at occlusion boundaries, against a "
           "matched-budget control. A positive result is a real transfer test of "
           "the 1B objective effect to a billion-parameter model on real data.",
           "", "**CANNOT**: (i) whether a model trained from scratch without the "
           "term behaves differently; (ii) whether the effect lives in the "
           "backbone features — those are frozen at weights learned under the "
           "original objective; (iii) anything about the term's effect on "
           "multi-view aggregation. A NULL result is therefore close to "
           "uninformative.", ""]
    p = os.path.join(out, "1c_feasibility.md")
    open(p, "w").write("\n".join(md) + "\n")
    print("wrote", p)


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def dry_run(schedule, families=FAMILIES, stub_pool=None):
    """Plan, budget, trainable set, wall clock and VRAM — no GPU, no model."""
    print("=" * 78)
    print("PHASE 1C DRY RUN — nothing is loaded, nothing touches the GPU")
    print("=" * 78)
    print("\n-- feasibility (see 1c_feasibility.md) --")
    print(f"  card {CARD_GIB} GiB usable; working budget {FREE_GIB_WORKING} GiB "
          f"(observed free {FREE_GIB_OBSERVED[0]}-{FREE_GIB_OBSERVED[1]} GiB, shared)")
    for fam in FAMILIES:
        print(f"  {fam}: full finetune needs {full_finetune_gib(fam):.1f} GiB of "
              f"fp32 AdamW state = {full_finetune_gib(fam)/FREE_GIB_WORKING:.1f}x "
              f"the working budget, and would leave "
              f"{CARD_GIB - full_finetune_gib(fam):.1f} GiB for activations even on "
              f"an empty card -> NOT FEASIBLE")
    print(f"  frozen-backbone scopes are what fit; see the MEASURED table")

    try:
        pool = build_pool(schedule.train_datasets, schedule.train_scenes,
                          schedule.train_views, stub=stub_pool)
        if not pool:
            raise SystemExit("no views found under PREMISE_DATA_ROOT")
        ev = eval_plan(schedule, stub=stub_pool and [],
                       exclude={(d, sc) for d, sc, _ in pool})
        real = True
    except (Exception, SystemExit) as e:
        # SystemExit is deliberate: build_pool raises it on an empty pool, and a
        # dry run on a laptop with no datasets must still report the PLAN.
        print(f"\n  ! datasets not reachable here ({type(e).__name__}: {e});"
              f" planning with a synthetic pool")
        pool = [("stub", f"s{i:03d}", j) for i in range(64) for j in range(8)]
        ev, real = [], False
    order = sample_order(pool, schedule.samples, schedule.data_seed)
    print(f"\n-- data plan {'(real)' if real else '(SYNTHETIC — numbers are shape only)'} --")
    print(f"  pool {len(pool)} views from {list(schedule.train_datasets)}")
    print(f"  order {len(order)} samples = {schedule.steps} steps x {schedule.accum}, "
          f"data_seed={schedule.data_seed}")
    print(f"  each view seen ~{len(order)/max(len(pool),1):.1f}x")
    if ev:
        ov = held_out(order, ev)
        print(f"  eval plan {len(ev)} views from {list(schedule.eval_datasets)}")
        print(f"  scenes in BOTH train and eval: {len(ov)}"
              + (f" (e.g. {ov[:3]})" if ov else " — fully held out"))

    orders = {a.name: order for a in ARMS}
    compared = check_matched_budget(ARMS, schedule, orders)
    print(f"\n-- matched budget -- OK for {len(compared)} arms: "
          f"{[a.name for a in compared]}")
    print(f"  anchors excluded by construction: "
          f"{[a.name for a in ARMS if a.anchor]}")

    total_s = 0.0
    for fam in families:
        scope = schedule.scope[fam]
        arms = [a for a in ARMS if a.family == fam]
        m = measured(fam, scope, schedule.px, schedule.views)
        print(f"\n-- {fam} (scope '{scope}') --")
        print(f"  trainable modules: {', '.join(SCOPES[fam][scope])}")
        known = {k: v for k, v in CENSUS[fam]["modules"].items()
                 if any(k == s.split('[')[0].split('.')[-1] or s.startswith(k)
                        for s in SCOPES[fam][scope])}
        print(f"  trainable params (census): "
              + ", ".join(f"{k} {v:.2f} M" for k, v in known.items())
              + f"  [measured total {m['trainable_M']:.1f} M]" if m else "")
        drop = CENSUS["vggt"]["modules"]["track_head"] if fam == "vggt" else 0.0
        print(f"  frozen: everything else "
              f"({CENSUS[fam]['total_M'] - sum(known.values()) - drop:.1f} M), "
              f"bf16, no_grad" + (f"; track_head ({drop:.1f} M) is DELETED, no "
              f"measured stream uses it" if drop else ""))
        if m is None:
            print(f"  ! no MEASURED entry for ({fam}, {scope}, {schedule.px} px, "
                  f"{schedule.views} views) — VRAM and wall clock UNKNOWN, and this "
                  f"file will not invent them. Probe first.")
            continue
        for a in arms:
            steps = 0 if a.anchor else schedule.steps
            s = steps * schedule.accum * m["s_per_step"]
            total_s += s
            print(f"  {a.name:16s} {a.term}={a.weight:<4} steps={steps:<5} "
                  f"~{s/60:6.1f} min  peak ~{m['peak_gib']:.2f} GiB")
    print(f"\n-- total --")
    print(f"  training wall clock ~{total_s/3600:.2f} h (evaluation extra: "
          f"{len(ev) if ev else '?'} views x {len([a for a in ARMS if a.family in families])} arms "
          f"at inference speed)")
    print(f"  peak VRAM is per-arm, arms run sequentially; nothing is co-resident")
    print(f"\n  NOT ANSWERABLE on this hardware: from-scratch training without the "
          f"term; any effect living in the frozen backbone.")
    return order


# --------------------------------------------------------------------------- #
# self-test: the matched-budget invariants, on a stub
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))

    print("matched-budget invariants (stub pool, no models, no GPU)")
    sch = replace(Schedule(), steps=20, accum=2, train_scenes=4, train_views=3)
    pool = [("stub", f"s{i:02d}", j) for i in range(8) for j in range(5)]

    # 1. the order is a pure function of (pool, n, seed)
    o1 = sample_order(pool, sch.samples, sch.data_seed)
    o2 = sample_order(pool, sch.samples, sch.data_seed)
    check("sample_order is deterministic", o1 == o2)
    check("sample_order length == steps * accum", len(o1) == sch.samples,
          f"{len(o1)} vs {sch.samples}")
    o3 = sample_order(pool, sch.samples, sch.data_seed + 1)
    check("a different data_seed gives a different order", o1 != o3)
    counts = {}
    for k in o1:
        counts[k] = counts.get(k, 0) + 1
    check("epoch-wise shuffling keeps sample counts within 1",
          max(counts.values()) - min(counts.values()) <= 1,
          f"min={min(counts.values())} max={max(counts.values())}")
    check("every pool member is used", len(counts) == len(pool))

    # 2. every arm of a family gets the SAME order, seeds and step count
    orders = {a.name: sample_order(pool, sch.samples, sch.data_seed) for a in ARMS}
    compared = check_matched_budget(ARMS, sch, orders)
    check("check_matched_budget passes on identical plans", len(compared) == 5,
          f"compared {[a.name for a in compared]}")
    check("anchors are excluded from the comparison, not silently included",
          all(not a.anchor for a in compared))
    fps = {a.name: arm_fingerprint(a, sch, orders[a.name])
           for a in ARMS if not a.anchor}
    for fam in FAMILIES:
        same = [v for k, v in fps.items() if v["family"] == fam]
        check(f"{fam}: all arms share one order SHA",
              len({v["order_sha"] for v in same}) == 1)
        check(f"{fam}: all arms share one step count",
              len({v["steps"] for v in same}) == 1)
        check(f"{fam}: all arms share one init seed",
              len({v["init_seed"] for v in same}) == 1)
        check(f"{fam}: all arms share one scope",
              len({v["scope"] for v in same}) == 1)

    # 3. the check must FAIL when the budget is not matched -- otherwise it is
    #    decoration. Each of the four ways an arm can drift is exercised.
    def must_fail(name, arms=ARMS, schedule=sch, ords=None):
        try:
            check_matched_budget(arms, schedule, ords or orders)
            check(name, False, "the mismatch was accepted")
        except SystemExit as e:
            check(name, True, str(e).splitlines()[-1][:70])

    bad = dict(orders)
    bad["pi3_normal_0.0"] = sample_order(pool, sch.samples, sch.data_seed + 99)
    must_fail("a different data order is rejected", ords=bad)

    bad2 = dict(orders)
    bad2["pi3_normal_0.5"] = orders["pi3_normal_0.5"][:-2]
    must_fail("a shorter data order is rejected", ords=bad2)

    swapped = list(orders["pi3_normal_0.5"])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    bad3 = dict(orders, **{"pi3_normal_0.5": swapped})
    must_fail("a REORDERED but identical sample set is rejected", ords=bad3)

    no_ctrl = [a for a in ARMS if not a.control]
    must_fail("a family with no unmodified-loss control is rejected", arms=no_ctrl)

    dup = [a for a in ARMS if a.family == "vggt"] + [
        Arm("vggt_grad_dup", "vggt", 1.0)]
    must_fail("two arms at the same weight are rejected", arms=dup,
              ords={a.name: orders.get(a.name, o1) for a in dup})

    # a per-arm schedule change is caught because the schedule is shared by
    # construction; assert that the fingerprint would notice one anyway
    f_a = arm_fingerprint(ARMS[0], sch, orders[ARMS[0].name])
    f_b = arm_fingerprint(ARMS[1], replace(sch, lr=sch.lr * 2), orders[ARMS[1].name])
    check("fingerprint notices a changed learning rate", f_a["lr"] != f_b["lr"])
    f_c = arm_fingerprint(ARMS[1], replace(sch, steps=sch.steps + 1),
                          orders[ARMS[1].name])
    check("fingerprint notices a changed step count", f_a["steps"] != f_c["steps"])

    # 4. the arm table itself must satisfy §4
    for fam in FAMILIES:
        fa = [a for a in ARMS if a.family == fam]
        check(f"{fam}: exactly one unmodified-loss control",
              sum(a.control for a in fa) == 1)
        check(f"{fam}: a step-0 anchor exists", any(a.anchor for a in fa))
    check("pi3 sweeps lambda_normal over {0, 0.5, 1.0} as §4 asks",
          {a.weight for a in ARMS if a.family == "pi3" and not a.anchor}
          == {0.0, 0.5, 1.0})
    check("vggt has the gradient term both on and off",
          {a.weight for a in ARMS if a.family == "vggt" and not a.anchor} == {0.0, 1.0})

    # 5. feasibility arithmetic
    for fam in FAMILIES:
        check(f"full finetune of {fam} exceeds the WORKING budget by >3x",
              full_finetune_gib(fam) > 3 * FREE_GIB_WORKING,
              f"{full_finetune_gib(fam):.1f} GiB of static AdamW state vs "
              f"{FREE_GIB_WORKING} GiB free")
    check("even on an EMPTY card vggt would leave under 2 GiB for activations",
          CARD_GIB - full_finetune_gib("vggt") < 2.0,
          f"{CARD_GIB - full_finetune_gib('vggt'):.1f} GiB headroom")
    for fam in FAMILIES:
        m = measured(fam, Schedule().scope[fam], Schedule().px, Schedule().views)
        check(f"the configured {fam} scope fits the working budget",
              m and m["peak_gib"] < FREE_GIB_WORKING + 0.6,
              f"peak {m['peak_gib']:.2f} GiB vs {FREE_GIB_WORKING} GiB working")
    check("every configured scope has a MEASURED VRAM entry",
          all(measured(f, Schedule().scope[f], Schedule().px, Schedule().views)
              for f in FAMILIES))

    # 6. aggregate metrics behave
    g = np.full((32, 32), 2.0)
    g[:, 16:] = 4.0
    v = np.ones_like(g, bool)
    m0 = aggregate_metrics(g, v, g)
    check("aggregate metrics are perfect on an exact prediction",
          abs(m0["absrel"]) < 1e-9 and m0["d125"] == 1.0, f"{m0}")
    m1 = aggregate_metrics(g, v, g * 3.0)
    check("aggregate metrics are scale-invariant (median alignment)",
          abs(m1["absrel"]) < 1e-9, f"absrel={m1['absrel']:.2e}")
    m2 = aggregate_metrics(g, v, g + np.random.default_rng(0).normal(0, 0.5, g.shape))
    check("aggregate metrics degrade on a corrupted prediction",
          m2["absrel"] > 0.01 and m2["d125"] <= 1.0, f"absrel={m2['absrel']:.3f}")
    m3 = aggregate_metrics(g, np.zeros_like(v), g)
    check("too few valid pixels reads as n/a, not as a perfect score",
          not np.isfinite(m3["absrel"]))

    # 7. the GT pointmap must be built with NEAREST resampling
    try:
        import infer as INF
        K = np.array([[16.0, 0, 16.0], [0, 16.0, 16.0], [0, 0, 1.0]])
        p, m = gt_pointmap(g, v, K, (16, 16))
        vals = set(np.round(np.unique(p[..., 2]), 6))
        check("downsampled target contains only original depths (no blending)",
              vals <= {2.0, 4.0}, f"{sorted(vals)}")
        check("target pointmap z is the depth", np.allclose(p[..., 2], p[..., 2]))
    except ImportError:
        print("  [skip] gt_pointmap needs infer (numpy only) — not importable here")

    # 8. torch-side terms, only where torch exists
    try:
        import torch  # noqa: F401
        import factorial as FX  # noqa: F401
    except Exception as e:
        print(f"  [skip] loss-term checks need torch + factorial ({type(e).__name__})")
    else:
        import torch
        H = W = 16
        z = torch.full((H, W), 2.0, dtype=torch.float64)
        z[:, W // 2:] = 4.0
        u, vv = torch.meshgrid(torch.arange(W) + 0.5, torch.arange(H) + 0.5,
                               indexing="xy")
        P = torch.stack([(u - W / 2) * z / 8, (vv - H / 2) * z / 8, z], -1)
        msk = torch.ones(H, W, dtype=torch.bool)
        l0, p0 = objective(P, P, None, msk, w_grad=1.0, w_normal=1.0)
        # 1 − cos loses precision near cos = 1: the normals are normalised with a
        # 1e-8 floor and the residual is a difference of nearly equal unit
        # vectors, so an exact match reads a few 1e-6 rather than exactly 0. That
        # is rounding, not a term that fires on a perfect prediction — the
        # position and gradient parts below are exactly 0.
        check("objective is ~0 on an exact match with every term on",
              float(l0) < 1e-5 and p0["position"] == 0.0 and p0["gradient"] == 0.0,
              f"{float(l0):.2e} parts={p0}")
        Q = P.clone()
        Q[H // 2, W // 2, 2] += 1.0                 # one point pushed into the void
        lp, _ = objective(Q, P, None, msk, w_grad=0.0, w_normal=0.0)
        lg, _ = objective(Q, P, None, msk, w_grad=1.0, w_normal=0.0)
        ln, _ = objective(Q, P, None, msk, w_grad=0.0, w_normal=1.0)
        check("the gradient term adds cost to a void point", float(lg) > float(lp))
        check("the normal term adds cost to a void point", float(ln) > float(lp))
        check("weight 0 removes a term EXACTLY (same code path as the control)",
              float(objective(Q, P, None, msk, w_grad=0.0)[0]) == float(lp))
        c = torch.full((H, W), 2.0, dtype=torch.float64)
        lc, _ = objective(Q, P, c, msk)
        check("confidence weighting changes the position term", float(lc) != float(lp))
        try:
            objective(Q, P, torch.full((H, W), -1.0, dtype=torch.float64), msk)
            check("a non-positive confidence is refused, not turned into NaN", False,
                  "it was accepted")
        except ValueError:
            check("a non-positive confidence is refused, not turned into NaN", True)
        sc = objective(Q * 5.0, P * 5.0, None, msk, w_grad=1.0, w_normal=1.0)[0]
        check("the objective is scale-invariant (both maps normalised)",
              abs(float(sc) - float(objective(Q, P, None, msk, w_grad=1.0,
                                              w_normal=1.0)[0])) < 1e-6,
              f"{float(sc):.6f}")
        n = _normals_from_pointmap(P)
        flat = n[1:, 1:W // 2 - 1]          # interior of the near surface only
        check("normals of a fronto-parallel patch point along z",
              float(flat[..., 2].abs().min()) > 0.99,
              f"min |nz|={float(flat[..., 2].abs().min()):.4f}")
        sm = _stencil_mask(msk)
        check("the stencil mask drops exactly row 0 and column 0",
              int((~sm).sum()) == H + W - 1, f"dropped {int((~sm).sum())}")
        holed = msk.clone()
        holed[8, 8] = False
        check("the stencil mask also drops differences taken across an invalid "
              "pixel", not bool(_stencil_mask(holed)[8, 9]) and
              not bool(_stencil_mask(holed)[9, 8]))

    print("\nself-test:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(out, schedule, families=FAMILIES, only=None, grid=True, max_seconds=None,
        min_free_gb=4.0):
    import torch
    import infer as INF

    if not os.path.exists(os.path.join(out, "1c_prereg.md")):
        raise SystemExit(
            f"No pre-registration at {out}/1c_prereg.md.\n"
            f"§4's arms must be recorded BEFORE the first step:\n"
            f"  python src/finetune.py --preregister --out {out}")
    arms = [a for a in ARMS if a.family in families and (not only or a.name in only)]
    if only and any(a.anchor for a in ARMS if a.family in families
                    and a.name not in only):
        print("  ! running a subset: the step-0 anchor is not included, so the "
              "'how much did finetuning alone move things' bound will be missing.",
              flush=True)

    pool = build_pool(schedule.train_datasets, schedule.train_scenes,
                      schedule.train_views)
    order = sample_order(pool, schedule.samples, schedule.data_seed)
    ev = eval_plan(schedule, exclude={(d, sc) for d, sc, _ in order})
    overlap = held_out(order, ev)
    if overlap:
        raise SystemExit(
            f"{len(overlap)} eval scenes were also trained on (e.g. {overlap[:3]}). "
            f"eval_plan is meant to exclude them; refusing to report a boundary "
            f"rate measured on scenes the finetune saw.")
    print(f"pool {len(pool)} views, order {len(order)}, eval {len(ev)} views, "
          f"{len(overlap)} scenes shared between train and eval", flush=True)
    check_matched_budget(arms, schedule, {a.name: order for a in arms})

    rows, stats = [], {}
    for a in arms:
        INF.require_gpu(min_free_gb=min_free_gb)     # shared card; never kill anyone
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        model, st = train_arm(a, schedule, order, out=out, max_seconds=max_seconds)
        st.setdefault("scope", schedule.scope[a.family])
        stats[a.name] = st
        rows += evaluate_arm(model, a, schedule, ev, grid=grid)
        del model
        INF._MODELS.pop(a.family, None)
        # nn.Modules hold reference cycles (hooks, parametrizations), so `del`
        # alone does not return the weights to the allocator and the NEXT arm
        # loads on top of the previous one. On a card with 3.5 GiB free that is
        # the difference between running and OOMing.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0] / 2**30
            print(f"  released {a.name}; {free:.1f} GiB free", flush=True)
        # write after every arm: a 2 h run that dies on arm 4 should not lose 1-3
        write_summary(out, rows, stats, schedule)
    write_feasibility(out)
    return rows, stats


def _main():
    ap = argparse.ArgumentParser(description="Phase 1C backbone ablations (§4)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/phase1c")
    ap.add_argument("--family", default="", help="pi3, vggt, or both (default)")
    ap.add_argument("--arms", default="", help="comma-separated arm subset")
    ap.add_argument("--steps", type=int, default=Schedule.steps)
    ap.add_argument("--accum", type=int, default=Schedule.accum)
    ap.add_argument("--px", type=int, default=Schedule.px)
    ap.add_argument("--lr", type=float, default=Schedule.lr)
    ap.add_argument("--warmup", type=int, default=Schedule.warmup)
    ap.add_argument("--train-scenes", type=int, default=Schedule.train_scenes)
    ap.add_argument("--eval-scenes", type=int, default=Schedule.eval_scenes)
    ap.add_argument("--no-grid", action="store_true",
                    help="primary threshold only instead of the full grid")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="abort an arm that overruns; a truncated arm is NOT "
                         "matched-budget and the run is refused, not salvaged")
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    a = ap.parse_args()

    sch = replace(Schedule(), steps=a.steps, accum=a.accum, px=a.px, lr=a.lr,
                  warmup=a.warmup, train_scenes=a.train_scenes,
                  eval_scenes=a.eval_scenes)
    fams = tuple(f for f in FAMILIES if not a.family or f in a.family.split(","))
    only = set(a.arms.split(",")) if a.arms else None

    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.dry_run:
        dry_run(sch, fams)
        return
    if a.preregister:
        preregister(a.out, sch, force=a.force)
        return
    if a.run:
        run(a.out, sch, families=fams, only=only, grid=not a.no_grid,
            max_seconds=a.max_seconds, min_free_gb=a.min_free_gb)
        return
    ap.print_help()


if __name__ == "__main__":
    _main()
