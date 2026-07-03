# Known methodological caveats (read before the next submission)

## 1. Our baseline is NOT D2USt3R's method. We do not improve upon D2USt3R.

**What we did.** We finetuned the *released D2USt3R checkpoint* with the plain
DUSt3R confidence-weighted regression that ships as the default criterion in the
released inference/eval code:

    ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)

and reweighted its per-pixel position residual by `w_i = 1 + gamma*|K_i|`
(gamma=0 = baseline, gamma=1 = curvature).

**Why this is wrong for the paper's framing.** D2USt3R **never released training
code**; its training objective was meant to be reimplemented from the paper. That
objective is (paper Eqs. 6-9):

- `L_static`: confidence-weighted L2 regression, view 1 all pixels + view 2
  **static** pixels only (masked by `1 - M_dyn`), against the rigid GT pointmap.
- `L_dyn`: dynamic, non-occluded pixels regressed against the **flow-aligned** GT
  point `X_bar^{1,1}_{i+b(i)}` (SDAP), with occlusion mask `M_occ`, plus the
  symmetric term.
- `L_total = L_static + L_dyn`.

Our `L_base` regresses **all** valid pixels of both views against the **rigid**
target. Decomposed against D2USt3R's split:

    L_base = [view-1 all + view-2 static]            (= L_static's pixels/targets)
           + [view-2 dynamic vs. the RIGID target]   (D2USt3R routes these to
                                                        L_dyn with a FLOW-ALIGNED
                                                        target instead)

So:
- gamma=0 is **plain DUSt3R regression on the D2USt3R checkpoint**, not D2USt3R.
- We **never implement `L_dyn`** (no SDAP flow-aligned targets, no M_occ in the
  loss). On the dynamic pixels that are the entire point of D2USt3R, we supervise
  against the rigid (static-assumption) target that D2USt3R was built to fix.
- Therefore "curvature-aware D2USt3R" and "improves dynamic reconstruction over
  D2USt3R" are **unsupported**. We reweighted a plain position regression and
  compared it to itself.

**What the experiments DO still support (the narrow, honest claim).** Within a
matched comparison (same loss, same data, same seeds), curvature-weighting the
position residual gives a small zero-shot depth improvement on Sintel, matched by
a first-order gradient control. That internal A/B is valid; the "vs. D2USt3R"
framing is not.

## Fix path for the next submission

1. **Reimplement D2USt3R's training loss** (`L_static + L_dyn`, Eqs. 6-9) so that
   gamma=0 is an actual D2USt3R reproduction. Most building blocks already exist
   from the (abandoned) L_cc exploration:
   - `curv/dynamic.py`: `cam_flow`, `dynamic_mask` (M_dyn, Eq. 5), `occlusion_mask`
     (M_occ, Eq. 4), `sdap_target` (flow-aligned dynamic target).
   - `curv/cc_loss.py`: `flow_warp`.
   What is missing is wiring these into a real training criterion (static term
   masked by 1-M_dyn on view 2; dynamic term against the flow-warped GT with
   M_dyn*(1-M_occ)) and the GT-flow dataloaders.
2. **GT dense flow per dataset.** Spring ships it (loader + verified scale in
   `curv/spring.py`). PointOdyssey ships only sparse tracks (needs densification
   or substitution). TartanAir is static (M_dyn empty).
3. **Validate the reproduction** against D2USt3R's reported numbers BEFORE claiming
   any improvement on top of it.
4. **Then** rerun the matched arms: gamma=0 = our D2USt3R reproduction,
   gamma=1 = + curvature weight. Only then is "curvature on top of D2USt3R" a
   legitimate claim.

## 2. Curvature is not motion-invariant for non-rigid content (relevant if L_cc revisited)

Mean curvature is invariant only under rigid motion. For deforming objects (cloth,
articulated humans) the true curvature changes across frames, so a cross-frame
curvature-consistency term cannot separate real deformation from prediction error.
See the L_cc premise analysis in `premise_cc.py` and the git history.
