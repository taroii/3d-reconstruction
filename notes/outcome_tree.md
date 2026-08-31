# Project outcome tree — diagnosis-first paper

Reference copy of the branching plan. Text version below; `OUTCOME_TREE.svg` is the diagram.

```
                    PREMISE CONFIRMED
                 8-31x over GT control
                (Middlebury GT 0.00028 vs models 0.25-0.32)
                            |
                            v
              TEST 1 - GENERALITY + CALIBRATION
       add DUSt3R / MASt3R (independent family)
       + fine-structure recalibration of the rate
                            |
             +--------------+--------------+
             v                             v
        UNIVERSAL                    FAMILY-SPECIFIC
    holds across families        narrower, still informative
             |                             |
             +--------------+--------------+
                            v
                  TEST 2 - ATTRIBUTION
            target / objective / hypothesis class
                            |
      +-------------+-------+-------+-------------+
      v             v               v             v
  OBJECTIVE       TARGET          MIXED        RESIDUAL
 aux coupling   supervision      several     smoothness
   terms          dirty        contribute       only
      |             |               |             |
      +------+------+               v             v
             v               SOLID PAPER      THIN - PIVOT
        STRONG PAPER        attribution w/    benchmark or
      corrects the field     effect sizes       workshop
```

## Notes on each branch

**Test 1** is cheap (inference only) and mostly de-risking — it widens or narrows the
claim but does not kill the project. It also fixes the independence problem: pi3
initializes from pretrained VGGT and freezes that encoder, so the three current
streams share a visual frontend and are not independent witnesses. DUSt3R/MASt3R are
a different lineage (CroCo-pretrained) and are clean on Middlebury, Infinigen, iBims,
ETH3D.

Fine-structure calibration is folded in here rather than treated separately: the
measured 0.25 rate includes real scene geometry that the two-surface metric
miscounts (genuine third surfaces in foliage and thin structure). If recalibration
drops the number substantially, the headline weakens regardless of which branch
Test 2 lands on.

**Test 2** decides the paper's fate.

- OBJECTIVE branch is now the leading hypothesis (see the loss-form finding: the
  position term cannot be responsible, but VGGT's gradient term and pi3's normal
  loss couple neighbouring pixels and can be). Best case: corrects a stated belief
  in the field and is actionable.
- TARGET branch is not a failure. If the supervision itself is contaminated at
  boundaries and models faithfully reproduce it, that implicates every benchmark in
  the field rather than one loss term - arguably a bigger paper. The premise check
  already found nonzero GT rates on Infinigen (0.025) and iBims (0.032), so this
  branch has real probability mass.
- RESIDUAL branch is the one to fear: "networks are smooth and cannot represent a
  step" is true, unsurprising, and thin. Running the cheap objective tests early is
  the fastest way to learn which side of the tree we are on.

## Standing decision rule

No fix is attempted until attribution returns a result. A fix for an unknown cause is
not a contribution.