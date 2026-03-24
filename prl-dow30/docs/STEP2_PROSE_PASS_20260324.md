# Step 2 Prose Pass

## Goal

Lock the manuscript tone to match the evidence from the rebuilt frozen-policy run.

## Main Changes

- Reframed the abstract as a `frozen-policy empirical protocol` rather than a broad training claim.
- Made the main asymmetry explicit:
  - strong evidence when `kappa > 0`
  - modest evidence when `kappa = 0`
- Clarified that the external-baseline result is about:
  - net Sharpe
  - turnover efficiency
  - realized-path quality
  - not universal CAGR dominance
- Strengthened the discussion emphasis on:
  - realized-path stabilization
  - cost alignment
  - not alpha creation
- Tightened the conclusion so that the main claim is:
  - cost-aligned execution control matters
  - validation-selected `eta=0.2` transfers to held-out test
  - the selected operating point is competitive on risk-adjusted metrics
  - no general retraining-superiority claim is made

## Sections Updated

- abstract
- held-out selected-vs-baseline results prose
- external-baseline interpretation paragraph
- discussion
- conclusion

## Tone Lock After Step 2

- Strong:
  - cost-alignment necessity
- Medium:
  - positive-cost net Sharpe improvement
- Weak:
  - `kappa=0` stabilization-consistent evidence
- Future work:
  - training-time alignment and broader comparator sets
