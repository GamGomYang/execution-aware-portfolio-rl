# Step 3 Related Work Pass

## Goal

Strengthen the literature positioning so the paper reads as a clearly placed contribution rather than as an isolated implementation note.

## What Changed

- Expanded the Related Work section from a minimal three-citation sketch into four linked axes:
  - classical portfolio choice and transaction-cost-aware optimization
  - portfolio RL and deep portfolio management
  - execution / partial adjustment / trading frictions
  - smoothing / control regularization in RL
- Added the key positioning distinction:
  - prior work often regularizes or penalizes the action path
  - this paper separates target portfolio decisions from realized executed holdings
  - therefore transaction costs are treated as an executed-path accounting problem, not only as action-level regularization

## References Added

- `jiang2017`
- `ye2020`
- `lien2023`
- `lobo2007`
- `garleanu2013`
- `almgren2000`
- `shen2020`
- `mysore2021`

## Positioning After Step 3

- The paper is now framed as:
  - adjacent to portfolio RL
  - informed by friction-aware portfolio optimization
  - closely connected to execution and partial-adjustment thinking
  - related in spirit, but not identical, to RL smoothing and policy-regularization work
- The contribution claim is now easier to read:
  - not “we have a universally better RL architecture”
  - but “we separate target and executed portfolios and evaluate on the realized executed path”

## Remaining Note

- The reference list is now structurally much stronger, but Step 4 should still add protocol detail so the empirical setup is defended as clearly as the literature positioning.
