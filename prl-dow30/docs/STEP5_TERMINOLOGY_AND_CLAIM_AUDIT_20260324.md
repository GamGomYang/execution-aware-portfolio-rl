# Step 5 Terminology and Claim Audit

## Goal

Ensure that the manuscript uses one stable name per core concept and that each contribution is paired with a matching limitation.

## Terminology Locks

- `target portfolio`
  - the policy output `w_tgt`
- `executed portfolio`
  - the actually held portfolio `w_exec`
- `immediate-execution baseline`
  - the `eta=1.0` comparison arm
- `validation-selected operating point`
  - the operating point selected on validation, here `eta=0.2`
- `net Sharpe`
  - the annualized Sharpe ratio computed from executed net linear returns
  - `sharpe_net_lin` is treated as the implementation label for the same object

## Main Claim / Limitation Pairing

- Contribution 1:
  - cost-aligned executed-path accounting
- Matching limitation:
  - established under a daily close-to-close abstraction, fixed universe snapshot, and no-cash long-only simplex

- Contribution 2:
  - fixed-eta execution mapping analysis
- Matching limitation:
  - adaptive execution schedules and richer control laws are not yet studied

- Contribution 3:
  - frozen-policy empirical study with locked validation-first selection
- Matching limitation:
  - no execution-aware retraining superiority claim and only a modest matched external baseline set

## Main Outcome

- The paper now reads more consistently:
  - less switching between `reference arm`, `baseline arm`, and `eta=1.0 arm`
  - less switching between `selected eta`, `selected operating point`, and `validation-selected eta`
  - clearer distinction between concept names in prose and implementation names in code

## Remaining Note

- Step 6 can now build on a cleaner terminology layer when adding Wilcoxon / bootstrap reporting.
