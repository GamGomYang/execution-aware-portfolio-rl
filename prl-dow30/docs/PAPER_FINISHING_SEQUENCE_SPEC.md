# Paper Finishing Sequence Spec

## Current Locked State

- Base rebuild run: `prl-dow30/outputs/paper_rebuild_20260324T065755Z`
- Validation-selected operating point: `eta=0.2`
- Safe main claim:
  - execution-aware formulation is cost-aligned
  - validation-selected `eta=0.2` remains valid on held-out test
  - the selected operating point is competitive in net Sharpe and turnover efficiency
- Unsafe claim for the current version:
  - general execution-aware training superiority

## Working Rule

- Steps `1` through `7` and `9` should be treated as finishing work on top of the current locked run.
- Most of this work does **not** require retraining.
- New training runs should be avoided unless a later step explicitly calls for them.

## Step 1. Final Audit of Numbers and Tables

### Goal

Catch any small inconsistency between tables, prose, captions, and generated artifacts before submission.

### Main Files

- `02.17.01.tex`
- `prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_eta1.csv`
- `prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/test_selected_vs_external_baselines.csv`
- `prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/tables/diagnostic_selected_eta.csv`
- `prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/stats/selected_eta_vs_eta1_stats.csv`
- `prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/diagnostics/diagnostic_selected_eta_v2.csv`

### Must Check

- Re-verify the `kappa=0` row in the diagnostic table.
- Confirm whether `tracking_error_l2` is non-zero while `mean_abs_return_gap`, `final_equity_gap`, and `max_abs_daily_gap` are zero because of:
  - rounding
  - print precision
  - trace-definition degeneracy
  - a bug
- Add a table note clarifying that a marginal median and a paired median are different objects.
- Check that `eta=0.2` is used consistently in:
  - abstract
  - results
  - discussion
  - conclusion
  - figure captions
  - appendix
- Check for copy/paste mistakes in:
  - Sharpe
  - CAGR
  - turnover
  - baseline vs selected-arm labels
- Check that validation-selected `eta` and held-out reported `eta` are not mixed up anywhere.

### Output

- Cleaned table notes in `02.17.01.tex`
- A short audit note in the working log or commit message summarizing what was checked

### Done When

- A first-time reader would not suspect a numeric inconsistency.
- Table values and prose values match exactly.

## Step 2. Final Prose Pass

### Goal

Make the manuscript read like a submission draft rather than an experiment memo.

### Focus Areas

- Abstract:
  - frozen-policy scope is explicit
  - `eta=0.2` is consistently described as validation-selected
  - “Sharpe stronger than CAGR” is visible
- Results:
  - positive-cost evidence is clearly strong
  - `kappa=0` remains explicitly weak or modest
- Discussion:
  - “realized-path stabilization, not alpha creation” stays central
- Conclusion:
  - no claim inflation
  - no hidden upgrade to execution-aware training superiority

### Tone Lock

- Strong:
  - cost-alignment necessity
- Medium:
  - positive-cost net Sharpe improvement
- Weak:
  - `kappa=0` stabilization-consistent evidence
- Future work:
  - training-time alignment

### Done When

- The writing feels natural and controlled.
- No section sounds stronger than the evidence it cites.

## Step 3. Related Work Expansion

### Goal

Strengthen the paper’s literature positioning.

### Required Axes

- portfolio RL
- transaction-cost-aware portfolio optimization
- execution / partial adjustment / trading frictions
- smoothing / control filtering / regularization

### Core Positioning Sentence

- Prior work often penalizes turnover at the action level.
- This paper distinguishes target actions from realized executed holdings.
- Therefore transaction costs are not only a penalty-tuning issue, but a realized-path accounting issue.

### Done When

- The reader can immediately see where the paper sits in the literature.
- Citations support the gap statement rather than merely increasing reference count.

## Step 4. Data / Protocol Appendix Strengthening

### Goal

Preempt reviewer questions about the backtest setup.

### Must Include

- Dow30 fixed snapshot usage
- survivorship-bias limitation
- close-to-close timing convention
- why adjusted close is used
- long-only fully-invested simplex limitation
- no-cash-asset limitation

### Good Additions

- decision at time `t`, realized return over `(t, t+1]`
- rebalancing and cost-application order
- how validation/test splits were frozen

### Done When

- It is hard for a reviewer to say the backtest protocol is underspecified.

## Step 5. Terminology and Claim Consistency Audit

### Goal

Ensure the paper uses one stable term per concept and that each contribution is matched by a limitation.

### Terminology Audit

- `target portfolio` / `target weights` / `policy output`
- `executed portfolio` / `realized portfolio` / `executed weights`
- `immediate execution baseline` / `baseline arm` / `eta=1.0 arm`
- `selected operating point` / `selected eta` / `validation-selected eta=0.2`
- `net Sharpe` / `executed-path Sharpe` / `sharpe_net_lin`

### Contribution-Limitation Audit

- cost-aligned executed turnover accounting
- frozen-policy empirical scope
- no universal retraining-superiority claim

### Done When

- The same concept is not read under multiple competing names.
- Contributions and limitations feel paired rather than contradictory.

## Step 6. Statistics Pass, Round 2

### Goal

Raise the credibility of the paired test table beyond directional wins alone.

### Already Present

- IQR
- win-rate
- exact sign test

### Still Recommended

- Wilcoxon signed-rank
- bootstrap confidence interval

### Main Target

- the selected-vs-eta1 held-out comparison table

### Done When

- The paired effect size and uncertainty are both visible.

## Step 7. Figure Caption and Annotation Refinement

### Goal

Make every figure self-contained and immediately interpretable.

### Check

- representative seed criterion is short and explicit
- selected `eta=0.2` is visually easy to find
- captions state:
  - what is being compared
  - which split is shown
  - whether quantities are median, paired, or representative

### Priority Figures

- Figure 1
- Figure 2
- Figure 3

### Done When

- The figures can mostly be understood without reading the surrounding paragraph.

## Step 8. Add 1--2 More External Baselines

### Goal

Strengthen the heuristic reference set if time permits.

### Recommended Additions

- minimum-variance
- simple mean-variance heuristic

### Priority Note

- This is useful, but less urgent than numeric integrity and manuscript consistency.

### Done When

- A reviewer is less likely to question whether the heuristic comparator set is too weak.

## Step 9. Final Submission-Format QA

### Goal

Avoid losing credibility on formatting and polish.

### Must Check

- table numbering continuity
- figure numbering continuity
- appendix cross-references
- axis, legend, and unit consistency
- reference formatting consistency
- punctuation consistency in captions
- notation consistency for `eta=0.2` and `kappa=10^{-3}`
- no TODO, placeholder, or internal note text

### Done When

- The paper no longer looks like a draft with internal scaffolding exposed.

## Step 10. Optional Expansion: Training-Time Alignment Experiments

### Goal

Upgrade the paper’s scope only if there is time and appetite for a larger empirical claim.

### Candidate Experiments

- immediate-exec training vs execution-aware training
- wrong-cost training vs correct-cost training

### Important Scope Warning

- This step expands the paper beyond the currently locked frozen-policy scope.
- It should not be treated as a finishing step.

### Done When

- Only if the paper is intentionally being upgraded to support a broader training-time claim.

## Additional Audit Requirements

### A. Self-Contained Tables and Figures

- Every table and figure should state:
  - what is compared
  - which split or regime is used
  - whether the statistic is median or mean
  - whether the comparison is paired or marginal

### B. Reference-Claim Linking

- Related work citations must connect to:
  - the introduction gap statement
  - the method positioning
  - the discussion claim

### C. Numeric Integrity Before Literature Breadth

- If time becomes constrained, prioritize:
  - Step 1 over Step 3
  - Step 2 over Step 8

## Recommended Execution Order

1. Final audit of numbers and tables
2. Final prose pass
3. Related work expansion
4. Data / protocol appendix strengthening
5. Terminology and claim consistency audit
6. Statistics pass, round 2
7. Figure caption and annotation refinement
8. Add 1--2 more external baselines
9. Final submission-format QA
10. Optional training-time alignment experiments

## Practical Guidance for the Next Interactive Passes

- Next pass should begin with Step 1.
- Step 1 should update `02.17.01.tex` notes and captions if any numeric ambiguity is found.
- Step 2 should happen immediately after Step 1 while the numbers are still fresh.
- Steps 3 and 4 can then be treated as one literature-plus-protocol pass.
- Step 5 should happen after Steps 2 through 4, because terminology consistency is easiest to audit after content has stabilized.
- Step 6 and Step 7 are polishing amplifiers, not blockers.
- Step 8 should only happen if there is still time after the paper is already coherent.
- Step 9 is the final gate before submission.
