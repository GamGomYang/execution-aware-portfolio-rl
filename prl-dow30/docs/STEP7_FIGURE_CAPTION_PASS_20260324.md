# Step 7: Figure Caption / Annotation Pass (2026-03-24)

## Goal

Make the paper figures readable without heavy reliance on the main text.

## Implemented

- Strengthened the visual highlight of the validation-selected operating point in the validation frontier figure with enlarged black-edged markers.
- Rewrote all three figure captions in the manuscript to be more self-contained.

## Caption changes

- Figure 1 now explicitly states:
  - held-out split (`2024--2025`)
  - regime (`kappa = 10^-3`)
  - that the target path is a hypothetical diagnostic path
  - how the representative seed is chosen
- Figure 2 now explicitly states:
  - validation split (`2022--2023`)
  - that each color is a cost regime
  - that each point is one eta
  - x/y semantics
  - that vertical bars are half-IQR of seedwise net Sharpe
  - that `eta=0.2` is emphasized with enlarged black-edged markers
- Figure 3 now explicitly states:
  - held-out split (`2024--2025`)
  - that each point is one paired-seed difference
  - that the horizontal dotted line is zero
  - that positive-cost panels are uniformly positive while `kappa=0` is mixed

## Files changed

- `/workspace/execution-aware-portfolio-rl/prl-dow30/scripts/build_paper_figures.py`
- `/workspace/execution-aware-portfolio-rl/02.17.01.tex`

## Regenerated outputs

- `/workspace/execution-aware-portfolio-rl/prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_selected_trace.png`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_validation_frontier.png`
- `/workspace/execution-aware-portfolio-rl/prl-dow30/outputs/paper_rebuild_20260324T065755Z/paper_pack/figures/fig_seed_scatter.png`
- Legacy copies for TeX inclusion:
  - `/workspace/execution-aware-portfolio-rl/fig_misalignment.png`
  - `/workspace/execution-aware-portfolio-rl/fig_frontier.png`
  - `/workspace/execution-aware-portfolio-rl/fig_seed_scatter.png`

## Intended effect

- Figure 1 should read as a diagnostic trace rather than as a cherry-picked performance plot.
- Figure 2 should make the selected eta visible immediately.
- Figure 3 should make the paired-seed interpretation obvious, especially the contrast between `kappa=0` and positive-cost regimes.
