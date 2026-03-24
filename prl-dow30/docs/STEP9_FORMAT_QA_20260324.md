# Step 9: Submission-Format QA (2026-03-24)

## Goal

Catch final presentational issues that can undermine trust or make the draft look unfinished even when the technical content is already stable.

## Static checks completed

- Removed the UTF-8 BOM at the top of `/workspace/execution-aware-portfolio-rl/02.17.01.tex`.
- Checked LaTeX label/reference integrity.
- Checked citation/bibliography integrity.
- Checked for placeholder or internal-review language such as:
  - `TODO`
  - `FIXME`
  - `TBD`
  - `placeholder`
  - `for reviewers`
  - stale appendix markers like `Appendix~A` / `Appendix~B`
- Checked notation consistency for the main `kappa` rows in the three core result tables.

## Fixes applied

- Corrected the appendix reference in the experimental setup paragraph to point to the actual data/protocol appendix:
  - `Appendix~\\ref{app:data_protocol}`
- Renamed the appendix section title:
  - from `Reproducibility Checklist (for reviewers)`
  - to `Reproducibility Checklist`
- Added appendix labels:
  - `app:repro`
  - `app:data_protocol`
  - `app:implementation_details`
- Renamed `Implementation Notes` to `Implementation Details`.
- Standardized the displayed `kappa` rows in the main held-out tables to:
  - `$0$`
  - `$5\\times10^{-4}$`
  - `$10^{-3}$`
- Updated the discussion prose so that the external-baseline paragraph matches the expanded five-baseline table rather than the older three-baseline wording.

## Verification results

- UTF-8 BOM: removed
- Undefined refs: none
- Undefined citations: none
- Reviewer/internal placeholders found after cleanup: none

## Residual risk

- This environment does not have `pdflatex`, `latexmk`, or `tectonic`, so true PDF-level QA was not possible here.
- That means page overflow, float placement, overfull boxes, and final numbering/layout still need one real TeX compile in a LaTeX-enabled environment.

## Recommended next check outside this environment

1. Compile `02.17.01.tex`.
2. Verify there are no `??` references in the PDF.
3. Check table widths and figure placements in two-column layout.
4. Check for overfull/underfull box warnings.
5. Confirm appendix section letters and table/figure numbering match the intended final submission style.
