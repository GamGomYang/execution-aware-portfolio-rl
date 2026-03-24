# Step 4 Data / Protocol Appendix Pass

## Goal

Make the backtest protocol self-contained enough that a reviewer can understand the empirical interface without reverse-engineering the code.

## What Was Added

- A dedicated appendix section in the manuscript for:
  - fixed universe and adjusted-close data
  - close-to-close timing and decision order
  - frozen train/validation/test split lock
  - frozen signal snapshot
  - long-only fully invested simplex
  - no-cash-asset limitation
  - backtest scope limitations

## Key Clarifications Now Present

- The paper uses a fixed 27-name Dow-style snapshot, not a point-in-time constituent reconstruction.
- Adjusted close is used so corporate actions are reflected in returns rather than as artificial jumps.
- The timing convention is explicit:
  - decide at close `t`
  - form executed weights
  - charge cost on executed turnover
  - realize return over `(t, t+1]`
- The split lock is explicit:
  - train `2010--2021`
  - validation `2022--2023`
  - test `2024--2025`
- The frozen signal snapshot is explicit:
  - `reversal_5d`
  - `short_term_reversal`
- Internal 2026 forward checks are explicitly excluded from paper model selection and tables.

## Why This Matters

- It reduces ambiguity about:
  - data source
  - timing
  - cost application order
  - selection protocol
  - empirical scope limitations
- It makes the paper easier to defend against common reviewer questions on backtest protocol clarity.

## Remaining Note

- PDF compilation has still not been verified in this environment because no TeX engine is installed.
