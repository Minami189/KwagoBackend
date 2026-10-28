# Technical Documentation & System Specifications

This directory contains technical documentation and mathematical specifications for KwagoBackend.

## Documents

1. **[`url_scoring_explanation.md`](url_scoring_explanation.md)**:
   - Detailed walkthrough of the VirusTotal threat scoring algorithm.
   - Maps category ratings (`malicious`, `suspicious`, `undetected`, `harmless`) to numerical values.
   - Explains vendor reputation weighting and single-vendor consensus boosts.

2. **[`weighted_verdict_plan.md`](weighted_verdict_plan.md)**:
   - Initial engineering architecture plan for multi-engine VirusTotal consensus scoring.
   - Outlines threshold boundaries: Malicious ($\ge 0.65$), Suspicious ($0.35 - 0.65$), Benign ($< 0.35$).

3. **[`optimal_threshold_calculation.md`](../Threshold%20testing/optimal_threshold_calculation.md)**:
   - Empirical statistical derivations on $4,474$ real SMS messages.
   - Documents Youden's $J$ statistic, cost curves, and multi-layer case boundaries.
