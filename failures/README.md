# /failures/ Directory

This directory is a mandatory component of our **Radical Integrity** protocol. It serves as a repository for runs, logs, and artifacts that demonstrate the model's failure points.

## Inclusion Policy
A run MUST be logged here if:
1. The simulation crashes (NaN, Overflow).
2. The diversity ($H$) drops below 0.5 in a "Robust" scenario.
3. Numerical drift exceeds 5% of the initial state.
4. Identical seeds produce different results (Non-determinism).

## Current Logs
- `stability_failure_v25_euler.log`: Documentation of entropy decay at T=150 using Euler.
- `clustering_audit_v25.json`: Artifact showing non-random heretic clustering (v2.5).
