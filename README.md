# TypeFuzz-style queue prioritisation: seeded simulation

Motivation: "TYPEFUZZ: Type Coverage Directed JavaScript Engine Fuzzing (Registered Report)" proposes guiding JavaScript engine fuzzing by type-state coverage, where each newly seen type state gets the same priority. Type coverage can be enumerated cheaply, but it does not say which type states are likely to hide bugs, so a fixed run budget may be spent on unproductive states. This experiment asks whether a learned bug-yield score can redirect that budget.

Question: Can a class-balanced gradient-boosting model that ranks type states by predicted bug yield find more seeded bugs within a fixed run budget than a type-coverage queue that treats every new type state equally?

Dataset: a seeded numpy simulation of fuzzing campaigns (no download). Each type state has a feature vector, and a small set hides bugs with low trigger probability. This is not the paper's own data; no public dataset of JavaScript engine fuzzing runs exists.

Baseline: a standard type-coverage-guided queue that gives every new type state equal priority and mutates seeds in first-come order.

Improved: the same queue and run budget, but a class-balanced LightGBM trained on earlier campaigns scores type states, and runs go to the highest-scoring state (discounted by runs already spent).

How to run:
```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and figures/bugs_found.png).

## Results (latest run)

On my seeded fuzzing-campaign simulation, a standard type-coverage-guided queue that treats every new type state equally found 1.86 distinct bugs per test campaign (mean, three seeds) within the fixed run budget, versus 9.28 for the class-balanced LightGBM ranker.

| Metric | Value |
|---|---|
| baseline_score | 1.861 |
| improved_score | 9.278 |
| improvement | 7.417 |
| baseline_score_std | 0.1273 |
| improved_score_std | 0.2406 |
| baseline_runs_to_first_bug | 805.3 |
| improved_runs_to_first_bug | 234 |
| baseline_states_covered | 398.6 |
| improved_states_covered | 398.8 |
| improved_model_test_auc | 0.961 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulation only; the hidden bug signal is linear in the features, which favours the tree model, so gains may be optimistic. Training labels come from baseline-style campaigns. Campaign-level split (not row-stratified) to avoid leakage. Bugs are rare, so the metric is noisy; standard deviations are across three seeds.

## Reference

This project is motivated by the following paper:

> T Wienand, L Bernhard, F Toffalini. *TYPEFUZZ: Type Coverage Directed JavaScript Engine Fuzzing (Registered Report)*. ndss-symposium.org. https://www.ndss-symposium.org/wp-content/uploads/fuzzing2026-5.pdf

Paper author: Prof. Jun.-Prof. Dr. Flavio Toffalini, Ruhr-Universität Bochum (flavio.toffalini@rub.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
