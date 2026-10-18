# Stage 9: Findings and Limitations

Everything below is recomputed by `python src/evaluate.py` from saved result
files (read-only) and is reproduced in `notebooks/10_final_evaluation.ipynb`.

## Findings

1. **Task 0 coincidence, resolved as reproducible.** Cross-checking the Azure
   labeler against two independent implementations (`src/labeling.py` and a
   hand-written counter) gives **2,659 positive windows in all four
   configurations** (`next24h`/`gap24_48h` x 16/32 machines). An earlier
   discrepancy was a bug in the hand counter (window end = t+72h instead of
   t+48h and a wrong tail margin), not in the labeler. The coincidence (both
   model1 and model4 hit exactly 2,659) is real and expected: every machine
   fails at least once in this dataset, so each model1 machine contributes one
   positive and each model4 machine is a union of two model1 machines.

2. **Exact reproducibility gate.** Re-running the Stage 4 FedProx setup
   (mu=0.01, seed 0) reproduces the reference `val30 = 16.502083720307283`
   with **absolute difference 0.0** (GPU, 161 s). This confirms the
   deterministic pipeline; all seeds 0-4 results below use identical code paths.

3. **FedProx shows no detectable difference from centralized, and it beats FedAvg on the held-out factory.**
   FD004 RMSE over 5 seeds: FedProx 17.91 +/- 0.96 vs Centralized 17.85 +/-
   0.95 (p = 0.95; with only 5 seeds a difference of about one RMSE point cannot be ruled out) and vs FedAvg 20.24 +/- 0.64 (paired t-test p = 0.0086,
   Holm-adj. 0.034, d = -2.15). Local-only factories are clearly worse
   (22.9-25.8). The regularization term stabilizes FL on heterogeneous clients.

4. **Privacy costs accuracy as expected.** RDP epsilon (30 rounds, delta 1e-5):
   FedProx = inf (no noise), DP sigma 0.03 = 1.75e4 -> RMSE 18.59 +/- 0.70,
   DP sigma 0.1 = 1.76e3 -> RMSE 22.90 +/- 4.20. DP 0.03 keeps most of the
   gain (delta vs FedProx = +0.69, p = 0.036, Holm-adj. 0.080); DP 0.1 is
   heavily degraded and unstable across seeds.

5. **Engine-level statistics (248 FD004 test engines).** Scoring the saved
   checkpoints per engine (mean absolute error in cycles) adds power:
   Centralized is best (JMAE 12.89), FedAvg worse than FedProx in mean
   absolute error but not significant (Wilcoxon p = 0.86 on JMAE, whereas the
   RMSE comparison was significant; RMSE weights large errors more),
   and DP 0.03 nearly matches FedProx while DP 0.1 is worse at the engine level (p ~ 1e-21, which measures variation across the 248 engines and not across training runs); at the seed level the Holm-adjusted p is 0.080, which is not significant.

6. **More factories usually help FedProx (3 seeds, descriptive only).** 1->3 clients: FedProx FD004 RMSE drops
   from ~23.5 (2 clients, FD001+FD002) down to 18.47 (3 clients) and is
   always <= FedAvg (FD004 RMSE: 23.48 vs 30.03 at 2 clients with hetero 0.057,
   21.07 vs 21.76 at 2 clients with hetero 0.163, 18.47 vs 20.10 at 3 clients;
   hetero = mean pairwise 1-Wasserstein on the last window, 0.057-0.163).
   Descriptive only: the FedAvg-FedProx gap is largest at the two-client
   configuration with the LOWEST heterogeneity (6.55 at 0.057) and smallest at
   the highest-heterogeneity configuration (0.69 at 0.163); the tests do NOT
   show a monotonic widening with heterogeneity. The best two-client set (FD002+FD003) reaches 18.24, so the third client did not improve on it, and FD001+FD003 (21.07) is slightly worse than FD003 alone (20.55).

7. **Azure failure prediction: FL beats local-only.** Held-out test PR-AUC
   (5 seeds): next24h FedAvg 0.986 +/- 0.004 > FedProx 0.975 / Centralized
   0.974 > LocalOnly 0.945. The 24-48h gap task is much harder (0.19 - 0.29);
   errors+telemetry together drive performance (ablation 0.99 for all features
   vs 0.72 errors-only, 0.24 telemetry-only on next24h).

8. **Grounded LLM explanations (B, with KG) hallucinate far less than
   ungrounded ones (A, without KG).** Hallucination rates (60 pairs/sample,
   rule-based checker): sample 42 A(without KG) 0.250 [0.150, 0.367] vs
   B(with KG) 0.000; sample 43 A 0.217 [0.117, 0.333] vs B 0.033 [0.000, 0.083];
   sample 44 A 0.300 [0.183, 0.417] vs B 0.033. McNemar over matches:
   p = 0.0074 (n=60, 13 A-only vs 2 B-only) and p = 0.0001 (17 vs 1).
   Note B's 0.033 residual comes from a single "refuse to answer" style
   response, which is not a factual hallucination.

9. **Communication cost.** 53,825-param LSTM: 0.431 MB/client/round, 12.9 MB
   over 30 rounds; Azure MLP (5,953 params): 0.143 MB/client/round, 4.3 MB
   total. FL is cheap to deploy.

## Limitations

1. **Small seed count (5).** The 5-seed paired t-tests have 4 degrees of
   freedom (Cohen's d is noisy; Holm multiplicity correction spans only 5
   comparisons). Engine-level tests and the DP curve reproductions mitigate but
   do not remove this. FedAvg/FedProx tune-based convergence uses 3 seeds
   (0 from `federated_tuning`, 3-4 from stage 9); DP curves use 5.

2. **Simulated factories.** C-MAPSS "factories" are slices of one benchmark
   (different op-conditions and fault modes), not real sites; Azure machines
   are a second, simpler dataset. Generalization claims are benchmark-level.

3. **PR-AUC floor.** Azure held-out test has an intrinsic PR-AUC floor from
   the imbalanced label distribution (prevalence 0.5-1.5%); the gap 24-48h
   numbers are low in absolute terms.

4. **DP leakage caveat.** DP is applied per round to client updates with
   clipping norm 0.32 (config), but the RDP budget uses a per-epoch
   (per-round) composition model and assumes the model is randomized;
   `rmse_unclipped` and intermediate checkpoints are not DP-protected.

5. **LLM cost/determinism.** Explanations rely on a local Ollama model
   (llama3.2), so generation is non-deterministic and caching/serving affects
   reproducibility; only hallucination checks (rule-based) were quantified,
   not semantic quality.

6. **KG scope.** The knowledge graph is small and schema-derived; explanations
   reference at most a handful of facts per engine.

7. **Engine-level metric mismatch.** Per-engine mean absolute error (JMAE) is
   not RMSE; aggregating by engine changes which method is "best" (e.g.
   FedProx ~ FedAvg on JMAE but better on RMSE). Both are reported to keep the
   aggregation honest.

8. **Checker scope.** The rule-based checker verifies that every sensor, component and action named in an explanation appears in the knowledge-graph facts, but it does not verify the quality or consistency of the reasoning. For example, for FD004 engine 1 the facts list five active sensor symptoms (s14, s9, s15, s8 and s13), yet version B states that there is a lack of sufficient evidence and recommends a routine inspection, while also mentioning the compressor and the fan; the checker still marks it as supported. This is a single example (descriptive only).

9. **Seed variability on Azure.** The Azure results vary between training seeds (for example, Centralized PR-AUC on the next-24h setting is 0.978 ± 0.001 over seeds 0-2 and 0.974 ± 0.010 over seeds 0-4), so differences between methods there are small relative to the seed-to-seed spread; the Holm-corrected paired t-test, not the bootstrap interval, is the primary evidence (descriptive only for differences that are not significant).