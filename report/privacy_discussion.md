# Stage 5 — What can still leak, even with differential privacy?

Our mechanism adds Gaussian noise to each client's clipped update and only then
sends it. That protects the *update channel*, but several things remain exposed.
All numbers below are from `results/privacy_results.csv` (median raw client
update norm 0.325, clipping bound **C = 0.32**, FedProx mu = 0.01, 30 rounds,
3 clients, 3 seeds).

**(a) Without noise, updates are usable in gradient-inversion and
membership-inference attacks.** The no-DP and clipping-only references reach
FD004 RMSE 18.47 and 18.64 with no privacy guarantee at all (epsilon = inf).
A server that sees an un-noised update can try to reconstruct representative
training windows or decide whether a specific engine record was in a client's
data. Clipping alone does not stop this; it only bounds the damage a single
update can cause. Noise is what makes the two cases indistinguishable — and the
noise needed to actually do that is large.

**(b) With only 3 clients and a sensitivity bound of C, the guarantee is weak,
and epsilon is huge for small noise.** Using the simple RDP bound
(epsilon = min_alpha [T*alpha/(2*sigma^2) + ln(1/delta)/(alpha-1)], T = 30,
delta = 1e-5), we get epsilon = 1.7e6 at sigma = 0.003, 1.5e5 at sigma = 0.01,
1.8e4 at sigma = 0.03, 1763 at sigma = 0.1, 254 at sigma = 0.3, 41 at sigma =
1.0, and 10.4 at sigma = 3.0. Accuracy stays essentially flat (FD004 RMSE
18.5–19.0) only while epsilon is above ~1.7e4 — i.e. no meaningful guarantee.
Accuracy only clearly degrades once epsilon is already large: FD004 RMSE rises
to 25.0 at sigma = 0.1 (epsilon 1763), 43.5 at sigma = 0.3 (254), 356 at sigma =
1.0 (41), and 1393 at sigma = 3.0 (10.4). There is no setting where the privacy
budget is both small and the model is still accurate. This is a demonstration of
the trade-off, not a strong real-world guarantee: three clients is too few for
strong client-level DP, and the bound itself is loose.

**(c) DP protects the updates, not the knowledge graph or the model.** Privacy
is applied only to the numeric weight deltas. It does nothing for the shared
knowledge-graph schema, any graph facts that are exchanged or looked up, node
and relation names, or the final model's predictions and explanations. Those
outputs can still reveal machine types, failure patterns, and operating
conditions even when the updates themselves are noised.

**(d) The server still sees participation and volume.** The server learns which
clients take part in each round and approximately how much data they have,
because our FedAvg weights updates by the number of training windows. Here the
three clients contribute 14,241 / 37,432 / 17,692 training windows and
3,490 / 8,787 / 4,128 validation windows. These counts are visible metadata that
DP on the updates does not hide.
