# gauss_train.py review memo

## CSV summary
- Raw rows: 28
- Rows after `dropna(subset=[target] + descriptors)`: 27
- Dropped rows: 1
- Numeric descriptors used: 15
- Descriptor names: homo_ev, lumo_ev, gap_ev, omega, dipole_moment_debye, molecular_volume_A3, h_nbo_charge, o_nbo_charge, polar, sasa, 3and6, 4and5, logp, hbd, hba

## Search-space size
- Number of descriptor combinations from 1 to 4: 1940
- With the current setting in the uploaded script:
  - outer CV: Leave-One-Out (27 folds after dropping missing rows)
  - inner CV: RepeatedKFold(n_splits=5, n_repeats=10) = 50 splits
  - kernels per grid search: 3
- Approximate number of model fits:
  - per combination: 4,200
  - total over all combinations: 8,148,000
- Note: the true computational burden is even larger because `GaussianProcessRegressor(n_restarts_optimizer=10)` repeats hyperparameter optimization inside each fit.

## Redundancy found in the uploaded CSV
- `gap_ev` is numerically identical to `lumo_ev - homo_ev` within floating-point tolerance.
  - max abs((lumo_ev - homo_ev) - gap_ev): 1.776e-15
- Highly correlated descriptor pairs (|r| >= 0.90):

  - molecular_volume_A3 vs polar: 0.980
  - lumo_ev vs omega: 0.977
  - polar vs logp: 0.914

## Main review points
1. Hard-coded Windows paths should be replaced with function arguments or CLI options.
2. Descriptor names should be inferred from the CSV instead of being hard-coded.
3. The current output mixes:
   - CV metrics from nested LOOCV
   - a single kernel refit on the full dataset
   These do not necessarily correspond to the same trained model.
4. Suppressing all `ConvergenceWarning` messages hides useful diagnostics.
5. If the final goal is to choose the single best descriptor set, the descriptor-set search itself should also be nested inside the outer CV. Otherwise the score of the finally selected best combination is optimistic.
6. Redundant/highly correlated descriptors can be skipped to reduce instability and runtime.
