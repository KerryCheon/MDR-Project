# Protocol contract

## Development boundary

Development code reads exactly `train.csv` and `val.csv`, rejects dates after 2022, and uses all nine current stations. Outer origins are 2020, 2021, and 2022. `forward_time` trains on all stations before an origin; `station_time` additionally removes every held-station row from candidate generation and model fitting. Zero-observation assigned station-year folds are fatal.

Station-time uncertainty uses partition seeds 42–46 with learner seed 42 and learner seeds 42–44 with partition seed 42, deduplicating `(42, 42)` for seven repeats. Forward-time uses learner seeds 42–44. Repeats are collapsed per candidate/family/origin/station/date: squared errors are averaged for primary risk, while predictions are averaged before secondary metrics. Repeats never become additional validation rows.

## Ranking and selection

The exact 1.3-lite learner is used for both ranking and evaluation. Ranking permutation utility is the change in station-year macro RMSE. Features are ordered by importance lower confidence bound, mean importance, then original column position. The direct/progressive screen uses one permutation repeat and selects direct only when the paired 95% upper confidence bound for direct minus progressive risk is below zero; otherwise progressive is frozen. The repeated run uses three permutation repeats and endpoint counts 150, 125, 100, 80, 65, 50, and 40. Progressive bridge steps follow `max(target_size, current_size - target_size)` and are never promotion endpoints.

V0 is the reference. All 496 predictors and the completed 2.0 original, crossed, and nested lists are diagnostic-only controls. Promotable forms are the pure selected list and the ordered V0 union. A candidate must significantly improve paired combined primary risk, avoid either fold-family point regression, pass station-year, worst-station, and monthly guards, and exactly match V0 coverage. The one-standard-error rule then prefers the smallest actual count, followed by stability and deterministic tie-breakers. No qualifying candidate means automatic V0 fallback and a recorded diagnostic-only best failed candidate.

Beta 0.2 is considered only after method, path source, form, and count freeze. It must significantly beat beta 0.0 and avoid either family regression. The final list is an equal-year 2020/2021/2022 consensus; year-specific models are prohibited.

## MoE boundary

The V0 K=2 router uses numeric coercion, infinity-to-missing conversion, train-frame mean imputation, `StandardScaler`, K-means with `n_init=10` and seed 42, and target-free centroid alignment to a 2017–2019 reference. Router preprocessing is fitted inside each outer training boundary. Every hard expert starts from a complete shared backbone; regime-specific selection may only add unused predictors.

MoE diagnostics always run. A failed global gate makes every MoE diagnostic-only and benchmark-ineligible. Otherwise a MoE must significantly beat the strongest corresponding single-global arm, avoid family regressions, and pass the same robustness guards.

## Benchmark boundary

`run_benchmark.py` requires `--confirm-benchmark` and a byte-valid `development_freeze.json` before the benchmark module is imported. It evaluates V0 and exactly one predeclared challenger with learner seed 42. Historical Model 16 predictions are accepted only after their 8,396 labels, metadata, metrics, and ordered target alignment are proven.

A project-benchmark SOTA requires the development gates, R² at least 0.6648718115185884, RMSE below 0.06042772002760553, paired station-year macro improvement, robustness guards, and historical alignment. Every claim discloses that the benchmark is reused and retrospective and that unbiased/external-generalization eligibility is false.

