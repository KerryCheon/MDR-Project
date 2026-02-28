# Results (Test $R^2$)

| Version     | Status    |                        Model / Variant | Test $R^2$ |
| ----------- | --------- | -------------------------------------: | -----------: |
| v1.0.0      | VALID     |                  Baseline (no derived) |     0.520847 |
| v1.1.0      | INVALID   |                            Leakage run |     0.017134 |
| v1.2.0      | INVALID   |                            Leakage run |     0.201255 |
| v2.1.0      | INVALID   |               Expanded split (leakage) |     0.712817 |
| v2.2.0      | VALID     |                          Derived fixed |     0.506571 |
| v2.3.0      | VALID     |                     Handpicked derived |     0.671525 |
| v3.1.0      | VALID     |                            All derived |     0.632453 |
| v3.2.0      | VALID     |             Pruned families (40 feats) |     0.628125 |
| v3.3.1      | VALID     |                        Tuned (row 177) |     0.714397 |
| v3.3.2      | VALID     |                        Tuned (row 405) |     0.711952 |
| v3.3.3      | VALID     |                        Tuned (row 267) |     0.711818 |
| v4.1.0      | VALID     |               Two-stage residual model |     0.579425 |
| v5.1.0      | VALID     |                  ElasticNet diagnostic |     0.562049 |
| v7.1.0      | VALID     |                   Tuned XGB (40 feats) |     0.757695 |
| v7.2.0      | VALID     |               Iterative pruning (diag) |       0.7051 |
| v7.3.0      | VALID     |                 XGB before calibration |     0.757695 |
| v7.3.0      | VALID     |                  XGB after calibration |     0.761296 |
| v7.4.0      | VALID     |               Stack (XGB + RF → Ridge) |     0.776814 |
| v7.5.0      | VALID     |                           Baseline XGB |     0.709898 |
| v7.5.0      | VALID     |            XGB after Ridge calibration |     0.750212 |
| v7.5.0      | VALID     |               Stack (XGB + RF → Ridge) |     0.753776 |
| v8.1.0      | VALID     |                 Rain feature additions |     0.674228 |
| v8.2.0      | VALID     |                    Main model baseline |     0.671544 |
| v8.2.0      | VALID     |                   Rain-only ShallowXGB |     0.759728 |
| v8.2.0      | VALID     |           Main + rain backbone feature |     0.684687 |
| v9.1.0      | VALID     |                         Soft mix (MoE) |     0.709101 |
| v9.2.0      | VALID     |    Soft mix (wet expert rain impulses) |     0.724468 |
| v9.3.0      | VALID     |           Soft mix (winter/non-winter) |     0.731950 |
| v9.4.0      | VALID     |      Soft mix (improved winter expert) |     0.743106 |
| v10.x       | VALID     |                       Final run (dump) |      0.77014 |
| v11.1.0     | VALID     |               Rain-only XGB diagnostic |            — |
| v11.2.0     | VALID     |                        Base soil model |       0.7231 |
| v11.2.0     | VALID     |           Final corrected test metrics |     0.757695 |
| v12.1.0     | VALID     |                    Shallow NN baseline |     0.628253 |
| v12.2.0     | VALID     |                      Shallow NN retune |     0.624292 |
| v12.3.0     | VALID     |                     Shallow NN variant |     0.681080 |
| v12.4.0     | VALID     |                  Stack (v12.1 + v12.3) |     0.689568 |
| v12.5.0     | VALID     |                     Shallow NN variant |     0.673340 |
| v12.6.0     | VALID     |             High-val / mid-test regime |     0.641860 |
| v12.7.0     | INVALID   |          Leakage (lag feature from GT) |     0.971928 |
| v13.1.0     | VALID     |                      Final ridge stack |     0.754492 |
| v13.2.0     | INVALID   |                    AR rollout mismatch |            — |
| v13.2.1     | VALID     |                  Baseline (no rollout) |     0.702751 |
| v13.2.1     | VALID     |                             AR rollout |     0.699530 |
| v14.1.0     | VALID     |              Temporal + static spatial |     0.637054 |
| v15.1.0     | VALID     |                   Base XGB (108 feats) |     0.731428 |
| v15.1.0     | VALID     |               Stack (XGB + RF → Ridge) |     0.760723 |
| v15.2.0     | VALID     |                           Stack re-run |     0.760723 |
| v15.3.0     | VALID     |               Base XGB (early-stopped) |     0.752211 |
| v15.3.0     | VALID     | Stack (early-stopped XGB + RF → Ridge) |     0.761775 |
| v16.1.0     | VALID     |               Base XGB (train+val fit) |     0.739264 |
| v16.1.0     | VALID     |               Stack (XGB + RF → Ridge) |     0.757054 |
| v16.2.0     | VALID     |               Baseline XGB (train+val) |     0.739264 |
| v16.2.0     | VALID     |                  Drift XGB (train+val) |     0.790900 |
| v16.2.0     | VALID     | Baseline train-only + ridge calibrator |     0.727043 |
| v16.2.0     | VALID     |    Drift train-only + ridge calibrator |     0.761099 |
| v16.3.0     | VALID     |                     Drift (no weights) |     0.802674 |
| v16.3.0     | VALID     |               Drift (beta=0.2 weights) |     0.807260 |
| v16.4.0     | VALID     |                     Drift (no weights) |     0.811571 |
| v16.4.0     | VALID     |                       Drift (weighted) |     0.799298 |
| v16.5.0     | VALID     |                     Drift (no weights) |     0.804560 |
| v16.5.0     | VALID     |                       Drift (weighted) |     0.810545 |
| v17.1.0     | VALID     |                     Drift (no weights) |      0.81424 |
| v17.1.0     | VALID     |                       Drift (weighted) |      0.81910 |
| v18.3.0     | VALID     |                     Drift (no weights) |      0.82186 |
| v18.3.0     | VALID     |                       Drift (weighted) |      0.82503 |
| v19.2.0     | VALID     |                     Drift (no weights) |      0.81418 |
| v19.2.0     | VALID     |                       Drift (weighted) |      0.82239 |
| v20.1.0     | VALID     |                     Drift (no weights) |      0.81418 |
| v20.1.0     | VALID     |                       Drift (weighted) |      0.82239 |
| **v20.3.0** | **VALID** |                    **Dry regime model** |   **0.85860** |
