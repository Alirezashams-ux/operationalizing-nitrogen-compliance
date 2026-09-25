# Precise sanity audit: leaderboard_point_by_horizon.csv

- Source: results/tables/leaderboard_point_by_horizon.csv
- Generated: 2026-03-03T17:54:22.957799
- Overall: PASS

## Check results

| check                                       | passed   | detail                                                                                                 |
|:--------------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------|
| schema_missing_columns                      | True     | missing=[]                                                                                             |
| schema_extra_columns                        | True     | extra=[]                                                                                               |
| horizon_set                                 | True     | observed=[1, 3, 5], expected=[1, 3, 5]                                                                 |
| source_group_set                            | True     | observed=['hybrid_rank_v2_guarded', 'main_linear'], expected=['hybrid_rank_v2_guarded', 'main_linear'] |
| main_linear_grid_complete                   | True     | observed=21, expected=21, models=7, horizons=[1, 3, 5]                                                 |
| hybrid_grid_complete                        | True     | observed=4, expected=4, models=4, horizons=[5]                                                         |
| main_linear_unique_model_horizon            | True     | duplicates=0                                                                                           |
| hybrid_rank_v2_guarded_unique_model_horizon | True     | duplicates=0                                                                                           |
| global_model_horizon_duplicates_expected    | True     | duplicates=[{'model': 'persistence', 'horizon': 5, 'n': 2}]                                            |
| rank_correct_within_horizon_source_group    | True     | errors=[] count=0                                                                                      |
| main_linear_monotonic_error_growth_H1_H3_H5 | True     | failed_models=[]                                                                                       |
| row_count_total                             | True     | rows=25 expected=25                                                                                    |
| row_count_main_linear                       | True     | rows=21 expected=21                                                                                    |
| row_count_hybrid                            | True     | rows=4 expected=4                                                                                      |
| nulls_in_critical_fields                    | True     | null_cells=0                                                                                           |

## Duplicate (model,horizon) rows across groups

| model       |   horizon |   n |
|:------------|----------:|----:|
| persistence |         5 |   2 |

## Mean MAE/RMSE by source_group and horizon

| source_group           |   horizon |     MAE |    RMSE |
|:-----------------------|----------:|--------:|--------:|
| hybrid_rank_v2_guarded |         5 | 1.83769 | 2.35672 |
| main_linear            |         1 | 1.58481 | 1.99914 |
| main_linear            |         3 | 1.79983 | 2.26502 |
| main_linear            |         5 | 1.87107 | 2.36061 |