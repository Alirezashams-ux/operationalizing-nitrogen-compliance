# Stage 2 Lag-Window Sensitivity Summary

This report is retrospective, offline, fixed-budget sensitivity evidence only.
No configuration optimization or post-hoc final-reference reassignment was performed.

| model | horizon | configuration_id | final_reference_configuration | mae_difference_vs_reference | rmse_difference_vs_reference | mase_difference_vs_reference | pearson_correlation_vs_reference | spearman_correlation_vs_reference | recall5_difference_vs_reference | training_row_difference_vs_reference | feature_count_difference_vs_reference | interpretation_classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HGBR | 1 | LONG | REFERENCE | -0.043462379121083305 | -0.0382987189938393 | -0.03273311222166542 | 0.918324619722347 | 0.9217302099849267 | -0.05405405405405406 | -48 | 4 | not_preclassified |
| HGBR | 1 | REFERENCE | REFERENCE | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | not_preclassified |
| HGBR | 1 | SHORT | REFERENCE | 0.004900776782592153 | 0.008180876717284846 | 0.003422676704527783 | 0.9831702664347721 | 0.983665919476959 | 0.013513513513513514 | 0 | -1 | not_preclassified |
| HGBR | 3 | LONG | REFERENCE | 0.05827860288580111 | 0.035911109186173906 | 0.048241838261425585 | 0.8975987979825292 | 0.9016343686754866 | -0.014084507042253502 | 0 | 4 | not_preclassified |
| HGBR | 3 | REFERENCE | REFERENCE | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | not_preclassified |
| HGBR | 3 | SHORT | REFERENCE | 0.06450691107176065 | 0.05429394986257963 | 0.05207760243333781 | 0.9130584658894286 | 0.9158042172245091 | -0.08450704225352111 | 0 | -1 | not_preclassified |
| HGBR | 5 | LONG | LONG | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | not_preclassified |
| HGBR | 5 | REFERENCE | LONG | -0.015792072366118015 | -0.015915856990335264 | -0.012327562442836282 | 0.9758154019249531 | 0.977019594507718 | 0.028169014084507032 | 0 | -4 | not_preclassified |
| HGBR | 5 | SHORT | LONG | -0.0046909315615000224 | -0.013360780632275304 | -0.0027020816082548027 | 0.960172591694624 | 0.9632993602794969 | 0.0 | 0 | -5 | not_preclassified |
| Ridge | 1 | LONG | REFERENCE | 0.00625420248736952 | 0.012578271806203078 | 0.005323299820940708 | 0.9901931699844246 | 0.9885713450365594 | 0.0 | -48 | 4 | not_preclassified |
| Ridge | 1 | REFERENCE | REFERENCE | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | not_preclassified |
| Ridge | 1 | SHORT | REFERENCE | 0.003170775438752882 | 0.002306834565689808 | 0.002507411631221368 | 0.998734429735546 | 0.9985892642152447 | 0.013513513513513514 | 0 | -1 | not_preclassified |
| Ridge | 3 | LONG | REFERENCE | -0.016789650576729098 | -0.010982737404672704 | -0.014285621808450166 | 0.9903505682111053 | 0.9881650730173349 | -0.01408450704225353 | 0 | 4 | not_preclassified |
| Ridge | 3 | REFERENCE | REFERENCE | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | not_preclassified |
| Ridge | 3 | SHORT | REFERENCE | -0.005082144635761177 | -0.014744220326138091 | -0.0044635964744679235 | 0.9742745129240603 | 0.9796979778729835 | -0.01408450704225353 | 0 | -1 | not_preclassified |
| Ridge | 5 | LONG | REFERENCE | -0.011298883683313221 | -0.02125746035050824 | -0.009581855039748177 | 0.9805294957646201 | 0.9790319541128486 | -0.04225352112676056 | 0 | 4 | not_preclassified |
| Ridge | 5 | REFERENCE | REFERENCE | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0 | 0 | not_preclassified |
| Ridge | 5 | SHORT | REFERENCE | -0.010705283289536904 | -0.015085042716538766 | -0.009185836712364503 | 0.9955891760545962 | 0.9953857378734304 | -0.01408450704225353 | 0 | -1 | not_preclassified |

Interpretation classification is fixed to not_preclassified because frozen design thresholds are quantitative-only.
