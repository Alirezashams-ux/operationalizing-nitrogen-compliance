# Seoul External Validation Report (ACS-Oriented)

## 1. Purpose and Role in the Paper

This report documents what was done with the Seoul dataset and explains how these results support the manuscript for ACS submission.

Primary role of Seoul in this project:
- External validation only (transferability and robustness check).
- Not used for model selection, hyperparameter tuning, or claim replacement from Ulsan blocked CV.
- Reported in SI as independent evidence that the decision framework generalizes beyond the development site.

This role aligns with the manuscript strategy: main claims are established on leakage-safe blocked CV at Ulsan, while Seoul demonstrates external plausibility.

## 2. Data Sources and Artifacts Used

Inputs and generated outputs used for Seoul external validation:
- Raw Seoul data: [data/raw/Seoul_Tancheon_1.csv](../data/raw/Seoul_Tancheon_1.csv)
- Raw Ulsan data (training source for transfer): [data/raw/Ulsan_Yongsan.csv](../data/raw/Ulsan_Yongsan.csv)
- External evaluation script: [src/si_make_seoul_external_eval.py](../src/si_make_seoul_external_eval.py)
- Cross-site variable map: [reports/main_manuscript_assets/SI/tables/TableS14_cross_site_variable_mapping_ulsan_to_seoul.csv](main_manuscript_assets/SI/tables/TableS14_cross_site_variable_mapping_ulsan_to_seoul.csv)
- External metrics table: [reports/main_manuscript_assets/SI/tables/TableS15_external_validation_summary_metrics_seoul.csv](main_manuscript_assets/SI/tables/TableS15_external_validation_summary_metrics_seoul.csv)
- Budget-recall curve data: [reports/main_manuscript_assets/SI/tables/TableS15a_seoul_budget_recall_curve_data.csv](main_manuscript_assets/SI/tables/TableS15a_seoul_budget_recall_curve_data.csv)
- Domain shift diagnostics: [reports/main_manuscript_assets/SI/tables/TableS15c_domain_shift_diagnostics_ulsan_vs_seoul.csv](main_manuscript_assets/SI/tables/TableS15c_domain_shift_diagnostics_ulsan_vs_seoul.csv)
- Seoul transferability figure: [reports/main_manuscript_assets/SI/figures/FigureS_Seoul_budget_recall_transferability.pdf](main_manuscript_assets/SI/figures/FigureS_Seoul_budget_recall_transferability.pdf)

## 3. What Was Done (Exact Method)

### 3.1 Site harmonization and feature compatibility

The pipeline first harmonized Seoul column names to the Ulsan schema where needed (for example weather fields) and retained overlapping variables.

Cross-site mapping outcome (Table S14):
- Core process and target variables were retained (Date, Inflow, BODin, TOCin, TNin, TNout, etc.).
- Several Ulsan-only meteorological quality fields and maxima windows were unavailable in Seoul and dropped.

### 3.2 Transfer protocol

For each horizon H in {1, 3, 5}:
- Feature engineering mirrored the main pipeline (lags, rolling windows, interactions, seasonality terms).
- Models were trained on Ulsan-derived features.
- Trained models were applied to Seoul without Seoul-side retuning.

### 3.3 Model set evaluated on Seoul

Primary SI verification models:
- Persistence
- ElasticNet (alpha=0.1, l1_ratio=0.5 with standardization)
- HGBR (Ulsan-tuned configuration)

### 3.4 Metrics and operating points

Point forecasting:
- MAE, RMSE, bias

Alarm-budget ranking utility:
- Thresholds tau = 15, 16, 17 mg/L
- Budgets r = 0.05 and 0.10
- k = ceil(r * n)
- Precision, recall, TP, FP, enrichment (recall / r)

## 4. Seoul Results Summary

### 4.1 Point forecasting on Seoul

Best MAE by horizon from Table S15:
- H1: ElasticNet, MAE = 1.327
- H3: Persistence, MAE = 1.598
- H5: Persistence, MAE = 1.627

Interpretation:
- Winner identity changes by horizon under domain transfer.
- This supports the paper's operating-point and context-dependent model selection message.

### 4.2 Alarm-budget recall winners on Seoul

Best recall by operating point (Table S15):

H1:
- tau=15, r=0.05: Persistence (0.348)
- tau=15, r=0.10: Persistence (0.551)
- tau=16, r=0.05: ElasticNet and HGBR tie (0.381)
- tau=16, r=0.10: Persistence and ElasticNet tie (0.571)
- tau=17, r=0.05: Persistence and HGBR tie (0.333)
- tau=17, r=0.10: ElasticNet (0.556)

H3:
- tau=15, r=0.05: ElasticNet (0.275)
- tau=15, r=0.10: ElasticNet (0.536)
- tau=16, r=0.05: HGBR (0.381)
- tau=16, r=0.10: ElasticNet (0.714)
- tau=17, r=0.05: HGBR (0.556)
- tau=17, r=0.10: HGBR (0.778)

H5:
- tau=15, r=0.05: ElasticNet (0.275)
- tau=15, r=0.10: ElasticNet (0.507)
- tau=16, r=0.05: Persistence (0.286)
- tau=16, r=0.10: ElasticNet (0.619)
- tau=17, r=0.05: Persistence (0.333)
- tau=17, r=0.10: Persistence (0.778)

Interpretation:
- There is no universal winner across all horizons and thresholds.
- This is consistent with the manuscript thesis that deployment decisions should be made at explicit operating points (H, tau, r), not by one global metric.

### 4.3 Enrichment evidence

Observed enrichment on Seoul (recall / r) is often well above 1.0, including values above 5 and up to 11.11 in selective low-prevalence settings.

Meaning for operations:
- Even under transfer, top-k ranking can concentrate exceedance days above random selection at fixed alarm capacity.
- This directly supports the bounded-attention decision framing in the paper.

## 5. Domain Shift Context (Why Seoul Is Hard and Useful)

Table S15c indicates strong cross-site shifts in several influent/process drivers:
- Inflow: very large standardized mean difference (SMD about 6.58), KS statistic 1.00.
- BODin: SMD about 2.56.
- TOCin: SMD about 1.77.
- TNin: SMD about 1.83.
- TNout distribution is shifted but less extremely than key influent variables.

Implication:
- Seoul is a meaningful stress test, not a near-duplicate environment.
- Positive enrichment under this shift strengthens transferability claims of the framework.

## 6. Alignment With ACS Journal Expectations

How this Seoul package helps ACS-facing quality:
- External validity: independent-site evaluation included.
- Decision relevance: fixed-budget alarm metrics reported, not only MAE/RMSE.
- Reproducibility: script-to-table-to-figure traceability with explicit SI artifacts.
- Transparency: variable mapping and domain shift diagnostics are provided.
- Scientific caution: external test is framed as robustness evidence, not post-hoc model re-selection.

## 7. Recommended Manuscript/SI Positioning

Suggested positioning for text and rebuttal readiness:
- Main manuscript: keep Ulsan blocked-CV as primary evidence for method efficacy.
- SI: present Seoul as external transferability verification with Table S14, Table S15, Table S15a, and Figure S.
- Discussion: emphasize context-dependent winner shifts and persistence of enrichment under domain shift.

Suggested one-sentence SI claim:
- External validation at Seoul (without retuning) preserved meaningful alarm-budget enrichment across multiple operating points despite substantial covariate shift, supporting framework-level transferability.

## 8. Limitations and Correct Interpretation

- Seoul evaluation currently covers the three primary models only (Persistence, ElasticNet, HGBR).
- Pseudo-block uncertainty on Seoul is based on contiguous temporal blocks (single external site), so uncertainty should be interpreted as stability indication, not full multi-site variance decomposition.
- Results should not be used to overwrite Ulsan-derived main-text winners.

## 9. Final Takeaway

What we have done on Seoul is methodologically correct for ACS-level external validation:
- We froze the Ulsan-developed framework.
- Applied it to an independent plant.
- Reported both point and decision metrics at operational thresholds and budgets.
- Quantified transfer under domain shift.

This strengthens the manuscript by showing the contribution is not a single-site overfit model, but a leakage-safe, decision-aligned evaluation framework with practical transferability evidence.
