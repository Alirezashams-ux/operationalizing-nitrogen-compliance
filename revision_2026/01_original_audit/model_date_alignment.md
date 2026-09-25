# Model Date Alignment

## Available predictions by family
- Enumerated Persistence, ElasticNet, HGBR, and BCR-TCN prediction files under results/predictions.

## Common-date feasibility
- H5 common-date set exists for all four families using current files.
- H1/H3 cannot include all four families because BCR-TCN is unavailable for H1/H3 and HGBR is unavailable for H1.

## Ridge reconstruction status
- Ridge prediction artifacts are absent.
- Ridge can be rerun with fixed existing alpha grid (0.01/0.1/1/10/100) from existing scripts/config without new hyperparameter search space expansion.

- Full alignment matrix: /home/alrezshams/acs_tnout_ulsan_revision/revision_2026/01_original_audit/model_date_alignment.csv