# Operationalizing Nitrogen Compliance

Reproducibility code and nonrestricted analysis artifacts supporting the published study:

**Operationalizing Nitrogen Compliance: A Leakage-Safe, Alarm-Budget Framework for Wastewater Treatment Early Warning**

*ACS ES&T Water* (2026)

**Paper DOI:** https://doi.org/10.1021/acsestwater.6c00549

**Repository:** https://github.com/Alirezashams-ux/operationalizing-nitrogen-compliance

---

## Overview

This repository contains the computational workflow and nonrestricted research artifacts supporting the published study on leakage-safe forecasting and alarm-budget early warning for wastewater nitrogen compliance.

The repository is intended to support transparency, auditability, and computational reproducibility of the reported analyses.

The public release includes code and permitted derived artifacts associated with:

- data preprocessing and feature engineering;
- leakage-safe and blocked temporal validation;
- model training and controlled reruns;
- point-prediction evaluation;
- nitrogen exceedance and alarm-budget evaluation;
- canonical fold-level prediction outputs;
- retrospective and sequential alarm-policy analyses;
- lag-window sensitivity analyses;
- figure and table source data;
- computational audit and reproducibility checks.

---

## Published article

Mahvelati Shamsabadi, A. et al. (2026).

**Operationalizing Nitrogen Compliance: A Leakage-Safe, Alarm-Budget Framework for Wastewater Treatment Early Warning.**

*ACS ES&T Water*.

DOI: https://doi.org/10.1021/acsestwater.6c00549

The ACS Version of Record is not redistributed in this repository.

---

## Repository structure

Key components of the public release include:

```text
.
├── src/                  # Core analysis and modeling scripts
├── revision_2026/        # Corrected protocols, controlled reruns, audits, and final integration
├── results/              # Nonrestricted predictions, metrics, tables, and figure-source artifacts
├── features/             # Permitted processed feature artifacts
├── configs/              # Configuration files
├── notebooks/            # Supporting computational notebooks
├── reports/              # Reproducibility and methodology reports
├── requirements.txt      # Python dependencies
├── requirements-lock.txt
├── CITATION.cff
├── LICENSE
└── LICENSE-DATA.md

## Reproducibility workflow

The repository includes materials supporting the principal computational stages of the study:

1. preprocessing and feature construction;
2. temporally leakage-safe evaluation design;
3. purged and blocked cross-validation definitions;
4. controlled model reruns;
5. canonical prediction assembly;
6. HybridRank and alarm-policy analysis;
7. retrospective alarm-budget evaluation;
8. sequential alarm-policy evaluation;
9. lag-window sensitivity analysis;
10. final computational integration and audit.

Environment specifications are provided in `requirements.txt` and `requirements-lock.txt`.

Additional configuration and run-specific metadata are retained alongside the corresponding analysis outputs.

---

## Data availability and restrictions

The raw wastewater-treatment datasets are **not redistributed in this repository**.

The following source datasets are intentionally excluded:

```text
Seoul_Tancheon_1.csv
Ulsan_Yongsan.csv

## Reproducibility workflow

The repository includes materials supporting the principal computational stages of the study:

1. preprocessing and feature construction;
2. temporally leakage-safe evaluation design;
3. purged and blocked cross-validation definitions;
4. controlled model reruns;
5. canonical prediction assembly;
6. HybridRank and alarm-policy analysis;
7. retrospective alarm-budget evaluation;
8. sequential alarm-policy evaluation;
9. lag-window sensitivity analysis;
10. final computational integration and audit.

Environment specifications are provided in `requirements.txt` and `requirements-lock.txt`.

Additional configuration and run-specific metadata are retained alongside the corresponding analysis outputs.

---

## Data availability and restrictions

The raw wastewater-treatment datasets are **not redistributed in this repository**.

The following source datasets are intentionally excluded:

```text
Seoul_Tancheon_1.csv
Ulsan_Yongsan.csv
```

The Ulsan wastewater-treatment dataset is subject to third-party data-sharing restrictions and cannot be publicly redistributed.

The Seoul wastewater-treatment data source is documented in the associated article. Its local source copy used in this project is not redistributed here.

Meteorological observations used in the study were obtained from the Korea Meteorological Administration Open MET Data Portal. Derived meteorological variables and permitted analysis artifacts are included where appropriate.

This repository therefore provides code, computational provenance, and permitted nonrestricted derived artifacts without redistributing the raw WWTP source files.

---

## Nonrestricted research artifacts

Where permitted, the repository contains derived materials used to reproduce or audit reported results, including:

- blocked cross-validation definitions;
- fold-level predictions;
- model metrics;
- alarm-budget outputs;
- operating-point analyses;
- figure-source CSV files;
- table-source CSV files;
- audit registries;
- checksums and run manifests.

These artifacts should not be interpreted as substitutes for the restricted raw source datasets.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For stricter environment reproduction, consult `requirements-lock.txt`.

---

## Licensing

### Source code

Source code authored for this project is released under the **MIT License**. See `LICENSE`.

### Nonrestricted derived research artifacts

Eligible nonrestricted derived research artifacts authored by the project team are released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** unless a file-specific notice states otherwise. See `LICENSE-DATA.md`.

### Exclusions

The repository licenses do **not** grant rights to:

- the raw Seoul wastewater-treatment dataset;
- the raw Ulsan wastewater-treatment dataset;
- third-party datasets or external source material;
- the ACS Version of Record;
- publisher-owned material;
- third-party imagery or other assets governed by separate terms.

---

## Citation

If you use this repository, please cite the associated published article:

**Operationalizing Nitrogen Compliance: A Leakage-Safe, Alarm-Budget Framework for Wastewater Treatment Early Warning**

*ACS ES&T Water* (2026)

https://doi.org/10.1021/acsestwater.6c00549

Machine-readable repository citation metadata are provided in `CITATION.cff`.

A permanent DOI for the archived software and reproducibility release will be added after the first public GitHub release is archived in Zenodo.

---

## Archival release

The publication-associated release will be tagged `v1.0.0`.

Following public release, the tagged GitHub release will be archived in Zenodo to provide an immutable, citable snapshot of the reproducibility package.

The resulting Zenodo DOI will subsequently be added to this repository.

---

## Contact

For questions concerning the computational workflow, reproducibility materials, or restricted-data access conditions, please refer to the corresponding author information in the published article.

---

## Detailed reproducibility documentation

Additional documentation is available in:

- `docs/REPRODUCIBILITY.md`
- `docs/DATA_AVAILABILITY.md`

The `revision_2026/` hierarchy contains frozen audit and provenance artifacts.
Some historical scripts intentionally retain original workstation paths, Git
branches, commit hashes, or source locations as part of the computational
audit trail. These historical identifiers are not installation requirements
for the public repository.
