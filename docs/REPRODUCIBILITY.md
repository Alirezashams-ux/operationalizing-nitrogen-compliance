# Reproducibility Guide

## Scope

This repository is the public reproducibility package associated with:

**Operationalizing Nitrogen Compliance: A Leakage-Safe, Alarm-Budget Framework for Wastewater Treatment Early Warning**

ACS ES&T Water (2026)

https://doi.org/10.1021/acsestwater.6c00549

The repository contains source code, environment specifications, validation
definitions, processed nonrestricted artifacts, model predictions, evaluation
outputs, figure/table source data, and computational audit materials.

Raw wastewater-treatment source datasets are not redistributed.

## Repository layers

### `src/`

The `src/` directory contains the principal project source code and is the
appropriate starting point for inspecting the preprocessing, modeling,
evaluation, and figure-generation workflow.

Repository roots in active source code are generally resolved relative to the
source file or project directory rather than to a specific workstation.

### `revision_2026/`

The `revision_2026/` directory is a frozen computational provenance and audit
layer created during the manuscript correction, rerun, reconciliation, and
final-integration process.

Some files in this directory intentionally preserve historical information
such as:

- original workstation paths;
- original Git branch names;
- original Git commit hashes;
- frozen tag identifiers;
- original run directories;
- historical source-file locations.

These values document the computational environment in which the published
results were produced. They should not be interpreted as required installation
paths for a fresh clone of the public repository.

Some historical audit builders also depended on private or submission-stage
materials that are intentionally not distributed in the public repository.
Their generated nonrestricted audit outputs are retained for transparency.

Accordingly, `revision_2026/` should be treated primarily as an auditable
frozen record rather than as a requirement that every historical script be
executed unchanged from the new public Git history.

## Data requirements

The following raw source files are intentionally absent:

- `Seoul_Tancheon_1.csv`
- `Ulsan_Yongsan.csv`

See:

`data/RESTRICTED_DATA_NOTICE.md`

The repository contains permitted processed and derived artifacts sufficient
for inspection and audit of the reported computational results.

## Environment

Primary dependency specifications are provided in:

- `requirements.txt`
- `requirements-lock.txt`

A typical environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Provenance principle

Historical paths, Git identifiers, manifests, and checksum records should not
be rewritten merely to match the location of a new clone. They are retained
where they form part of the audit trail of the published analysis.

Public-release documentation and active workflows use the current repository
identity:

https://github.com/Alirezashams-ux/operationalizing-nitrogen-compliance
