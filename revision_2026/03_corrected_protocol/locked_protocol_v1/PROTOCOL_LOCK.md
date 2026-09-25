# Corrected Protocol Lock — Version 1

This directory contains the approved and immutable temporal evaluation protocol for the ACS ES&T Water revision.

Canonical dataset:
- Ulsan_Yongsan.csv
- SHA-256: 6ed147cd585e5e8083292683f9be3cf585e97e4d46fa0e3224d24a4e2b42cbbb

Mandatory split inputs:
- corrected_split_assignment.csv
- corrected_split_summary.csv

Forecast horizons:
- H1
- H3
- H5

Outer purge totals:
- H1: 3 rows
- H3: 9 rows
- H5: 15 rows

Inner purge totals:
- H1: 3 rows
- H3: 9 rows
- H5: 15 rows

Protocol status:
- All automated leakage assertions passed.
- Canonical outer test dates were preserved.
- These split files must be used unchanged for every controlled rerun.
- Models must not independently recreate or alter the folds.
- Any checksum mismatch invalidates the affected run.
