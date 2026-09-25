# Future Alarm Tie Rule Registry

The original submitted implementation did not define a deterministic secondary key for equal hybrid scores at a future top-k cutoff.

This score-construction stage does not perform top-k selection.

The later retrospective alarm-evaluation stage will use this revised deterministic ordering:
1. descending risk score;
2. ascending target_date;
3. ascending feature_date;
4. ascending canonical row identifier (outer_fold, feature_date, target_date).

Explicit scope statement:
- this rule is not used in the present score-construction stage;
- this rule is a revision-era deterministic evaluation rule;
- it is not claimed as an exact reproduction of the submitted implementation;
- it does not use outcomes or event labels.

