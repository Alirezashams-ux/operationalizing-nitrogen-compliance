# Sequential vs Retrospective Alarm Definitions

## Retrospective top-k
- Ranks the complete held-out fold.
- Issues exactly k alarms per fold.
- Is an offline benchmark.

## Sequential policy
- Processes dates one at a time.
- Uses only earlier scores.
- Does not force an exact alarm count.
- Reports realized burden.
- Is a historical simulation of deployable logic.
- Is not a live prospective deployment.

No policy winner was selected from test outcomes.
