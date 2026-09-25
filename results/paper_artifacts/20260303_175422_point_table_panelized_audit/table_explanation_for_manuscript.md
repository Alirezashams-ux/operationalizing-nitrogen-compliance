# Point-forecast leaderboard explanation (panelized)

- Panel A (`main_linear`) is a complete 7×3 model–horizon grid (H=1,3,5).
- Panel B (`hybrid_rank_v2_guarded`) is an H=5-only benchmark context (4 models).
- Ranks are computed within each (horizon, source_group) block and are not cross-group comparable.
- Cross-group duplicate at (persistence, H=5) is intentional because rows correspond to different source_group contexts.
- Sanity result: PASS.
