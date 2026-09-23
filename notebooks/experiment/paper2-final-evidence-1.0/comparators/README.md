# Sensitivity comparator snapshot

`formal_eval_1.0_temporal_seed_summary.csv` is a byte-for-byte copy of the
existing executed `derived_8.4-formal-eval-1.0` temporal seed table. It supplies
the V0, global, target-gating, dynamic, seasonal, and G_API context rows in F1.
These rows are labeled as sensitivity references in the evidence notebook;
they are not paired with the new Guarded run.

The neighboring pinned configuration and README preserve the source settings
and reported provenance. New paired Guarded-versus-Backbone results are
recomputed in this evidence run and stored at the experiment directory root.
