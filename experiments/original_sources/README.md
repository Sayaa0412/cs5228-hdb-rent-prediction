# Historical source snapshots

These are unmodified scripts from the recorded study. `manifest.json` maps each
file to its original relative path and SHA-256. They preserve the original
experimental definitions, including historical directory and control references.

Do not execute them directly from this folder: their old dependency chain and
private transaction caches are not bundled. The portable replacements are
`python -m experiments.run` and `python -m experiments.diagnose`; see the parent
README for coverage, commands and the no-retraining validation limit.
