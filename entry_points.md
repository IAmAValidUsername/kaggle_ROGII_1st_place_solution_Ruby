# Entry Points

All commands below must be run with the current working directory set to the
`submission_model` directory. The defaults in `SETTINGS.json` are relative to
that file and already point to the bundled data, six bundled snapshots,
internal cache directories, and `reproduction_outputs/`. Running from the
repository root or from a result subdirectory is not the documented workflow.

The launcher is Bash, not Python. Set `PYTHON_BIN` when the active Python is
not the environment containing the pinned packages. Before a reproduction,
raise the open-file limit in the same shell:

```bash
cd /path/to/submission_model
ulimit -n 8192
```

The launcher uses `python3` by default. To use a custom environment, set the
interpreter once before running the commands below:

```bash
export PYTHON_BIN=/path/to/python
```

`reproduce_workflow.sh` first rebuilds
`data/train_png_typewell_map.csv` from the 773 bundled training horizontal
CSVs. It verifies the generated `well_id,horizontal_avg_X,horizontal_avg_Y`
projection against the existing file before writing the compact map. The
historical PNG-derived/group columns are intentionally not generated because
the sequence CV code never reads them.

## 1. Use the self-contained defaults

No path editing or archive download is required for this repository. The
following command should work as-is after dependencies are installed:

```bash
./reproduce_workflow.sh --train 0801_V2 --device cuda
```

All supported IDs are `0719_V1`, `0724_V1`, `0729_V3`, `0801_V1`, `0801_V2`,
and `0803_V2`. `SETTINGS.json` remains the single path authority if the
package is intentionally relocated or a different test directory is supplied;
relative values continue to resolve relative to `SETTINGS.json`.

## 2. Optional setup verification

This is a fast, debug-oriented check of all six source snapshots and recipe
configs. It imports each snapshot, discovers the bundled train/test wells,
serializes a temporary config, and compares every non-environment field with
the bundled `cfg.pkl`. It stops before model allocation or training. It can be
skipped when the goal is simply to run part 3.

```bash
./reproduce_workflow.sh --verify
```

Use `--keep-temp` only when inspecting the generated temporary configs:

```bash
./reproduce_workflow.sh --verify --keep-temp
```

## 3. Train an exact recipe

Train one family, including its 15 folds and individual test inference:

```bash
./reproduce_workflow.sh --train 0801_V2 --device cuda
```

The result is written to `reproduction_outputs/0801_V2/`. Replace the ID to
run another family. The destination must not already exist; choose a new
`paths.output_dir` in `SETTINGS.json` or pass `--force` when overwriting is
intentional.

To train all six sequentially in canonical order:

```bash
./reproduce_workflow.sh --train-all --device cuda
```

The launcher validates all six bundled snapshots and output destinations before
starting the first run, then waits for each complete family before starting the
next. Each run copies its exact `seq_NN*.py` files beside its newly written
`models.pkl`, config, log, and prediction artifacts. No archived `models.pkl`
is required because these commands retrain the families.

CPU smoke overrides, reduced folds, and shortened epochs are intentionally not
part of the documented reproduction command; they change the recipe and are
only appropriate for local debugging.

## 4. Optional direct entrypoint

This is an optional equivalent to part 3 for users who want to call Python
directly. It produces the same `reproduction_outputs/<ID>/` artifacts and
automatically selects `reference_results/<ID>/` from `SETTINGS.json`:

```bash
"${PYTHON_BIN:-python3}" ./generate_train_geo_map.py \
  --train-dir data/train \
  --output data/train_png_typewell_map.csv
"${PYTHON_BIN:-python3}" ./seq_NN_main_reproduce.py \
  --id 0801_V2 \
  --device cuda
```

The map command is shown explicitly so the direct path has the same first step
as the launcher. `--settings`, `--source-dir`, and `--output-dir` are available
for an intentional relocation or an isolated output, but are unnecessary for
the self-contained defaults.

## Final ensemble inference

After the six individual families are trained, use the existing notebook for
saved-model inference and the final ensemble:

<https://www.kaggle.com/code/w5833946/submit-reproduce>

Point its six result inputs at the corresponding
`reproduction_outputs/<ID>/` directories. Keep each `models.pkl` paired with
the `seq_NN*.py` snapshot copied into the same directory because pickle class
imports depend on that snapshot.
