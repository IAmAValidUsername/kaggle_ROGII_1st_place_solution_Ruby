# Submission Model Reproduction

**Start with [entry_points.md](entry_points.md). It is the authoritative
step-by-step reproduction guide.** Unless a command says otherwise, run the
commands from the repository root, the directory containing this README.

For the original competition context and the first-place solution details, see
the [Kaggle technical write-up](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution).

## What is bundled

This repository is a self-contained reproduction package. `SETTINGS.json` uses
only paths relative to itself, the train/test CSV data is copied under `data/`,
and the six archived source/config/log bundles are under
`reference_results/<ID>/`. No six-version archive needs to be downloaded from
the website or from the original project checkout. The multi-gigabyte
`models.pkl` files are deliberately not included: reproduction means retraining
the 15-fold model family from the bundled source snapshot.

The bundled data copy also retains the training PNG files because they are part
of the original training download. The sequence pipeline does not read those
images; the geographic map is rebuilt only from horizontal CSV coordinates.

The first action of `reproduce_workflow.sh` is to run
`generate_train_geo_map.py`. It computes the full-sequence mean `X` and `Y` for
each training well, compares those three required columns byte-for-byte (with
six-decimal formatting) to any existing map, and writes the compact map. The
historical map had five additional PNG-derived/group columns
(`typewell_name`, `typewell_avg_X`, `typewell_avg_Y`, `typewell_well_count`,
and `typewell_row_count`). CV consumes none of them, so they are intentionally
omitted and are not treated as a reproduction dependency.

## Reproduction repository structure

```text
<repository-root>/
|-- README.md
|-- entry_points.md
|-- SETTINGS.json
|-- requirements.txt
|-- reproduce_workflow.sh
|-- generate_train_geo_map.py
|-- seq_NN_main_reproduce.py
|-- verify_cfg.py
|-- data/
|   |-- train/
|   |-- test/
|   |-- train_png_typewell_map.csv
|   `-- sample_submission.csv
|-- reference_results/
|   `-- <ID>/
|       |-- seq_NN*.py
|       |-- cfg.pkl
|       `-- seq_nn.log
|-- PF_cache/                 # generated when a recipe uses PF channels
`-- reproduction_outputs/    # generated training and inference artifacts
```

`reproduce_workflow.sh` is the user-facing controller. It reads the relative
paths from `SETTINGS.json`, checks the pinned environment, regenerates the
geographic CV map, selects one or all six recipes, and launches the Python
wrapper. `entry_points.md` documents its supported commands, while
`requirements.txt` defines the Python environment.

`generate_train_geo_map.py` owns the only derived input that must be rebuilt
before a run. `seq_NN_main_reproduce.py` then loads the selected archived source
from `reference_results/<ID>/`, applies the package paths, selects the exact
historical config, and delegates training and inference to that snapshot's
`seq_NN_main.py`. `verify_cfg.py` provides the setup-only comparison against
the archived `cfg.pkl` used by `--verify`.

The `data/` and `reference_results/` trees are bundled, read-only inputs.
`PF_cache/` is a regenerable acceleration cache, and
`reproduction_outputs/<ID>/` receives newly trained models, copied source,
OOF predictions, and test predictions. The normal execution flow is therefore
`reproduce_workflow.sh` -> map generation -> archived recipe wrapper ->
`reproduction_outputs/<ID>/`.

## Six recipes

| ID | Historical config source | Models | Archived OOF RMSE |
|---|---|---:|---:|
| `0719_V1` | snapshot default `CFG()` | 15 | 5.0910 ft |
| `0724_V1` | snapshot default `CFG()` | 15 | 4.8586 ft |
| `0729_V3` | `SEQ_TRAIN_CFGS[submit_ver_0729_V3]` | 15 | 5.5360 ft |
| `0801_V1` | `SEQ_TRAIN_CFGS[submit_ver_0801_V1]` | 15 | 5.1668 ft |
| `0801_V2` | `SEQ_TRAIN_CFGS[submit_ver_0801_V2]` | 15 | 4.8045 ft |
| `0803_V2` | `SEQ_TRAIN_CFGS[submit_ver_0803_V2]` | 15 | 5.0055 ft |

Each run is three geographic repeats times five folds. The wrapper selects the
correct default or named registry entry and delegates model/data behavior to
that run's archived `seq_NN_*.py` files. It does not modernize the recipe.

## Archived recipe source structure

`0803_V2` is the template for the bundled sequence source. The same module
responsibilities apply to the other five snapshots, with historical additions
or omissions preserved per snapshot:

| File | Responsibility |
|---|---|
| `seq_NN_cfg.py` | `CFG` defaults and the named `SEQ_TRAIN_CFGS` registry. |
| `seq_NN_main.py` | Well discovery, output handling, `run_common()`, training entry, and test prediction. |
| `seq_NN_train.py` | Geographic CV splits, fold training, checkpoint selection, OOF assembly, and inference orchestration. |
| `seq_NN_dataset.py` | Horizontal/Typewell loading, window construction, simulation, augmentation, and feature tensors. |
| `seq_NN_data_prep.py` | Static channel construction plus production PF heatmap generation and caching. |
| `seq_NN_geo_prior.py` | Fold-safe XY geo-prior construction and its diagnostic fields. |
| `seq_NN_models.py` | Sequence U-Net model, heads, and prediction outputs. |
| `seq_NN_pretrained_unet.py` | Pretrained ConvNeXt encoder/decoder building blocks. |
| `seq_NN_trf_backbones.py` | Transformer backbone adapters used by the snapshot's optional model paths. |
| `seq_NN_trf_unet.py` | Transformer U-Net wrapper used when that model path is selected. |
| `cfg.pkl` | The archived resolved configuration for the reference run, retained for recipe comparison. |
| `seq_nn.log` | The archived run log, retained as provenance and an audit reference. |

Standalone research utilities from the submission archives are not bundled.
In particular, `seq_NN_data_prep_pf_opt.py` is a PF optimization/benchmark
harness, and `seq_NN_geo_condition_analysis.py` is a post-training geo-prior
analysis harness. No training or inference module imports or invokes either
file. The active pipeline implementations are in `seq_NN_data_prep.py` and
`seq_NN_geo_prior.py`, respectively.

`seq_NN_main_reproduce.py` is the package-level orchestration wrapper. It loads
one bundled snapshot, applies the relative settings paths, selects one recipe,
and copies the exact source modules beside newly trained artifacts. It is not a
replacement for `seq_NN_main.py`.

## Installation and hardware

Use Python 3.10 with the pinned dependencies. The commands below assume the
desired environment's `python` and `pip` are already active:

```bash
python -m pip install --upgrade pip
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r requirements.txt
```

The launcher uses `python3` by default. Set `PYTHON_BIN` to the executable in
another environment before invoking it. A CUDA run requires a compatible
CUDA-capable GPU and may need the public
`timm/convnext_small.in12k_ft_in1k_384` checkpoint in the standard Hugging Face
cache. That checkpoint is the only network download; all competition data and
historical sequence sources are bundled here.

Before a full run, raise the inherited open-file limit:

```bash
ulimit -n 8192
```

## Outputs

Training one ID writes under `reproduction_outputs/<ID>/`:

- `seq_NN*.py`: the exact source snapshot used by that run;
- `models.pkl`: newly trained 15-fold model dictionary;
- `oof_df.pqt`: repeat-averaged OOF predictions and diagnostics;
- `submission_details.pqt` and `submission.csv`: individual-family test output;
- `cfg.pkl` and `seq_nn.log`: resolved config and execution log.

The settings and data inputs are read-only. PF heatmap caches are derived under
the internal `PF_cache/` path and can be regenerated when absent. Seeds are
fixed, but strict CUDA determinism is intentionally disabled in the historical
recipes; retrained weights and predictions can therefore vary slightly across
GPU/software stacks.

For the final six-family ensemble and saved-model inference, use the existing
notebook implementation:

<https://www.kaggle.com/code/w5833946/submit-reproduce>

It can consume the six newly trained output directories. The notebook is not
needed to reproduce any individual family.
