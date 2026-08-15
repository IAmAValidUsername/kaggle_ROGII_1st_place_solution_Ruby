#!/usr/bin/env python3
"""Single-run reproduction entrypoint for the archived sequence NN snapshots.

This file deliberately contains orchestration only.  The actual data loading,
model definition, loss, fold construction, checkpoint selection, and inference
remain in the bundled ``reference_results/<ID>/seq_NN_*.py`` snapshot.  Run it
from the ``submission_model`` directory, for example::

    python seq_NN_main_reproduce.py --id 0801_V2

The six short IDs are stable aliases for the recipes documented in README.md.
All filesystem locations come from SETTINGS.json; this is important because
the archived cfg modules compute their defaults relative to their snapshot
directory, not relative to the original project checkout.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any


# ``common`` means that the archived run used ``CFG()`` directly.  ``registry``
# means that the archived run was one named entry in SEQ_TRAIN_CFGS and must be
# selected before calling the snapshot's common training function.  The f value
# is retained from the archived cfg.pkl (0803 used f=1 when its directory was
# created); it has no effect for a new, non-existent output directory.
RUN_RECIPES = {
    "0719_V1": {"mode": "common", "registry_name": None, "archived_f": 0},
    "0724_V1": {"mode": "common", "registry_name": None, "archived_f": 0},
    "0729_V3": {"mode": "registry", "registry_name": "submit_ver_0729_V3", "archived_f": 0},
    "0801_V1": {"mode": "registry", "registry_name": "submit_ver_0801_V1", "archived_f": 0},
    "0801_V2": {"mode": "registry", "registry_name": "submit_ver_0801_V2", "archived_f": 0},
    "0803_V2": {"mode": "registry", "registry_name": "submit_ver_0803_V2", "archived_f": 1},
}

ID_ALIASES = {
    **{name: name for name in RUN_RECIPES},
    **{f"submit_ver_{name}": name for name in RUN_RECIPES},
}

REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "timm": "timm",
    "pyarrow": "pyarrow",
    "numba": "numba",
    "tqdm": "tqdm",
}

REQUIRED_SETTINGS_PATHS = (
    "project_root",
    "data_dir",
    "train_dir",
    "test_dir",
    "output_dir",
    "pf_cache_dir",
    "cv_geo_map_path",
)


def _log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _settings_candidates() -> list[Path]:
    """Find SETTINGS.json without embedding a checkout-specific absolute path.

    In the documented workflow the wrapper is run from ``submission_model``,
    but searching parent package directories also keeps explicit relocation
    and notebook use convenient.  An explicit ``--settings`` always wins.
    """

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    bases = [script_dir, cwd, *script_dir.parents, *cwd.parents]
    candidates: list[Path] = []
    for base in bases:
        candidates.extend((base / "SETTINGS.json", base / "submission_model" / "SETTINGS.json"))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _resolve_path(value: Any, settings_path: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = settings_path.parent / path
    return path.resolve()


def load_settings(settings_arg: Path | None) -> tuple[dict[str, Any], Path]:
    settings_path = settings_arg.expanduser().resolve() if settings_arg else None
    if settings_path is None:
        settings_path = next((p for p in _settings_candidates() if p.is_file()), None)
    if settings_path is None:
        searched = "\n  ".join(str(p) for p in _settings_candidates()[:8])
        raise FileNotFoundError(
            "SETTINGS.json was not found. Copy it beside this script or pass "
            f"--settings PATH. Searched:\n  {searched}"
        )
    with settings_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    paths = raw.get("paths", raw)
    missing = [key for key in REQUIRED_SETTINGS_PATHS if key not in paths]
    if missing:
        raise ValueError(f"{settings_path} is missing required paths keys: {missing}")
    resolved = dict(raw)
    resolved["paths"] = {
        key: _resolve_path(paths[key], settings_path) for key in REQUIRED_SETTINGS_PATHS
    }
    references = raw.get("reference_results", {})
    resolved["reference_results"] = {
        run_id: _resolve_path(path, settings_path) for run_id, path in references.items()
    }
    return resolved, settings_path


def check_runtime_dependencies() -> None:
    """Fail early with an actionable message before importing archived code."""

    missing: list[str] = []
    versions: list[str] = []
    for distribution, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            versions.append(f"{distribution}=={version}")
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            missing.append(f"{distribution} ({exc})")
    if missing:
        raise RuntimeError(
            "Missing sequence-NN dependencies:\n  " + "\n  ".join(missing) +
            "\nInstall the pinned packages from requirements.txt before training."
        )
    _log("runtime dependencies: " + ", ".join(versions))


def rebuild_geo_map(settings: dict[str, Any]) -> None:
    """Rebuild and verify the compact geo-CV map before loading a snapshot."""

    from generate_train_geo_map import build_map, compare_reference, write_atomically

    paths = settings["paths"]
    generated = build_map(paths["train_dir"])
    compare_reference(generated, paths["cv_geo_map_path"])
    write_atomically(generated, paths["cv_geo_map_path"])
    _log(f"verified/generated geo map: {paths['cv_geo_map_path']}")


def resolve_source_dir(
    source_arg: Path | None,
    settings: dict[str, Any],
    run_id: str,
) -> Path:
    candidates = []
    if source_arg is not None:
        candidates.append(source_arg.expanduser().resolve())
    reference_dir = settings.get("reference_results", {}).get(run_id)
    if reference_dir is not None:
        candidates.append(Path(reference_dir).resolve())
    candidates.extend((Path.cwd().resolve(), Path(__file__).resolve().parent))
    for candidate in candidates:
        if (candidate / "seq_NN_cfg.py").is_file() and (candidate / "seq_NN_main.py").is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find archived seq_NN_cfg.py and seq_NN_main.py. "
        f"Pass --source-dir RESULT_DIR (checked: {rendered})."
    )


def copy_snapshot_sources(source_dir: Path, output_dir: Path) -> None:
    """Place the exact archived sequence source beside the reproduced model."""

    source_files = sorted(source_dir.glob("seq_NN*.py"))
    for source_path in source_files:
        # copy2 preserves the archived bytes and filesystem timestamps.  Keeping
        # these modules beside models.pkl also provides its pickle class imports.
        copy2(source_path, output_dir / source_path.name)
    _log(f"copied {len(source_files)} archived seq_NN*.py files to {output_dir}")


def import_snapshot(source_dir: Path):
    """Import the snapshot from ``source_dir`` rather than the checkout's src/."""

    source_dir = source_dir.resolve()
    # A long-lived Python process (notably a notebook or the verification
    # helper) may have imported another snapshot already.  Remove only the
    # sequence modules so their absolute imports resolve to this directory.
    for module_name in list(sys.modules):
        if module_name == "seq_NN_cfg" or module_name.startswith("seq_NN_"):
            del sys.modules[module_name]
    sys.path.insert(0, str(source_dir))
    importlib.invalidate_caches()
    cfg_module = importlib.import_module("seq_NN_cfg")
    main_module = importlib.import_module("seq_NN_main")
    return cfg_module, main_module


def canonical_run_id(run_id: str) -> str:
    try:
        return ID_ALIASES[run_id]
    except KeyError as exc:
        choices = ", ".join(RUN_RECIPES)
        raise ValueError(f"unknown --id {run_id!r}; choose one of: {choices}") from exc


def select_recipe(run_id: str, cfg_module):
    canonical = canonical_run_id(run_id)
    recipe = RUN_RECIPES[canonical]
    if recipe["mode"] == "common":
        cfg = cfg_module.CFG()
    else:
        registry = dict(cfg_module.SEQ_TRAIN_CFGS)
        name = recipe["registry_name"]
        if name not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"snapshot registry does not contain {name!r}; available entries: {available}"
            )
        # Registry builders intentionally return independent CFG objects, but
        # deepcopy makes that contract explicit for historical snapshots too.
        from copy import deepcopy

        cfg = deepcopy(registry[name])
    cfg.refresh()
    return canonical, recipe, cfg


def apply_settings_paths(cfg, settings: dict[str, Any], output_dir: Path) -> None:
    paths = settings["paths"]
    cfg.project_root = paths["project_root"]
    cfg.local_data_dir = paths["data_dir"]
    cfg.data_dir = paths["data_dir"]
    cfg.train_path = paths["train_dir"]
    cfg.test_path = paths["test_dir"]
    cfg.output_dir = output_dir.resolve()

    cfg.PF_heatmap_cache_dir = paths["pf_cache_dir"]

    # Every archived CFG contains this field, but all six selected recipes use
    # diff_block_bootstrap and never read the PF sample cache.  Keep the class's
    # historical project-relative value without exposing an inactive setting.
    cfg.PF_sample_cache_dir = paths["project_root"] / "PF_sample_cache"

    # Archived configs intentionally store None here and resolve the implicit
    # map as data_dir/train_png_typewell_map.csv.  Preserve that exact value
    # whenever the map is present under data_dir.  In a new environment where
    # only submission_model/data/train_png_typewell_map.csv is available, use
    # the explicit settings path as the equivalent fallback.
    implicit_map = paths["data_dir"] / "train_png_typewell_map.csv"
    if getattr(cfg, "cv_geo_map_path", None) is None and not implicit_map.is_file():
        cfg.cv_geo_map_path = paths["cv_geo_map_path"]
    elif getattr(cfg, "cv_geo_map_path", None) is not None:
        cfg.cv_geo_map_path = paths["cv_geo_map_path"]
    cfg.submit_mode = False
    cfg.refresh()


def apply_cli_overrides(cfg, args: argparse.Namespace) -> None:
    for arg_name, cfg_name in (
        ("device", "device"),
        ("well_limit", "well_limit"),
        ("epochs", "epochs"),
        ("fold_count", "fold_count"),
        ("cv_repeats", "cv_repeats"),
        ("batch_size", "batch_size"),
        ("val_batch_size", "val_batch_size"),
        ("num_workers", "num_workers"),
        ("grad_accum_steps", "grad_accum_steps"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if args.persistent_workers is not None:
        cfg.persistent_workers = bool(args.persistent_workers)
    if args.f is not None:
        cfg.f = args.f
    if args.smoke:
        # A bounded smoke run is intentionally opt-in; the default command
        # retains every archived epoch, fold, augmentation, and batch setting.
        cfg.device = args.device or "cpu"
        cfg.well_limit = 2
        cfg.epochs = 1
        cfg.min_epochs = 1
        cfg.early_stopping_rounds = 1
        cfg.fold_count = min(2, int(cfg.fold_count))
        cfg.cv_repeats = 1
        cfg.num_workers = 0
        cfg.persistent_workers = False
        cfg.f = 1
    cfg.refresh()


def validate_inputs(cfg, main_module) -> tuple[list[str], list[str]]:
    if not cfg.train_path.is_dir():
        raise FileNotFoundError(f"train directory does not exist: {cfg.train_path}")
    if not cfg.test_path.is_dir():
        raise FileNotFoundError(f"test directory does not exist: {cfg.test_path}")
    train_wells, test_wells = main_module.discover_run_wells(cfg, include_test=True)
    if not train_wells:
        raise ValueError(f"no train wells found under {cfg.train_path}")
    if not test_wells:
        raise ValueError(f"no test wells found under {cfg.test_path}")
    map_path = main_module.seq_NN_train.resolve_cv_geo_map_path(cfg)
    if cfg.cv_split_mode == "geo_skfold" and not map_path.is_file():
        raise FileNotFoundError(f"geo CV map does not exist: {map_path}")
    return train_wells, test_wells


def run(args: argparse.Namespace) -> int:
    if args.list_runs:
        for run_id, recipe in RUN_RECIPES.items():
            source = "CFG()" if recipe["mode"] == "common" else recipe["registry_name"]
            print(f"{run_id}: {source}")
        return 0

    check_runtime_dependencies()
    settings, settings_path = load_settings(args.settings)
    rebuild_geo_map(settings)
    canonical_requested = canonical_run_id(args.id)
    source_dir = resolve_source_dir(args.source_dir, settings, canonical_requested)
    cfg_module, main_module = import_snapshot(source_dir)
    canonical, recipe, cfg = select_recipe(canonical_requested, cfg_module)
    cfg.f = int(recipe["archived_f"])

    paths = settings["paths"]
    if args.output_dir is None:
        output_dir = paths["output_dir"] / canonical
    else:
        output_dir = args.output_dir.expanduser().resolve()
    apply_settings_paths(cfg, settings, output_dir)
    apply_cli_overrides(cfg, args)
    train_wells, test_wells = validate_inputs(cfg, main_module)

    if not args.startup_only and str(cfg.device).startswith("cuda"):
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"cfg.device={cfg.device!r}, but torch.cuda.is_available() is False; "
                "install the pinned CUDA build and expose a compatible NVIDIA GPU"
            )

    _log(f"recipe={canonical}; source_dir={source_dir}; settings={settings_path}")
    _log(f"train_wells={len(train_wells):,}; test_wells={len(test_wells):,}; output_dir={cfg.output_dir}")
    _log(f"effective geo map={main_module.seq_NN_train.resolve_cv_geo_map_path(cfg)}")

    output_dir_existed = cfg.output_dir.exists()
    if not main_module.prepare_output_dir(cfg.output_dir, cfg.f):
        return 0
    copy_snapshot_sources(source_dir, cfg.output_dir)

    if args.startup_only:
        # This is the verification boundary: it executes imports, config
        # refresh, data discovery, geo-map resolution, and cfg serialization,
        # then exits before allocating a model or entering k-fold training.
        main_module.log_section(f"reproduction setup {canonical}")
        main_module.log_cfg_summary(cfg, train_wells, test_wells, include_test=True)
        main_module.save_cfg(cfg)
        _log("startup-only requested; training was not entered")
        return 0

    log_path = cfg.output_dir / cfg.log_filename
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = main_module.Tee(sys.stdout, log_file)
        with redirect_stdout(tee):
            if output_dir_existed and cfg.f == 1:
                main_module.log(f"WARNING: output directory already exists and f=1 was set: {cfg.output_dir}")
            main_module.run_common(cfg)
            main_module.log(f"saved print log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce one archived sequence-NN recipe using the local seq_NN_* snapshot."
    )
    parser.add_argument("--id", choices=sorted(ID_ALIASES), help="Recipe ID, e.g. 0801_V2.")
    parser.add_argument("--list-runs", action="store_true", help="List supported IDs and exit.")
    parser.add_argument("--settings", type=Path, help="SETTINGS.json; auto-discovered beside this script when omitted.")
    parser.add_argument("--source-dir", type=Path, help="Directory containing the archived seq_NN_*.py files.")
    parser.add_argument("--output-dir", type=Path, help="Exact output directory; default is SETTINGS.paths.output_dir/<id>.")
    parser.add_argument("--device", help="Torch device override, such as cuda or cpu.")
    parser.add_argument("--well-limit", type=int, help="Use only the first N train/test wells (smoke/debug only).")
    parser.add_argument("--epochs", type=int, help="Epoch override; omitted means the archived value.")
    parser.add_argument("--fold-count", type=int, help="Fold-count override; omitted means the archived value.")
    parser.add_argument("--cv-repeats", type=int, help="CV-repeat override; omitted means the archived value.")
    parser.add_argument("--batch-size", type=int, help="Training batch-size override.")
    parser.add_argument("--val-batch-size", type=int, help="Validation/inference batch-size override.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker-count override.")
    parser.add_argument("--grad-accum-steps", type=int, help="Gradient accumulation override.")
    parser.add_argument("--persistent-workers", type=int, choices=[0, 1], help="DataLoader persistent worker override.")
    parser.add_argument("--f", type=int, choices=[0, 1], help="Allow writing an existing output directory.")
    parser.add_argument(
        "--startup-only", "--stop-after-setup", dest="startup_only", action="store_true",
        help="Serialize/validate cfg.pkl and stop before model training.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny two-well, one-epoch smoke training job.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_runs:
        return run(args)
    if not args.id:
        parser.error("--id is required unless --list-runs is used")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
