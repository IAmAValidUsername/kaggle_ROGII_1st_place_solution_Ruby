#!/usr/bin/env python3
"""Compare a reproduced cfg.pkl with its archived reference recipe.

The output directory and other machine-local paths must change when a run is
reproduced in a temporary or host-provided environment.  This verifier removes
only those I/O locations, then compares every remaining serialized field and
prints a stable SHA-256 digest of the effective training recipe.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any


ENVIRONMENT_PATH_FIELDS = {
    "project_root",
    "local_data_dir",
    "kaggle_data_dir",
    "data_dir",
    "train_path",
    "test_path",
    "output_dir",
    "PF_heatmap_cache_dir",
    "PF_sample_cache_dir",
    "cv_geo_map_path",
}


def import_snapshot(source_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    for module_name in list(sys.modules):
        if module_name == "seq_NN_cfg" or module_name.startswith("seq_NN_"):
            del sys.modules[module_name]
    sys.path.insert(0, str(source_dir))
    importlib.invalidate_caches()
    importlib.import_module("seq_NN_cfg")


def load_cfg(path: Path):
    with path.expanduser().resolve().open("rb") as handle:
        # These are trusted, locally generated competition artifacts.
        return pickle.load(handle)


def normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return "<environment-path>"
    if dataclasses.is_dataclass(value):
        return {
            "__dataclass__": type(value).__name__,
            "fields": normalize(dataclasses.asdict(value)),
        }
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, tuple):
        return {"__tuple__": [normalize(item) for item in value]}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, set):
        return {"__set__": sorted(normalize(item) for item in value)}
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return normalize(vars(value))
    return repr(value)


def normalized_cfg(cfg) -> dict[str, Any]:
    return {
        key: normalize(value)
        for key, value in sorted(vars(cfg).items())
        if key not in ENVIRONMENT_PATH_FIELDS
    }


def render(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare archived and reproduced sequence-NN configs.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Archived directory containing seq_NN_cfg.py.")
    parser.add_argument("--reference-cfg", type=Path, required=True, help="Archived cfg.pkl.")
    parser.add_argument("--candidate-cfg", type=Path, required=True, help="Reproduced cfg.pkl.")
    parser.add_argument("--id", required=True, help="Human-readable run ID for the report.")
    args = parser.parse_args(argv)

    import_snapshot(args.source_dir)
    reference = normalized_cfg(load_cfg(args.reference_cfg))
    candidate = normalized_cfg(load_cfg(args.candidate_cfg))
    reference_digest = digest(reference)
    candidate_digest = digest(candidate)

    if reference != candidate:
        print(f"{args.id}: CONFIG MISMATCH")
        print(f"reference recipe sha256: {reference_digest}")
        print(f"candidate recipe sha256: {candidate_digest}")
        diff = difflib.unified_diff(
            render(reference).splitlines(),
            render(candidate).splitlines(),
            fromfile="archived cfg.pkl",
            tofile="reproduced cfg.pkl",
            lineterm="",
        )
        for line_no, line in enumerate(diff):
            if line_no >= 240:
                print("... diff truncated ...")
                break
            print(line)
        return 1

    print(f"{args.id}: EXACT RECIPE MATCH (environment paths excluded)")
    print(f"recipe sha256: {candidate_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
