#!/usr/bin/env python3
"""Rebuild the geographic CV map from the packaged training CSVs.

The sequence recipes only consume ``well_id``, ``horizontal_avg_X`` and
``horizontal_avg_Y``.  Each centroid is the arithmetic mean over the complete
horizontal sequence, including both the visible prefix and the target suffix;
the suffix labels are not read.  The original project map also carried
PNG-derived Typewell labels and group statistics.  Those fields are omitted
here because they are not a training input and cannot be reconstructed from
the horizontal/typewell CSV pair alone.

When an existing map is present, the required-column projection is compared as
serialized six-decimal CSV text before the output is replaced.  This makes the
first self-contained reproduction fail loudly if the packaged data and map
ever drift, while allowing the historical eight-column map to be migrated to
the compact three-column form.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd


MAP_COLUMNS = ("well_id", "horizontal_avg_X", "horizontal_avg_Y")
HORIZONTAL_SUFFIX = "__horizontal_well.csv"


def build_map(train_dir: Path) -> pd.DataFrame:
    """Return sorted full-sequence centroids for every training horizontal CSV."""

    paths = sorted(train_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"no *{HORIZONTAL_SUFFIX} files found under {train_dir}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in paths:
        well_id = path.name[: -len(HORIZONTAL_SUFFIX)]
        if well_id in seen:
            raise ValueError(f"duplicate training well_id {well_id!r}")
        seen.add(well_id)
        frame = pd.read_csv(path, usecols=["X", "Y"])
        if frame.empty:
            raise ValueError(f"training horizontal file is empty: {path}")
        means = frame.mean(numeric_only=True)
        if not means.loc[["X", "Y"]].notna().all():
            raise ValueError(f"non-finite X/Y mean in {path}")
        rows.append(
            {
                "well_id": well_id,
                "horizontal_avg_X": means["X"],
                "horizontal_avg_Y": means["Y"],
            }
        )

    result = pd.DataFrame(rows, columns=MAP_COLUMNS)
    result["well_id"] = result["well_id"].astype(str)
    return result


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize the map with the stable formatting used by the package."""

    return frame.to_csv(index=False, columns=MAP_COLUMNS, float_format="%.6f").encode("utf-8")


def compare_reference(generated: pd.DataFrame, reference_path: Path) -> None:
    """Compare required columns, retaining compatibility with the old map."""

    if not reference_path.is_file():
        return
    reference = pd.read_csv(reference_path, dtype={"well_id": str})
    missing = [column for column in MAP_COLUMNS if column not in reference.columns]
    if missing:
        raise ValueError(f"reference map {reference_path} is missing columns: {missing}")
    reference_bytes = csv_bytes(reference.loc[:, MAP_COLUMNS])
    generated_bytes = csv_bytes(generated)
    if reference_bytes != generated_bytes:
        raise ValueError(
            f"generated centroids do not exactly match {reference_path}; "
            "check the training CSV contents and map definition"
        )
    omitted = [column for column in reference.columns if column not in MAP_COLUMNS]
    if omitted:
        print(
            "verified exact centroid match; omitting non-training PNG/group columns: "
            + ", ".join(omitted)
        )
    else:
        print(f"verified exact centroid match: {reference_path}")


def write_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(csv_bytes(frame))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Existing map to compare before replacement; defaults to --output.",
    )
    args = parser.parse_args(argv)

    train_dir = args.train_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    reference_path = (args.reference or args.output).expanduser().resolve()
    generated = build_map(train_dir)
    compare_reference(generated, reference_path)
    write_atomically(generated, output_path)
    print(f"wrote {len(generated):,} training centroids to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
