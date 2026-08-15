#!/usr/bin/env bash
set -euo pipefail

# One-command front end for users who do not know the project layout.  It
# validates the pinned runtime, rebuilds the geographic CV map from the bundled
# training CSVs, selects the requested bundled snapshot from SETTINGS.json, and
# starts the requested reproduction.
# The caller should run `ulimit -n 8192` in the invoking shell first.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_PATH="${SETTINGS_PATH:-${SCRIPT_DIR}/SETTINGS.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WRAPPER_SOURCE="${SCRIPT_DIR}/seq_NN_main_reproduce.py"
MAP_GENERATOR="${SCRIPT_DIR}/generate_train_geo_map.py"
CFG_VERIFIER="${SCRIPT_DIR}/verify_cfg.py"
REQUIREMENTS_PATH="${SCRIPT_DIR}/requirements.txt"
SUPPORTED_IDS=(0719_V1 0724_V1 0729_V3 0801_V1 0801_V2 0803_V2)

usage() {
  sed -n '1,220p' "${SCRIPT_DIR}/entry_points.md"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 2
  fi
}

generate_geo_map() {
  local train_dir map_path
  train_dir="$(json_path paths train_dir)"
  map_path="$(json_path paths cv_geo_map_path)"
  "${PYTHON_BIN}" "${MAP_GENERATOR}" \
    --train-dir "${train_dir}" \
    --output "${map_path}" \
    --reference "${map_path}"
}

check_requirements() {
  require_file "${REQUIREMENTS_PATH}"
  "${PYTHON_BIN}" - "${REQUIREMENTS_PATH}" <<'PY'
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

requirements_path = Path(sys.argv[1])
required = {
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "torchvision",
    "timm", "pyarrow", "numba", "tqdm", "pillow", "huggingface-hub",
    "safetensors", "filelock", "fsspec", "pyyaml"
}
pins = {}
for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "==" not in line:
        continue
    name, version = line.split("==", 1)
    pins[name.lower()] = version

missing_pins = sorted(required - pins.keys())
if missing_pins:
    raise SystemExit(f"requirements.txt is missing required pins: {missing_pins}")

problems = []
for name in sorted(required):
    expected = pins[name]
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        problems.append(f"{name}: not installed (expected {expected})")
        continue
    if actual != expected:
        problems.append(f"{name}: installed {actual}, expected {expected}")

if problems:
    raise SystemExit("Pinned runtime check failed:\n  " + "\n  ".join(problems))
print("Pinned runtime check passed for the sequence-NN dependencies.")
PY
}

json_path() {
  local section="$1"
  local key="$2"
  "${PYTHON_BIN}" - "${SETTINGS_PATH}" "${section}" "${key}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

settings_path = Path(sys.argv[1]).expanduser().resolve()
section, key = sys.argv[2], sys.argv[3]
data = json.loads(settings_path.read_text(encoding="utf-8"))
value = Path(os.path.expandvars(str(data[section][key]))).expanduser()
if not value.is_absolute():
    value = settings_path.parent / value
print(value.resolve())
PY
}

check_snapshot() {
  local run_id="$1"
  local result_dir="$2"
  local required
  if [[ ! -d "${result_dir}" ]]; then
    echo "Reference result directory for ${run_id} does not exist: ${result_dir}" >&2
    exit 2
  fi
  for required in seq_NN_cfg.py seq_NN_main.py seq_NN_train.py cfg.pkl; do
    if [[ ! -f "${result_dir}/${required}" ]]; then
      echo "${run_id}: missing ${result_dir}/${required}" >&2
      exit 2
    fi
  done
}

run_from_snapshot() {
  local run_id="$1"
  local result_dir="$2"
  local exact_output_dir="$3"
  shift 3

  "${PYTHON_BIN}" "${WRAPPER_SOURCE}" \
    --settings "${SETTINGS_PATH}" \
    --source-dir "${result_dir}" \
    --id "${run_id}" \
    --output-dir "${exact_output_dir}" \
    "$@"
}

verify_all() {
  local output_root verify_root run_id result_dir candidate_dir
  output_root="$(json_path paths output_dir)"
  mkdir -p -- "${output_root}"
  verify_root="$(mktemp -d "${output_root}/cfg_verify.XXXXXX")"
  echo "Temporary verification root: ${verify_root}"

  for run_id in "${SUPPORTED_IDS[@]}"; do
    result_dir="$(json_path reference_results "${run_id}")"
    candidate_dir="${verify_root}/${run_id}"
    check_snapshot "${run_id}" "${result_dir}"
    echo
    echo "===== ${run_id}: snapshot startup ====="
    run_from_snapshot "${run_id}" "${result_dir}" "${candidate_dir}" --startup-only
    "${PYTHON_BIN}" "${CFG_VERIFIER}" \
      --source-dir "${result_dir}" \
      --reference-cfg "${result_dir}/cfg.pkl" \
      --candidate-cfg "${candidate_dir}/cfg.pkl" \
      --id "${run_id}"
  done

  echo
  echo "All six archived recipes passed startup and cfg.pkl comparison."
  if [[ "${KEEP_TEMP}" == "1" ]]; then
    echo "Verification outputs retained at: ${verify_root}"
  else
    rm -rf -- "${verify_root}"
  fi
}

train_one() {
  local run_id="$1"
  local result_dir output_root exact_output_dir
  result_dir="$(json_path reference_results "${run_id}")"
  output_root="$(json_path paths output_dir)"
  exact_output_dir="${output_root}/${run_id}"
  check_snapshot "${run_id}" "${result_dir}"
  mkdir -p -- "${output_root}"

  if [[ -e "${exact_output_dir}" && "${FORCE}" != "1" ]]; then
    echo "Output already exists: ${exact_output_dir}" >&2
    echo "Choose a new SETTINGS.paths.output_dir or rerun with --force." >&2
    exit 2
  fi

  local forwarded=()
  if [[ -n "${DEVICE}" ]]; then
    forwarded+=(--device "${DEVICE}")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    forwarded+=(--f 1)
  fi
  echo "Starting ${run_id} from ${result_dir}"
  echo "Artifacts will be written to ${exact_output_dir}"
  run_from_snapshot "${run_id}" "${result_dir}" "${exact_output_dir}" "${forwarded[@]}"
}

preflight_train_all() {
  local output_root run_id result_dir exact_output_dir
  output_root="$(json_path paths output_dir)"
  mkdir -p -- "${output_root}"

  # Validate every snapshot and destination before spending time on the first
  # training run.  The per-run checks in train_one() remain the final guard.
  for run_id in "${SUPPORTED_IDS[@]}"; do
    result_dir="$(json_path reference_results "${run_id}")"
    check_snapshot "${run_id}" "${result_dir}"
    exact_output_dir="${output_root}/${run_id}"
    if [[ -e "${exact_output_dir}" && "${FORCE}" != "1" ]]; then
      echo "Output already exists: ${exact_output_dir}" >&2
      echo "Choose a new SETTINGS.paths.output_dir or rerun with --force." >&2
      exit 2
    fi
  done
}

train_all() {
  local run_id
  preflight_train_all
  echo "Starting all six recipes in order: ${SUPPORTED_IDS[*]}"
  for run_id in "${SUPPORTED_IDS[@]}"; do
    echo
    echo "===== ${run_id}: full training ====="
    train_one "${run_id}"
  done
  echo
  echo "All six recipe trainings completed successfully."
}

MODE=""
RUN_ID=""
DEVICE=""
FORCE=0
KEEP_TEMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify)
      MODE="verify"
      shift
      ;;
    --train)
      MODE="train"
      RUN_ID="${2:-}"
      shift 2
      ;;
    --train-all)
      MODE="train_all"
      shift
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_file "${SETTINGS_PATH}"
require_file "${WRAPPER_SOURCE}"
require_file "${MAP_GENERATOR}"
require_file "${CFG_VERIFIER}"
generate_geo_map
check_requirements

case "${MODE}" in
  verify)
    verify_all
    ;;
  train)
    valid=0
    for candidate in "${SUPPORTED_IDS[@]}"; do
      if [[ "${candidate}" == "${RUN_ID}" ]]; then
        valid=1
        break
      fi
    done
    if [[ "${valid}" != "1" ]]; then
      echo "--train requires one of: ${SUPPORTED_IDS[*]}" >&2
      exit 2
    fi
    train_one "${RUN_ID}"
    ;;
  train_all)
    train_all
    ;;
  *)
    usage
    exit 2
    ;;
esac
