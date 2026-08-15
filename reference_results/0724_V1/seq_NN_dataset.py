import os
from pathlib import Path
from functools import lru_cache
from copy import deepcopy

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter

from seq_NN_data_prep import (
    load_pf_heatmap_cache,
    load_pf_sample_cache,
    uses_pf_heatmap_channels,
    uses_pf_sample_trend,
)
from seq_NN_cfg import TYPEWELL_GR_GRID_TYPES


UNET_MODE = "unet"
LEGACY_STAGE_UNET_MODE = "stage_unet"
Z_SHIFT_NOISE_MODES = {
    "copy",
    "real_block",
    "structured_random",
    "TVT_bias",
    "TVT_bias_v2",
    "mixed_V1",
    "conditional_GR",
    "decomposed_GR",
    "global_pool",
}
Z_SHIFT_NAN_MODES = {"same", "none", "shift", "global"}
MATCHED_GR_METRIC_CHANNELS = {
    "geo_tvt_matched_gr_rmse",
    "geo_tvt_matched_gr_corr",
    "pf_tvt_matched_gr_rmse",
    "pf_tvt_matched_gr_corr",
    "pf_tvt_ffbsi_matched_gr_rmse",
    "pf_tvt_ffbsi_matched_gr_corr",
}
PF_TVT_MATCHED_GR_CHANNELS = {
    "pf_tvt_matched_gr",
    "pf_tvt_ffbsi_matched_gr",
}
PF_PARTICLE_DENSITY_CHANNELS = {
    "pf_particle_density_prob",
}
PF_ROTATE_SHIFT_VARIANTS = {
    "filtered": {
        "prob": "pf_prob",
        "abs_diff": "pf_mean_abs_diff",
        "tvt": "pf_tvt",
        "matched_gr": "pf_tvt_matched_gr",
    },
    "ffbsi": {
        "prob": "pf_prob_ffbsi",
        "abs_diff": "pf_mean_abs_diff_ffbsi",
        "tvt": "pf_tvt_ffbsi",
        "matched_gr": "pf_tvt_ffbsi_matched_gr",
    },
}
PF_ANCHOR_MASS_CHANNELS = {
    "pf_anchor_mass_5": 5.0,
    "pf_anchor_mass_10": 10.0,
    "pf_anchor_mass_20": 20.0,
}
GEO_PRIOR_STATIC_CHANNELS = {
    "geo_s_rel",
    "geo_tvt_rel",
    "geo_tvt_abs_diff",
    "geo_dS",
    "geo_tvt_diff",
    "delta_geo_trend_rbf_5",
}
PF_RELIABILITY_CHANNELS = {
    "pf_entropy_ffbsi",
    "pf_ess_frac_ffbsi",
    "pf_max_prob_ffbsi",
    "pf_geo_abs_diff",
    "pf_anchor_abs_diff",
    "pf_filtered_ffbsi_abs_diff",
    *PF_ANCHOR_MASS_CHANNELS.keys(),
}
TYPEWELL_GR_JITTER_MODES = {"coherent_residual_copy", "reference_only"}
TYPEWELL_SHIFTED_GR_OFFSETS = {
    "m0p25": -0.25,
    "p0p25": 0.25,
}
TYPEWELL_SHIFTED_GR_CHANNELS = {
    f"tw_gr_{suffix}": suffix for suffix in TYPEWELL_SHIFTED_GR_OFFSETS
}
TYPEWELL_SHIFTED_GR_ABS_DIFF_CHANNELS = {
    f"gr_abs_diff_tw_{suffix}": suffix for suffix in TYPEWELL_SHIFTED_GR_OFFSETS
}
GR_QUADRATIC_CHANNELS = (
    "gr_quadratic_a",
    "gr_quadratic_b",
    "gr_quadratic_c",
    "gr_quadratic_rmse",
)
GR_QUADRATIC_MIN_POINTS = 6
XY_DIFF_DERIVED_CHANNELS = (
    "x_twice_diff",
    "y_twice_diff",
    "xy_diff_ac",
)
XY_DIFF_DERIVED_DEPENDENCIES = {
    "x_twice_diff": ("x_diff",),
    "y_twice_diff": ("y_diff",),
    "xy_diff_ac": ("x_diff", "y_diff"),
}


def discover_well_ids(path):
    path = Path(path)
    return sorted(p.name.split("__")[0] for p in path.glob("*__typewell.csv"))


def train_rm_wells_set(cfg):
    return {str(well_id) for well_id in getattr(cfg, "train_rm_wells", [])}


def remove_train_rm_wells(well_ids, cfg):
    removed_set = train_rm_wells_set(cfg)
    if not removed_set:
        return list(well_ids), []
    kept = [well_id for well_id in well_ids if str(well_id) not in removed_set]
    removed = [well_id for well_id in well_ids if str(well_id) in removed_set]
    return kept, removed


def _normalize(arr, mean, std):
    return (arr - mean) / std


@lru_cache(maxsize=None)
def _downsample_bins(raw_len, downsample, out_len):
    bin_idx = np.arange(raw_len, dtype=np.int64) // downsample
    return bin_idx[bin_idx < out_len]


def _downsample_mean(raw_arr, raw_valid, downsample, out_len):
    out = np.full(out_len, np.nan, dtype="float32")
    bin_idx = _downsample_bins(raw_arr.shape[0], downsample, out_len)
    valid = raw_valid[: bin_idx.size] & np.isfinite(raw_arr[: bin_idx.size])
    if not valid.any():
        return out
    sums = np.bincount(bin_idx[valid], weights=raw_arr[: bin_idx.size][valid], minlength=out_len)
    counts = np.bincount(bin_idx[valid], minlength=out_len)
    has_value = counts > 0
    out[has_value] = (sums[has_value] / counts[has_value]).astype("float32")
    return out


def _downsample_rate(raw_arr, downsample, out_len, raw_valid=None, empty_value=0.0):
    out = np.full(out_len, empty_value, dtype="float32")
    bin_idx = _downsample_bins(raw_arr.shape[0], downsample, out_len)
    values = raw_arr[: bin_idx.size].astype("float32", copy=False)
    if raw_valid is not None:
        valid = raw_valid[: bin_idx.size]
        if not valid.any():
            return out
        bin_idx = bin_idx[valid]
        values = values[valid]
    sums = np.bincount(bin_idx, weights=values, minlength=out_len)
    counts = np.bincount(bin_idx, minlength=out_len)
    has_value = counts > 0
    out[has_value] = (sums[has_value] / counts[has_value]).astype("float32")
    return out


def _downsample_count(raw_valid, downsample, out_len):
    bin_idx = _downsample_bins(raw_valid.shape[0], downsample, out_len)
    valid = raw_valid[: bin_idx.size]
    return np.bincount(bin_idx[valid], minlength=out_len).astype("float32")


def _downsample_std(raw_arr, raw_valid, downsample, out_len):
    out = np.full(out_len, np.nan, dtype="float32")
    bin_idx = _downsample_bins(raw_arr.shape[0], downsample, out_len)
    valid = raw_valid[: bin_idx.size] & np.isfinite(raw_arr[: bin_idx.size])
    if not valid.any():
        return out
    values = raw_arr[: bin_idx.size][valid].astype("float64", copy=False)
    valid_bins = bin_idx[valid]
    sums = np.bincount(valid_bins, weights=values, minlength=out_len)
    sq_sums = np.bincount(valid_bins, weights=values * values, minlength=out_len)
    counts = np.bincount(valid_bins, minlength=out_len).astype("float64")
    has_value = counts > 0.0
    mean = np.zeros(out_len, dtype="float64")
    mean_sq = np.zeros(out_len, dtype="float64")
    mean[has_value] = sums[has_value] / counts[has_value]
    mean_sq[has_value] = sq_sums[has_value] / counts[has_value]
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    out[has_value] = np.sqrt(variance[has_value]).astype("float32")
    return out


def _downsample_extreme(raw_arr, raw_valid, downsample, out_len, reducer):
    out = np.full(out_len, np.nan, dtype="float32")
    bin_idx = _downsample_bins(raw_arr.shape[0], downsample, out_len)
    valid = raw_valid[: bin_idx.size] & np.isfinite(raw_arr[: bin_idx.size])
    if not valid.any():
        return out
    valid_bins = bin_idx[valid]
    values = raw_arr[: bin_idx.size][valid].astype("float32", copy=False)
    unique_bins, first_idx = np.unique(valid_bins, return_index=True)
    out[unique_bins] = reducer.reduceat(values, first_idx).astype("float32")
    return out


def _downsample_first_last_delta(raw_arr, raw_valid, downsample, out_len):
    out = np.full(out_len, np.nan, dtype="float32")
    bin_idx = _downsample_bins(raw_arr.shape[0], downsample, out_len)
    valid = raw_valid[: bin_idx.size] & np.isfinite(raw_arr[: bin_idx.size])
    if not valid.any():
        return out
    values = raw_arr[: bin_idx.size][valid].astype("float32", copy=False)
    valid_bins = bin_idx[valid]
    unique_bins, first_idx, counts = np.unique(valid_bins, return_index=True, return_counts=True)
    last_idx = first_idx + counts - 1
    out[unique_bins] = values[last_idx] - values[first_idx]
    return out


def _downsample_slope(raw_y, raw_x, raw_valid, downsample, out_len):
    out = np.full(out_len, np.nan, dtype="float32")
    bin_idx = _downsample_bins(raw_y.shape[0], downsample, out_len)
    valid = (
        raw_valid[: bin_idx.size]
        & np.isfinite(raw_y[: bin_idx.size])
        & np.isfinite(raw_x[: bin_idx.size])
    )
    if not valid.any():
        return out
    x = raw_x[: bin_idx.size][valid].astype("float64", copy=False)
    y = raw_y[: bin_idx.size][valid].astype("float64", copy=False)
    valid_bins = bin_idx[valid]
    counts = np.bincount(valid_bins, minlength=out_len).astype("float64")
    sum_x = np.bincount(valid_bins, weights=x, minlength=out_len)
    sum_y = np.bincount(valid_bins, weights=y, minlength=out_len)
    sum_xx = np.bincount(valid_bins, weights=x * x, minlength=out_len)
    sum_xy = np.bincount(valid_bins, weights=x * y, minlength=out_len)
    denom = counts * sum_xx - sum_x * sum_x
    has_slope = (counts >= 2.0) & (np.abs(denom) > 1e-12)
    slope = (counts[has_slope] * sum_xy[has_slope] - sum_x[has_slope] * sum_y[has_slope]) / denom[has_slope]
    out[has_slope] = slope.astype("float32")
    return out


def _downsample_gr_quadratic(raw_gr, raw_valid, downsample, out_len):
    """Fit GR = a*x^2 + b*x + c in every sufficiently observed bin."""
    outputs = {
        name: np.full(out_len, np.nan, dtype="float32")
        for name in GR_QUADRATIC_CHANNELS
    }
    bin_idx = _downsample_bins(raw_gr.shape[0], downsample, out_len)
    valid_positions = np.flatnonzero(
        raw_valid[: bin_idx.size] & np.isfinite(raw_gr[: bin_idx.size])
    )
    if valid_positions.size < GR_QUADRATIC_MIN_POINTS:
        return outputs

    valid_bins = bin_idx[valid_positions]
    unique_bins, group_starts, group_counts = np.unique(
        valid_bins,
        return_index=True,
        return_counts=True,
    )
    eligible_groups = group_counts >= GR_QUADRATIC_MIN_POINTS
    if not eligible_groups.any():
        return outputs

    # Scaling each observed span avoids unstable extrapolation when finite GR
    # occupies only a small part of a wider downsample bin.
    relative_idx = (valid_positions % int(downsample)).astype("float64")
    group_ends = group_starts + group_counts - 1
    group_centers = (
        relative_idx[group_starts[eligible_groups]]
        + relative_idx[group_ends[eligible_groups]]
    ) / 2.0
    group_half_spans = (
        relative_idx[group_ends[eligible_groups]]
        - relative_idx[group_starts[eligible_groups]]
    ) / 2.0
    eligible_counts = group_counts[eligible_groups]
    eligible_points = np.repeat(eligible_groups, group_counts)
    fit_bins = valid_bins[eligible_points]
    fit_positions = valid_positions[eligible_points]
    x = (
        relative_idx[eligible_points] - np.repeat(group_centers, eligible_counts)
    ) / np.repeat(group_half_spans, eligible_counts)
    y = raw_gr[fit_positions].astype("float64", copy=False)

    x2 = x * x
    sum_x = np.bincount(fit_bins, weights=x, minlength=out_len)
    sum_x2 = np.bincount(fit_bins, weights=x2, minlength=out_len)
    sum_x3 = np.bincount(fit_bins, weights=x2 * x, minlength=out_len)
    sum_x4 = np.bincount(fit_bins, weights=x2 * x2, minlength=out_len)
    sum_y = np.bincount(fit_bins, weights=y, minlength=out_len)
    sum_xy = np.bincount(fit_bins, weights=x * y, minlength=out_len)
    sum_x2y = np.bincount(fit_bins, weights=x2 * y, minlength=out_len)
    sum_y2 = np.bincount(fit_bins, weights=y * y, minlength=out_len)

    fit_bin_idx = unique_bins[eligible_groups]
    counts = eligible_counts.astype("float64", copy=False)
    # Assemble all normal equations from grouped moments, then solve the bins
    # together instead of calling a polynomial fitter in a Python loop.
    normal_matrix = np.stack(
        (
            np.stack(
                (sum_x4[fit_bin_idx], sum_x3[fit_bin_idx], sum_x2[fit_bin_idx]),
                axis=-1,
            ),
            np.stack(
                (sum_x3[fit_bin_idx], sum_x2[fit_bin_idx], sum_x[fit_bin_idx]),
                axis=-1,
            ),
            np.stack(
                (sum_x2[fit_bin_idx], sum_x[fit_bin_idx], counts),
                axis=-1,
            ),
        ),
        axis=1,
    )
    rhs = np.stack(
        (sum_x2y[fit_bin_idx], sum_xy[fit_bin_idx], sum_y[fit_bin_idx]),
        axis=-1,
    )
    coefficients = np.linalg.solve(normal_matrix, rhs[..., None])[..., 0]
    residual_sse = np.maximum(
        sum_y2[fit_bin_idx] - np.sum(coefficients * rhs, axis=1),
        0.0,
    )

    for coefficient_idx, name in enumerate(GR_QUADRATIC_CHANNELS[:3]):
        outputs[name][fit_bin_idx] = coefficients[:, coefficient_idx].astype("float32")
    outputs["gr_quadratic_rmse"][fit_bin_idx] = np.sqrt(
        residual_sse / counts
    ).astype("float32")
    return outputs


def _raw_gradient_over_finite_runs(raw_arr, raw_valid):
    gradient = np.full(raw_arr.shape, np.nan, dtype="float32")
    valid_idx = np.flatnonzero(raw_valid & np.isfinite(raw_arr))
    if valid_idx.size < 2:
        return gradient
    split_points = np.flatnonzero(np.diff(valid_idx) > 1) + 1
    for run_idx in np.split(valid_idx, split_points):
        if run_idx.size < 2:
            continue
        gradient[run_idx] = np.gradient(raw_arr[run_idx].astype("float32", copy=False)).astype("float32")
    return gradient


def _seen_surface_rel(raw):
    return raw["tvt_input_rel"] + raw["z_rel"]


def _seen_surface_valid(raw, raw_has_row):
    return raw_has_row & np.isfinite(raw["tvt_input_rel"]) & np.isfinite(raw["z_rel"])


def _finish_feature(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _noise_mode_weights(noise_mode, valid_modes, *, context):
    if isinstance(noise_mode, str):
        weights = {noise_mode: 1.0}
    elif isinstance(noise_mode, dict):
        weights = {str(mode): float(weight) for mode, weight in noise_mode.items()}
    else:
        raise TypeError(f"{context} must be a string or a dict of mode probabilities, got {type(noise_mode).__name__}")

    unknown = sorted(set(weights) - set(valid_modes))
    if unknown:
        raise ValueError(f"unknown {context}: {unknown}; expected one of {sorted(valid_modes)}")

    active = {}
    for mode, weight in weights.items():
        if not np.isfinite(weight):
            raise ValueError(f"{context} probability for {mode!r} must be finite, got {weight}")
        if weight < 0.0:
            raise ValueError(f"{context} probability for {mode!r} must be non-negative, got {weight}")
        if weight > 0.0:
            active[mode] = weight
    if not active:
        raise ValueError(f"{context} must give positive probability to at least one mode")
    return active


def _sample_noise_mode(noise_mode, valid_modes, *, context):
    active = _noise_mode_weights(noise_mode, valid_modes, context=context)
    total = sum(active.values())
    draw = random.random() * total
    cumsum = 0.0
    last_mode = None
    for mode, weight in active.items():
        cumsum += weight
        last_mode = mode
        if draw <= cumsum:
            return mode
    return last_mode


def _linear_interpolate_gr_after_simulation(well_data):
    horizontal = well_data["horizontal"]
    gr = horizontal["GR"]
    finite = np.isfinite(gr)
    if finite.all():
        return well_data
    if not finite.any():
        return well_data
    gr_isnan_mask = ~finite
    idx = np.arange(gr.shape[0], dtype=np.float32)
    filled_gr = np.interp(idx, idx[finite], gr[finite]).astype("float32")
    horizontal = dict(horizontal)
    horizontal["GR"] = filled_gr
    horizontal["GR_isnan_mask"] = gr_isnan_mask
    well_data = dict(well_data)
    well_data["horizontal"] = horizontal
    return well_data


def _gr_interpolate_enabled(cfg):
    return bool(getattr(cfg, "gr_interpolate", getattr(cfg, "GR_interpolate", False)))


def _out_of_range_gr_mask_enabled(cfg):
    return bool(getattr(cfg, "out_of_range_GR_mask", False))


def _local_typewell_gr_bounds(typewell, tvt0, cfg, well_id):
    local = (
        (typewell["TVT"] >= np.float32(tvt0 - cfg.typewell_window))
        & (typewell["TVT"] <= np.float32(tvt0 + cfg.typewell_window))
        & np.isfinite(typewell["GR"])
    )
    if not local.any():
        raise ValueError(f"{well_id}: out_of_range_GR_mask found no finite clipped Typewell GR")
    gr_values = typewell["GR"][local]
    return float(np.min(gr_values)), float(np.max(gr_values))


def _mask_out_of_range_horizontal_gr(well_data, cfg):
    if not _out_of_range_gr_mask_enabled(cfg):
        return well_data

    well_id = well_data["well_id"]
    horizontal = well_data["horizontal"]
    last_seen_idx = _last_seen_idx(horizontal, well_id)
    gr_min, gr_max = _local_typewell_gr_bounds(
        well_data["typewell"],
        horizontal["TVT_input"][last_seen_idx],
        cfg,
        well_id,
    )

    gr = horizontal["GR"]
    finite_gr = np.isfinite(gr)
    out_of_range = finite_gr & ((gr < np.float32(gr_min)) | (gr > np.float32(gr_max)))
    if not out_of_range.any():
        return well_data

    horizontal = dict(horizontal)
    masked_gr = gr.astype("float32", copy=True)
    raw_gr_isnan = horizontal.get("GR_isnan_mask")
    if raw_gr_isnan is None:
        raw_gr_isnan = ~finite_gr
    else:
        raw_gr_isnan = np.asarray(raw_gr_isnan, dtype=bool).copy()
    masked_gr[out_of_range] = np.nan
    raw_gr_isnan[out_of_range] = True
    horizontal["GR"] = masked_gr
    horizontal["GR_isnan_mask"] = raw_gr_isnan

    well_data = dict(well_data)
    well_data["horizontal"] = horizontal
    return well_data


def _normalize_if_known(name, arr, stats):
    if name in stats:
        arr = _normalize(arr, *stats[name])
    return _finish_feature(arr)


def _cfg_mode(cfg, name, allowed):
    mode = str(getattr(cfg, name, "global")).lower()
    if mode not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {mode!r}")
    return mode


def _finite_mean_or(values, fallback):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return float(fallback)
    return float(np.mean(values[finite]))


def _pearson_corr_or_zero(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape[0] < 2:
        return 0.0
    left_centered = left - np.float32(np.mean(left))
    right_centered = right - np.float32(np.mean(right))
    denom = float(np.sqrt(np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered)))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.sum(left_centered * right_centered) / denom)
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _last_seen_idx(horizontal, well_id):
    finite_seen = np.flatnonzero(np.isfinite(horizontal["TVT_input"]))
    if finite_seen.size == 0:
        raise ValueError(f"{well_id}: no finite TVT_input anchor rows")
    return int(finite_seen[-1])


def _md_stretch_cfg(cfg):
    aug_cfg = getattr(cfg, "aug_cfg", {})
    return aug_cfg.get("MD_streching", {}) if isinstance(aug_cfg, dict) else {}


def _stretch_to_full_size_enabled(cfg):
    # This is a pipeline-level switch.  MD_streching remains an independent
    # source-level simulation/augmentation and must not control canonical
    # train/inference geometry.
    return bool(getattr(cfg, "strech_to_full_size", False))


def _linear_interp_finite_runs(source_coord, values, target_coord):
    """Interpolate finite runs without filling NaN gaps.

    Run boundaries are handled with vectorized ``searchsorted``/adjacency
    checks.  A singleton finite source value is assigned to its nearest
    destination coordinate so it remains observable after stretching.
    """
    source_coord = np.asarray(source_coord, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    target_coord = np.asarray(target_coord, dtype=np.float64)
    out = np.full(target_coord.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(source_coord) & np.isfinite(values)
    source_idx = np.flatnonzero(finite)
    target_idx = np.flatnonzero(np.isfinite(target_coord))
    if source_idx.size == 0 or target_idx.size == 0:
        return out

    x = source_coord[source_idx]
    y = values[source_idx]
    query = target_coord[target_idx]
    source_count = x.size
    right = np.searchsorted(x, query, side="left")
    right_valid = right < source_count
    right_safe = np.clip(right, 0, source_count - 1)
    left_safe = np.clip(right_safe - 1, 0, source_count - 1)
    exact = right_valid & (x[right_safe] == query)
    adjacent = right_valid & (right_safe > 0) & (
        source_idx[right_safe] - source_idx[left_safe] == 1
    )
    values_out = np.interp(query, x, values[source_idx], left=np.nan, right=np.nan)
    valid = exact | adjacent
    values_out[~valid] = np.nan

    run_starts = np.flatnonzero(np.r_[True, np.diff(source_idx) > 1])
    run_ends = np.r_[run_starts[1:] - 1, source_count - 1]
    singleton = run_starts[run_starts == run_ends]
    singleton_mask = np.zeros(source_count, dtype=bool)
    singleton_mask[singleton] = True
    singleton_query = exact & singleton_mask[right_safe]
    values_out[singleton_query] = np.nan
    out[target_idx] = values_out.astype(np.float32)
    if singleton.size:
        singleton_x = x[singleton]
        target_right = np.searchsorted(query, singleton_x, side="left")
        target_right = np.clip(target_right, 0, query.size - 1)
        target_left = np.clip(target_right - 1, 0, query.size - 1)
        use_right = np.abs(query[target_right] - singleton_x) < np.abs(
            query[target_left] - singleton_x
        )
        nearest = np.where(use_right, target_right, target_left)
        destination = target_idx[nearest]
        keep = ~np.isfinite(out[destination])
        out[destination[keep]] = y[singleton[keep]]
    return out


def _nearest_resample(source_coord, values, target_coord):
    """Resample a row-validity/missingness mask with nearest-neighbor rules."""
    source_coord = np.asarray(source_coord, dtype=np.float64)
    values = np.asarray(values)
    target_coord = np.asarray(target_coord, dtype=np.float64)
    out = np.zeros(target_coord.shape, dtype=values.dtype)
    finite = np.isfinite(source_coord)
    source_idx = np.flatnonzero(finite)
    target_idx = np.flatnonzero(np.isfinite(target_coord))
    if source_idx.size == 0 or target_idx.size == 0:
        return out

    x = source_coord[source_idx]
    query = target_coord[target_idx]
    right = np.searchsorted(x, query, side="left")
    right = np.clip(right, 0, x.size - 1)
    left = np.clip(right - 1, 0, x.size - 1)
    use_right = np.abs(x[right] - query) < np.abs(x[left] - query)
    nearest = np.where(use_right, right, left)
    out[target_idx] = values[source_idx[nearest]]
    return out


def _stretch_horizontal_suffix(horizontal, last_seen_idx, cfg):
    """Resample only the prediction suffix onto the canonical raw canvas."""
    suffix_indices = np.arange(
        last_seen_idx + 1,
        horizontal["MD"].shape[0],
        dtype=np.int64,
    )
    suffix_len = suffix_indices.size
    target_len = int(cfg.target_len)
    if suffix_len == 0 or target_len == 0:
        return dict(horizontal), np.empty(0, dtype=np.float32), 1.0

    apply_to_md = bool(_md_stretch_cfg(cfg).get("apply_to_MD", True))
    source_coord = (
        np.asarray(horizontal["MD"][suffix_indices], dtype=np.float64)
        if apply_to_md
        else np.arange(suffix_len, dtype=np.float64)
    )
    if suffix_len > 1 and np.any(np.diff(source_coord) <= 0.0):
        raise ValueError("full MD stretching requires a strictly increasing suffix coordinate")
    destination_coord = (
        np.linspace(source_coord[0], source_coord[-1], target_len, dtype=np.float64)
        if suffix_len > 1 and target_len > 1
        else np.full(target_len, source_coord[0], dtype=np.float64)
    )
    inverse_pos = (
        np.interp(source_coord, destination_coord, np.arange(target_len, dtype=np.float64))
        .astype(np.float32)
        if suffix_len > 1 and target_len > 1
        else np.zeros(suffix_len, dtype=np.float32)
    )
    out = {}
    suffix_start = int(last_seen_idx) + 1
    for name, values in horizontal.items():
        values = np.asarray(values)
        prefix = values[:suffix_start]
        if name == "MD":
            suffix = destination_coord if apply_to_md else np.arange(
                suffix_start,
                suffix_start + target_len,
                dtype=np.float64,
            )
        elif values.dtype == np.bool_ or name.endswith("_mask"):
            suffix = _nearest_resample(source_coord, values[suffix_indices], destination_coord)
        else:
            suffix = _linear_interp_finite_runs(
                source_coord,
                values[suffix_indices],
                destination_coord,
            )
        out[name] = np.concatenate((prefix, suffix)).astype(
            np.float32 if values.dtype.kind in "fc" else values.dtype,
            copy=False,
        )
    if not apply_to_md:
        out["MD"] = np.arange(suffix_start + target_len, dtype=np.float32)
    ratio = (
        float((target_len - 1) / (suffix_len - 1))
        if suffix_len > 1 and target_len > 1
        else 1.0
    )
    return out, inverse_pos, ratio


def _make_original_scale_target(horizontal, suffix_slice, tvt0, z0, cfg):
    mode = getattr(cfg, "seq_target_mode", UNET_MODE)
    if mode == LEGACY_STAGE_UNET_MODE:
        mode = UNET_MODE
    if mode != UNET_MODE:
        raise ValueError("sequence NN dataset now supports only seq_target_mode='unet'")
    tvt = horizontal["TVT"][suffix_slice].astype("float32", copy=False)
    del z0
    return tvt - np.float32(tvt0)


def _build_raw_window(horizontal, last_seen_idx, cfg):
    if _stretch_to_full_size_enabled(cfg):
        original_suffix_len = horizontal["MD"].shape[0] - last_seen_idx - 1
        stretched, inverse_pos, scale_ratio = _stretch_horizontal_suffix(
            horizontal,
            last_seen_idx,
            cfg,
        )
        raw, gr_nan, raw_has_row, raw_is_suffix, anchor = _build_raw_window_native(
            stretched,
            last_seen_idx,
            cfg,
        )
        anchor.update(
            {
                "suffix_len": int(original_suffix_len),
                "kept_suffix_len": int(original_suffix_len),
                "stretch_to_full_size": True,
                "stretch_suffix_dest_pos": inverse_pos,
                "stretch_scale_ratio": float(scale_ratio),
            }
        )
        return raw, gr_nan, raw_has_row, raw_is_suffix, anchor
    return _build_raw_window_native(horizontal, last_seen_idx, cfg)


def _build_raw_window_native(horizontal, last_seen_idx, cfg):
    raw_len = cfg.raw_len
    prefix_len = cfg.prefix_len
    target_len = cfg.target_len
    raw = {
        "tvt_input_rel": np.full(raw_len, np.nan, dtype="float32"),
        "gr": np.full(raw_len, np.nan, dtype="float32"),
        "x": np.full(raw_len, np.nan, dtype="float32"),
        "y": np.full(raw_len, np.nan, dtype="float32"),
        "x_rel": np.full(raw_len, np.nan, dtype="float32"),
        "y_rel": np.full(raw_len, np.nan, dtype="float32"),
        "z_rel": np.full(raw_len, np.nan, dtype="float32"),
        "md_rel": np.full(raw_len, np.nan, dtype="float32"),
        "x_abs": np.full(raw_len, np.nan, dtype="float32"),
        "y_abs": np.full(raw_len, np.nan, dtype="float32"),
        "z_abs": np.full(raw_len, np.nan, dtype="float32"),
        "orig_index": np.full(raw_len, np.nan, dtype="float32"),
        "target_value": np.full(raw_len, np.nan, dtype="float32"),
        "tvt_target_value": np.full(raw_len, np.nan, dtype="float32"),
        "geo_s_rel_prior": np.full(raw_len, np.nan, dtype="float32"),
        "geo_nbr_std": np.full(raw_len, np.nan, dtype="float32"),
        "geo_nbr_dS_std": np.full(raw_len, np.nan, dtype="float32"),
        "geo_radial_extrap_score": np.full(raw_len, np.nan, dtype="float32"),
        "surface_ds_value": np.full(raw_len, np.nan, dtype="float32"),
        "z_diff_value": np.full(raw_len, np.nan, dtype="float32"),
    }
    gr_nan = np.ones(raw_len, dtype=bool)
    raw_has_row = np.zeros(raw_len, dtype=bool)
    raw_is_suffix = np.zeros(raw_len, dtype=bool)

    tvt_input = horizontal["TVT_input"]
    tvt0 = float(tvt_input[last_seen_idx])
    x0 = float(horizontal["X"][last_seen_idx])
    y0 = float(horizontal["Y"][last_seen_idx])
    z0 = float(horizontal["Z"][last_seen_idx])
    md0 = float(horizontal["MD"][last_seen_idx])
    row_count = horizontal["MD"].shape[0]

    prefix_start = max(0, last_seen_idx + 1 - prefix_len)
    prefix_size = last_seen_idx + 1 - prefix_start
    prefix_dest_start = prefix_len - prefix_size
    suffix_end = min(row_count, last_seen_idx + 1 + target_len)
    suffix_size = suffix_end - (last_seen_idx + 1)

    source = {
        "tvt_input_rel": horizontal["TVT_input"] - tvt0,
        "gr": horizontal["GR"],
        "x": horizontal["X"],
        "y": horizontal["Y"],
        "x_rel": horizontal["X"] - x0,
        "y_rel": horizontal["Y"] - y0,
        "z_rel": horizontal["Z"] - z0,
        "md_rel": horizontal["MD"] - md0,
        "x_abs": horizontal["X"],
        "y_abs": horizontal["Y"],
        "z_abs": horizontal["Z"],
        "orig_index": horizontal.get(
            "orig_index",
            np.arange(row_count, dtype=np.float32),
        ),
    }
    if "geo_s_rel_prior_abs" in horizontal:
        source["geo_s_rel_prior"] = horizontal["geo_s_rel_prior_abs"] - np.float32(tvt0 + z0)
    for name in ("geo_nbr_std", "geo_nbr_dS_std", "geo_radial_extrap_score"):
        if name in horizontal:
            source[name] = horizontal[name]
    gr_isnan_source = horizontal.get("GR_isnan_mask")
    if gr_isnan_source is None:
        gr_isnan_source = ~np.isfinite(source["gr"])
    else:
        gr_isnan_source = np.asarray(gr_isnan_source, dtype=bool)
    if prefix_size > 0:
        src = slice(prefix_start, last_seen_idx + 1)
        dst = slice(prefix_dest_start, prefix_len)
        for name, arr in source.items():
            raw[name][dst] = arr[src]
        raw_has_row[dst] = True
        gr_nan[dst] = gr_isnan_source[src]
    if suffix_size > 0:
        src = slice(last_seen_idx + 1, suffix_end)
        dst = slice(prefix_len, prefix_len + suffix_size)
        for name, arr in source.items():
            raw[name][dst] = arr[src]
        raw_has_row[dst] = True
        raw_is_suffix[dst] = True
        gr_nan[dst] = gr_isnan_source[src]
        suffix_indices = np.arange(last_seen_idx + 1, suffix_end)
        previous_indices = suffix_indices - 1
        z = horizontal["Z"]
        raw["z_diff_value"][dst] = z[suffix_indices] - z[previous_indices]
        if "TVT" in horizontal:
            raw["target_value"][dst] = _make_original_scale_target(horizontal, src, tvt0, z0, cfg)
            raw["tvt_target_value"][dst] = horizontal["TVT"][src]
            tvt = horizontal["TVT"]
            raw["surface_ds_value"][dst] = (tvt[suffix_indices] + z[suffix_indices]) - (
                tvt[previous_indices] + z[previous_indices]
            )

    anchor = {
        "X0": x0,
        "Y0": y0,
        "Z0": z0,
        "TVT0": tvt0,
        "last_seen_idx": last_seen_idx,
        "prefix_start": prefix_start,
        "suffix_len": row_count - (last_seen_idx + 1),
        "kept_suffix_len": suffix_size,
    }
    return raw, gr_nan, raw_has_row, raw_is_suffix, anchor


def _make_typewell_rel_grid(cfg):
    return np.linspace(
        -cfg.typewell_window + cfg.typewell_window / cfg.typewell_len,
        cfg.typewell_window - cfg.typewell_window / cfg.typewell_len,
        cfg.typewell_len,
        dtype="float32",
    )


def _interpolate_typewell_gr_no_extrapolate(typewell, grid_tvt):
    valid = (grid_tvt >= typewell["TVT"][0]) & (grid_tvt <= typewell["TVT"][-1])
    interp_gr = np.full(grid_tvt.shape, np.nan, dtype="float32")
    if valid.any():
        interp_gr[valid] = np.interp(
            grid_tvt[valid],
            typewell["TVT"],
            typewell["GR"],
        ).astype("float32")
    return interp_gr, valid


def _piecewise_linear_cumulative_area(typewell_tvt, typewell_gr):
    """Return the integral of the native piecewise-linear Typewell curve."""
    segment_width = np.diff(typewell_tvt)
    if typewell_tvt.size < 2 or np.any(segment_width <= 0.0):
        raise ValueError("Typewell TVT must contain at least two strictly increasing points for GR averaging")
    segment_area = 0.5 * (typewell_gr[:-1] + typewell_gr[1:]) * segment_width
    cumulative_area = np.empty(typewell_tvt.shape, dtype=np.float64)
    cumulative_area[0] = 0.0
    cumulative_area[1:] = np.cumsum(segment_area, dtype=np.float64)
    return cumulative_area


def _piecewise_linear_integral_at(typewell_tvt, typewell_gr, cumulative_area, query_tvt):
    """Evaluate the integral from the first Typewell TVT to each query TVT."""
    query_tvt = np.asarray(query_tvt, dtype=np.float64)
    segment_idx = np.searchsorted(typewell_tvt, query_tvt, side="right") - 1
    segment_idx = np.clip(segment_idx, 0, typewell_tvt.size - 2)
    left_tvt = typewell_tvt[segment_idx]
    left_gr = typewell_gr[segment_idx]
    slope = (typewell_gr[segment_idx + 1] - left_gr) / (
        typewell_tvt[segment_idx + 1] - left_tvt
    )
    delta_tvt = query_tvt - left_tvt
    query_gr = left_gr + slope * delta_tvt
    return cumulative_area[segment_idx] + 0.5 * delta_tvt * (left_gr + query_gr)


def _average_typewell_gr_no_extrapolate(
    typewell_tvt,
    typewell_gr,
    cumulative_area,
    grid_tvt,
    half_width,
):
    """Average the piecewise-linear Typewell GR over each local grid interval."""
    grid_tvt = np.asarray(grid_tvt, dtype=np.float64)
    lower_tvt = grid_tvt - np.float64(half_width)
    upper_tvt = grid_tvt + np.float64(half_width)
    valid = (lower_tvt >= typewell_tvt[0]) & (upper_tvt <= typewell_tvt[-1])
    avg_gr = np.full(grid_tvt.shape, np.nan, dtype="float32")
    if valid.any():
        lower_area = _piecewise_linear_integral_at(
            typewell_tvt,
            typewell_gr,
            cumulative_area,
            lower_tvt[valid],
        )
        upper_area = _piecewise_linear_integral_at(
            typewell_tvt,
            typewell_gr,
            cumulative_area,
            upper_tvt[valid],
        )
        avg_gr[valid] = (
            (upper_area - lower_area) / (2.0 * np.float64(half_width))
        ).astype("float32")
    return avg_gr, valid


def _make_typewell_grid_no_extrapolate(typewell, tvt0, cfg):
    rel_grid = _make_typewell_rel_grid(cfg)
    grid_tvt = tvt0 + rel_grid
    grid_type = str(getattr(cfg, "tw_gr_grid_type", "interpolate")).lower()
    if grid_type not in TYPEWELL_GR_GRID_TYPES:
        raise ValueError(
            f"unknown tw_gr_grid_type={grid_type!r}; expected one of {TYPEWELL_GR_GRID_TYPES}"
        )

    if grid_type == "interpolate":
        interp_gr, _ = _interpolate_typewell_gr_no_extrapolate(typewell, grid_tvt)
        out = {"tvt_rel": rel_grid, "gr": interp_gr, "gr_is_nan": ~np.isfinite(interp_gr)}
        for suffix, offset in TYPEWELL_SHIFTED_GR_OFFSETS.items():
            shifted_gr, _ = _interpolate_typewell_gr_no_extrapolate(
                typewell,
                grid_tvt + np.float32(offset),
            )
            out[f"gr_{suffix}"] = shifted_gr
            out[f"gr_{suffix}_is_nan"] = ~np.isfinite(shifted_gr)
        return out

    typewell_tvt = np.asarray(typewell["TVT"], dtype=np.float64)
    typewell_gr = np.asarray(typewell["GR"], dtype=np.float64)
    half_width = float(cfg.typewell_window) / float(cfg.typewell_len)
    if half_width <= 0.0:
        raise ValueError(
            "tw_gr_grid_type='avg' requires positive typewell_window/typewell_len"
        )
    cumulative_area = _piecewise_linear_cumulative_area(typewell_tvt, typewell_gr)

    grid_gr, _ = _average_typewell_gr_no_extrapolate(
        typewell_tvt,
        typewell_gr,
        cumulative_area,
        grid_tvt,
        half_width,
    )
    out = {"tvt_rel": rel_grid, "gr": grid_gr, "gr_is_nan": ~np.isfinite(grid_gr)}
    for suffix, offset in TYPEWELL_SHIFTED_GR_OFFSETS.items():
        shifted_gr, _ = _average_typewell_gr_no_extrapolate(
            typewell_tvt,
            typewell_gr,
            cumulative_area,
            grid_tvt + np.float32(offset),
            half_width,
        )
        out[f"gr_{suffix}"] = shifted_gr
        out[f"gr_{suffix}_is_nan"] = ~np.isfinite(shifted_gr)
    return out


def _typewell_gr_from_grid(typewell_grid, suffix=None):
    key = "gr" if suffix is None else f"gr_{suffix}"
    nan_key = "gr_is_nan" if suffix is None else f"gr_{suffix}_is_nan"
    tw_gr_is_nan = typewell_grid[nan_key].astype(bool)
    tw_gr = typewell_grid[key].astype("float32", copy=True)
    tw_gr[tw_gr_is_nan] = np.nan
    return tw_gr, tw_gr_is_nan


def _make_tw_gr_channel(tw_gr, tw_gr_is_nan, horizontal_len, typewell_len, cfg, gr_norm):
    if gr_norm is None:
        values = _normalize(tw_gr, *cfg.typewell_stats["gr"])
    else:
        values = _normalize(tw_gr, *gr_norm["typewell_gr"])
    values = values.astype("float32", copy=False)
    values[tw_gr_is_nan] = np.nan
    return np.broadcast_to(_finish_feature(values)[None, :], (horizontal_len, typewell_len))


def _make_gr_abs_diff_channel(gr, gr_nan_rate, tw_gr, tw_gr_is_nan, diff_std):
    if diff_std <= 0.0:
        raise ValueError(f"GR difference std must be positive, got {diff_std}")
    channel = np.abs((gr[:, None] - tw_gr[None, :]) / np.float32(diff_std))
    channel[gr_nan_rate >= 1.0, :] = 0.0
    channel[:, tw_gr_is_nan] = 0.0
    return _finish_feature(channel)


def _downsample_horizontal_feature(raw, gr_nan, raw_has_row, name, cfg):
    if name == "gr_nan_rate":
        return _downsample_rate(
            gr_nan.astype("float32", copy=False),
            cfg.downsample,
            cfg.num_bins,
            raw_valid=raw_has_row,
            empty_value=1.0,
        )
    if name == "gr_std":
        return _downsample_std(
            raw["gr"],
            raw_has_row & np.isfinite(raw["gr"]),
            cfg.downsample,
            cfg.num_bins,
        )
    if name == "gr_max":
        return _downsample_extreme(
            raw["gr"],
            raw_has_row & np.isfinite(raw["gr"]),
            cfg.downsample,
            cfg.num_bins,
            np.maximum,
        )
    if name == "gr_min":
        return _downsample_extreme(
            raw["gr"],
            raw_has_row & np.isfinite(raw["gr"]),
            cfg.downsample,
            cfg.num_bins,
            np.minimum,
        )
    if name == "gr_first_last_delta":
        return _downsample_first_last_delta(
            raw["gr"],
            raw_has_row & np.isfinite(raw["gr"]),
            cfg.downsample,
            cfg.num_bins,
        )
    if name == "gr_slope":
        return _downsample_slope(
            raw["gr"],
            raw["md_rel"],
            raw_has_row & np.isfinite(raw["gr"]) & np.isfinite(raw["md_rel"]),
            cfg.downsample,
            cfg.num_bins,
        )
    if name in GR_QUADRATIC_CHANNELS:
        return _downsample_gr_quadratic(
            raw["gr"],
            raw_has_row & ~gr_nan,
            cfg.downsample,
            cfg.num_bins,
        )[name]
    if name in {"gr_grad_mean", "gr_grad_std"}:
        raw_grad = _raw_gradient_over_finite_runs(raw["gr"], raw_has_row & ~gr_nan)
        raw_grad_valid = np.isfinite(raw_grad)
        if name == "gr_grad_mean":
            return _downsample_mean(raw_grad, raw_grad_valid, cfg.downsample, cfg.num_bins)
        return _downsample_std(raw_grad, raw_grad_valid, cfg.downsample, cfg.num_bins)
    if name in {"gr_diff_mean", "gr_diff_std"}:
        raw_diff = np.full_like(raw["gr"], np.nan, dtype="float32")
        raw_diff[1:] = np.diff(raw["gr"])
        raw_diff_valid = raw_has_row & np.r_[False, raw_has_row[:-1]] & np.isfinite(raw_diff)
        if name == "gr_diff_mean":
            return _downsample_mean(raw_diff, raw_diff_valid, cfg.downsample, cfg.num_bins)
        return _downsample_std(raw_diff, raw_diff_valid, cfg.downsample, cfg.num_bins)
    if name == "seen_tvt_rel":
        return _downsample_mean(
            raw["tvt_input_rel"],
            raw_has_row & np.isfinite(raw["tvt_input_rel"]),
            cfg.downsample,
            cfg.num_bins,
        )
    if name == "seen_S_rel":
        seen_s_rel = _seen_surface_rel(raw)
        return _downsample_mean(
            seen_s_rel,
            _seen_surface_valid(raw, raw_has_row),
            cfg.downsample,
            cfg.num_bins,
        )
    if name == "seen_S_diff":
        seen_s_rel = _seen_surface_rel(raw)
        seen_s_valid = _seen_surface_valid(raw, raw_has_row)
        seen_s_diff = np.full_like(seen_s_rel, np.nan)
        has_previous = seen_s_valid & np.r_[False, seen_s_valid[:-1]]
        seen_s_diff[has_previous] = seen_s_rel[has_previous] - seen_s_rel[np.flatnonzero(has_previous) - 1]
        return _downsample_mean(
            seen_s_diff,
            seen_s_valid & np.isfinite(seen_s_diff),
            cfg.downsample,
            cfg.num_bins,
        )
    if name == "seen_tvt_is_seen":
        return _downsample_rate(
            np.isfinite(raw["tvt_input_rel"]).astype("float32", copy=False),
            cfg.downsample,
            cfg.num_bins,
            raw_valid=raw_has_row,
            empty_value=0.0,
        )
    if (diff_source := _relative_diff_source_name(name)) is not None:
        source = raw[diff_source]
        values = np.full_like(source, np.nan, dtype="float32")
        valid = raw_has_row & np.isfinite(source)
        has_previous = valid & np.r_[False, valid[:-1]]
        idx = np.flatnonzero(has_previous)
        values[idx] = source[idx] - source[idx - 1]
        return _downsample_mean(values, has_previous, cfg.downsample, cfg.num_bins)
    if (xy_source := _xy_fourier_source_name(name)) is not None:
        axis, kind, frequency = xy_source
        source_name = f"{axis}_abs"
        values = _make_xy_fourier(raw[source_name], axis, kind, frequency, cfg)
        return _downsample_mean(values, np.isfinite(raw[source_name]), cfg.downsample, cfg.num_bins)
    if name not in raw:
        raise ValueError(f"unknown horizontal feature: {name}")
    return _downsample_mean(raw[name], np.isfinite(raw[name]), cfg.downsample, cfg.num_bins)


def _adjacent_feature_diff(values):
    values = np.asarray(values, dtype=np.float32)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    valid_pair = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    idx = np.flatnonzero(valid_pair) + 1
    out[idx] = values[idx] - values[idx - 1]
    return out


def _make_xy_diff_derived_features(feature_map, requested_names):
    requested = set(requested_names)
    features = {}
    if "x_twice_diff" in requested:
        features["x_twice_diff"] = _adjacent_feature_diff(feature_map["x_diff"])
    if "y_twice_diff" in requested:
        features["y_twice_diff"] = _adjacent_feature_diff(feature_map["y_diff"])
    if "xy_diff_ac" in requested:
        x_diff = np.asarray(feature_map["x_diff"], dtype=np.float32)
        y_diff = np.asarray(feature_map["y_diff"], dtype=np.float32)
        values = np.full(x_diff.shape, np.nan, dtype=np.float32)
        valid_pair = (
            np.isfinite(x_diff[1:])
            & np.isfinite(x_diff[:-1])
            & np.isfinite(y_diff[1:])
            & np.isfinite(y_diff[:-1])
        )
        idx = np.flatnonzero(valid_pair) + 1
        denominator = np.hypot(x_diff[idx], y_diff[idx]) * np.hypot(
            x_diff[idx - 1],
            y_diff[idx - 1],
        )
        nonzero = denominator > 0.0
        idx = idx[nonzero]
        cosine = (
            x_diff[idx] * x_diff[idx - 1]
            + y_diff[idx] * y_diff[idx - 1]
        ) / denominator[nonzero]
        values[idx] = np.clip(cosine, -1.0, 1.0)
        features["xy_diff_ac"] = values
    return features


def _make_horizontal_feature_map(raw, gr_nan, raw_has_row, feature_names, cfg):
    feature_names = tuple(dict.fromkeys(feature_names))
    requested_derived = tuple(
        name for name in XY_DIFF_DERIVED_CHANNELS if name in feature_names
    )
    base_feature_names = [name for name in feature_names if name not in XY_DIFF_DERIVED_CHANNELS]
    for name in requested_derived:
        for dependency in XY_DIFF_DERIVED_DEPENDENCIES[name]:
            if dependency not in base_feature_names:
                base_feature_names.append(dependency)

    feature_map = {}
    if set(base_feature_names) & set(GR_QUADRATIC_CHANNELS):
        quadratic_features = _downsample_gr_quadratic(
            raw["gr"],
            raw_has_row & ~gr_nan,
            cfg.downsample,
            cfg.num_bins,
        )
        for name in GR_QUADRATIC_CHANNELS:
            if name in base_feature_names:
                feature_map[name] = quadratic_features[name]
    for name in base_feature_names:
        if name not in feature_map:
            feature_map[name] = _downsample_horizontal_feature(
                raw,
                gr_nan,
                raw_has_row,
                name,
                cfg,
            )
    feature_map.update(_make_xy_diff_derived_features(feature_map, requested_derived))
    return {name: feature_map[name] for name in feature_names}


def _make_anchor_features(anchor, cfg):
    return {
        "x0": np.float32((anchor["X0"] - cfg.global_stats["X0"][0]) / cfg.global_stats["X0"][1]),
        "y0": np.float32((anchor["Y0"] - cfg.global_stats["Y0"][0]) / cfg.global_stats["Y0"][1]),
        "z0": np.float32((anchor["Z0"] - cfg.global_stats["Z0"][0]) / cfg.global_stats["Z0"][1]),
        "tvt0": np.float32((anchor["TVT0"] - cfg.global_stats["TVT0"][0]) / cfg.global_stats["TVT0"][1]),
    }


def _make_target(raw, raw_is_suffix, cfg, training):
    target = _downsample_mean(
        raw["target_value"],
        raw_is_suffix & np.isfinite(raw["target_value"]),
        cfg.downsample,
        cfg.num_bins,
    )
    target_mask = _downsample_rate(raw_is_suffix.astype("float32"), cfg.downsample, cfg.num_bins) > 0
    if training:
        target_mask &= np.isfinite(target)
        target = _normalize(target, *cfg.target_stats[UNET_MODE])
    return _finish_feature(target), target_mask.astype(bool)


def _make_tvt_target(raw, raw_is_suffix, cfg):
    tvt_target = _downsample_mean(
        raw["tvt_target_value"],
        raw_is_suffix & np.isfinite(raw["tvt_target_value"]),
        cfg.downsample,
        cfg.num_bins,
    )
    return _finish_feature(tvt_target)


def _make_typewell_nearest2_target_probs(tvt_target, target_mask, tvt0, cfg):
    probs = np.zeros((cfg.num_bins, cfg.typewell_len), dtype="float32")
    valid = target_mask & np.isfinite(tvt_target)
    if not valid.any():
        return probs

    axis = _make_typewell_rel_grid(cfg).astype("float64", copy=False)
    target_rel = tvt_target[valid].astype("float64", copy=False) - float(tvt0)
    # The model cannot predict outside the local Typewell support, so edge
    # targets are projected to the nearest grid endpoint before interpolation.
    target_rel = np.clip(target_rel, axis[0], axis[-1])
    right = np.searchsorted(axis, target_rel, side="right")
    right = np.clip(right, 1, axis.size - 1)
    left = right - 1
    denom = axis[right] - axis[left]
    right_prob = (target_rel - axis[left]) / denom
    left_prob = 1.0 - right_prob

    rows = np.flatnonzero(valid)
    probs[rows, left] = left_prob.astype("float32")
    probs[rows, right] = right_prob.astype("float32")
    return probs


def _make_typewell_exp_smooth_target_probs(tvt_target, target_mask, tvt0, cfg):
    probs = np.zeros((cfg.num_bins, cfg.typewell_len), dtype="float32")
    valid = target_mask & np.isfinite(tvt_target)
    if not valid.any():
        return probs

    sigma = float(getattr(cfg, "alignment_exp_smooth_sigma", 1.0))
    if sigma <= 0.0:
        raise ValueError(f"alignment_exp_smooth_sigma must be positive: {sigma}")
    axis = _make_typewell_rel_grid(cfg).astype("float64", copy=False)
    target_rel = tvt_target[valid].astype("float64", copy=False) - float(tvt0)
    target_rel = np.clip(target_rel, axis[0], axis[-1])
    diff = axis[None, :] - target_rel[:, None]
    target_probs = np.exp(-0.5 * np.square(diff / sigma))
    target_probs /= target_probs.sum(axis=1, keepdims=True)
    probs[np.flatnonzero(valid)] = target_probs.astype("float32")
    return probs


def _make_typewell_target_probs(tvt_target, target_mask, tvt0, cfg):
    target_mode = str(getattr(cfg, "alignment_target_mode", "exp_smooth"))
    if target_mode in {"nbr_two_avg", "nearest2", "nearest_two"}:
        return _make_typewell_nearest2_target_probs(tvt_target, target_mask, tvt0, cfg)
    if target_mode == "exp_smooth":
        return _make_typewell_exp_smooth_target_probs(tvt_target, target_mask, tvt0, cfg)
    raise ValueError(
        f"unknown alignment_target_mode={target_mode!r}; "
        "expected 'nearest2' or 'exp_smooth'"
    )


def _make_aux_target(raw, raw_is_suffix, raw_name, stats_name, cfg, training):
    target = _downsample_mean(
        raw[raw_name],
        raw_is_suffix & np.isfinite(raw[raw_name]),
        cfg.downsample,
        cfg.num_bins,
    )
    if training:
        target = _normalize(target, *cfg.target_stats[stats_name])
    return _finish_feature(target)


def _make_unet_gr_rmse_target(raw, raw_is_suffix, gr_nan, typewell, cfg):
    matched_gr = np.interp(
        raw["tvt_target_value"],
        typewell["TVT"],
        typewell["GR"],
        left=np.nan,
        right=np.nan,
    ).astype("float32")
    valid = (
        raw_is_suffix
        & ~gr_nan
        & np.isfinite(raw["gr"])
        & np.isfinite(raw["tvt_target_value"])
        & np.isfinite(matched_gr)
    )
    residual_sq = np.full(raw["gr"].shape, np.nan, dtype="float32")
    residual_sq[valid] = np.square(raw["gr"][valid] - matched_gr[valid]).astype("float32")
    mean_sq = _downsample_mean(residual_sq, valid, cfg.downsample, cfg.num_bins)
    target_mask = np.isfinite(mean_sq)
    target = np.full(cfg.num_bins, 0.0, dtype="float32")
    target[target_mask] = np.sqrt(mean_sq[target_mask]).astype("float32")
    return target, target_mask.astype("float32")


def _make_common_item(well_data, cfg, training):
    well_id = well_data["well_id"]
    horizontal = well_data["horizontal"]
    typewell = well_data["typewell"]

    last_seen_idx = _last_seen_idx(horizontal, well_id)

    raw, gr_nan, raw_has_row, raw_is_suffix, anchor = _build_raw_window(horizontal, last_seen_idx, cfg)
    target, target_mask = _make_target(raw, raw_is_suffix, cfg, training)
    tvt_target = _make_tvt_target(raw, raw_is_suffix, cfg)
    if training:
        typewell_target_probs = _make_typewell_target_probs(tvt_target, target_mask, anchor["TVT0"], cfg)
    else:
        typewell_target_probs = np.zeros((0,), dtype="float32")
    aux_targets = {}
    geo_s_rel_prior = _finish_feature(
        _downsample_mean(
            raw["geo_s_rel_prior"],
            raw_is_suffix & np.isfinite(raw["geo_s_rel_prior"]),
            cfg.downsample,
            cfg.num_bins,
        )
    )
    geo_nbr_std = _finish_feature(
        _downsample_mean(
            raw["geo_nbr_std"],
            raw_is_suffix & np.isfinite(raw["geo_nbr_std"]),
            cfg.downsample,
            cfg.num_bins,
        )
    )
    geo_nbr_dS_std = _finish_feature(
        _downsample_mean(
            raw["geo_nbr_dS_std"],
            raw_is_suffix & np.isfinite(raw["geo_nbr_dS_std"]),
            cfg.downsample,
            cfg.num_bins,
        )
    )
    geo_radial_extrap_score = _finish_feature(
        _downsample_mean(
            raw["geo_radial_extrap_score"],
            raw_is_suffix & np.isfinite(raw["geo_radial_extrap_score"]),
            cfg.downsample,
            cfg.num_bins,
        )
    )
    row_mask = _downsample_rate(raw_has_row.astype("float32"), cfg.downsample, cfg.num_bins) > 0
    suffix_mask = _downsample_rate(raw_is_suffix.astype("float32"), cfg.downsample, cfg.num_bins) > 0
    tvt0 = anchor["TVT0"]
    submit_indices = np.arange(last_seen_idx + 1, horizontal["MD"].shape[0], dtype=np.int64)

    z_rel = _finish_feature(
        _downsample_mean(
            raw["z_rel"],
            raw_is_suffix & np.isfinite(raw["z_rel"]),
            cfg.downsample,
            cfg.num_bins,
        )
    )
    geo_prior_tvt = (
        horizontal["geo_s_rel_prior_abs"][submit_indices] - horizontal["Z"][submit_indices]
    ).astype("float32", copy=True)
    meta = {
        "well_id": well_id,
        "submit_indices": submit_indices,
        "last_seen_idx": last_seen_idx,
        "tvt0": tvt0,
        "z0": anchor["Z0"],
        "suffix_z": horizontal["Z"][submit_indices].astype("float32", copy=True),
        "suffix_len": int(anchor["suffix_len"]),
        "kept_suffix_len": int(anchor["kept_suffix_len"]),
        "geo_prior_TVT": geo_prior_tvt,
    }
    if anchor.get("stretch_to_full_size", False):
        meta["stretch_to_full_size"] = True
        meta["stretch_suffix_dest_pos"] = np.asarray(
            anchor["stretch_suffix_dest_pos"],
            dtype="float32",
        ).copy()
        meta["stretch_scale_ratio"] = float(anchor["stretch_scale_ratio"])
    if PF_TVT_MATCHED_GR_CHANNELS & set(getattr(cfg, "unet_static_channels", ())):
        meta["typewell_tvt"] = typewell["TVT"].astype("float32", copy=True)
        meta["typewell_gr"] = typewell["GR"].astype("float32", copy=True)
    if training:
        meta["TVT"] = horizontal["TVT"][submit_indices]

    return {
        "raw": raw,
        "gr_nan": gr_nan,
        "raw_has_row": raw_has_row,
        "raw_is_suffix": raw_is_suffix,
        "anchor_features": _make_anchor_features(anchor, cfg),
        "z_rel": z_rel,
        "bin_count": _downsample_count(raw_is_suffix, cfg.downsample, cfg.num_bins),
        "z_diff": _finish_feature(
            _downsample_mean(
                raw["z_diff_value"],
                raw_is_suffix & np.isfinite(raw["z_diff_value"]),
                cfg.downsample,
                cfg.num_bins,
            )
        ),
        "geo_s_rel_prior": geo_s_rel_prior,
        "geo_nbr_std": geo_nbr_std,
        "geo_nbr_dS_std": geo_nbr_dS_std,
        "geo_radial_extrap_score": geo_radial_extrap_score,
        "target": target,
        "tvt_target": tvt_target,
        "typewell_target_probs": typewell_target_probs,
        "aux_targets": aux_targets,
        "matched_gr_metrics": well_data.get("matched_gr_metrics", {}),
        "target_mask": target_mask,
        "row_mask": row_mask.astype(bool),
        "suffix_mask": suffix_mask.astype(bool),
        "meta": meta,
    }


def _make_unet_aux(typewell_grid):
    tw_gr_is_nan = typewell_grid["gr_is_nan"].astype(bool)
    tw_gr = typewell_grid["gr"].astype("float32", copy=True)
    tw_gr[tw_gr_is_nan] = np.nan
    typewell_aux = np.stack([_finish_feature(tw_gr), tw_gr_is_nan.astype("float32")], axis=0).astype("float32")
    return typewell_aux


def _finalize_item(common, unet_static, typewell_aux):
    return {
        "geo_s_rel_prior": common["geo_s_rel_prior"],
        "unet_static": unet_static,
        "typewell_aux": typewell_aux,
        "z_rel": common["z_rel"],
        "bin_count": common["bin_count"],
        "z_diff": common["z_diff"],
        "target": common["target"],
        "typewell_target_probs": common["typewell_target_probs"],
        "aux_targets": common["aux_targets"],
        "target_mask": common["target_mask"],
        "row_mask": common["row_mask"],
        "suffix_mask": common["suffix_mask"],
        "meta": common["meta"],
    }


def _xy_fourier_source_name(channel_name):
    parts = channel_name.split("_")
    if len(parts) != 4 or parts[1] != "abs" or parts[2] not in {"cos", "sin"}:
        return None
    if parts[0] not in {"x", "y", "z"}:
        return None
    try:
        frequency = int(parts[3])
    except ValueError:
        return None
    if frequency < 0:
        return None
    return parts[0], parts[2], frequency


def _make_xy_fourier(values, axis, kind, frequency, cfg):
    source_name = axis.upper()
    normalized = _normalize(values, *cfg.xy_stats[source_name])
    angle = (2**frequency) * normalized
    if kind == "cos":
        return _finish_feature(np.cos(angle))
    return _finish_feature(np.sin(angle))


def _relative_diff_source_name(channel_name):
    return {
        "x_diff": "x_rel",
        "y_diff": "y_rel",
        "z_diff": "z_rel",
    }.get(channel_name)


def _normalize_tvt_diff(values, cfg):
    clipped = np.clip(
        values,
        -float(getattr(cfg, "tvt_diff_clip", 100.0)),
        float(getattr(cfg, "tvt_diff_clip", 100.0)),
    )
    return _finish_feature(_normalize(clipped, *cfg.typewell_stats["tvt_rel"]))


def _geo_dS_from_s_rel(geo_s_rel, suffix_mask, cfg):
    geo_s_rel = np.asarray(geo_s_rel, dtype=np.float32)
    suffix_mask = np.asarray(suffix_mask, dtype=bool)
    values = np.zeros(geo_s_rel.shape, dtype=np.float32)
    valid_idx = np.flatnonzero(suffix_mask & np.isfinite(geo_s_rel))
    if valid_idx.size < 2:
        return values
    gap = np.maximum(valid_idx[1:] - valid_idx[:-1], 1).astype(np.float32)
    scale = gap * np.float32(cfg.downsample)
    values[valid_idx[1:]] = (
        (geo_s_rel[valid_idx[1:]] - geo_s_rel[valid_idx[:-1]]) / scale
    ).astype(np.float32)
    return values


def _geo_tvt_diff_from_s_rel(geo_s_rel, z_diff, suffix_mask, cfg):
    values = _geo_dS_from_s_rel(geo_s_rel, suffix_mask, cfg) - np.asarray(
        z_diff,
        dtype=np.float32,
    )
    values[~np.asarray(suffix_mask, dtype=bool)] = 0.0
    return values.astype(np.float32)


def _geo_tvt_trend_rel_from_s(geo_s_rel, z_rel, suffix_mask):
    geo_s_rel = np.asarray(geo_s_rel, dtype=np.float32)
    z_rel = np.asarray(z_rel, dtype=np.float32)
    suffix_mask = np.asarray(suffix_mask, dtype=bool)
    values = np.zeros(geo_s_rel.shape, dtype=np.float32)
    valid_idx = np.flatnonzero(suffix_mask & np.isfinite(geo_s_rel) & np.isfinite(z_rel))
    if valid_idx.size == 0:
        return values
    geo_tvt_rel = (geo_s_rel - z_rel).astype(np.float32)
    values[valid_idx] = (geo_tvt_rel[valid_idx] - geo_tvt_rel[valid_idx[0]]).astype(np.float32)
    return values


def _delta_geo_trend_rbf_channel(geo_s_rel, z_rel, suffix_mask, tw_tvt_rel, sigma):
    if sigma <= 0.0:
        raise ValueError(f"delta_geo_trend_rbf sigma must be positive, got {sigma}")
    geo_tvt_trend_rel = _geo_tvt_trend_rel_from_s(geo_s_rel, z_rel, suffix_mask)
    delta = tw_tvt_rel[None, :] - geo_tvt_trend_rel[:, None]
    channel = np.exp(-0.5 * np.square(delta / np.float32(sigma))).astype(np.float32)
    channel[~np.asarray(suffix_mask, dtype=bool), :] = 0.0
    return channel


def _interp_prob_rows(source_prob, source_pos):
    source_prob = np.asarray(source_prob, dtype=np.float32)
    source_pos = np.asarray(source_pos, dtype=np.float32)
    out = np.zeros((source_pos.shape[0], source_prob.shape[1]), dtype=np.float32)
    valid = np.isfinite(source_pos) & (source_pos >= 0.0) & (source_pos <= source_prob.shape[0] - 1)
    if not valid.any():
        return out
    left = np.floor(source_pos[valid]).astype(np.int64)
    right = np.clip(left + 1, 0, source_prob.shape[0] - 1)
    frac = (source_pos[valid] - left.astype(np.float32))[:, None]
    out[valid] = source_prob[left] * (1.0 - frac) + source_prob[right] * frac
    return out


def _interp_vector(values, source_pos, default=0.0):
    values = np.asarray(values, dtype=np.float32)
    source_pos = np.asarray(source_pos, dtype=np.float32)
    out = np.full(source_pos.shape[0], float(default), dtype=np.float32)
    valid = np.isfinite(source_pos) & (source_pos >= 0.0) & (source_pos <= values.shape[0] - 1)
    if valid.any():
        out[valid] = np.interp(
            source_pos[valid].astype(np.float32),
            np.arange(values.shape[0], dtype=np.float32),
            values,
        ).astype(np.float32)
    return out


def _cache_source_pos_from_orig_index(pf_cache, orig_index):
    orig_index = np.asarray(orig_index, dtype=np.float32)
    cache_orig_index = np.asarray(pf_cache["window_orig_index"], dtype=np.float32)
    valid = np.isfinite(cache_orig_index) & (cache_orig_index >= 0.0)
    out = np.full(orig_index.shape[0], np.nan, dtype=np.float32)
    if valid.sum() < 2:
        return out
    cache_pos = np.arange(cache_orig_index.shape[0], dtype=np.float32)
    out[:] = np.interp(
        orig_index,
        cache_orig_index[valid],
        cache_pos[valid],
        left=np.nan,
        right=np.nan,
    ).astype(np.float32)
    return out


def _normalize_prob_rows(prob):
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    prob[prob < 0.0] = 0.0
    row_sum = prob.sum(axis=1, keepdims=True)
    np.divide(prob, row_sum, out=prob, where=row_sum > 0.0)
    return prob


def _pf_reliability_values_from_components(
    name,
    prob_ffbsi,
    tvt_ffbsi,
    valid_ffbsi,
    tw_tvt_rel,
    cfg,
    *,
    geo_tvt_rel=None,
    tvt_filtered=None,
    valid_filtered=None,
):
    prob_ffbsi = np.asarray(prob_ffbsi, dtype=np.float32)
    tvt_ffbsi = np.asarray(tvt_ffbsi, dtype=np.float32)
    tw_tvt_rel = np.asarray(tw_tvt_rel, dtype=np.float32)
    row_sum = prob_ffbsi.sum(axis=1)
    valid_ffbsi = np.asarray(valid_ffbsi, dtype=bool) & (row_sum > 0.0)
    values = np.zeros(prob_ffbsi.shape[0], dtype=np.float32)

    if name == "pf_entropy_ffbsi":
        if valid_ffbsi.any():
            p = prob_ffbsi[valid_ffbsi].astype(np.float32, copy=False)
            positive = p > 0.0
            log_p = np.zeros_like(p, dtype=np.float32)
            log_p[positive] = np.log(p[positive]).astype(np.float32)
            entropy = -np.sum(np.where(positive, p * log_p, 0.0), axis=1)
            values[valid_ffbsi] = (entropy / np.float32(np.log(max(prob_ffbsi.shape[1], 2)))).astype(np.float32)
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    if name == "pf_ess_frac_ffbsi":
        if valid_ffbsi.any():
            p = prob_ffbsi[valid_ffbsi].astype(np.float32, copy=False)
            denom = np.sum(np.square(p, dtype=np.float32), axis=1) * np.float32(prob_ffbsi.shape[1])
            ess = np.zeros(denom.shape, dtype=np.float32)
            np.divide(1.0, denom, out=ess, where=denom > 0.0)
            values[valid_ffbsi] = ess
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    if name == "pf_max_prob_ffbsi":
        if valid_ffbsi.any():
            values[valid_ffbsi] = np.max(prob_ffbsi[valid_ffbsi], axis=1).astype(np.float32)
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    if name in PF_ANCHOR_MASS_CHANNELS:
        if valid_ffbsi.any():
            radius = float(PF_ANCHOR_MASS_CHANNELS[name])
            in_window = np.abs(tw_tvt_rel) <= np.float32(radius)
            if in_window.any():
                values[valid_ffbsi] = prob_ffbsi[valid_ffbsi][:, in_window].sum(axis=1).astype(np.float32)
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    if name == "pf_geo_abs_diff":
        if geo_tvt_rel is None:
            raise ValueError("pf_geo_abs_diff requires geo_tvt_rel")
        geo_tvt_rel = np.asarray(geo_tvt_rel, dtype=np.float32)
        valid = valid_ffbsi & np.isfinite(tvt_ffbsi) & np.isfinite(geo_tvt_rel)
        if valid.any():
            values[valid] = _normalize_tvt_diff(np.abs(tvt_ffbsi[valid] - geo_tvt_rel[valid]), cfg)
        return values

    if name == "pf_anchor_abs_diff":
        valid = valid_ffbsi & np.isfinite(tvt_ffbsi)
        if valid.any():
            values[valid] = _normalize_tvt_diff(np.abs(tvt_ffbsi[valid]), cfg)
        return values

    if name == "pf_filtered_ffbsi_abs_diff":
        if tvt_filtered is None or valid_filtered is None:
            raise ValueError("pf_filtered_ffbsi_abs_diff requires filtered PF path components")
        tvt_filtered = np.asarray(tvt_filtered, dtype=np.float32)
        valid_filtered = np.asarray(valid_filtered, dtype=bool)
        valid = valid_ffbsi & valid_filtered & np.isfinite(tvt_ffbsi) & np.isfinite(tvt_filtered)
        if valid.any():
            values[valid] = _normalize_tvt_diff(np.abs(tvt_filtered[valid] - tvt_ffbsi[valid]), cfg)
        return values

    raise ValueError(f"unknown PF reliability channel: {name}")


def _pf_reliability_values(name, common, pf_features, tw_tvt_rel, cfg):
    geo_tvt_rel = common["geo_s_rel_prior"].astype(np.float32, copy=False) - common["z_rel"].astype(
        np.float32,
        copy=False,
    )
    return _pf_reliability_values_from_components(
        name,
        pf_features["pf_prob_ffbsi"],
        pf_features["pf_tvt_rel_pred_ffbsi"],
        pf_features["source_valid_ffbsi"],
        tw_tvt_rel,
        cfg,
        geo_tvt_rel=geo_tvt_rel,
        tvt_filtered=pf_features["pf_tvt_rel_pred"],
        valid_filtered=pf_features["source_valid"],
    )


def _shift_prob_axis(prob, delta_rel, cfg):
    prob = np.asarray(prob, dtype=np.float32)
    delta_rel = np.asarray(delta_rel, dtype=np.float32)
    axis = _make_typewell_rel_grid(cfg).astype(np.float32, copy=False)
    step = np.float32(axis[1] - axis[0])
    source_pos = (axis[None, :] - delta_rel[:, None] - axis[0]) / step
    left = np.floor(source_pos).astype(np.int64)
    frac = (source_pos - left.astype(np.float32)).astype(np.float32)
    right = left + 1
    row_idx = np.arange(prob.shape[0], dtype=np.int64)[:, None]

    shifted = np.zeros_like(prob, dtype=np.float32)
    valid_left = (left >= 0) & (left < prob.shape[1])
    if valid_left.any():
        safe_left = np.clip(left, 0, prob.shape[1] - 1)
        shifted += np.where(valid_left, prob[row_idx, safe_left] * (1.0 - frac), 0.0)
    valid_right = (right >= 0) & (right < prob.shape[1])
    if valid_right.any():
        safe_right = np.clip(right, 0, prob.shape[1] - 1)
        shifted += np.where(valid_right, prob[row_idx, safe_right] * frac, 0.0)
    return _normalize_prob_rows(shifted)


def _shift_axis_image(channel, delta_rel, cfg, fill_value=0.0):
    channel = np.asarray(channel, dtype=np.float32)
    delta_rel = np.asarray(delta_rel, dtype=np.float32)
    axis = _make_typewell_rel_grid(cfg).astype(np.float32, copy=False)
    step = np.float32(axis[1] - axis[0])
    source_pos = (axis[None, :] - delta_rel[:, None] - axis[0]) / step
    left = np.floor(source_pos).astype(np.int64)
    frac = (source_pos - left.astype(np.float32)).astype(np.float32)
    right = left + 1
    row_idx = np.arange(channel.shape[0], dtype=np.int64)[:, None]

    shifted = np.full_like(channel, np.float32(fill_value), dtype=np.float32)
    valid_left = (left >= 0) & (left < channel.shape[1])
    valid_right = (right >= 0) & (right < channel.shape[1])
    has_any = valid_left | valid_right
    if not has_any.any():
        return shifted

    interp = np.zeros_like(channel, dtype=np.float32)
    if valid_left.any():
        safe_left = np.clip(left, 0, channel.shape[1] - 1)
        interp += np.where(valid_left, channel[row_idx, safe_left] * (1.0 - frac), 0.0)
    if valid_right.any():
        safe_right = np.clip(right, 0, channel.shape[1] - 1)
        interp += np.where(valid_right, channel[row_idx, safe_right] * frac, 0.0)
    shifted[has_any] = interp[has_any]
    return shifted.astype(np.float32, copy=False)


def _make_pf_unet_features(common, horizontal_map, pf_cache, cfg):
    if pf_cache is None:
        raise ValueError("PF U-Net channels requested but no PF cache was loaded")

    orig_index = horizontal_map["orig_index"].astype(np.float32, copy=False)
    source_pos = _cache_source_pos_from_orig_index(pf_cache, orig_index)
    orig_tvt_rel = _interp_vector(pf_cache["window_tvt_rel"], source_pos, default=0.0)

    current_tvt_rel = np.zeros(cfg.num_bins, dtype=np.float32)
    seen_tvt_rel = horizontal_map["seen_tvt_rel"].astype(np.float32, copy=False)
    has_current_tvt = np.isfinite(seen_tvt_rel)
    current_tvt_rel[has_current_tvt] = seen_tvt_rel[has_current_tvt]
    if "TVT" in common["meta"]:
        target_tvt_rel = common["tvt_target"].astype(np.float32, copy=False) - np.float32(common["meta"]["tvt0"])
        has_target_tvt = common["suffix_mask"] & np.isfinite(target_tvt_rel)
        current_tvt_rel[has_target_tvt] = target_tvt_rel[has_target_tvt]
        has_current_tvt |= has_target_tvt
    source_has_tvt = _interp_vector(
        pf_cache["window_has_tvt"].astype(np.float32),
        source_pos,
        default=0.0,
    ) > 0.5
    delta_rel = np.zeros(cfg.num_bins, dtype=np.float32)
    shift_mask = has_current_tvt & source_has_tvt
    delta_rel[shift_mask] = current_tvt_rel[shift_mask] - orig_tvt_rel[shift_mask]

    # PF cache is generated once on original wells. z_shift changes TVT while
    # preserving row order, xy_shift changes X/Y plus the induced TVT, reverse_path
    # flips/rebases rows, MD_streching resamples rows, and tail_cut truncates the
    # suffix. All are represented by orig_index plus this anchor-relative TVT-axis
    # shift. GR_noise_shift intentionally leaves PF channels unchanged.
    # start_point_shift stays disabled
    # because it changes the visible-prefix boundary used by fixed split caches.
    no_source = ~(np.isfinite(source_pos) & (source_pos >= 0.0) & (source_pos <= pf_cache["pf_prob"].shape[0] - 1))

    def remap_variant(prob_key, tvt_rel_key):
        pf_prob = _interp_prob_rows(pf_cache[prob_key], source_pos)
        pf_tvt_rel_pred = _interp_vector(pf_cache[tvt_rel_key], source_pos, default=np.nan)
        pf_prob = _shift_prob_axis(pf_prob, delta_rel, cfg)
        pf_tvt_rel_pred = pf_tvt_rel_pred + delta_rel
        invalid_source = no_source | ~np.isfinite(pf_tvt_rel_pred) | (pf_prob.sum(axis=1) <= 0.0)
        pf_prob[invalid_source] = 0.0
        pf_tvt_rel_pred[invalid_source] = 0.0
        return pf_prob, pf_tvt_rel_pred.astype(np.float32), ~invalid_source

    def remap_prob_image(prob_key):
        pf_prob = _interp_prob_rows(pf_cache[prob_key], source_pos)
        pf_prob = _shift_prob_axis(pf_prob, delta_rel, cfg)
        invalid_source = no_source | (pf_prob.sum(axis=1) <= 0.0)
        pf_prob[invalid_source] = 0.0
        return pf_prob.astype(np.float32), ~invalid_source

    pf_prob, pf_tvt_rel_pred, source_valid = remap_variant("pf_prob", "pf_tvt_rel_pred")
    pf_prob_ffbsi, pf_tvt_rel_pred_ffbsi, source_valid_ffbsi = remap_variant(
        "pf_prob_ffbsi",
        "pf_tvt_rel_pred_ffbsi",
    )
    features = {
        "pf_prob": pf_prob,
        "pf_tvt_rel_pred": pf_tvt_rel_pred,
        "source_valid": source_valid,
        "pf_prob_ffbsi": pf_prob_ffbsi,
        "pf_tvt_rel_pred_ffbsi": pf_tvt_rel_pred_ffbsi,
        "source_valid_ffbsi": source_valid_ffbsi,
    }
    if PF_PARTICLE_DENSITY_CHANNELS & set(getattr(cfg, "unet_static_channels", ())):
        pf_particle_density_prob, source_valid_particle_density = remap_prob_image("pf_particle_density_prob")
        features["pf_particle_density_prob"] = pf_particle_density_prob
        features["source_valid_particle_density"] = source_valid_particle_density
    return features


def _pf_tvt_paths_from_cache_for_original_rows(pf_cache, row_count):
    orig_index = np.arange(row_count, dtype=np.float32)
    source_pos = _cache_source_pos_from_orig_index(pf_cache, orig_index)
    return {
        "pf_tvt": _interp_vector(pf_cache["pf_tvt_pred"], source_pos, default=np.nan),
        "pf_tvt_ffbsi": _interp_vector(pf_cache["pf_tvt_pred_ffbsi"], source_pos, default=np.nan),
    }


def _full_typewell_lookup_gr(typewell, tvt_values):
    tvt_values = np.asarray(tvt_values, dtype=np.float32)
    matched_gr = np.full(tvt_values.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(tvt_values)
        & (tvt_values >= typewell["TVT"][0])
        & (tvt_values <= typewell["TVT"][-1])
    )
    if valid.any():
        matched_gr[valid] = np.interp(
            tvt_values[valid],
            typewell["TVT"],
            typewell["GR"],
        ).astype("float32")
    return matched_gr


def _full_typewell_lookup_gr_from_axis(typewell_tvt, typewell_gr, tvt_values):
    tvt_values = np.asarray(tvt_values, dtype=np.float32)
    typewell_tvt = np.asarray(typewell_tvt, dtype=np.float32)
    typewell_gr = np.asarray(typewell_gr, dtype=np.float32)
    matched_gr = np.full(tvt_values.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(tvt_values)
        & (tvt_values >= typewell_tvt[0])
        & (tvt_values <= typewell_tvt[-1])
    )
    if valid.any():
        matched_gr[valid] = np.interp(
            tvt_values[valid],
            typewell_tvt,
            typewell_gr,
        ).astype("float32")
    return matched_gr


def _normalize_matched_gr(values, cfg, gr_norm=None):
    if gr_norm is None:
        return _normalize_if_known("gr", values, cfg.typewell_stats)
    return _finish_feature(_normalize(values, *gr_norm["typewell_gr"]))


def _matched_gr_metrics_for_candidate(horizontal, typewell, candidate_tvt):
    gr = horizontal["GR"]
    matched_gr = _full_typewell_lookup_gr(typewell, candidate_tvt)
    suffix_mask = np.isnan(horizontal["TVT_input"])
    valid = suffix_mask & np.isfinite(gr) & np.isfinite(matched_gr)
    if not valid.any():
        return 0.0, 0.0
    residual = gr[valid].astype(np.float32, copy=False) - matched_gr[valid].astype(np.float32, copy=False)
    rmse = float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))
    corr = _pearson_corr_or_zero(gr[valid], matched_gr[valid])
    return rmse, corr


def _make_static_candidate_sidecars(horizontal, typewell, pf_cache, cfg):
    static_channels = set(getattr(cfg, "unet_static_channels", ()))
    need_metrics = bool(MATCHED_GR_METRIC_CHANNELS & static_channels)
    if not need_metrics:
        return {}, {}

    candidate_tvt = {}
    if need_metrics and {"geo_tvt_matched_gr_rmse", "geo_tvt_matched_gr_corr"} & static_channels:
        candidate_tvt["geo_tvt"] = (
            horizontal["geo_s_rel_prior_abs"] - horizontal["Z"]
        ).astype("float32")

    pf_tvt_by_row = None
    need_any_pf = bool(
        {
            "pf_tvt",
            "pf_tvt_ffbsi",
            "pf_tvt_matched_gr_rmse",
            "pf_tvt_matched_gr_corr",
            "pf_tvt_ffbsi_matched_gr_rmse",
            "pf_tvt_ffbsi_matched_gr_corr",
        }
        & static_channels
    )
    if need_any_pf:
        if pf_cache is None:
            raise ValueError("PF TVT static channels requested but no PF cache was loaded")
        pf_tvt_by_row = _pf_tvt_paths_from_cache_for_original_rows(
            pf_cache,
            horizontal["MD"].shape[0],
        )
        candidate_tvt.update(pf_tvt_by_row)

    metric_sidecars = {}
    if need_metrics:
        metric_candidates = (
            ("geo_tvt", "geo_tvt_matched_gr"),
            ("pf_tvt", "pf_tvt_matched_gr"),
            ("pf_tvt_ffbsi", "pf_tvt_ffbsi_matched_gr"),
        )
        for candidate_name, metric_prefix in metric_candidates:
            rmse_name = f"{metric_prefix}_rmse"
            corr_name = f"{metric_prefix}_corr"
            if rmse_name not in static_channels and corr_name not in static_channels:
                continue
            rmse, corr = _matched_gr_metrics_for_candidate(
                horizontal,
                typewell,
                candidate_tvt[candidate_name],
            )
            metric_sidecars[rmse_name] = np.float32(rmse)
            metric_sidecars[corr_name] = np.float32(corr)

    return {}, metric_sidecars


def _make_gr_normalization_context(horizontal_map, common, well_data, cfg):
    mean_mode = _cfg_mode(cfg, "gr_normalize_mean_mode", {"global", "local"})
    std_mode = _cfg_mode(cfg, "gr_normalize_std_mode", {"global", "local"})

    horizontal_gr_mean, horizontal_gr_std = cfg.horizontal_stats["gr"]
    typewell_gr_mean, typewell_gr_std = cfg.typewell_stats["gr"]
    gr_diff_std = float(getattr(cfg, "gr_diff_std", horizontal_gr_std))

    if mean_mode == "local":
        suffix_gr = horizontal_map["gr"][common["suffix_mask"]]
        local_mean = _finite_mean_or(suffix_gr, horizontal_gr_mean)
        horizontal_gr_mean = local_mean
        typewell_gr_mean = local_mean

    if std_mode == "local":
        floor = float(getattr(cfg, "gr_normalize_local_std_floor", 3.0))
        if floor <= 0.0:
            raise ValueError(f"gr_normalize_local_std_floor must be positive, got {floor}")
        local_std = float(well_data.get("gr_normalize_local_std", gr_diff_std))
        if not np.isfinite(local_std):
            local_std = gr_diff_std
        local_std = max(local_std, floor)
        horizontal_gr_std = local_std
        typewell_gr_std = local_std
        gr_diff_std = local_std

    return {
        "horizontal_gr": (float(horizontal_gr_mean), float(horizontal_gr_std)),
        "typewell_gr": (float(typewell_gr_mean), float(typewell_gr_std)),
        "gr_abs_diff_std": float(gr_diff_std),
    }


def _make_same_mean_power_compression(values, power):
    values = np.asarray(values, dtype=np.float32)
    compressed = np.full(values.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    if not valid.any():
        return compressed
    if np.any(values[valid] <= 0.0):
        raise ValueError("gr_compression_residual requires positive finite Typewell GR values")
    powered = np.power(values[valid].astype(np.float64), float(power))
    scale = float(np.mean(values[valid].astype(np.float64)) / np.mean(powered))
    compressed[valid] = (scale * powered).astype(np.float32)
    return compressed


def _typewell_gr_slope_norm(tw_gr, tw_tvt_rel, diff_std, cfg):
    valid = np.isfinite(tw_gr) & np.isfinite(tw_tvt_rel)
    slope = np.full(tw_gr.shape, np.nan, dtype=np.float32)
    if valid.sum() < 2:
        return slope
    slope[valid] = np.gradient(
        tw_gr[valid].astype(np.float32, copy=False),
        tw_tvt_rel[valid].astype(np.float32, copy=False),
    ).astype(np.float32)
    tvt_step = np.float32(np.nanmedian(np.diff(tw_tvt_rel[valid].astype(np.float32, copy=False))))
    values = slope * tvt_step / np.float32(diff_std)
    slope_clip = float(getattr(cfg, "gr_tw_slope_norm_clip", 8.0))
    if slope_clip > 0.0:
        values = np.clip(values, -slope_clip, slope_clip)
    return values.astype(np.float32, copy=False)


def _prefix_direct_gr_rmse_after_downsample(horizontal, typewell, cfg):
    tvt_input = horizontal["TVT_input"]
    gr = horizontal["GR"]
    finite_seen = np.flatnonzero(np.isfinite(tvt_input))
    if finite_seen.size == 0:
        return float(cfg.gr_diff_std)

    last_seen_idx = int(finite_seen[-1])
    prefix_start = max(0, last_seen_idx + 1 - cfg.prefix_len)
    prefix_size = last_seen_idx + 1 - prefix_start
    prefix_dest_start = cfg.prefix_len - prefix_size
    src = slice(prefix_start, last_seen_idx + 1)
    dst = slice(prefix_dest_start, cfg.prefix_len)

    matched_gr = np.interp(
        tvt_input[src],
        typewell["TVT"],
        typewell["GR"],
        left=np.nan,
        right=np.nan,
    ).astype("float32")
    raw_residual = np.full(cfg.raw_len, np.nan, dtype="float32")
    raw_has_row = np.zeros(cfg.raw_len, dtype=bool)
    raw_residual[dst] = (gr[src] - matched_gr).astype("float32")
    raw_has_row[dst] = True

    residual = _downsample_mean(
        raw_residual,
        raw_has_row & np.isfinite(raw_residual),
        cfg.downsample,
        cfg.num_bins,
    )
    finite = np.isfinite(residual)
    if not finite.any():
        return float(cfg.gr_diff_std)
    return float(np.sqrt(np.mean(np.square(residual[finite], dtype=np.float64))))


def _make_unet_static_input(
    horizontal_map,
    typewell_grid,
    cfg,
    common=None,
    pf_cache=None,
    gr_norm=None,
    pf_features=None,
):
    horizontal_len = cfg.num_bins
    typewell_len = cfg.typewell_len
    gr = horizontal_map["gr"].astype("float32", copy=False)
    gr_nan_rate = _finish_feature(horizontal_map["gr_nan_rate"])
    seen_tvt_rel = _finish_feature(horizontal_map["seen_tvt_rel"])
    tw_gr, tw_gr_is_nan = _typewell_gr_from_grid(typewell_grid)
    tw_tvt_rel = typewell_grid["tvt_rel"].astype("float32", copy=False)
    horizontal_gr_std = cfg.horizontal_stats["gr"][1] if gr_norm is None else gr_norm["horizontal_gr"][1]

    channels = []
    for name in cfg.unet_static_channels:
        if name == "tw_gr":
            channel = _make_tw_gr_channel(tw_gr, tw_gr_is_nan, horizontal_len, typewell_len, cfg, gr_norm)
        elif name in TYPEWELL_SHIFTED_GR_CHANNELS:
            shifted_tw_gr, shifted_tw_gr_is_nan = _typewell_gr_from_grid(
                typewell_grid,
                TYPEWELL_SHIFTED_GR_CHANNELS[name],
            )
            channel = _make_tw_gr_channel(
                shifted_tw_gr,
                shifted_tw_gr_is_nan,
                horizontal_len,
                typewell_len,
                cfg,
                gr_norm,
            )
        elif name == "gr":
            if gr_norm is None:
                values = _normalize_if_known("gr", gr, cfg.horizontal_stats)
            else:
                values = _finish_feature(_normalize(gr, *gr_norm["horizontal_gr"]))
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name in TYPEWELL_SHIFTED_GR_ABS_DIFF_CHANNELS:
            diff_std = float(cfg.gr_diff_std if gr_norm is None else gr_norm["gr_abs_diff_std"])
            shifted_tw_gr, shifted_tw_gr_is_nan = _typewell_gr_from_grid(
                typewell_grid,
                TYPEWELL_SHIFTED_GR_ABS_DIFF_CHANNELS[name],
            )
            channel = _make_gr_abs_diff_channel(
                gr,
                gr_nan_rate,
                shifted_tw_gr,
                shifted_tw_gr_is_nan,
                diff_std,
            )
        elif name in {"gr_abs_diff", "gr_signed_diff", "gr_compression_residual", "gr_ratio_diff", "gr_diff_x_tw_slope"}:
            diff_std = float(cfg.gr_diff_std if gr_norm is None else gr_norm["gr_abs_diff_std"])
            if diff_std <= 0.0:
                raise ValueError(f"GR difference std must be positive, got {diff_std}")
            if name == "gr_abs_diff":
                channel = _make_gr_abs_diff_channel(gr, gr_nan_rate, tw_gr, tw_gr_is_nan, diff_std)
                channels.append(channel.astype("float32"))
                continue
            signed_diff = (gr[:, None] - tw_gr[None, :]) / np.float32(diff_std)
            if name == "gr_signed_diff":
                channel = signed_diff
            elif name == "gr_compression_residual":
                compressed_tw_gr = _make_same_mean_power_compression(
                    tw_gr,
                    float(getattr(cfg, "gr_compression_power", 0.9308)),
                )
                channel = (gr[:, None] - compressed_tw_gr[None, :]) / np.float32(diff_std)
            elif name == "gr_ratio_diff":
                floor = float(getattr(cfg, "gr_ratio_floor", 1.0))
                if floor <= 0.0:
                    raise ValueError(f"gr_ratio_floor must be positive, got {floor}")
                denom = np.maximum(tw_gr, np.float32(floor))
                channel = (gr[:, None] - tw_gr[None, :]) / denom[None, :]
            else:
                tw_slope_norm = _typewell_gr_slope_norm(tw_gr, tw_tvt_rel, diff_std, cfg)
                channel = signed_diff * tw_slope_norm[None, :]
            channel[gr_nan_rate >= 1.0, :] = 0.0
            channel[:, tw_gr_is_nan] = 0.0
            channel = _finish_feature(channel)
        elif name == "gr_isnan_rate":
            channel = np.broadcast_to(gr_nan_rate[:, None], (horizontal_len, typewell_len))
        elif name == "gr_std":
            values = _finish_feature(horizontal_map[name] / horizontal_gr_std)
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name in {"gr_max", "gr_min"}:
            if gr_norm is None:
                values = _normalize_if_known("gr", horizontal_map[name], cfg.horizontal_stats)
            else:
                values = _finish_feature(_normalize(horizontal_map[name], *gr_norm["horizontal_gr"]))
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name == "gr_first_last_delta":
            values = _finish_feature(horizontal_map[name] / horizontal_gr_std)
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name == "gr_slope":
            values = horizontal_map[name] * float(cfg.downsample) / horizontal_gr_std
            slope_clip = float(getattr(cfg, "gr_slope_norm_clip", 8.0))
            if slope_clip > 0.0:
                values = np.clip(values, -slope_clip, slope_clip)
            values = _finish_feature(values)
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name in GR_QUADRATIC_CHANNELS:
            gr_mean = (
                cfg.horizontal_stats["gr"][0]
                if gr_norm is None
                else gr_norm["horizontal_gr"][0]
            )
            values = horizontal_map[name].astype("float32", copy=False)
            if name == "gr_quadratic_c":
                values = (values - np.float32(gr_mean)) / np.float32(horizontal_gr_std)
            else:
                values = values / np.float32(horizontal_gr_std)
            stats = cfg.gr_quadratic_stats[int(cfg.downsample)][name]
            values = _normalize(values, *stats)
            quadratic_clip = float(getattr(cfg, "gr_quadratic_norm_clip", 8.0))
            if quadratic_clip > 0.0:
                values = np.clip(values, -quadratic_clip, quadratic_clip)
            channel = np.broadcast_to(
                _finish_feature(values)[:, None],
                (horizontal_len, typewell_len),
            )
        elif name in {"gr_grad_mean", "gr_grad_std", "gr_diff_mean", "gr_diff_std"}:
            values = _finish_feature(horizontal_map[name] / horizontal_gr_std)
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name in XY_DIFF_DERIVED_CHANNELS:
            stats = cfg.xy_diff_derived_stats[int(cfg.downsample)][name]
            values = _normalize(horizontal_map[name], *stats)
            norm_clip = float(getattr(cfg, "xy_diff_derived_norm_clip", 8.0))
            if norm_clip > 0.0:
                values = np.clip(values, -norm_clip, norm_clip)
            channel = np.broadcast_to(
                _finish_feature(values)[:, None],
                (horizontal_len, typewell_len),
            )
        elif name in {"md_rel", "x_diff", "y_diff", "z_rel", "z_diff"}:
            values = _normalize_if_known(name, horizontal_map[name], cfg.horizontal_stats)
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif _xy_fourier_source_name(name) is not None:
            values = _finish_feature(horizontal_map[name])
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name == "tw_gr_is_nan":
            channel = np.broadcast_to(tw_gr_is_nan.astype("float32")[None, :], (horizontal_len, typewell_len))
        elif name == "tw_tvt_rel":
            values = _normalize_if_known("tvt_rel", tw_tvt_rel, cfg.typewell_stats)
            channel = np.broadcast_to(values[None, :], (horizontal_len, typewell_len))
        elif name == "seen_tvt_rel":
            values = _normalize_if_known("tvt_input_rel", seen_tvt_rel, cfg.horizontal_stats)
            channel = np.broadcast_to(values[:, None], (horizontal_len, typewell_len))
        elif name == "tw_seen_tvt_abs_diff":
            channel = _normalize_tvt_diff(np.abs(tw_tvt_rel[None, :] - seen_tvt_rel[:, None]), cfg)
        elif name in GEO_PRIOR_STATIC_CHANNELS:
            if common is None:
                raise ValueError("geo prior U-Net channel construction requires common item context")
            geo_s_rel = common["geo_s_rel_prior"].astype("float32", copy=False)
            geo_tvt_rel = geo_s_rel - common["z_rel"].astype("float32", copy=False)
            if name == "geo_s_rel":
                values = _normalize(geo_s_rel, *cfg.surface_stats["S_rel"])
                channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
            elif name == "geo_tvt_rel":
                values = _normalize(geo_tvt_rel, *cfg.target_stats[UNET_MODE])
                channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
            elif name == "geo_tvt_abs_diff":
                channel = _normalize_tvt_diff(np.abs(tw_tvt_rel[None, :] - geo_tvt_rel[:, None]), cfg)
            elif name == "geo_dS":
                values = _normalize(
                    _geo_dS_from_s_rel(geo_s_rel, common["suffix_mask"], cfg),
                    *cfg.target_stats["d(TVT+Z)"],
                )
                values[~common["suffix_mask"]] = 0.0
                channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
            elif name == "geo_tvt_diff":
                values = _normalize(
                    _geo_tvt_diff_from_s_rel(geo_s_rel, common["z_diff"], common["suffix_mask"], cfg),
                    *cfg.target_stats["d(TVT+Z)"],
                )
                values[~common["suffix_mask"]] = 0.0
                channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
            else:
                channel = _delta_geo_trend_rbf_channel(
                    geo_s_rel,
                    common["z_rel"],
                    common["suffix_mask"],
                    tw_tvt_rel,
                    sigma=5.0,
                )
        elif name in {"geo_nbr_std", "geo_nbr_dS_std", "geo_radial_extrap_score"}:
            if common is None:
                raise ValueError("geo prior diagnostic U-Net channel construction requires common item context")
            values = common[name].astype("float32", copy=False)
            if name == "geo_nbr_dS_std":
                values = values / np.float32(cfg.target_stats["d(TVT+Z)"][1])
            else:
                values = np.log1p(np.maximum(values, 0.0)).astype("float32", copy=False)
            channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
        elif name in {"pf_tvt", "pf_tvt_ffbsi"}:
            if pf_features is None:
                if common is None:
                    raise ValueError("PF TVT U-Net channel construction requires common item context")
                pf_features = _make_pf_unet_features(common, horizontal_map, pf_cache, cfg)
            is_ffbsi = name.endswith("_ffbsi")
            tvt_key = "pf_tvt_rel_pred_ffbsi" if is_ffbsi else "pf_tvt_rel_pred"
            valid_key = "source_valid_ffbsi" if is_ffbsi else "source_valid"
            values = _normalize(pf_features[tvt_key], *cfg.target_stats[UNET_MODE])
            values[~pf_features[valid_key]] = 0.0
            channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
        elif name in PF_TVT_MATCHED_GR_CHANNELS:
            if pf_features is None:
                if common is None:
                    raise ValueError("PF TVT matched-GR U-Net channel construction requires common item context")
                pf_features = _make_pf_unet_features(common, horizontal_map, pf_cache, cfg)
            is_ffbsi = name.startswith("pf_tvt_ffbsi")
            tvt_key = "pf_tvt_rel_pred_ffbsi" if is_ffbsi else "pf_tvt_rel_pred"
            valid_key = "source_valid_ffbsi" if is_ffbsi else "source_valid"
            if common is None:
                raise ValueError("PF TVT matched-GR U-Net channel construction requires common item context")
            typewell_tvt = common["meta"]["typewell_tvt"]
            typewell_gr = common["meta"]["typewell_gr"]
            pf_tvt_abs = np.float32(common["meta"]["tvt0"]) + pf_features[tvt_key]
            values = _normalize_matched_gr(
                _full_typewell_lookup_gr_from_axis(typewell_tvt, typewell_gr, pf_tvt_abs),
                cfg,
                gr_norm=gr_norm,
            )
            values[~pf_features[valid_key]] = 0.0
            channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
        elif name in MATCHED_GR_METRIC_CHANNELS:
            if common is None:
                raise ValueError("matched-GR metric U-Net channels require common item context")
            value = float(common["matched_gr_metrics"].get(name, 0.0))
            if name in getattr(cfg, "matched_gr_metric_stats", {}):
                value = float(_normalize(value, *cfg.matched_gr_metric_stats[name]))
            channel = np.full((horizontal_len, typewell_len), value, dtype=np.float32)
        elif name in PF_RELIABILITY_CHANNELS:
            if pf_features is None:
                if common is None:
                    raise ValueError("PF reliability U-Net channel construction requires common item context")
                pf_features = _make_pf_unet_features(common, horizontal_map, pf_cache, cfg)
            if common is None:
                raise ValueError("PF reliability U-Net channel construction requires common item context")
            values = _pf_reliability_values(name, common, pf_features, tw_tvt_rel, cfg)
            channel = np.broadcast_to(_finish_feature(values)[:, None], (horizontal_len, typewell_len))
        elif name in PF_PARTICLE_DENSITY_CHANNELS:
            if pf_features is None:
                if common is None:
                    raise ValueError("PF particle-density U-Net channel construction requires common item context")
                pf_features = _make_pf_unet_features(common, horizontal_map, pf_cache, cfg)
            channel = np.log1p(pf_features[name] * np.float32(typewell_len))
        elif name in {"pf_prob", "pf_mean_abs_diff", "pf_prob_ffbsi", "pf_mean_abs_diff_ffbsi"}:
            if pf_features is None:
                if common is None:
                    raise ValueError("PF U-Net channel construction requires common item context")
                pf_features = _make_pf_unet_features(common, horizontal_map, pf_cache, cfg)
            is_ffbsi = name.endswith("_ffbsi")
            prob_key = "pf_prob_ffbsi" if is_ffbsi else "pf_prob"
            tvt_key = "pf_tvt_rel_pred_ffbsi" if is_ffbsi else "pf_tvt_rel_pred"
            valid_key = "source_valid_ffbsi" if is_ffbsi else "source_valid"
            if name in {"pf_prob", "pf_prob_ffbsi"}:
                channel = np.log1p(pf_features[prob_key] * np.float32(typewell_len))
            else:
                channel = _normalize_tvt_diff(
                    np.abs(tw_tvt_rel[None, :] - pf_features[tvt_key][:, None]),
                    cfg,
                )
                channel[~pf_features[valid_key], :] = 0.0
        else:
            raise ValueError(f"unknown unet static channel: {name}")
        channels.append(channel.astype("float32"))
    return np.stack(channels, axis=0).astype("float32")


def _pf_tvt_meta_key(cfg):
    static_channels = set(getattr(cfg, "unet_static_channels", ()))
    if {"pf_prob", "pf_mean_abs_diff", "pf_tvt", "pf_tvt_matched_gr"} & static_channels:
        return "pf_tvt_rel_pred"
    if (
        {"pf_prob_ffbsi", "pf_mean_abs_diff_ffbsi", "pf_tvt_ffbsi", "pf_tvt_ffbsi_matched_gr"}
        | PF_RELIABILITY_CHANNELS
    ) & static_channels:
        return "pf_tvt_rel_pred_ffbsi"
    return None


def _add_analysis_sidecars_to_meta(common, pf_features, cfg):
    pf_key = _pf_tvt_meta_key(cfg)
    if pf_key is None:
        return
    common["meta"]["pf_tvt_rel_pred"] = pf_features[pf_key].astype("float32", copy=True)


def _refresh_pf_tvt_matched_gr_channel(
    static_input,
    channel_idx,
    channel_name,
    tvt_rel,
    valid_rows,
    item,
    cfg,
    row_mask=None,
):
    matched_name = f"{channel_name}_matched_gr"
    if matched_name not in channel_idx:
        return
    meta = item["meta"]
    matched_gr = _full_typewell_lookup_gr_from_axis(
        meta["typewell_tvt"],
        meta["typewell_gr"],
        np.float32(meta["tvt0"]) + tvt_rel.astype(np.float32, copy=False),
    )
    norm = item.get("_pf_matched_gr_norm")
    if norm is None:
        values = _normalize_matched_gr(matched_gr, cfg)
    else:
        values = _finish_feature(_normalize(matched_gr, float(norm[0]), float(norm[1])))
    values[~valid_rows] = 0.0
    channel = np.broadcast_to(
        _finish_feature(values)[:, None],
        static_input[channel_idx[matched_name]].shape,
    ).astype(np.float32)
    if row_mask is None:
        static_input[channel_idx[matched_name]] = channel
    else:
        row_mask = np.asarray(row_mask, dtype=bool)
        static_input[channel_idx[matched_name], row_mask, :] = channel[row_mask]


def _refresh_pf_reliability_channels(static_input, channel_idx, item, cfg, row_mask=None):
    active_names = PF_RELIABILITY_CHANNELS & set(channel_idx)
    if not active_names:
        return
    required = (
        "_pf_prob_ffbsi",
        "_pf_tvt_rel_pred_ffbsi",
        "_pf_source_valid_ffbsi",
        "_pf_tvt_rel_pred",
        "_pf_source_valid",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"PF reliability channels require local PF sidecars: missing {missing}")

    geo_tvt_rel = item["geo_s_rel_prior"].astype(np.float32, copy=False) - item["z_rel"].astype(
        np.float32,
        copy=False,
    )
    tw_tvt_rel = _make_typewell_rel_grid(cfg).astype(np.float32, copy=False)
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
    for name in active_names:
        values = _pf_reliability_values_from_components(
            name,
            item["_pf_prob_ffbsi"],
            item["_pf_tvt_rel_pred_ffbsi"],
            item["_pf_source_valid_ffbsi"],
            tw_tvt_rel,
            cfg,
            geo_tvt_rel=geo_tvt_rel,
            tvt_filtered=item["_pf_tvt_rel_pred"],
            valid_filtered=item["_pf_source_valid"],
        )
        channel = np.broadcast_to(
            _finish_feature(values)[:, None],
            static_input[channel_idx[name]].shape,
        ).astype(np.float32)
        if row_mask is None:
            static_input[channel_idx[name]] = channel
        else:
            static_input[channel_idx[name], row_mask, :] = channel[row_mask]


def _add_pf_feature_sidecars_to_item(item, pf_features, cfg, gr_norm):
    if pf_features is None:
        return item
    static_channels = set(getattr(cfg, "unet_static_channels", ()))
    need_matched_gr = bool(PF_TVT_MATCHED_GR_CHANNELS & static_channels)
    need_reliability = bool(PF_RELIABILITY_CHANNELS & static_channels)
    if not (need_matched_gr or need_reliability):
        return item
    item["_pf_tvt_rel_pred"] = pf_features["pf_tvt_rel_pred"].astype("float32", copy=True)
    item["_pf_source_valid"] = pf_features["source_valid"].astype(bool, copy=True)
    item["_pf_tvt_rel_pred_ffbsi"] = pf_features["pf_tvt_rel_pred_ffbsi"].astype("float32", copy=True)
    item["_pf_source_valid_ffbsi"] = pf_features["source_valid_ffbsi"].astype(bool, copy=True)
    if need_reliability:
        item["_pf_prob_ffbsi"] = pf_features["pf_prob_ffbsi"].astype("float32", copy=True)
    if need_matched_gr:
        norm = gr_norm["typewell_gr"] if gr_norm is not None else cfg.typewell_stats["gr"]
        item["_pf_matched_gr_norm"] = np.asarray(norm, dtype=np.float32)
    return item


def _make_gr_penalty_error(horizontal_map, typewell_grid, cfg, gr_norm=None):
    gr = horizontal_map["gr"].astype("float32", copy=False)
    gr_nan_rate = _finish_feature(horizontal_map["gr_nan_rate"])
    tw_gr_is_nan = typewell_grid["gr_is_nan"].astype(bool)
    tw_gr = typewell_grid["gr"].astype("float32", copy=True)
    tw_gr[tw_gr_is_nan] = np.nan
    diff_std = float(cfg.gr_diff_std if gr_norm is None else gr_norm["gr_abs_diff_std"])
    if diff_std <= 0.0:
        raise ValueError(f"GR penalty diff std must be positive, got {diff_std}")
    penalty = np.square(((tw_gr[None, :] - gr[:, None]) / diff_std).astype("float32"))
    clip = float(getattr(cfg, "GR_penalty_clip", 25.0))
    if clip > 0.0:
        penalty = np.minimum(penalty, np.float32(clip))
    row_valid = np.isfinite(gr) & (gr_nan_rate < 1.0)
    penalty[~row_valid, :] = 0.0
    invalid_value = np.float32(clip if clip > 0.0 else 1.0)
    penalty[row_valid[:, None] & tw_gr_is_nan[None, :]] = invalid_value
    return _finish_feature(penalty)


def _gr_penalty_enabled(cfg):
    return float(getattr(cfg, "unet_loss_weights", {}).get("GR_penalty", 0.0)) > 0.0


def _unet_gr_rmse_enabled(cfg):
    return float(getattr(cfg, "unet_loss_weights", {}).get("unet_GR_RMSE", 0.0)) > 0.0


def make_unet_item(well_data, cfg, training):
    well_data = _mask_out_of_range_horizontal_gr(well_data, cfg)
    if _gr_interpolate_enabled(cfg):
        well_data = _linear_interpolate_gr_after_simulation(well_data)
        if _cfg_mode(cfg, "gr_normalize_std_mode", {"global", "local"}) == "local":
            well_data = dict(well_data)
            well_data["gr_normalize_local_std"] = _prefix_direct_gr_rmse_after_downsample(
                well_data["horizontal"],
                well_data["typewell"],
                cfg,
            )
    elif (
        _out_of_range_gr_mask_enabled(cfg)
        and _cfg_mode(cfg, "gr_normalize_std_mode", {"global", "local"}) == "local"
    ):
        well_data = dict(well_data)
        well_data["gr_normalize_local_std"] = _prefix_direct_gr_rmse_after_downsample(
            well_data["horizontal"],
            well_data["typewell"],
            cfg,
        )
    common = _make_common_item(well_data, cfg, training)
    unet_sources = ["gr", "gr_nan_rate", "seen_tvt_rel"]
    for source_name in (
        "md_rel",
        "x_diff",
        "y_diff",
        "x_twice_diff",
        "y_twice_diff",
        "xy_diff_ac",
        "z_rel",
        "z_diff",
        "gr_std",
        "gr_max",
        "gr_min",
        "gr_first_last_delta",
        "gr_slope",
        "gr_quadratic_a",
        "gr_quadratic_b",
        "gr_quadratic_c",
        "gr_quadratic_rmse",
        "gr_grad_mean",
        "gr_grad_std",
        "gr_diff_mean",
        "gr_diff_std",
    ):
        if source_name in cfg.unet_static_channels:
            unet_sources.append(source_name)
    for source_name in cfg.unet_static_channels:
        if _xy_fourier_source_name(source_name) is not None:
            unet_sources.append(source_name)
    if uses_pf_heatmap_channels(cfg):
        unet_sources.append("orig_index")
    horizontal_sources = tuple(dict.fromkeys(tuple(unet_sources)))
    horizontal_map = _make_horizontal_feature_map(
        common["raw"],
        common["gr_nan"],
        common["raw_has_row"],
        horizontal_sources,
        cfg,
    )
    typewell_grid = _make_typewell_grid_no_extrapolate(
        well_data["typewell"],
        common["meta"]["tvt0"],
        cfg,
    )
    gr_norm = None
    mean_mode = _cfg_mode(cfg, "gr_normalize_mean_mode", {"global", "local"})
    std_mode = _cfg_mode(cfg, "gr_normalize_std_mode", {"global", "local"})
    if mean_mode != "global" or std_mode != "global":
        gr_norm = _make_gr_normalization_context(horizontal_map, common, well_data, cfg)
    pf_features = None
    if uses_pf_heatmap_channels(cfg):
        pf_features = _make_pf_unet_features(
            common,
            horizontal_map,
            well_data.get("pf_cache"),
            cfg,
        )
        _add_analysis_sidecars_to_meta(common, pf_features, cfg)
    unet_static = _make_unet_static_input(
        horizontal_map,
        typewell_grid,
        cfg,
        common=common,
        pf_cache=well_data.get("pf_cache"),
        gr_norm=gr_norm,
        pf_features=pf_features,
    )
    if _gr_penalty_enabled(cfg):
        common["aux_targets"]["GR_penalty_error"] = _make_gr_penalty_error(
            horizontal_map,
            typewell_grid,
            cfg,
            gr_norm=gr_norm,
        )
    if training and _unet_gr_rmse_enabled(cfg):
        gr_rmse_target, gr_rmse_mask = _make_unet_gr_rmse_target(
            common["raw"],
            common["raw_is_suffix"],
            common["gr_nan"],
            well_data["typewell"],
            cfg,
        )
        common["aux_targets"]["unet_GR_RMSE"] = gr_rmse_target
        common["aux_targets"]["unet_GR_RMSE_mask"] = gr_rmse_mask
    typewell_aux = _make_unet_aux(typewell_grid)
    item = _finalize_item(common, unet_static, typewell_aux)
    item = _add_pf_feature_sidecars_to_item(item, pf_features, cfg, gr_norm)
    return item


def _apply_channel_mask_2d(item, cfg):
    aug_cfg = getattr(cfg, "aug_cfg", {}).get("channel_mask_2d", {})
    if random.random() >= float(aug_cfg.get("apply_prob", 0.0)):
        return item

    static_channels = tuple(cfg.unet_static_channels)
    static_input = item["unet_static"]
    if static_input.shape[0] != len(static_channels):
        raise ValueError(
            "channel_mask_2d expects item['unet_static'] channel count to match "
            "cfg.unet_static_channels"
        )

    mask_prob = float(aug_cfg.get("mask_prob", 0.0))
    pf_mask_prob = float(aug_cfg.get("PF_mask_prob", 0.0))
    pf_mask = pf_mask_prob > 0.0 and random.random() < pf_mask_prob
    for channel_idx, channel_name in enumerate(static_channels):
        if channel_name in MATCHED_GR_METRIC_CHANNELS:
            continue
        if channel_name.startswith("pf_"):
            if pf_mask:
                static_input[channel_idx] = 0.0
            continue
        if mask_prob > 0.0 and random.random() < mask_prob:
            static_input[channel_idx] = 0.0
    return item


def _sample_pf_rotate_shift_vec(item, cfg, aug_cfg):
    target_mask = np.asarray(item["target_mask"], dtype=bool)
    suffix_mask = np.asarray(item.get("suffix_mask", target_mask), dtype=bool)
    valid = target_mask & suffix_mask & np.isfinite(item["target"])
    if valid.sum() < 2:
        return None

    target = item["target"].astype(np.float32, copy=False)
    target_rel = _finish_feature(
        target * np.float32(cfg.target_stats[UNET_MODE][1]) + np.float32(cfg.target_stats[UNET_MODE][0])
    )
    valid_idx = np.flatnonzero(valid)
    center_mode = str(aug_cfg.get("center_mode", "first_valid"))
    if center_mode == "zero":
        center = 0.0
    elif center_mode == "first_valid":
        center = float(target_rel[valid_idx[0]])
    else:
        raise ValueError(f"unknown PF_rotate_shift center_mode={center_mode!r}")

    base_valid = target_rel[valid] - np.float32(center)
    denom = float(np.dot(base_valid.astype(np.float64), base_valid.astype(np.float64)))
    min_denom = float(aug_cfg.get("min_fit_denom", 1e-6))
    if denom <= min_denom:
        return None
    y_valid = target_rel[valid].astype(np.float64, copy=False)
    k = float(np.dot(base_valid.astype(np.float64), y_valid) / denom)
    if not np.isfinite(k):
        return None

    k_clip = float(aug_cfg.get("k_clip", 0.0))
    if k_clip > 0.0:
        k = float(np.clip(k, -k_clip, k_clip))
    ratio_jitter = float(aug_cfg.get("ratio_jitter", 0.25))
    if ratio_jitter < 0.0:
        raise ValueError(f"PF_rotate_shift ratio_jitter must be non-negative, got {ratio_jitter}")
    ratio = float(np.random.uniform(1.0 - ratio_jitter, 1.0 + ratio_jitter))
    shift_mode = str(aug_cfg.get("shift_mode", "absolute"))
    base = np.zeros(target_rel.shape[0], dtype=np.float32)
    base[valid] = target_rel[valid] - np.float32(center)
    if shift_mode == "delta":
        shift_vec = np.float32(k * (ratio - 1.0)) * base
    elif shift_mode == "absolute":
        shift_vec = np.float32(k * ratio) * base
    else:
        raise ValueError(f"unknown PF_rotate_shift shift_mode={shift_mode!r}")

    max_abs_shift = float(aug_cfg.get("max_abs_shift", 20.0))
    if max_abs_shift > 0.0:
        shift_vec = np.clip(shift_vec, -max_abs_shift, max_abs_shift).astype(np.float32)
    shift_vec[~valid] = 0.0
    if not np.isfinite(shift_vec).all() or float(np.max(np.abs(shift_vec))) <= 1e-6:
        return None
    return shift_vec.astype(np.float32, copy=False)


def _prob_channel_to_prob(channel, typewell_len):
    prob = np.expm1(np.maximum(channel.astype(np.float32, copy=False), 0.0)) / np.float32(typewell_len)
    return _normalize_prob_rows(prob)


def _row_valid_from_prob_channel(channel, typewell_len):
    prob = _prob_channel_to_prob(channel, typewell_len)
    return prob, prob.sum(axis=1) > 0.0


def _row_valid_from_static_channel(channel):
    return np.any(np.abs(channel.astype(np.float32, copy=False)) > 1e-6, axis=1)


def _odd_positive_window(value, max_len=None):
    window = int(value)
    if max_len is not None:
        window = min(window, int(max_len))
    if window <= 1:
        return 1
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def _box_smooth_1d(values, window):
    values = np.asarray(values, dtype=np.float32)
    window = _odd_positive_window(window, values.shape[0])
    if window <= 1 or values.shape[0] <= 1:
        return values.astype(np.float32, copy=True)
    kernel = np.full(window, 1.0 / float(window), dtype=np.float32)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _box_smooth_1d_cumsum(values, window):
    values = np.asarray(values, dtype=np.float32)
    window = _odd_positive_window(window, values.shape[0])
    if window <= 1 or values.shape[0] <= 1:
        return values.astype(np.float32, copy=True)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    cumsum = np.empty(padded.shape[0] + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(padded, dtype=np.float64, out=cumsum[1:])
    return ((cumsum[window:] - cumsum[:-window]) / float(window)).astype(np.float32)


def _sample_boxcar_normal_curve(length, mean, std, window):
    """Sample a smooth Gaussian curve without changing its row marginals."""
    raw = np.random.normal(0.0, 1.0, size=int(length) + int(window) - 1)
    cumulative = np.empty(raw.shape[0] + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(raw, dtype=np.float64, out=cumulative[1:])
    moving_sum = cumulative[int(window):] - cumulative[:-int(window)]
    standard_normal = moving_sum / np.sqrt(float(window))
    return (float(mean) + float(std) * standard_normal).astype(np.float32)


def _blur_low_variance_gr_region(values, max_ratio):
    values = np.asarray(values, dtype=np.float32)
    out = values.astype(np.float32, copy=True)
    if values.shape[0] <= 2 or max_ratio <= 0.0:
        return out

    region_ratio = float(np.random.uniform(0.0, max_ratio))
    region_len = min(values.shape[0], int(round(region_ratio * values.shape[0])))
    if region_len <= 2:
        return out

    finite = np.isfinite(values)
    if not finite.any():
        return out
    row_idx = np.arange(values.shape[0], dtype=np.float64)
    filled = np.interp(row_idx, row_idx[finite], values[finite]).astype(np.float32)

    value_sum = np.empty(values.shape[0] + 1, dtype=np.float64)
    value_sq_sum = np.empty(values.shape[0] + 1, dtype=np.float64)
    finite_count = np.empty(values.shape[0] + 1, dtype=np.int64)
    value_sum[0] = 0.0
    value_sq_sum[0] = 0.0
    finite_count[0] = 0
    np.cumsum(filled, dtype=np.float64, out=value_sum[1:])
    np.cumsum(filled * filled, dtype=np.float64, out=value_sq_sum[1:])
    np.cumsum(finite, dtype=np.int64, out=finite_count[1:])

    rolling_sum = value_sum[region_len:] - value_sum[:-region_len]
    rolling_sq_sum = value_sq_sum[region_len:] - value_sq_sum[:-region_len]
    rolling_mean = rolling_sum / float(region_len)
    rolling_var = np.maximum(
        rolling_sq_sum / float(region_len) - rolling_mean * rolling_mean,
        0.0,
    )
    rolling_finite_count = finite_count[region_len:] - finite_count[:-region_len]
    rolling_var[rolling_finite_count == 0] = np.inf
    region_start = int(np.argmin(rolling_var))
    region_end = region_start + region_len

    region_filled = filled[region_start:region_end]
    region_finite = finite[region_start:region_end]
    blurred = _box_smooth_1d_cumsum(region_filled, region_len)
    mean_delta = np.mean(values[region_start:region_end][region_finite], dtype=np.float64) - np.mean(
        blurred[region_finite],
        dtype=np.float64,
    )
    blurred += np.float32(mean_delta)
    out_region = out[region_start:region_end]
    out_region[region_finite] = blurred[region_finite]
    return out


def _sample_smooth_noise(length, std, window_choices):
    length = int(length)
    std = float(std)
    if length <= 0 or std <= 0.0:
        return np.zeros(max(length, 0), dtype=np.float32)
    noise = np.random.normal(0.0, 1.0, size=length).astype(np.float32)
    choices = tuple(window_choices)
    window = int(random.choice(choices)) if choices else 1
    noise = _box_smooth_1d(noise, window)
    current_std = float(np.std(noise))
    if current_std > 1e-6:
        noise *= np.float32(std / current_std)
    else:
        noise[:] = 0.0
    return noise.astype(np.float32)


def _sample_range_uniform(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"range config must have length 2, got {value!r}")
        low, high = float(value[0]), float(value[1])
        if low == high:
            return low
        return float(np.random.uniform(low, high))
    return float(value)


def _has_common_mix_augmentation(item):
    meta = item.get("meta", {})
    return bool(meta.get("mixup", False) or meta.get("cutmix", False))


def _apply_pf_rotate_shift(item, cfg):
    aug_cfg = getattr(cfg, "aug_cfg", {}).get("PF_rotate_shift", {})
    if _has_common_mix_augmentation(item):
        return item
    if random.random() >= float(aug_cfg.get("apply_prob", 0.0)):
        return item

    static_channels = tuple(cfg.unet_static_channels)
    active_pf_channels = set(static_channels) & {
        name
        for variant in PF_ROTATE_SHIFT_VARIANTS.values()
        for name in variant.values()
    } | (set(static_channels) & PF_RELIABILITY_CHANNELS)
    if not active_pf_channels:
        return item

    static_input = item["unet_static"]
    if static_input.shape[0] != len(static_channels):
        raise ValueError(
            "PF_rotate_shift expects item['unet_static'] channel count to match "
            "cfg.unet_static_channels"
        )

    shift_vec = _sample_pf_rotate_shift_vec(item, cfg, aug_cfg)
    if shift_vec is None:
        return item

    channel_idx = {name: idx for idx, name in enumerate(static_channels)}
    abs_fill = float(_normalize_tvt_diff(np.asarray([getattr(cfg, "tvt_diff_clip", 100.0)], dtype=np.float32), cfg)[0])
    target_mean, target_std = cfg.target_stats[UNET_MODE]
    typewell_len = int(cfg.typewell_len)
    reliability_active = bool(PF_RELIABILITY_CHANNELS & set(channel_idx))

    for variant in PF_ROTATE_SHIFT_VARIANTS.values():
        prob = None
        valid_rows = None
        prob_name = variant["prob"]
        if prob_name in channel_idx:
            idx = channel_idx[prob_name]
            prob, valid_rows = _row_valid_from_prob_channel(static_input[idx], typewell_len)
            shifted_prob = _shift_prob_axis(prob, shift_vec, cfg)
            valid_rows = valid_rows & (shifted_prob.sum(axis=1) > 0.0)
            shifted_prob[~valid_rows] = 0.0
            static_input[idx] = np.log1p(shifted_prob * np.float32(typewell_len)).astype(np.float32)
            if prob_name == "pf_prob_ffbsi" and "_pf_prob_ffbsi" in item:
                item["_pf_prob_ffbsi"] = shifted_prob.astype(np.float32, copy=True)
                item["_pf_source_valid_ffbsi"] = valid_rows.astype(bool, copy=True)
        elif prob_name == "pf_prob_ffbsi" and reliability_active and "_pf_prob_ffbsi" in item:
            prob = item["_pf_prob_ffbsi"].astype(np.float32, copy=False)
            valid_rows = np.asarray(item["_pf_source_valid_ffbsi"], dtype=bool) & (prob.sum(axis=1) > 0.0)
            shifted_prob = _shift_prob_axis(prob, shift_vec, cfg)
            valid_rows = valid_rows & (shifted_prob.sum(axis=1) > 0.0)
            shifted_prob[~valid_rows] = 0.0
            item["_pf_prob_ffbsi"] = shifted_prob.astype(np.float32, copy=True)
            item["_pf_source_valid_ffbsi"] = valid_rows.astype(bool, copy=True)

        abs_name = variant["abs_diff"]
        if abs_name in channel_idx:
            idx = channel_idx[abs_name]
            abs_valid = valid_rows
            if abs_valid is None:
                abs_valid = _row_valid_from_static_channel(static_input[idx])
            shifted_abs = _shift_axis_image(
                static_input[idx],
                shift_vec,
                cfg,
                fill_value=abs_fill,
            )
            shifted_abs[~abs_valid] = 0.0
            static_input[idx] = shifted_abs.astype(np.float32)

        tvt_name = variant["tvt"]
        if tvt_name in channel_idx:
            idx = channel_idx[tvt_name]
            tvt_valid = valid_rows
            if tvt_valid is None:
                suffix = "_ffbsi" if tvt_name.endswith("_ffbsi") else ""
                sidecar_valid_key = f"_pf_source_valid{suffix}"
                if sidecar_valid_key in item:
                    tvt_valid = np.asarray(item[sidecar_valid_key], dtype=bool).copy()
                else:
                    tvt_valid = _row_valid_from_static_channel(static_input[idx])
            tvt_rel = (
                static_input[idx, :, 0].astype(np.float32, copy=False) * np.float32(target_std)
                + np.float32(target_mean)
            )
            tvt_rel[tvt_valid] += shift_vec[tvt_valid]
            shifted_tvt = _normalize(tvt_rel, target_mean, target_std).astype(np.float32)
            static_input[idx] = np.broadcast_to(shifted_tvt[:, None], static_input[idx].shape).astype(np.float32)
            static_input[idx, ~tvt_valid, :] = 0.0
            suffix = "_ffbsi" if tvt_name.endswith("_ffbsi") else ""
            sidecar_key = f"_pf_tvt_rel_pred{suffix}"
            sidecar_valid_key = f"_pf_source_valid{suffix}"
            if sidecar_key in item:
                sidecar = item[sidecar_key].astype(np.float32, copy=True)
                sidecar[tvt_valid] = tvt_rel[tvt_valid]
                sidecar[~tvt_valid] = 0.0
                item[sidecar_key] = sidecar
            if sidecar_valid_key in item:
                item[sidecar_valid_key] = np.asarray(tvt_valid, dtype=bool).copy()
            _refresh_pf_tvt_matched_gr_channel(
                static_input,
                channel_idx,
                tvt_name,
                tvt_rel,
                tvt_valid,
                item,
                cfg,
            )
        else:
            matched_name = variant["matched_gr"]
            suffix = "_ffbsi" if variant["tvt"].endswith("_ffbsi") else ""
            sidecar_key = f"_pf_tvt_rel_pred{suffix}"
            valid_key = f"_pf_source_valid{suffix}"
            update_path_sidecar = matched_name in channel_idx or (
                reliability_active and sidecar_key in item and valid_key in item
            )
            if update_path_sidecar:
                if sidecar_key not in item or valid_key not in item:
                    raise ValueError(f"{matched_name} requires local PF TVT sidecars for PF_rotate_shift")
                tvt_valid = np.asarray(item[valid_key], dtype=bool).copy()
                if valid_rows is not None:
                    tvt_valid &= valid_rows
                tvt_rel = item[sidecar_key].astype(np.float32, copy=True)
                tvt_rel[tvt_valid] += shift_vec[tvt_valid]
                tvt_rel[~tvt_valid] = 0.0
                item[sidecar_key] = tvt_rel
                item[valid_key] = tvt_valid.astype(bool, copy=True)
                if matched_name in channel_idx:
                    _refresh_pf_tvt_matched_gr_channel(
                        static_input,
                        channel_idx,
                        variant["tvt"],
                        tvt_rel,
                        tvt_valid,
                        item,
                        cfg,
                    )

    if reliability_active:
        _refresh_pf_reliability_channels(static_input, channel_idx, item, cfg)
    return item


def _sample_low_freq_delta(active_idx, cfg_dict, *, prefix, max_abs_key, base_delta=None):
    active_idx = np.asarray(active_idx, dtype=np.int64)
    if active_idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    if base_delta is not None:
        delta = np.asarray(base_delta, dtype=np.float32).copy()
        if delta.shape[0] != active_idx.size:
            raise ValueError(f"{prefix} base_delta length does not match active rows")
    else:
        bias = np.float32(np.random.normal(0.0, float(cfg_dict.get(f"{prefix}_bias_std", 0.0))))
        ramp_end = np.float32(np.random.normal(0.0, float(cfg_dict.get(f"{prefix}_ramp_std", 0.0))))
        if active_idx.size > 1:
            ramp = np.linspace(0.0, float(ramp_end), active_idx.size, dtype=np.float32)
        else:
            ramp = np.zeros(active_idx.size, dtype=np.float32)
        smooth_noise = _sample_smooth_noise(
            active_idx.size,
            float(cfg_dict.get(f"{prefix}_smooth_noise_std", 0.0)),
            cfg_dict.get(f"{prefix}_smooth_window_choices", (1,)),
        )
        delta = (bias + ramp + smooth_noise).astype(np.float32)

    jump_prob = float(cfg_dict.get(f"{prefix}_jump_prob", 0.0))
    jump_std = float(cfg_dict.get(f"{prefix}_jump_std", 0.0))
    if active_idx.size > 1 and jump_prob > 0.0 and jump_std > 0.0 and random.random() < jump_prob:
        jump_pos = random.randint(0, active_idx.size - 1)
        delta[jump_pos:] += np.float32(np.random.normal(0.0, jump_std))

    max_abs = float(cfg_dict.get(max_abs_key, 0.0))
    if max_abs > 0.0:
        delta = np.clip(delta, -max_abs, max_abs).astype(np.float32)
    return delta.astype(np.float32, copy=False)


def _active_suffix_rows(item):
    row_mask = np.asarray(item.get("row_mask", np.ones_like(item["suffix_mask"])), dtype=bool)
    suffix_mask = np.asarray(item.get("suffix_mask", row_mask), dtype=bool)
    return np.flatnonzero(row_mask & suffix_mask)


def _apply_geo_delta_to_channels(item, cfg, channel_idx, active_idx, delta):
    if active_idx.size == 0 or delta.size == 0:
        return False
    static_input = item["unet_static"]
    geo_names = GEO_PRIOR_STATIC_CHANNELS & set(channel_idx)
    if not geo_names and "geo_s_rel_prior" not in item:
        return False

    geo_s_rel = item["geo_s_rel_prior"].astype(np.float32, copy=True)
    geo_s_rel[active_idx] += delta.astype(np.float32, copy=False)
    item["geo_s_rel_prior"] = geo_s_rel.astype(np.float32)
    z_rel = item["z_rel"].astype(np.float32, copy=False)
    suffix_mask = np.asarray(item["suffix_mask"], dtype=bool)
    geo_tvt_rel = (geo_s_rel - z_rel).astype(np.float32)

    if "geo_s_rel" in channel_idx:
        values = _normalize(geo_s_rel, *cfg.surface_stats["S_rel"])
        static_input[channel_idx["geo_s_rel"]] = np.broadcast_to(
            _finish_feature(values)[:, None],
            static_input[channel_idx["geo_s_rel"]].shape,
        ).astype(np.float32)
    if "geo_tvt_rel" in channel_idx:
        values = _normalize(geo_tvt_rel, *cfg.target_stats[UNET_MODE])
        static_input[channel_idx["geo_tvt_rel"]] = np.broadcast_to(
            _finish_feature(values)[:, None],
            static_input[channel_idx["geo_tvt_rel"]].shape,
        ).astype(np.float32)
    if "geo_tvt_abs_diff" in channel_idx:
        tw_tvt_rel = _make_typewell_rel_grid(cfg).astype(np.float32, copy=False)
        channel = _normalize_tvt_diff(np.abs(tw_tvt_rel[None, :] - geo_tvt_rel[:, None]), cfg)
        static_input[channel_idx["geo_tvt_abs_diff"]] = channel.astype(np.float32)
    if "geo_dS" in channel_idx:
        values = _normalize(
            _geo_dS_from_s_rel(geo_s_rel, suffix_mask, cfg),
            *cfg.target_stats["d(TVT+Z)"],
        )
        values[~suffix_mask] = 0.0
        static_input[channel_idx["geo_dS"]] = np.broadcast_to(
            _finish_feature(values)[:, None],
            static_input[channel_idx["geo_dS"]].shape,
        ).astype(np.float32)
    if "geo_tvt_diff" in channel_idx:
        values = _normalize(
            _geo_tvt_diff_from_s_rel(geo_s_rel, item["z_diff"], suffix_mask, cfg),
            *cfg.target_stats["d(TVT+Z)"],
        )
        values[~suffix_mask] = 0.0
        static_input[channel_idx["geo_tvt_diff"]] = np.broadcast_to(
            _finish_feature(values)[:, None],
            static_input[channel_idx["geo_tvt_diff"]].shape,
        ).astype(np.float32)
    if "delta_geo_trend_rbf_5" in channel_idx:
        tw_tvt_rel = _make_typewell_rel_grid(cfg).astype(np.float32, copy=False)
        static_input[channel_idx["delta_geo_trend_rbf_5"]] = _delta_geo_trend_rbf_channel(
            geo_s_rel,
            z_rel,
            suffix_mask,
            tw_tvt_rel,
            sigma=5.0,
        ).astype(np.float32)
    return True


def _apply_geo_prior_corrupt(item, cfg, aug_cfg, channel_idx, active_idx, base_delta=None):
    if not (GEO_PRIOR_STATIC_CHANNELS & set(channel_idx)):
        return None
    delta = _sample_low_freq_delta(
        active_idx,
        aug_cfg,
        prefix="geo",
        max_abs_key="geo_max_abs_shift",
        base_delta=base_delta,
    )
    if delta.size == 0 or float(np.max(np.abs(delta))) <= 1e-6:
        return delta
    _apply_geo_delta_to_channels(item, cfg, channel_idx, active_idx, delta)
    return delta


def _sample_pf_delta(active_idx, aug_cfg, base_delta=None):
    active_idx = np.asarray(active_idx, dtype=np.int64)
    if active_idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    if base_delta is not None:
        delta = np.asarray(base_delta, dtype=np.float32).copy()
        if delta.shape[0] != active_idx.size:
            raise ValueError("PF base_delta length does not match active rows")
    else:
        delta = _sample_smooth_noise(
            active_idx.size,
            float(aug_cfg.get("pf_path_noise_std", 0.0)),
            aug_cfg.get("pf_path_smooth_window_choices", (1,)),
        )

    wrong_prob = float(aug_cfg.get("pf_wrong_mode_prob", 0.0))
    wrong_range = aug_cfg.get("pf_wrong_mode_shift_range", (0.0, 0.0))
    if active_idx.size > 0 and wrong_prob > 0.0 and random.random() < wrong_prob:
        low, high = float(wrong_range[0]), float(wrong_range[1])
        if high > 0.0:
            low = min(low, high)
            sign = -1.0 if random.random() < 0.5 else 1.0
            wrong_shift = np.float32(sign * np.random.uniform(low, high))
            if active_idx.size > 1 and random.random() < 0.5:
                start = random.randint(0, active_idx.size - 1)
                delta[start:] += wrong_shift
            else:
                delta += wrong_shift

    max_abs = float(aug_cfg.get("pf_path_max_abs_shift", 0.0))
    if max_abs > 0.0:
        delta = np.clip(delta, -max_abs, max_abs).astype(np.float32)
    return delta.astype(np.float32, copy=False)


def _temperature_prob_rows(prob, temperature, valid_rows):
    temperature = float(temperature)
    if temperature <= 0.0 or abs(temperature - 1.0) <= 1e-6:
        return _normalize_prob_rows(prob)
    out = prob.astype(np.float32, copy=True)
    active = np.asarray(valid_rows, dtype=bool) & (out.sum(axis=1) > 0.0)
    if active.any():
        exponent = np.float32(1.0 / temperature)
        out[active] = np.power(out[active], exponent).astype(np.float32)
    return _normalize_prob_rows(out)


def _apply_pf_prior_variant_corrupt(item, cfg, aug_cfg, channel_idx, variant, delta_full, active_rows):
    static_input = item["unet_static"]
    typewell_len = int(cfg.typewell_len)
    target_mean, target_std = cfg.target_stats[UNET_MODE]
    prob = None
    valid_rows = None
    reliability_active = bool(PF_RELIABILITY_CHANNELS & set(channel_idx))

    prob_name = variant["prob"]
    if prob_name in channel_idx:
        idx = channel_idx[prob_name]
        prob, valid_rows = _row_valid_from_prob_channel(static_input[idx], typewell_len)
        shifted_prob = _shift_prob_axis(prob, delta_full, cfg)
        valid_rows = valid_rows & (shifted_prob.sum(axis=1) > 0.0)
        temp_low, temp_high = aug_cfg.get("pf_temperature_range", (1.0, 1.0))
        temperature = np.random.uniform(float(temp_low), float(temp_high))
        shifted_prob = _temperature_prob_rows(shifted_prob, temperature, valid_rows & active_rows)
        dropout_prob = float(aug_cfg.get("pf_row_dropout_prob", 0.0))
        if dropout_prob > 0.0 and (valid_rows & active_rows).any():
            drop = (np.random.random(valid_rows.shape[0]) < dropout_prob) & valid_rows & active_rows
            valid_rows[drop] = False
        shifted_prob[~valid_rows] = 0.0
        channel = np.log1p(shifted_prob * np.float32(typewell_len)).astype(np.float32)
        static_input[idx, active_rows, :] = channel[active_rows]
        if prob_name == "pf_prob_ffbsi" and "_pf_prob_ffbsi" in item:
            sidecar_prob = item["_pf_prob_ffbsi"].astype(np.float32, copy=True)
            sidecar_prob[active_rows] = shifted_prob[active_rows]
            item["_pf_prob_ffbsi"] = sidecar_prob
            valid_sidecar = np.asarray(item["_pf_source_valid_ffbsi"], dtype=bool).copy()
            valid_sidecar[active_rows] = valid_rows[active_rows]
            item["_pf_source_valid_ffbsi"] = valid_sidecar
    elif prob_name == "pf_prob_ffbsi" and reliability_active and "_pf_prob_ffbsi" in item:
        prob = item["_pf_prob_ffbsi"].astype(np.float32, copy=False)
        valid_rows = np.asarray(item["_pf_source_valid_ffbsi"], dtype=bool) & (prob.sum(axis=1) > 0.0)
        shifted_prob = _shift_prob_axis(prob, delta_full, cfg)
        valid_rows = valid_rows & (shifted_prob.sum(axis=1) > 0.0)
        temp_low, temp_high = aug_cfg.get("pf_temperature_range", (1.0, 1.0))
        temperature = np.random.uniform(float(temp_low), float(temp_high))
        shifted_prob = _temperature_prob_rows(shifted_prob, temperature, valid_rows & active_rows)
        dropout_prob = float(aug_cfg.get("pf_row_dropout_prob", 0.0))
        if dropout_prob > 0.0 and (valid_rows & active_rows).any():
            drop = (np.random.random(valid_rows.shape[0]) < dropout_prob) & valid_rows & active_rows
            valid_rows[drop] = False
        shifted_prob[~valid_rows] = 0.0
        sidecar_prob = item["_pf_prob_ffbsi"].astype(np.float32, copy=True)
        sidecar_prob[active_rows] = shifted_prob[active_rows]
        item["_pf_prob_ffbsi"] = sidecar_prob
        valid_sidecar = np.asarray(item["_pf_source_valid_ffbsi"], dtype=bool).copy()
        valid_sidecar[active_rows] = valid_rows[active_rows]
        item["_pf_source_valid_ffbsi"] = valid_sidecar

    abs_name = variant["abs_diff"]
    if abs_name in channel_idx:
        idx = channel_idx[abs_name]
        abs_valid = valid_rows
        if abs_valid is None:
            abs_valid = _row_valid_from_static_channel(static_input[idx])
        abs_fill = float(_normalize_tvt_diff(np.asarray([getattr(cfg, "tvt_diff_clip", 100.0)], dtype=np.float32), cfg)[0])
        shifted_abs = _shift_axis_image(static_input[idx], delta_full, cfg, fill_value=abs_fill)
        shifted_abs[~abs_valid] = 0.0
        static_input[idx, active_rows, :] = shifted_abs[active_rows].astype(np.float32)

    tvt_name = variant["tvt"]
    suffix = "_ffbsi" if tvt_name.endswith("_ffbsi") else ""
    sidecar_key = f"_pf_tvt_rel_pred{suffix}"
    valid_key = f"_pf_source_valid{suffix}"
    tvt_rel_for_refresh = None
    tvt_valid = valid_rows
    if tvt_name in channel_idx:
        idx = channel_idx[tvt_name]
        if tvt_valid is None:
            if valid_key in item:
                tvt_valid = np.asarray(item[valid_key], dtype=bool).copy()
            else:
                tvt_valid = _row_valid_from_static_channel(static_input[idx])
        tvt_rel = (
            static_input[idx, :, 0].astype(np.float32, copy=False) * np.float32(target_std)
            + np.float32(target_mean)
        )
        tvt_rel[tvt_valid] += delta_full[tvt_valid]
        shifted_tvt = np.broadcast_to(
            _normalize(tvt_rel, target_mean, target_std)[:, None],
            static_input[idx].shape,
        ).astype(np.float32)
        shifted_tvt[~tvt_valid, :] = 0.0
        static_input[idx, active_rows, :] = shifted_tvt[active_rows]
        tvt_rel_for_refresh = tvt_rel
        if sidecar_key in item:
            sidecar = item[sidecar_key].astype(np.float32, copy=True)
            sidecar[active_rows & tvt_valid] = tvt_rel[active_rows & tvt_valid]
            sidecar[active_rows & ~tvt_valid] = 0.0
            item[sidecar_key] = sidecar
        if valid_key in item:
            valid_sidecar = np.asarray(item[valid_key], dtype=bool).copy()
            valid_sidecar[active_rows] = np.asarray(tvt_valid, dtype=bool)[active_rows]
            item[valid_key] = valid_sidecar
    elif variant["matched_gr"] in channel_idx or (reliability_active and sidecar_key in item and valid_key in item):
        if sidecar_key not in item or valid_key not in item:
            raise ValueError(f"{variant['matched_gr']} requires local PF TVT sidecars for prior_corrupt_2d")
        tvt_valid = np.asarray(item[valid_key], dtype=bool).copy()
        if valid_rows is not None:
            tvt_valid &= valid_rows
            valid_sidecar = np.asarray(item[valid_key], dtype=bool).copy()
            valid_sidecar[active_rows] = tvt_valid[active_rows]
            item[valid_key] = valid_sidecar
        tvt_rel = item[sidecar_key].astype(np.float32, copy=True)
        active_valid = active_rows & tvt_valid
        tvt_rel[active_valid] += delta_full[active_valid]
        tvt_rel[active_rows & ~tvt_valid] = 0.0
        item[sidecar_key] = tvt_rel
        tvt_rel_for_refresh = tvt_rel

    if tvt_rel_for_refresh is not None and tvt_valid is not None:
        _refresh_pf_tvt_matched_gr_channel(
            static_input,
            channel_idx,
            tvt_name,
            tvt_rel_for_refresh,
            np.asarray(tvt_valid, dtype=bool),
            item,
            cfg,
            row_mask=active_rows,
        )
    elif valid_rows is not None:
        matched_name = variant["matched_gr"]
        if matched_name in channel_idx:
            static_input[channel_idx[matched_name], active_rows & ~valid_rows, :] = 0.0
    if reliability_active:
        _refresh_pf_reliability_channels(static_input, channel_idx, item, cfg, row_mask=active_rows)


def _apply_pf_prior_corrupt(item, cfg, aug_cfg, channel_idx, active_idx, base_delta=None):
    active_pf_channels = set(channel_idx) & {
        name
        for variant in PF_ROTATE_SHIFT_VARIANTS.values()
        for name in variant.values()
    } | (set(channel_idx) & PF_RELIABILITY_CHANNELS)
    if not active_pf_channels:
        return None
    delta = _sample_pf_delta(active_idx, aug_cfg, base_delta=base_delta)
    if delta.size == 0 or float(np.max(np.abs(delta))) <= 1e-6:
        return delta
    delta_full = np.zeros(item["unet_static"].shape[1], dtype=np.float32)
    delta_full[active_idx] = delta
    active_rows = np.zeros_like(delta_full, dtype=bool)
    active_rows[active_idx] = True
    reliability_active = bool(PF_RELIABILITY_CHANNELS & set(channel_idx))
    for variant in PF_ROTATE_SHIFT_VARIANTS.values():
        if reliability_active or set(variant.values()) & set(channel_idx):
            _apply_pf_prior_variant_corrupt(item, cfg, aug_cfg, channel_idx, variant, delta_full, active_rows)
    return delta


def _apply_prior_corrupt_2d(item, cfg):
    aug_cfg = getattr(cfg, "aug_cfg", {}).get("prior_corrupt_2d", {})
    if _has_common_mix_augmentation(item):
        return item
    if random.random() >= float(aug_cfg.get("apply_prob", 0.0)):
        return item

    static_channels = tuple(cfg.unet_static_channels)
    static_input = item["unet_static"]
    if static_input.shape[0] != len(static_channels):
        raise ValueError(
            "prior_corrupt_2d expects item['unet_static'] channel count to match "
            "cfg.unet_static_channels"
        )

    active_idx = _active_suffix_rows(item)
    if active_idx.size == 0:
        return item

    channel_idx = {name: idx for idx, name in enumerate(static_channels)}
    geo_apply = random.random() < float(aug_cfg.get("geo_apply_prob", 0.0))
    pf_apply = random.random() < float(aug_cfg.get("pf_apply_prob", 0.0))
    disagreement = (
        geo_apply
        and pf_apply
        and float(aug_cfg.get("geo_pf_disagreement_prob", 0.0)) > 0.0
        and random.random() < float(aug_cfg.get("geo_pf_disagreement_prob", 0.0))
    )

    geo_delta = None
    if geo_apply:
        geo_delta = _apply_geo_prior_corrupt(item, cfg, aug_cfg, channel_idx, active_idx)
    if pf_apply:
        pf_base_delta = None
        if disagreement and geo_delta is not None:
            pf_base_delta = (-0.5 * geo_delta).astype(np.float32)
        _apply_pf_prior_corrupt(item, cfg, aug_cfg, channel_idx, active_idx, base_delta=pf_base_delta)
    return item


def _apply_seq_mask(item, cfg):
    aug_cfg = getattr(cfg, "aug_cfg", {}).get("seq_mask", {})
    if random.random() >= float(aug_cfg.get("apply_prob", 0.0)):
        return item

    max_ratio = float(aug_cfg.get("mask_ratio", 0.3))
    if max_ratio <= 0.0:
        return item
    suffix_idx = np.flatnonzero(item["suffix_mask"])
    if suffix_idx.size == 0:
        return item

    mask_ratio = float(np.random.uniform(0.0, max_ratio))
    mask_len = min(suffix_idx.size, max(1, int(np.ceil(mask_ratio * suffix_idx.size))))
    start_offset = random.randint(0, suffix_idx.size - mask_len)
    start = int(suffix_idx[start_offset])
    end = int(suffix_idx[start_offset + mask_len - 1]) + 1

    item["geo_s_rel_prior"][start:end] = 0.0
    keep_channels = np.array(
        [name in MATCHED_GR_METRIC_CHANNELS for name in cfg.unet_static_channels],
        dtype=bool,
    )
    if keep_channels.any():
        item["unet_static"][~keep_channels, start:end, :] = 0.0
    else:
        item["unet_static"][:, start:end, :] = 0.0
    return item


def _apply_hflip(item, cfg):
    aug_cfg = getattr(cfg, "aug_cfg", {}).get("hflip", {})
    if random.random() >= float(aug_cfg.get("apply_prob", 0.0)):
        return item

    horizontal_len = item["unet_static"].shape[1]
    item["unet_static"] = item["unet_static"][:, ::-1, :].copy()

    for key in (
        "geo_s_rel_prior",
        "z_rel",
        "bin_count",
        "z_diff",
        "target",
        "target_mask",
        "row_mask",
        "suffix_mask",
    ):
        item[key] = item[key][::-1].copy()

    item["typewell_target_probs"] = item["typewell_target_probs"][::-1, :].copy()

    for key, value in list(item["aux_targets"].items()):
        if value.ndim == 1 and value.shape[0] == horizontal_len:
            item["aux_targets"][key] = value[::-1].copy()
        elif value.ndim == 2 and value.shape[0] == horizontal_len:
            item["aux_targets"][key] = value[::-1, :].copy()

    for key in (
        "_pf_tvt_rel_pred",
        "_pf_source_valid",
        "_pf_tvt_rel_pred_ffbsi",
        "_pf_source_valid_ffbsi",
    ):
        if key in item and item[key].ndim == 1 and item[key].shape[0] == horizontal_len:
            item[key] = item[key][::-1].copy()

    item["meta"]["hflip"] = True
    return item


def apply_unet_item_augmentations(item, cfg):
    item = _apply_pf_rotate_shift(item, cfg)
    item = _apply_prior_corrupt_2d(item, cfg)
    item = _apply_channel_mask_2d(item, cfg)
    item = _apply_seq_mask(item, cfg)
    item = _apply_hflip(item, cfg)
    return item


def _mix_item_rows(primary, secondary, row_idx, secondary_row_idx, lam):
    inv_lam = np.float32(1.0 - lam)
    lam = np.float32(lam)

    primary["unet_static"][:, row_idx, :] = (
        lam * primary["unet_static"][:, row_idx, :]
        + inv_lam * secondary["unet_static"][:, secondary_row_idx, :]
    ).astype("float32")
    for key in ("geo_s_rel_prior", "z_rel", "bin_count", "z_diff", "target"):
        primary[key][row_idx] = (
            lam * primary[key][row_idx]
            + inv_lam * secondary[key][secondary_row_idx]
        ).astype("float32")

    primary["typewell_target_probs"][row_idx, :] = (
        lam * primary["typewell_target_probs"][row_idx, :]
        + inv_lam * secondary["typewell_target_probs"][secondary_row_idx, :]
    ).astype("float32")

    for key in ("target_mask", "row_mask", "suffix_mask"):
        primary[key][row_idx] = primary[key][row_idx] & secondary[key][secondary_row_idx]

    for key, value in primary["aux_targets"].items():
        secondary_value = secondary["aux_targets"][key]
        if key.endswith("_mask"):
            value[row_idx] = np.minimum(value[row_idx], secondary_value[secondary_row_idx]).astype("float32")
        elif value.ndim == 1:
            value[row_idx] = (lam * value[row_idx] + inv_lam * secondary_value[secondary_row_idx]).astype("float32")
        elif value.ndim == 2:
            value[row_idx, :] = (
                lam * value[row_idx, :]
                + inv_lam * secondary_value[secondary_row_idx, :]
            ).astype("float32")
        else:
            raise ValueError(f"mixup_2d does not support aux target {key!r} with ndim={value.ndim}")


def _copy_item_rows(primary, secondary, row_idx, secondary_row_idx):
    primary["unet_static"][:, row_idx, :] = secondary["unet_static"][:, secondary_row_idx, :].astype("float32")
    for key in ("geo_s_rel_prior", "z_rel", "bin_count", "z_diff", "target"):
        primary[key][row_idx] = secondary[key][secondary_row_idx].astype("float32")

    primary["typewell_target_probs"][row_idx, :] = secondary["typewell_target_probs"][
        secondary_row_idx,
        :,
    ].astype("float32")

    for key in ("target_mask", "row_mask", "suffix_mask"):
        primary[key][row_idx] = secondary[key][secondary_row_idx]

    for key, value in primary["aux_targets"].items():
        secondary_value = secondary["aux_targets"][key]
        if value.ndim == 1:
            value[row_idx] = secondary_value[secondary_row_idx].astype(value.dtype, copy=False)
        elif value.ndim == 2:
            value[row_idx, :] = secondary_value[secondary_row_idx, :].astype(value.dtype, copy=False)
        else:
            raise ValueError(f"cutmix_2d does not support aux target {key!r} with ndim={value.ndim}")

    for key in (
        "_pf_tvt_rel_pred",
        "_pf_source_valid",
        "_pf_tvt_rel_pred_ffbsi",
        "_pf_source_valid_ffbsi",
        "_pf_prob_ffbsi",
    ):
        if key in primary and key in secondary:
            if primary[key].ndim == 1:
                primary[key][row_idx] = secondary[key][secondary_row_idx]
            elif primary[key].ndim == 2:
                primary[key][row_idx, :] = secondary[key][secondary_row_idx, :]


def _common_valid_suffix_rows(item):
    valid = np.asarray(item["target_mask"], dtype=bool) & np.asarray(item["suffix_mask"], dtype=bool)
    return np.flatnonzero(valid)


def apply_unet_common_mixup(item, secondary_item, cfg):
    alpha = float(getattr(cfg, "mixup_alpha", 1.0))
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError(f"mixup_alpha must be positive and finite, got {alpha}")
    if sorted(item["aux_targets"]) != sorted(secondary_item["aux_targets"]):
        raise ValueError("mixup_2d expects both items to have the same aux target keys")

    rows = _common_valid_suffix_rows(item)
    secondary_rows = _common_valid_suffix_rows(secondary_item)
    common_len = min(rows.size, secondary_rows.size)
    if common_len <= 0:
        return item

    rows = rows[:common_len]
    secondary_rows = secondary_rows[:common_len]
    lam = float(np.random.beta(alpha, alpha))
    _mix_item_rows(item, secondary_item, rows, secondary_rows, lam)
    item["meta"]["mixup"] = True
    item["meta"]["mixup_lambda"] = np.float32(lam)
    item["meta"]["mixup_well_id"] = secondary_item["meta"].get("well_id")
    return item


def apply_unet_common_cutmix(item, secondary_item, cfg):
    alpha = float(getattr(cfg, "cutmix_alpha", 1.0))
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError(f"cutmix_alpha must be positive and finite, got {alpha}")
    if sorted(item["aux_targets"]) != sorted(secondary_item["aux_targets"]):
        raise ValueError("cutmix_2d expects both items to have the same aux target keys")

    rows = _common_valid_suffix_rows(item)
    secondary_rows = _common_valid_suffix_rows(secondary_item)
    common_len = min(rows.size, secondary_rows.size)
    if common_len <= 0:
        return item

    sampled_lam = float(np.random.beta(alpha, alpha))
    span_len = int(round((1.0 - sampled_lam) * common_len))
    span_len = max(1, min(common_len, span_len))
    start_offset = random.randint(0, common_len - span_len)
    row_slice = slice(start_offset, start_offset + span_len)
    cut_rows = rows[:common_len][row_slice]
    secondary_cut_rows = secondary_rows[:common_len][row_slice]

    _copy_item_rows(item, secondary_item, cut_rows, secondary_cut_rows)
    actual_lam = 1.0 - (float(span_len) / float(common_len))
    item["meta"]["cutmix"] = True
    item["meta"]["cutmix_lambda"] = np.float32(actual_lam)
    item["meta"]["cutmix_sampled_lambda"] = np.float32(sampled_lam)
    item["meta"]["cutmix_start_offset"] = int(start_offset)
    item["meta"]["cutmix_span_len"] = int(span_len)
    item["meta"]["cutmix_well_id"] = secondary_item["meta"].get("well_id")
    return item


def resize_vec(x, target_len):
    old_idx = np.linspace(0, 1, len(x))
    new_idx = np.linspace(0, 1, target_len)
    return np.interp(new_idx, old_idx, x)

class TrendSampler:
    """
    Sample suffix TVT trends for z_shift.

    The default model keeps the historical behavior of sampling true train
    suffix TVT paths. ``target_with_rules`` uses the same target-derived pool
    but applies clip/reverse/stretch/scale transforms after sampling and before
    resizing. ``diff_block_bootstrap`` resamples empirical local derivatives
    from the same pool and integrates them into a fresh anchor-zero trend.
    ``PF_sample`` consumes cached PF suffix paths generated without target TVT.
    ``mixed`` appends both sources into one pool.
    """

    def __init__(self, well_data_seq, z_shift_cfg=None):
        z_shift_cfg = {} if z_shift_cfg is None else z_shift_cfg
        self.sample_model = str(z_shift_cfg.get("sample_model", z_shift_cfg.get("trend_sample_model", "target")))
        self.sample_model_key = self.sample_model.lower()
        self.sample_from_same_well = bool(z_shift_cfg.get("sample_from_same_well", False))
        self.rule_apply_prob = float(z_shift_cfg.get("rule_apply_prob", 1.0))
        self.rule_reverse_prob = float(z_shift_cfg.get("rule_reverse_prob", 0.5))
        self.rule_clip_prob = float(z_shift_cfg.get("rule_clip_prob", z_shift_cfg.get("rule_crop_prob", 1.0)))
        self.rule_clip_min_len = int(z_shift_cfg.get("rule_clip_min_len", z_shift_cfg.get("rule_crop_min_len", 4000)))
        self.rule_stretch_range = z_shift_cfg.get("rule_stretch_range", (1.0, 1.0))
        self.rule_scale_range = z_shift_cfg.get("rule_scale_range", (1.0, 1.0))
        self.pf_sg_window = int(z_shift_cfg.get("pf_sample_sg_window", 127))
        self.pf_sg_polyorder = int(z_shift_cfg.get("pf_sample_sg_polyorder", 2))
        self.diff_block_lengths = tuple(int(v) for v in z_shift_cfg.get("diff_block_lengths", (128, 256, 512)))
        self.diff_block_weights = self._normalize_weights(
            z_shift_cfg.get("diff_block_weights", (0.25, 0.50, 0.25)),
            len(self.diff_block_lengths),
            "diff_block_weights",
        )
        self.diff_block_join_width = int(z_shift_cfg.get("diff_block_join_width", 16))
        self.diff_block_reverse_prob = float(z_shift_cfg.get("diff_block_reverse_prob", 0.20))
        self.diff_block_candidate_count = int(z_shift_cfg.get("diff_block_candidate_count", 1))
        self.TVT_rel_pool = []
        self.TVT_rel_pool_by_well = {}

        for well_data in well_data_seq:
            if self.sample_model_key in {
                "target",
                "truth",
                "real",
                "tvt",
                "target_with_rules",
                "diff_block_bootstrap",
                "block_bootstrap",
                "mixed",
            }:
                self._load_target_trend(well_data)
            if self.sample_model_key in {"pf_sample", "mixed"}:
                self._load_pf_sample_trends(well_data)
            if self.sample_model_key not in {
                "target",
                "truth",
                "real",
                "tvt",
                "target_with_rules",
                "diff_block_bootstrap",
                "block_bootstrap",
                "pf_sample",
                "mixed",
            }:
                raise ValueError(
                    f"unknown z_shift sample_model={self.sample_model!r}; "
                    "expected 'target', 'target_with_rules', 'diff_block_bootstrap', 'PF_sample', or 'mixed'"
                )
        self.pool_size = len(self.TVT_rel_pool)
        if self.pool_size == 0:
            raise ValueError(f"TrendSampler pool is empty for sample_model={self.sample_model!r}")
        return

    @staticmethod
    def _normalize_weights(weights, expected_len, context):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != expected_len:
            raise ValueError(f"{context} must contain {expected_len} values, got {weights.shape[0]}")
        if not np.isfinite(weights).all():
            raise ValueError(f"{context} must be finite")
        if (weights < 0.0).any():
            raise ValueError(f"{context} must be non-negative")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(f"{context} must give positive probability to at least one value")
        return (weights / total).astype("float64")

    def _smooth_pf_path(self, tvt_path):
        window = self.pf_sg_window
        polyorder = self.pf_sg_polyorder
        if window <= 1:
            return tvt_path
        window = min(int(window), int(tvt_path.shape[0]))
        if window % 2 == 0:
            window -= 1
        if window <= polyorder:
            return tvt_path
        return savgol_filter(
            tvt_path.astype(np.float64, copy=False),
            window_length=window,
            polyorder=polyorder,
            mode="interp",
        ).astype("float32")

    def _append_tvt_path(self, well_id, tvt_path, source):
        tvt_path = np.asarray(tvt_path, dtype=np.float32)
        finite = np.isfinite(tvt_path)
        if finite.sum() < 2:
            return
        if not finite.all():
            x = np.arange(tvt_path.shape[0], dtype=np.float32)
            tvt_path = np.interp(x, x[finite], tvt_path[finite]).astype("float32")
        if source == "pf_sample":
            tvt_path = self._smooth_pf_path(tvt_path)
        tvt_diff = np.diff(tvt_path)
        tvt_diff = tvt_diff[np.isfinite(tvt_diff) & (np.abs(tvt_diff) < 0.15)]
        if tvt_diff.size == 0:
            return
        tvt_rel = np.cumsum(tvt_diff).astype("float32")
        entry = {
            "well_id": well_id,
            "source": source,
            "diff": tvt_diff.astype("float32", copy=False),
            "tvt_rel": tvt_rel,
        }
        self.TVT_rel_pool.append(entry)
        self.TVT_rel_pool_by_well.setdefault(well_id, []).append(entry)

    def _load_target_trend(self, well_data):
        well_id = well_data["well_id"]
        TVT_input = well_data["horizontal"]["TVT_input"]
        TVT = well_data["horizontal"]["TVT"]
        test_cond = np.isnan(TVT_input)
        if test_cond.sum() < 1000:
            return
        self._append_tvt_path(well_id, TVT[test_cond].copy(), source="target")

    def _load_pf_sample_trends(self, well_data):
        well_id = well_data["well_id"]
        pf_sample_cache = well_data.get("pf_sample_cache")
        if pf_sample_cache is None:
            raise ValueError(f"{well_id}: PF_sample trend requested but no PF sample cache was loaded")
        for tvt_path in pf_sample_cache["pf_tvt_samples"]:
            self._append_tvt_path(well_id, tvt_path, source="pf_sample")

    def _sample_range(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError(f"range config must have length 2, got {value!r}")
            low, high = float(value[0]), float(value[1])
            if low == high:
                return low
            return float(np.random.uniform(low, high))
        return float(value)

    def _random_clip_long_vec(self, vec):
        min_len = self.rule_clip_min_len
        if min_len <= 0 or len(vec) <= min_len:
            return vec
        if random.random() >= self.rule_clip_prob:
            return vec
        max_start = len(vec) - min_len - 1
        if max_start <= 0:
            return vec
        start = random.randint(0, max_start)
        vec = vec[start:].astype("float32", copy=True)
        return (vec - vec[0]).astype("float32")

    def _apply_rule_transform(self, vec):
        if random.random() >= self.rule_apply_prob:
            return vec.astype("float32", copy=True)

        vec = self._random_clip_long_vec(vec)
        if len(vec) > 1 and random.random() < self.rule_reverse_prob:
            vec = vec[::-1].astype("float32", copy=True)
            vec = (vec - vec[0]).astype("float32")
        else:
            vec = vec.astype("float32", copy=True)

        stretch = self._sample_range(self.rule_stretch_range)
        if stretch > 0.0 and stretch != 1.0 and len(vec) > 1:
            stretch_len = max(2, int(round(len(vec) * stretch)))
            vec = resize_vec(vec, stretch_len).astype("float32")
            vec = (vec - vec[0]).astype("float32")

        scale = self._sample_range(self.rule_scale_range)
        if scale != 1.0:
            vec = (vec * np.float32(scale)).astype("float32")
        return vec

    @staticmethod
    def _zero_anchor_path(vec):
        vec = np.asarray(vec, dtype=np.float32)
        out = np.empty(vec.shape[0] + 1, dtype=np.float32)
        out[0] = 0.0
        out[1:] = vec
        return out

    @staticmethod
    def _resize_or_crop_zero_path(path, L):
        path = np.asarray(path, dtype=np.float32)
        L = int(L)
        if path.shape[0] > L:
            start = random.randint(0, path.shape[0] - L)
            path = path[start:start + L].astype("float32", copy=True)
        elif path.shape[0] < L:
            path = resize_vec(path, L).astype("float32")
        else:
            path = path.astype("float32", copy=True)
        return (path - path[0]).astype("float32")

    @staticmethod
    def _path_stats(path):
        path = np.asarray(path, dtype=np.float32)
        d = np.diff(path)
        dd = np.diff(d)
        nonzero = d[np.abs(d) > 1e-6]
        if nonzero.size >= 2:
            sign_change = float(np.mean(np.sign(nonzero[1:]) != np.sign(nonzero[:-1])))
        else:
            sign_change = 0.0
        return {
            "max_abs": float(np.max(np.abs(path))) if path.size else 0.0,
            "std": float(np.std(path)) if path.size else 0.0,
            "d_std": float(np.std(d)) if d.size else 0.0,
            "dd_std": float(np.std(dd)) if dd.size else 0.0,
            "sign_change": sign_change,
        }

    @staticmethod
    def _stat_ratio_loss(value, target, eps=1e-4):
        return abs(np.log((float(value) + eps) / (float(target) + eps)))

    def _sample_target_stat_path(self, pool, L, idx=None):
        if idx is None:
            entry = pool[random.randint(0, len(pool) - 1)]
        else:
            entry = pool[int(idx) % len(pool)]
        vec = self._apply_rule_transform(entry["tvt_rel"])
        path = self._zero_anchor_path(vec)
        return self._resize_or_crop_zero_path(path, L)

    def _sample_diff_block_length(self, remaining):
        block_len = int(np.random.choice(self.diff_block_lengths, p=self.diff_block_weights))
        return max(1, min(block_len, int(remaining)))

    def _sample_diff_block_sequence(self, pool, length):
        length = int(length)
        if length <= 0:
            return np.empty(0, dtype=np.float32)
        out = np.empty(length, dtype=np.float32)
        pos = 0
        while pos < length:
            requested_len = self._sample_diff_block_length(length - pos)
            entry = pool[random.randint(0, len(pool) - 1)]
            diff = entry["diff"]
            use_len = min(requested_len, int(diff.shape[0]), length - pos)
            if use_len <= 0:
                continue
            if diff.shape[0] > use_len:
                start = random.randint(0, diff.shape[0] - use_len)
            else:
                start = 0
            block = diff[start:start + use_len].astype("float32", copy=True)
            if block.shape[0] > 1 and random.random() < self.diff_block_reverse_prob:
                block = (-block[::-1]).astype("float32", copy=True)
            if pos > 0 and self.diff_block_join_width > 0:
                blend_len = min(int(self.diff_block_join_width), int(block.shape[0]))
                if blend_len > 0:
                    weights = (np.arange(blend_len, dtype=np.float32) + 1.0) / np.float32(blend_len + 1)
                    block[:blend_len] = (1.0 - weights) * out[pos - 1] + weights * block[:blend_len]
            out[pos:pos + block.shape[0]] = block
            pos += block.shape[0]
        return out

    def _score_bootstrap_path(self, path, target_stats):
        stats = self._path_stats(path)
        return (
            self._stat_ratio_loss(stats["max_abs"], target_stats["max_abs"])
            + 0.8 * self._stat_ratio_loss(stats["std"], target_stats["std"])
            + 0.6 * self._stat_ratio_loss(stats["d_std"], target_stats["d_std"])
            + 0.4 * self._stat_ratio_loss(stats["dd_std"], target_stats["dd_std"])
            + 0.25 * abs(stats["sign_change"] - target_stats["sign_change"])
        )

    def _sample_diff_block_bootstrap(self, L, pool, idx=None):
        target_stats = None
        if self.diff_block_candidate_count > 1:
            target_path = self._sample_target_stat_path(pool, L, idx=idx)
            target_stats = self._path_stats(target_path)
        candidate_count = max(1, self.diff_block_candidate_count)
        best_path = None
        best_score = np.inf
        for _ in range(candidate_count):
            diff = self._sample_diff_block_sequence(pool, L - 1)
            raw_path = np.empty(L, dtype=np.float32)
            raw_path[0] = 0.0
            raw_path[1:] = np.cumsum(diff).astype("float32")
            path = (raw_path - raw_path[0]).astype("float32")
            if target_stats is None:
                return path
            score = self._score_bootstrap_path(path, target_stats)
            if score < best_score:
                best_score = score
                best_path = path
        return best_path.astype("float32")

    def get(self, L, idx=None, well_id=None):
        L = int(L)
        if L <= 0:
            return np.empty(0, dtype=np.float32)
        pool = self.TVT_rel_pool
        if self.sample_from_same_well:
            pool = self.TVT_rel_pool_by_well.get(well_id, [])
            if len(pool) == 0:
                raise ValueError(f"TrendSampler same-well pool is empty for well_id={well_id!r}")
        if len(pool) == 0:
            raise ValueError(f"TrendSampler pool is empty for well_id={well_id!r}")
        if idx is None:
            idx = random.randint(0, len(pool) - 1)
        entry = pool[idx]
        vec = entry["tvt_rel"]
        if self.sample_model_key in {"diff_block_bootstrap", "block_bootstrap"}:
            return self._sample_diff_block_bootstrap(L, pool, idx=idx)
        if self.sample_model_key == "target_with_rules":
            vec = self._apply_rule_transform(vec)
        # Compressing a long sequence can create large dTVT; slice when possible.
        if len(vec) >= L:
            return vec[:L].astype("float32", copy=True)
        return resize_vec(vec, L).astype("float32")

def piecewise_ols_2var(
    X,
    Y,
    Z,
    thr,
    min_seg_len=20,
    fit_intercept=True,
    ridge=1e-6,
    a_bound=None,
    b_bound=None,
):
    """
    Piecewise regression:

        Z = a*X + b*Y + c

    Split points occur where:

        abs(second_derivative(Z)) > thr

    Parameters
    ----------
    X,Y,Z : 1D arrays
    thr : float
        Curvature threshold.
    min_seg_len : int
        Merge nearby splits.
    fit_intercept : bool
    ridge : float

    Returns
    -------
    a_full,b_full,c_full
        Same length as input.
    """

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)

    n = len(Z)

    # ------------------------
    # curvature
    # ------------------------

    d2 = np.abs(np.gradient(np.gradient(Z)))

    split_idx = np.where(d2 > thr)[0]

    # ------------------------
    # enforce min segment length
    # ------------------------

    splits = []
    last = -10**9

    for idx in split_idx:
        if idx - last >= min_seg_len:
            splits.append(idx)
            last = idx

    bounds = [0] + splits + [n]

    segment_bounds = []
    segment_coefs = []

    # ------------------------
    # fit each segment
    # ------------------------

    for start, end in zip(bounds[:-1], bounds[1:]):
        segment_bounds.append((start, end))

        if end - start < 3:
            segment_coefs.append((np.nan, np.nan, np.nan))
            continue

        xs = X[start:end]
        ys = Y[start:end]
        zs = Z[start:end]
        x_center = float(np.mean(xs))
        y_center = float(np.mean(ys))
        xs_centered = xs - x_center
        ys_centered = ys - y_center

        if fit_intercept:

            A = np.column_stack([
                xs_centered,
                ys_centered,
                np.ones_like(xs)
            ])

        else:

            A = np.column_stack([
                xs_centered,
                ys_centered
            ])

        ATA = A.T @ A
        ATA += ridge * np.eye(ATA.shape[0])

        coef = np.linalg.pinv(ATA)@(A.T @ zs)

        if fit_intercept:
            a, b, c_centered = coef
        else:
            a, b = coef
            c_centered = 0.0

        if a_bound is not None:
            a = np.clip(a, -float(a_bound), float(a_bound))
        if b_bound is not None:
            b = np.clip(b, -float(b_bound), float(b_bound))
        c = c_centered - a * x_center - b * y_center

        segment_coefs.append((a, b, c))

    segment_centers = np.asarray([(start + end - 1) * 0.5 for start, end in segment_bounds], dtype=np.float64)
    segment_coefs = np.asarray(segment_coefs, dtype=np.float64)
    valid_segments = np.isfinite(segment_coefs).all(axis=1)
    if not valid_segments.any():
        return np.zeros(n), np.zeros(n), np.zeros(n)

    filled_coefs = segment_coefs.copy()
    for coef_idx in range(3):
        filled_coefs[:, coef_idx] = np.interp(
            segment_centers,
            segment_centers[valid_segments],
            segment_coefs[valid_segments, coef_idx],
        )

    a_full = np.zeros(n)
    b_full = np.zeros(n)
    c_full = np.zeros(n)
    for (start, end), (a, b, c) in zip(segment_bounds, filled_coefs):
        a_full[start:end] = a
        b_full[start:end] = b
        c_full[start:end] = c

    return a_full, b_full, c_full


def piecewise_delta_tangent_gradient_2var(
    X,
    Y,
    Z,
    thr,
    min_seg_len=20,
    ridge=1e-6,
    a_bound=None,
    b_bound=None,
):
    """
    Estimate a local horizontal gradient from row differences:

        dZ = a*dX + b*dY

    Horizontal wells mostly trace a 1D path through XY, so the transverse
    gradient is weakly identified. Each segment is therefore fit only along
    the dominant local tangent direction, yielding the minimum-risk gradient
    for small XY translations.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    n = len(Z)

    d2 = np.abs(np.gradient(np.gradient(Z)))
    split_idx = np.where(d2 > thr)[0]
    splits = []
    last = -10**9
    for idx in split_idx:
        if idx - last >= min_seg_len:
            splits.append(idx)
            last = idx

    bounds = [0] + splits + [n]
    segment_bounds = []
    segment_coefs = []
    segment_centers = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        segment_bounds.append((start, end))
        segment_centers.append((start + end - 1) * 0.5)

        if end - start < 3:
            segment_coefs.append((np.nan, np.nan))
            continue

        dx = np.diff(X[start:end])
        dy = np.diff(Y[start:end])
        dz = np.diff(Z[start:end])
        valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz) & ((dx * dx + dy * dy) > 1e-8)
        if valid.sum() < 2:
            segment_coefs.append((np.nan, np.nan))
            continue

        steps = np.column_stack([dx[valid], dy[valid]])
        gram = steps.T @ steps
        eigvals, eigvecs = np.linalg.eigh(gram)
        tangent = eigvecs[:, int(np.argmax(eigvals))]
        if float(tangent @ steps.mean(axis=0)) < 0.0:
            tangent = -tangent

        step_along_tangent = steps @ tangent
        denom = float(step_along_tangent @ step_along_tangent + ridge)
        if denom <= 0.0:
            segment_coefs.append((np.nan, np.nan))
            continue
        slope = float((step_along_tangent @ dz[valid]) / denom)
        a, b = slope * tangent
        if a_bound is not None:
            a = np.clip(a, -float(a_bound), float(a_bound))
        if b_bound is not None:
            b = np.clip(b, -float(b_bound), float(b_bound))
        segment_coefs.append((a, b))

    segment_centers = np.asarray(segment_centers, dtype=np.float64)
    segment_coefs = np.asarray(segment_coefs, dtype=np.float64)
    valid_segments = np.isfinite(segment_coefs).all(axis=1)
    if not valid_segments.any():
        return np.zeros(n), np.zeros(n)

    filled_coefs = segment_coefs.copy()
    for coef_idx in range(2):
        filled_coefs[:, coef_idx] = np.interp(
            segment_centers,
            segment_centers[valid_segments],
            segment_coefs[valid_segments, coef_idx],
        )

    a_full = np.zeros(n)
    b_full = np.zeros(n)
    for (start, end), (a, b) in zip(segment_bounds, filled_coefs):
        a_full[start:end] = a
        b_full[start:end] = b

    return a_full, b_full


def _fill_nan_linear(values):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if finite.all():
        return values.copy()
    if not finite.any():
        raise ValueError("cannot fill a residual vector with no finite values")
    idx = np.arange(values.shape[0], dtype=np.float32)
    return np.interp(idx, idx[finite], values[finite]).astype("float32")


def _finite_mean_std(values):
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("cannot compute residual stats with no finite values")
    return float(np.mean(finite_values)), float(np.std(finite_values))


def _sample_range(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"range config must have length 2, got {value!r}")
        return float(np.random.uniform(float(value[0]), float(value[1])))
    return float(value)


def _sample_weighted_range(ranges, weights, *, context):
    if len(ranges) == 0:
        raise ValueError(f"{context} ranges must not be empty")
    if len(ranges) != len(weights):
        raise ValueError(f"{context} ranges and weights must have the same length")

    weights_arr = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_arr).all():
        raise ValueError(f"{context} weights must be finite")
    if (weights_arr < 0.0).any():
        raise ValueError(f"{context} weights must be non-negative")
    total = float(weights_arr.sum())
    if total <= 0.0:
        raise ValueError(f"{context} weights must give positive probability to at least one range")
    weights_arr = weights_arr / total

    idx = int(np.random.choice(len(ranges), p=weights_arr))
    value_range = ranges[idx]
    if len(value_range) != 2:
        raise ValueError(f"{context} range entries must have length 2, got {value_range!r}")
    low = float(value_range[0])
    high = float(value_range[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{context} range bounds must be finite, got {value_range!r}")
    if low > high:
        raise ValueError(f"{context} range lower bound must be <= upper bound, got {value_range!r}")
    return float(np.random.uniform(low, high))


def _match_residual_stats(candidate, target, finite_mask, scale_clip=None):
    candidate = np.asarray(candidate, dtype=np.float32).copy()
    if not finite_mask.any():
        return candidate

    target_mean, target_std = _finite_mean_std(np.asarray(target, dtype=np.float32)[finite_mask])
    cand_mean, cand_std = _finite_mean_std(candidate[finite_mask])
    if cand_std <= 1e-6:
        scale = 1.0
    else:
        scale = target_std / cand_std
    if scale_clip is not None:
        scale = float(np.clip(scale, float(scale_clip[0]), float(scale_clip[1])))
    return (target_mean + (candidate - cand_mean) * scale).astype("float32")


def _standardize_component(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(np.std(values))
    if std <= 1e-6:
        return values - float(np.mean(values))
    return ((values - float(np.mean(values))) / std).astype("float32")


def _standardize_finite_component(values):
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    finite_values = values[finite]
    std = float(np.std(finite_values))
    if std <= 1e-6:
        out[finite] = finite_values - float(np.mean(finite_values))
    else:
        out[finite] = (finite_values - float(np.mean(finite_values))) / std
    return out.astype("float32")


def _boxcar_random_component(length, window):
    window = int(window)
    if window <= 1:
        return _standardize_component(np.random.normal(0.0, 1.0, size=length).astype("float32"))
    raw = np.random.normal(0.0, 1.0, size=length + window - 1).astype("float32")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return _standardize_component(np.convolve(raw, kernel, mode="valid").astype("float32"))


def _boxcar_smooth(values, window):
    window = int(window)
    if window <= 1:
        return np.asarray(values, dtype=np.float32).copy()
    values = np.asarray(values, dtype=np.float32)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype("float32")


def _tvt_bias_component(length, window):
    long_bias = _boxcar_random_component(length, window)
    raw_jitter = _boxcar_random_component(length, 1)
    long_weight = np.float32(np.sqrt(0.85))
    jitter_weight = np.float32(np.sqrt(0.15))
    return _standardize_component(
        long_weight * long_bias + jitter_weight * raw_jitter
    )


def _tvt_bias_v2_component(length, windows, weights):
    if len(windows) != 3:
        raise ValueError(f"TVT_bias_v2_windows must contain exactly three windows, got {windows!r}")
    if len(weights) != 3:
        raise ValueError(f"TVT_bias_v2_weights must contain exactly three weights, got {weights!r}")
    weights = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("TVT_bias_v2_weights must be finite")
    if (weights < 0.0).any():
        raise ValueError("TVT_bias_v2_weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("TVT_bias_v2_weights must give positive weight to at least one window")
    weights = weights / total

    components = []
    for window, weight in zip(windows, weights):
        components.append(np.sqrt(float(weight)) * _boxcar_random_component(length, int(window)))
    return _standardize_component(np.sum(components, axis=0))


def _conditional_gr_random_component(length, windows, weights, tail_df):
    if len(weights) != len(windows) + 1:
        raise ValueError("conditional_GR_weights must contain one weight per window plus one tail weight")
    weights = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("conditional_GR_weights must be finite")
    if (weights < 0.0).any():
        raise ValueError("conditional_GR_weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("conditional_GR_weights must give positive weight to at least one component")
    weights = weights / total

    components = []
    for window, weight in zip(windows, weights[:-1]):
        components.append(np.sqrt(float(weight)) * _boxcar_random_component(length, int(window)))
    tail = np.random.standard_t(max(1e-6, float(tail_df)), size=length).astype("float32")
    components.append(np.sqrt(float(weights[-1])) * _standardize_component(tail))
    return _standardize_component(np.sum(components, axis=0)).astype("float32")


class Simulator:
    def __init__(self,well_data_seq, sim_cfg={},aug_cfg={}, cfg=None):
        self.sim_cfg=sim_cfg
        self.aug_cfg=aug_cfg
        self.prefix_len = int(getattr(cfg, "prefix_len", 1024))
        self.target_len = int(getattr(cfg, "target_len", 10000))
        self.sampler=TrendSampler(well_data_seq, z_shift_cfg=sim_cfg.get('z_shift',{}))
        self.noise_bank=None
        self.global_noise_pool=None
        self.conditional_gr_model=None
        self.typewell_bank=None
        self.init_typewell_bank(well_data_seq)
        z_noise_modes=_noise_mode_weights(
            sim_cfg.get('z_shift',{}).get('noise_mode', {'TVT_bias': 1.0}),
            Z_SHIFT_NOISE_MODES,
            context='z_shift noise_mode',
        )
        z_nan_modes=self._resolve_z_nan_mode(sim_cfg.get('z_shift',{}))
        gr_noise_mode=aug_cfg.get('GR_noise_shift',{}).get('noise_mode','copy')
        if 'global_pool' in z_noise_modes or 'global' in z_nan_modes or gr_noise_mode=='global_pool':
            self.init_global_noise_pool(well_data_seq)
        if 'real_block' in z_noise_modes:
            self.init_noise_bank(well_data_seq)
        if 'conditional_GR' in z_noise_modes or 'decomposed_GR' in z_noise_modes:
            self.init_conditional_gr_model(well_data_seq)
        self.xy_local_fits=None
        if sim_cfg['xy_shift']['apply_prob']>0:
            self.init_xy_local_fits(well_data_seq)
        return

    def init_global_noise_pool(self,well_data_seq):
        residuals=[]
        for well_data in well_data_seq:
            horizontal=well_data['horizontal']
            TVT=horizontal['TVT']
            GR=horizontal['GR']
            v_GR=well_data['typewell']['GR']
            v_TVT=well_data['typewell']['TVT']
            matched_GR=np.interp(TVT, v_TVT, v_GR, left=np.nan, right=np.nan)
            residuals.append((GR-matched_GR).astype("float32"))
        if len(residuals)==0:
            raise ValueError("global residual noise pool is empty")
        self.global_noise_pool=np.concatenate(residuals).astype("float32",copy=False)
        if self.global_noise_pool.shape[0]==0:
            raise ValueError("global residual noise pool is empty")
        return

    def init_noise_bank(self,well_data_seq):
        self.noise_bank=[]
        for well_data in well_data_seq:
            horizontal=well_data['horizontal']
            TVT_input=horizontal['TVT_input']
            TVT=horizontal['TVT']
            GR=horizontal['GR']
            test_cond=np.isnan(TVT_input)
            if test_cond.sum()==0:
                continue
            v_GR=well_data['typewell']['GR']
            v_TVT=well_data['typewell']['TVT']
            matched_GR=np.interp(TVT, v_TVT, v_GR, left=np.nan, right=np.nan)
            suffix_noise=(GR-matched_GR)[test_cond].astype("float32")
            finite_noise=np.isfinite(suffix_noise)
            if finite_noise.sum()<16:
                continue
            mean,std=_finite_mean_std(suffix_noise)
            self.noise_bank.append({
                'well_id': well_data['well_id'],
                'noise': suffix_noise,
                'finite_rate': float(finite_noise.mean()),
                'mean': mean,
                'std': std,
            })
        if len(self.noise_bank)==0:
            raise ValueError("z_shift residual noise bank is empty")
        return

    def init_conditional_gr_model(self,well_data_seq):
        cfg=self.sim_cfg.get('z_shift',{})
        matched_parts=[]
        residual_parts=[]
        for well_data in well_data_seq:
            horizontal=well_data['horizontal']
            TVT_input=horizontal['TVT_input']
            TVT=horizontal['TVT']
            GR=horizontal['GR']
            test_cond=np.isnan(TVT_input)
            if test_cond.sum()==0:
                continue
            v_GR=well_data['typewell']['GR']
            v_TVT=well_data['typewell']['TVT']
            matched_GR=np.interp(TVT, v_TVT, v_GR, left=np.nan, right=np.nan).astype("float32")
            residual=(GR-matched_GR).astype("float32")
            valid=test_cond & np.isfinite(GR) & np.isfinite(matched_GR) & (np.abs(matched_GR)>1e-6)
            if valid.sum()==0:
                continue
            matched_parts.append(matched_GR[valid])
            residual_parts.append(residual[valid])
        if len(matched_parts)==0:
            raise ValueError("conditional_GR residual model found no finite suffix residuals")

        matched=np.concatenate(matched_parts).astype("float32",copy=False)
        residual=np.concatenate(residual_parts).astype("float32",copy=False)
        global_mean,global_std=_finite_mean_std(residual)
        min_count=max(1,int(cfg.get('conditional_GR_min_count',512)))
        max_bins=max(2,int(cfg.get('conditional_GR_bins',28)))
        bin_count=max(2,min(max_bins,matched.shape[0]//min_count))
        edges=np.quantile(matched.astype("float64"),np.linspace(0.0,1.0,bin_count+1))
        edges=np.unique(edges)
        if edges.shape[0]<3:
            center=float(np.mean(matched))
            centers=np.asarray([center-1e-3,center+1e-3],dtype=np.float32)
            means=np.asarray([global_mean,global_mean],dtype=np.float32)
            stds=np.asarray([global_std,global_std],dtype=np.float32)
        else:
            groups=np.searchsorted(edges[1:-1],matched,side="right")
            centers=[]
            means=[]
            stds=[]
            for group_idx in range(edges.shape[0]-1):
                group_mask=groups==group_idx
                if group_mask.sum()==0:
                    continue
                group_matched=matched[group_mask]
                group_residual=residual[group_mask]
                centers.append(float(np.mean(group_matched)))
                means.append(float(np.mean(group_residual)))
                std=float(np.std(group_residual))
                stds.append(std if std>1e-6 else global_std)
            centers=np.asarray(centers,dtype=np.float32)
            means=np.asarray(means,dtype=np.float32)
            stds=np.asarray(stds,dtype=np.float32)
            if centers.shape[0]==1:
                centers=np.asarray([centers[0]-1e-3,centers[0]+1e-3],dtype=np.float32)
                means=np.asarray([means[0],means[0]],dtype=np.float32)
                stds=np.asarray([stds[0],stds[0]],dtype=np.float32)

        order=np.argsort(centers)
        self.conditional_gr_model={
            'centers': centers[order].astype("float32",copy=True),
            'mean': means[order].astype("float32",copy=True),
            'std': stds[order].astype("float32",copy=True),
            'global_mean': np.float32(global_mean),
            'global_std': np.float32(global_std),
            'matched_mean': np.float32(np.mean(matched)),
            'matched_power_mean': np.float32(np.mean(matched.astype(np.float64)**0.9308)),
            'matched_values': matched.astype("float32",copy=True),
        }
        return

    def init_typewell_bank(self,well_data_seq):
        self.typewell_bank=[]
        self.typewell_bank_index={}
        for bank_idx,well_data in enumerate(well_data_seq):
            typewell=well_data['typewell']
            self.typewell_bank.append({
                'well_id': well_data['well_id'],
                'TVT': typewell['TVT'],
                'GR': typewell['GR'],
                'tvt_min': float(typewell['TVT'][0]),
                'tvt_max': float(typewell['TVT'][-1]),
            })
            self.typewell_bank_index[well_data['well_id']]=bank_idx
        if len(self.typewell_bank)==0:
            raise ValueError("switch_typewell Typewell bank is empty")
        return

    def init_xy_local_fits(self,well_data_seq):
        self.xy_local_fits={}
        for well_data in well_data_seq:
            well_id=well_data['well_id']
            # Locally dS = a*dX + b*dY for S = TVT+Z. S_avg differs from S
            # by a near-constant offset and has better row-level resolution.
            # The well path is nearly 1D in XY, so estimate only the tangent
            # component instead of an unstable full 2D plane gradient.
            X=well_data['horizontal']['X']
            Y=well_data['horizontal']['Y']
            S=well_data['horizontal']['S_avg']
            a,b=piecewise_delta_tangent_gradient_2var(
                X,
                Y,
                S,
                thr=self.sim_cfg['xy_shift']['ddS_jump_thr'],
                min_seg_len=self.sim_cfg['xy_shift']['min_window'],
                ridge=1e-6,
                a_bound=self.sim_cfg['xy_shift'].get('a_bound'),
                b_bound=self.sim_cfg['xy_shift'].get('b_bound'),
            )
            self.xy_local_fits[well_id]={
                'a':a,
                'b':b,
            }
        return

    def _sample_z_shift(self,well_data,z_shift_range):
        if z_shift_range <= 0:
            return 0.0
        TVT_input=well_data['horizontal']['TVT_input']
        finite_tvt_input=np.flatnonzero(np.isfinite(TVT_input))
        TVT0=float(TVT_input[finite_tvt_input[-1]])
        v_TVT=well_data['typewell']['TVT']
        anchor_window=100.0
        shift_min=TVT0-(float(v_TVT[-1])-anchor_window)
        shift_max=TVT0-(float(v_TVT[0])+anchor_window)
        low=max(-float(z_shift_range),shift_min)
        high=min(float(z_shift_range),shift_max)
        if low > high:
            return 0.0
        return float(np.random.uniform(low,high))

    def _sample_noise_bank_entry(self,well_id):
        if self.noise_bank is None:
            raise ValueError("z_shift residual noise bank was not initialized")
        if len(self.noise_bank)==1:
            return self.noise_bank[0]
        for _ in range(8):
            entry=self.noise_bank[random.randint(0,len(self.noise_bank)-1)]
            if entry['well_id']!=well_id:
                return entry
        return self.noise_bank[random.randint(0,len(self.noise_bank)-1)]

    def _sample_global_pool_noise(self,length):
        if self.global_noise_pool is None:
            raise ValueError("global residual noise pool was not initialized")
        length=int(length)
        if length==0:
            return np.empty(0,dtype=np.float32)
        pool_len=int(self.global_noise_pool.shape[0])
        if length>pool_len:
            raise ValueError(f"global residual noise pool is shorter than requested segment: {pool_len} < {length}")
        start=random.randint(0,pool_len-length)
        return self.global_noise_pool[start:start+length].astype("float32",copy=True)

    @staticmethod
    def _resolve_z_nan_mode(z_shift_cfg):
        if 'nan_mode' in z_shift_cfg:
            nan_mode=z_shift_cfg['nan_mode']
        else:
            nan_mode='same' if bool(z_shift_cfg.get('same_nan', True)) else 'none'
        return _noise_mode_weights(
            nan_mode,
            Z_SHIFT_NAN_MODES,
            context='z_shift nan_mode',
        )

    @staticmethod
    def _sample_shifted_nan_mask(nan_mask):
        nan_mask = np.asarray(nan_mask, dtype=bool)
        length = int(nan_mask.shape[0])
        if length <= 1:
            return nan_mask.copy()
        shift = random.randrange(length)
        indices = (np.arange(length, dtype=np.int64) + shift) % length
        return nan_mask[indices].astype(bool, copy=True)

    def _sample_real_block_noise(self,well_data,target_noise):
        cfg=self.sim_cfg['z_shift']
        target_noise=np.asarray(target_noise,dtype=np.float32)
        finite_target=np.isfinite(target_noise)
        copy_noise=_fill_nan_linear(target_noise)
        block_len=max(1,int(cfg.get('real_block_len',512)))
        scale_clip=cfg.get('real_block_scale_clip',(0.6,1.6))
        alpha=float(np.clip(_sample_range(cfg.get('real_block_alpha_range',(0.65,0.90))),0.0,1.0))

        out=np.full(target_noise.shape[0],np.nan,dtype=np.float32)
        pos=0
        while pos<out.shape[0]:
            entry=self._sample_noise_bank_entry(well_data['well_id'])
            donor=entry['noise']
            use_len=min(block_len,out.shape[0]-pos,donor.shape[0])
            if donor.shape[0]>use_len:
                start=random.randint(0,donor.shape[0]-use_len)
            else:
                start=0
            out[pos:pos+use_len]=donor[start:start+use_len]
            pos+=use_len

        out=np.where(np.isfinite(out),out,copy_noise).astype("float32")
        out=_match_residual_stats(out,target_noise,finite_target,scale_clip=scale_clip)
        out=alpha*copy_noise+(1.0-alpha)*out
        out=_match_residual_stats(out,target_noise,finite_target,scale_clip=scale_clip)
        return out.astype("float32")

    def _sample_structured_random_noise(self,target_noise):
        cfg=self.sim_cfg['z_shift']
        target_noise=np.asarray(target_noise,dtype=np.float32)
        finite_target=np.isfinite(target_noise)
        copy_noise=_fill_nan_linear(target_noise)
        scale_clip=cfg.get('structured_scale_clip',(0.6,1.6))
        alpha=float(np.clip(_sample_range(cfg.get('structured_alpha_range',(0.25,0.65))),0.0,1.0))
        windows=cfg.get('structured_windows',(160,64,16))
        weights=cfg.get('structured_weights',(0.50,0.18,0.25,0.07))
        if len(weights)!=len(windows)+1:
            raise ValueError("structured_weights must contain one weight per window plus one tail weight")

        weights=np.asarray(weights,dtype=np.float32)
        weights=weights/weights.sum()
        components=[]
        for window,weight in zip(windows,weights[:-1]):
            components.append(np.sqrt(float(weight))*_boxcar_random_component(target_noise.shape[0],window))
        tail=np.random.standard_t(5,size=target_noise.shape[0]).astype("float32")
        components.append(np.sqrt(float(weights[-1]))*_standardize_component(tail))
        out=_standardize_component(np.sum(components,axis=0))

        target_mean,target_std=_finite_mean_std(target_noise[finite_target])
        out=(target_mean+target_std*out).astype("float32")
        clip_abs=float(cfg.get('structured_clip_abs',45.0))
        if clip_abs>0:
            out=np.clip(out,target_mean-clip_abs,target_mean+clip_abs).astype("float32")
            out=_match_residual_stats(out,target_noise,finite_target,scale_clip=scale_clip)

        out=alpha*copy_noise+(1.0-alpha)*out
        out=_match_residual_stats(out,target_noise,finite_target,scale_clip=scale_clip)
        return out.astype("float32")

    def _sample_TVT_bias_noise(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg=None):
        if cfg is None:
            cfg=self.sim_cfg['z_shift']
        TVT_sim=np.asarray(TVT_sim,dtype=np.float32)
        sim_matched_GR=np.asarray(sim_matched_GR,dtype=np.float32)
        tvt_std=float(cfg.get('TVT_bias_std',0.95))
        window=max(1,int(cfg.get('TVT_bias_window',1280)))
        if tvt_std <= 0.0:
            return np.zeros(TVT_sim.shape[0],dtype=np.float32)
        tvt_noise=(np.float32(tvt_std)*_tvt_bias_component(TVT_sim.shape[0],window)).astype("float32")
        corrupted_TVT=np.clip(TVT_sim+tvt_noise,float(v_TVT[0]),float(v_TVT[-1])).astype("float32")
        corrupted_GR=np.interp(corrupted_TVT,v_TVT,v_GR,left=np.nan,right=np.nan).astype("float32")
        return (corrupted_GR-sim_matched_GR).astype("float32")

    def _sample_TVT_bias_v2_noise(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg=None):
        if cfg is None:
            cfg=self.sim_cfg['z_shift']
        TVT_sim=np.asarray(TVT_sim,dtype=np.float32)
        sim_matched_GR=np.asarray(sim_matched_GR,dtype=np.float32)
        tvt_std=float(cfg.get('TVT_bias_v2_tvt_std',cfg.get('TVT_bias_std',0.95)))
        windows=cfg.get('TVT_bias_v2_windows',(1536,256,2))
        weights=cfg.get('TVT_bias_v2_weights',(0.68,0.20,0.12))
        if tvt_std <= 0.0:
            return np.zeros(TVT_sim.shape[0],dtype=np.float32)

        tvt_noise=(np.float32(tvt_std)*_tvt_bias_v2_component(TVT_sim.shape[0],windows,weights)).astype("float32")
        corrupted_TVT=np.clip(TVT_sim+tvt_noise,float(v_TVT[0]),float(v_TVT[-1])).astype("float32")
        corrupted_GR=np.interp(corrupted_TVT,v_TVT,v_GR,left=np.nan,right=np.nan).astype("float32")
        out=(corrupted_GR-sim_matched_GR).astype("float32")

        finite=np.isfinite(out)
        if finite.any():
            scale_mode=str(cfg.get('TVT_bias_v2_scale_mode','std'))
            if scale_mode != 'none':
                target_scale=_sample_weighted_range(
                    cfg.get('TVT_bias_v2_gr_std_ranges',((5.6,8.4),(8.4,12.2),(12.2,17.5))),
                    cfg.get('TVT_bias_v2_gr_std_weights',(0.24,0.51,0.25)),
                    context='TVT_bias_v2_gr_std',
                )
                if scale_mode=='std':
                    current_scale=float(np.std(out[finite]))
                elif scale_mode=='rmse':
                    current_scale=float(np.sqrt(np.mean(out[finite]*out[finite])))
                else:
                    raise ValueError(f"unknown TVT_bias_v2_scale_mode: {scale_mode!r}")
                if current_scale>1e-6:
                    out=(out*np.float32(target_scale/current_scale)).astype("float32")

            offset_std=float(cfg.get('TVT_bias_v2_gr_offset_std',2.0))
            if offset_std>0.0:
                out=(out+np.float32(np.random.normal(0.0,offset_std))).astype("float32")

        clip_abs=float(cfg.get('TVT_bias_v2_clip_abs',36.0))
        if clip_abs>0.0:
            out=np.clip(out,-clip_abs,clip_abs).astype("float32")
        return out.astype("float32")

    def _sample_TVT_bias_noise_with_slope_clip(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg):
        TVT_sim=np.asarray(TVT_sim,dtype=np.float32)
        sim_matched_GR=np.asarray(sim_matched_GR,dtype=np.float32)
        tvt_std=float(cfg.get('TVT_bias_std',0.95))
        window=max(1,int(cfg.get('TVT_bias_window',1280)))
        if tvt_std <= 0.0:
            return np.zeros(TVT_sim.shape[0],dtype=np.float32)
        tvt_noise=(np.float32(tvt_std)*_tvt_bias_component(TVT_sim.shape[0],window)).astype("float32")

        slope_ref=float(cfg.get('mixed_V1_slope_clip_ref',0.0))
        if slope_ref > 0.0:
            slope_window=max(1,int(cfg.get('mixed_V1_slope_smooth_window',9)))
            smooth_gr=_boxcar_smooth(v_GR,slope_window)
            slope=np.abs(np.gradient(smooth_gr,v_TVT)).astype("float32")
            local_slope=np.interp(TVT_sim,v_TVT,slope,left=np.nan,right=np.nan).astype("float32")
            clip_scale=np.minimum(1.0,slope_ref/np.maximum(local_slope,1e-6)).astype("float32")
            clip_scale=np.nan_to_num(clip_scale,nan=1.0,posinf=1.0,neginf=1.0).astype("float32")
            clip_power=float(cfg.get('mixed_V1_slope_clip_power',1.0))
            if clip_power != 1.0:
                clip_scale=np.power(clip_scale,clip_power).astype("float32")
            tvt_noise=(tvt_noise*clip_scale).astype("float32")

        corrupted_TVT=np.clip(TVT_sim+tvt_noise,float(v_TVT[0]),float(v_TVT[-1])).astype("float32")
        corrupted_GR=np.interp(corrupted_TVT,v_TVT,v_GR,left=np.nan,right=np.nan).astype("float32")
        return (corrupted_GR-sim_matched_GR).astype("float32")

    def _sample_mixed_V1_noise(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg=None):
        if cfg is None:
            cfg=self.sim_cfg['z_shift']
        TVT_sim=np.asarray(TVT_sim,dtype=np.float32)
        tvt_bias=self._sample_TVT_bias_noise_with_slope_clip(
            TVT_sim,
            sim_matched_GR,
            v_TVT,
            v_GR,
            cfg,
        )
        finite_tvt_bias=np.isfinite(tvt_bias)

        windows=cfg.get('mixed_V1_windows',(320,96,24))
        weights=cfg.get('mixed_V1_weights',(0.45,0.25,0.22,0.08))
        if len(weights)!=len(windows)+1:
            raise ValueError("mixed_V1_weights must contain one weight per window plus one tail weight")
        weights=np.asarray(weights,dtype=np.float32)
        weights=weights/weights.sum()
        components=[]
        for window,weight in zip(windows,weights[:-1]):
            components.append(np.sqrt(float(weight))*_boxcar_random_component(TVT_sim.shape[0],window))
        tail_df=max(1e-6,float(cfg.get('mixed_V1_tail_df',6.0)))
        tail=np.random.standard_t(tail_df,size=TVT_sim.shape[0]).astype("float32")
        components.append(np.sqrt(float(weights[-1]))*_standardize_component(tail))
        structured=_standardize_component(np.sum(components,axis=0)).astype("float32")

        tvt_weight=float(cfg.get('mixed_V1_tvt_weight',0.4))
        target_std=float(cfg.get('mixed_V1_target_std',10.7))
        offset_std=float(cfg.get('mixed_V1_offset_std',1.1))
        tvt_part=(np.float32(tvt_weight)*np.nan_to_num(tvt_bias,nan=0.0,posinf=0.0,neginf=0.0)).astype("float32")
        if finite_tvt_bias.any():
            tvt_part_std=float(np.std(tvt_part[finite_tvt_bias]))
        else:
            tvt_part_std=0.0
        structured_std=np.sqrt(max(target_std*target_std-tvt_part_std*tvt_part_std,1e-6))
        offset=np.float32(np.random.normal(0.0,offset_std)) if offset_std > 0.0 else np.float32(0.0)
        out=(tvt_part+np.float32(structured_std)*structured+offset).astype("float32")
        clip_abs=float(cfg.get('mixed_V1_clip_abs',45.0))
        if clip_abs > 0.0:
            out=np.clip(out,-clip_abs,clip_abs).astype("float32")
        out[~finite_tvt_bias]=np.nan
        return out.astype("float32")

    def _sample_conditional_GR_noise(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg=None):
        if cfg is None:
            cfg=self.sim_cfg['z_shift']
        if self.conditional_gr_model is None:
            raise ValueError("conditional_GR residual model was not initialized")
        TVT_sim=np.asarray(TVT_sim,dtype=np.float32)
        sim_matched_GR=np.asarray(sim_matched_GR,dtype=np.float32)
        finite=np.isfinite(sim_matched_GR)
        out=np.full(TVT_sim.shape[0],np.nan,dtype=np.float32)
        if not finite.any():
            return out

        model=self.conditional_gr_model
        centers=model['centers']
        curve_mean=np.interp(
            sim_matched_GR[finite],
            centers,
            model['mean'],
            left=float(model['mean'][0]),
            right=float(model['mean'][-1]),
        ).astype("float32")
        curve_std=np.interp(
            sim_matched_GR[finite],
            centers,
            model['std'],
            left=float(model['std'][0]),
            right=float(model['std'][-1]),
        ).astype("float32")

        mean_shrink=float(cfg.get('conditional_GR_mean_shrink',0.45))
        scale_blend=float(np.clip(cfg.get('conditional_GR_scale_blend',0.85),0.0,1.0))
        global_std=max(float(model['global_std']),1e-6)
        curve_mean=(np.float32(mean_shrink)*curve_mean).astype("float32")
        curve_std=(
            np.float32(scale_blend)*curve_std
            + np.float32(1.0-scale_blend)*np.float32(global_std)
        ).astype("float32")
        curve_std=np.maximum(curve_std,np.float32(1e-3)).astype("float32")

        windows=cfg.get('conditional_GR_windows',(1024,256,32))
        weights=cfg.get('conditional_GR_weights',(0.48,0.25,0.20,0.07))
        tail_df=float(cfg.get('conditional_GR_tail_df',6.0))
        random_field=_conditional_gr_random_component(TVT_sim.shape[0],windows,weights,tail_df)

        texture_weight=float(np.clip(cfg.get('conditional_GR_texture_weight',0.25),0.0,1.0))
        if texture_weight>0.0:
            texture_cfg=dict(cfg)
            texture_cfg['TVT_bias_std']=float(
                cfg.get('conditional_GR_tvt_std',cfg.get('TVT_bias_std',0.95))
            )
            if 'conditional_GR_tvt_window' in cfg:
                texture_cfg['TVT_bias_window']=int(cfg['conditional_GR_tvt_window'])
            texture=self._sample_TVT_bias_noise_with_slope_clip(
                TVT_sim,
                sim_matched_GR,
                v_TVT,
                v_GR,
                texture_cfg,
            )
            texture=_standardize_finite_component(texture)
            innovation=(
                np.float32(np.sqrt(1.0-texture_weight))*random_field
                + np.float32(np.sqrt(texture_weight))*texture
            ).astype("float32")
            innovation=_standardize_component(innovation)
        else:
            innovation=random_field

        std_ranges=cfg.get(
            'conditional_GR_std_ranges',
            ((5.6,8.4),(8.4,12.2),(12.2,17.5)),
        )
        std_weights=cfg.get('conditional_GR_std_weights',(0.24,0.51,0.25))
        if std_ranges:
            target_std=_sample_weighted_range(std_ranges,std_weights,context='conditional_GR_std')
            well_scale=target_std/global_std
        else:
            well_scale=1.0

        offset_std=float(cfg.get('conditional_GR_offset_std',1.2))
        offset=np.float32(np.random.normal(0.0,offset_std)) if offset_std>0.0 else np.float32(0.0)
        values=(curve_mean+np.float32(well_scale)*curve_std*innovation[finite]+offset).astype("float32")
        clip_abs=float(cfg.get('conditional_GR_clip_abs',45.0))
        if clip_abs>0.0:
            values=np.clip(values,-clip_abs,clip_abs).astype("float32")
        out[finite]=values
        return out.astype("float32")

    def _conditional_gr_curve(self,sim_matched_GR,cfg,finite):
        model=self.conditional_gr_model
        mean_mode=str(cfg.get('decomposed_GR_mean_mode','power'))
        if mean_mode=='empirical':
            values=np.interp(
                sim_matched_GR[finite],
                model['centers'],
                model['mean'],
                left=float(model['mean'][0]),
                right=float(model['mean'][-1]),
            ).astype("float32")
            return values
        if mean_mode!='power':
            raise ValueError(f"unknown decomposed_GR_mean_mode: {mean_mode!r}")

        matched=sim_matched_GR[finite].astype(np.float64,copy=False)
        power=float(cfg.get('decomposed_GR_power',0.9308))
        power_mean=float(cfg.get('decomposed_GR_power_mean',model.get('matched_mean',np.nan)))
        power_norm=float(cfg.get('decomposed_GR_power_norm',np.nan))
        if not np.isfinite(power_mean) or power_mean<=0.0:
            power_mean=float(np.mean(model['centers']))
        if not np.isfinite(power_norm) or power_norm<=0.0:
            matched_source=model.get('matched_values')
            if matched_source is not None:
                power_norm=float(np.mean(np.asarray(matched_source,dtype=np.float64)**power))
            else:
                power_norm=float(np.mean(model['centers'].astype(np.float64)**power))
        scale=power_mean/power_norm
        values=(scale*np.power(np.maximum(matched,1e-6),power)-matched).astype("float32")
        return values

    def _sample_tvt_shift_burst_noise(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg):
        length=int(np.asarray(TVT_sim).shape[0])
        out=np.zeros(length,dtype=np.float32)
        base_std=float(cfg.get('decomposed_GR_tvt_base_std',0.26))
        if base_std>0.0:
            windows=cfg.get('decomposed_GR_tvt_base_windows',(512,96,8))
            weights=cfg.get('decomposed_GR_tvt_base_weights',(0.50,0.32,0.18))
            out=(out+np.float32(base_std)*_tvt_bias_v2_component(length,windows,weights)).astype("float32")

        event_prob=float(cfg.get('decomposed_GR_tvt_event_prob',0.50))
        max_events=int(cfg.get('decomposed_GR_tvt_event_max',3))
        event_std=float(cfg.get('decomposed_GR_tvt_event_std',1.25))
        if max_events>0 and event_std>0.0 and np.random.random()<event_prob and length>1:
            event_count=np.random.randint(1,max_events+1)
            min_len=max(1,int(cfg.get('decomposed_GR_tvt_event_min_len',24)))
            max_len=max(min_len,int(cfg.get('decomposed_GR_tvt_event_max_len',160)))
            smooth=max(1,int(cfg.get('decomposed_GR_tvt_event_smooth',9)))
            for _ in range(event_count):
                span=int(np.random.randint(min_len,min(max_len,length)+1))
                start=int(np.random.randint(0,max(1,length-span+1)))
                amp=np.float32(np.random.laplace(0.0,event_std))
                event=np.full(span,amp,dtype=np.float32)
                if smooth>1 and span>2:
                    event=_boxcar_smooth(event,smooth)
                if span>1:
                    ramp=min(max(1,smooth),span//2)
                    if ramp>1:
                        taper=np.ones(span,dtype=np.float32)
                        edge=np.linspace(0.0,1.0,ramp,dtype=np.float32)
                        taper[:ramp]=edge
                        taper[-ramp:]=edge[::-1]
                        event=(event*taper).astype("float32")
                out[start:start+span]=(out[start:start+span]+event).astype("float32")

        if not np.isfinite(out).any() or float(np.nanstd(out))<=1e-6:
            return np.zeros(length,dtype=np.float32)
        corrupted_TVT=np.clip(
            np.asarray(TVT_sim,dtype=np.float32)+out,
            float(v_TVT[0]),
            float(v_TVT[-1]),
        ).astype("float32")
        corrupted_GR=np.interp(corrupted_TVT,v_TVT,v_GR,left=np.nan,right=np.nan).astype("float32")
        return (corrupted_GR-np.asarray(sim_matched_GR,dtype=np.float32)).astype("float32")

    def _sample_decomposed_GR_noise(self,TVT_sim,sim_matched_GR,v_TVT,v_GR,cfg=None):
        if cfg is None:
            cfg=self.sim_cfg['z_shift']
        if self.conditional_gr_model is None:
            raise ValueError("decomposed_GR residual model was not initialized")
        TVT_sim=np.asarray(TVT_sim,dtype=np.float32)
        sim_matched_GR=np.asarray(sim_matched_GR,dtype=np.float32)
        finite=np.isfinite(sim_matched_GR)
        out=np.full(TVT_sim.shape[0],np.nan,dtype=np.float32)
        if not finite.any():
            return out

        model=self.conditional_gr_model
        global_std=max(float(model['global_std']),1e-6)
        curve_mean=self._conditional_gr_curve(sim_matched_GR,cfg,finite)
        mean_shrink=float(cfg.get('decomposed_GR_mean_shrink',0.70))
        curve_mean=(np.float32(mean_shrink)*curve_mean).astype("float32")

        curve_std=np.interp(
            sim_matched_GR[finite],
            model['centers'],
            model['std'],
            left=float(model['std'][0]),
            right=float(model['std'][-1]),
        ).astype("float32")
        scale_blend=float(np.clip(cfg.get('decomposed_GR_scale_blend',0.55),0.0,1.0))
        curve_std=(
            np.float32(scale_blend)*curve_std
            + np.float32(1.0-scale_blend)*np.float32(global_std)
        ).astype("float32")
        curve_std=np.maximum(curve_std,np.float32(1e-3)).astype("float32")

        random_windows=cfg.get('decomposed_GR_random_windows',(256,64,8))
        random_weights=cfg.get('decomposed_GR_random_weights',(0.34,0.28,0.22,0.16))
        tail_df=float(cfg.get('decomposed_GR_tail_df',6.0))
        random_field=_conditional_gr_random_component(TVT_sim.shape[0],random_windows,random_weights,tail_df)

        tvt_component=self._sample_tvt_shift_burst_noise(
            TVT_sim,
            sim_matched_GR,
            v_TVT,
            v_GR,
            cfg,
        )
        tvt_component=_standardize_finite_component(tvt_component)
        random_field=_standardize_component(random_field)

        tvt_weight=float(np.clip(cfg.get('decomposed_GR_tvt_weight',0.26),0.0,1.0))
        innovation=(
            np.float32(np.sqrt(1.0-tvt_weight))*random_field
            + np.float32(np.sqrt(tvt_weight))*tvt_component
        ).astype("float32")
        innovation=_standardize_component(innovation)

        std_ranges=cfg.get(
            'decomposed_GR_std_ranges',
            cfg.get('conditional_GR_std_ranges',((5.6,8.4),(8.4,12.2),(12.2,17.5))),
        )
        std_weights=cfg.get(
            'decomposed_GR_std_weights',
            cfg.get('conditional_GR_std_weights',(0.24,0.51,0.25)),
        )
        if std_ranges:
            target_std=_sample_weighted_range(std_ranges,std_weights,context='decomposed_GR_std')
            well_scale=target_std/global_std
        else:
            well_scale=1.0

        offset_std=float(cfg.get('decomposed_GR_offset_std',0.9))
        offset=np.float32(np.random.normal(0.0,offset_std)) if offset_std>0.0 else np.float32(0.0)
        values=(curve_mean+np.float32(well_scale)*curve_std*innovation[finite]+offset).astype("float32")

        target_rmse=float(cfg.get('decomposed_GR_target_rmse',10.7))
        if target_rmse>0.0 and np.isfinite(values).any():
            current_rmse=float(np.sqrt(np.mean(values.astype(np.float64)*values.astype(np.float64))))
            if current_rmse>1e-6:
                values=(values*np.float32(target_rmse/current_rmse)).astype("float32")
        clip_abs=float(cfg.get('decomposed_GR_clip_abs',45.0))
        if clip_abs>0.0:
            values=np.clip(values,-clip_abs,clip_abs).astype("float32")
        out[finite]=values
        return out.astype("float32")

    def _sample_typewell_entry(self,well_data,TVT):
        if self.typewell_bank is None:
            raise ValueError("switch_typewell Typewell bank was not initialized")
        bank_len=len(self.typewell_bank)
        if bank_len<=1:
            return None
        finite_tvt=TVT[np.isfinite(TVT)]
        if finite_tvt.size==0:
            return None

        tvt_low=float(np.min(finite_tvt))
        tvt_high=float(np.max(finite_tvt))
        current_idx=self.typewell_bank_index[well_data['well_id']]
        # Hot path: sample one other Typewell and skip invalid donors instead of scanning the bank.
        donor_idx=random.randint(0,bank_len-2)
        if donor_idx>=current_idx:
            donor_idx+=1
        entry=self.typewell_bank[donor_idx]
        if entry['tvt_min']>tvt_low or entry['tvt_max']<tvt_high:
            return None
        return entry

    def z_shift(
        self,
        well_data,
        S_jump_thr=0.15,
        z_shift_range=0.0,
        noise_mode='copy',
        nan_mode='same',
        drift_ratio=0.0,
        fault_ratio=0.0,
        fault_max_num=0,
        fault_std=0.0,
    ):
        '''
        x,y stay the same, shift tvt while hold (tvt+z) unchanged
        '''
        nan_mode = _sample_noise_mode(
            nan_mode,
            Z_SHIFT_NAN_MODES,
            context='z_shift nan_mode',
        )
        noise_mode = _sample_noise_mode(
            noise_mode,
            Z_SHIFT_NOISE_MODES,
            context='z_shift noise_mode',
        )
        TVT_input=well_data['horizontal']['TVT_input']
        TVT=well_data['horizontal']['TVT']
        Z=well_data['horizontal']['Z']
        test_cond=np.isnan(TVT_input)
        TVT_test=TVT[test_cond]
        # Sparse surface-jump augmentation. This intentionally leaves the
        # independent geo-S prior unchanged, so faulted samples can teach the
        # model to handle local prior mismatch.
        fault_ratio = float(fault_ratio)
        fault_max_num = int(fault_max_num)
        fault_std = float(fault_std)
        if fault_ratio > 0.0 and fault_max_num > 0 and fault_std > 0.0 and random.random()<fault_ratio:
            L=len(TVT_test)
            fault_drift=np.zeros(L,dtype=np.float32)
            if L > 1:
                k = np.random.randint(0, min(fault_max_num, L - 1) + 1)
                if k > 0:
                    idx = np.random.choice(np.arange(1, L), size=k, replace=False)
                    fault_drift[idx] = np.random.laplace(0, fault_std, size=k)
                    fault_drift =np.cumsum(fault_drift)
                    TVT_test=TVT_test+fault_drift

        # this is target TVT we aim to reach
        trend = self.sampler.get(len(TVT_test), well_id=well_data['well_id'])
        # simulate large drift in rare cases
        if random.random()<drift_ratio:
            bound=100
            axis=np.arange(len(trend))
            k=(trend*axis).sum()/(axis**2).sum()
            max_drift=np.max(np.abs(trend))
            if max_drift<bound:
                drift_k=np.sign(k)*(bound-max_drift)/axis[-1]
                trend=trend+drift_k*axis**2/axis[-1]
        
        TVT_test_sim=TVT_test[0]+trend
        # we shall keep the jump point caused by surface change
        TVT_shift=np.diff(TVT_test)
        TVT_shift[np.abs(TVT_shift)<S_jump_thr]=0.0
        TVT_shift=np.cumsum(TVT_shift)
        TVT_test_sim[1:]+=TVT_shift

        TVT_sim=TVT.copy()
        TVT_sim[test_cond]=TVT_test_sim
        # keep TVT+Z unchanged
        Z_sim=Z.copy()
        Z_sim[test_cond]=TVT_test+Z[test_cond]-TVT_test_sim

        range_shift=self._sample_z_shift(well_data,z_shift_range)
        if range_shift != 0.0:
            TVT_sim=TVT_sim-range_shift
            Z_sim=Z_sim+range_shift
            TVT_input_sim=TVT_input.copy()
            finite_tvt_input=np.isfinite(TVT_input_sim)
            TVT_input_sim[finite_tvt_input]=TVT_input_sim[finite_tvt_input]-range_shift
        else:
            TVT_input_sim=TVT_input

        ## fix MD axis based on relative (dZ)^2 change
        ## we don't fix md in this 1st version since |dZ|<<|dX|+|dY| in test phase, even rounding noise is larger than that
        ## this also saves our time to match all stats to unit gap MD axis
        # Z_diff=np.diff(Z)
        # Z_sim_diff=np.diff(Z_sim)
        # MD_change=Z_sim_diff**2-Z_diff**2
        # dMD_sim=(np.ones(len(Z_diff),dtype=np.float32)+MD_change)**0.5

        # create new GR by keep noise component unchanged
        GR=well_data['horizontal']['GR']
        v_GR=well_data['typewell']['GR']
        v_TVT=well_data['typewell']['TVT']
        sim_matched_GR=np.interp(TVT_sim, v_TVT, v_GR, left=np.nan, right=np.nan)
        matched_GR=np.interp(TVT, v_TVT, v_GR, left=np.nan, right=np.nan)
        base_noise=GR-matched_GR
        GR_noise=base_noise.copy()
        global_nan_source = None
        if noise_mode=='copy':
            pass
        elif noise_mode=='real_block':
            GR_noise[test_cond]=self._sample_real_block_noise(well_data,base_noise[test_cond])
        elif noise_mode=='structured_random':
            GR_noise[test_cond]=self._sample_structured_random_noise(base_noise[test_cond])
        elif noise_mode=='TVT_bias':
            GR_noise[test_cond]=self._sample_TVT_bias_noise(
                TVT_sim[test_cond],
                sim_matched_GR[test_cond],
                v_TVT,
                v_GR,
            )
        elif noise_mode=='TVT_bias_v2':
            GR_noise[test_cond]=self._sample_TVT_bias_v2_noise(
                TVT_sim[test_cond],
                sim_matched_GR[test_cond],
                v_TVT,
                v_GR,
            )
        elif noise_mode=='mixed_V1':
            GR_noise[test_cond]=self._sample_mixed_V1_noise(
                TVT_sim[test_cond],
                sim_matched_GR[test_cond],
                v_TVT,
                v_GR,
            )
        elif noise_mode=='conditional_GR':
            GR_noise[test_cond]=self._sample_conditional_GR_noise(
                TVT_sim[test_cond],
                sim_matched_GR[test_cond],
                v_TVT,
                v_GR,
            )
        elif noise_mode=='decomposed_GR':
            GR_noise[test_cond]=self._sample_decomposed_GR_noise(
                TVT_sim[test_cond],
                sim_matched_GR[test_cond],
                v_TVT,
                v_GR,
            )
        elif noise_mode=='global_pool':
            global_nan_source = self._sample_global_pool_noise(test_cond.sum())
            GR_noise[test_cond]=global_nan_source
        else:
            raise ValueError(f"unknown z_shift noise_mode: {noise_mode!r}")
        GR_sim=sim_matched_GR+GR_noise
        if nan_mode=='same':
            GR_sim[np.isnan(GR)]=np.nan
        elif nan_mode=='shift':
            suffix_gr=GR_sim[test_cond].copy()
            finite_suffix_gr=np.isfinite(suffix_gr)
            if finite_suffix_gr.any():
                suffix_gr=_fill_nan_linear(suffix_gr)
            else:
                suffix_gr=np.zeros_like(suffix_gr,dtype=np.float32)
            shifted_nan_mask=self._sample_shifted_nan_mask(np.isnan(GR[test_cond]))
            suffix_gr[shifted_nan_mask]=np.nan
            GR_sim[test_cond]=suffix_gr
        elif nan_mode=='global':
            if global_nan_source is None:
                global_nan_source = self._sample_global_pool_noise(test_cond.sum())
                suffix_noise = np.nan_to_num(
                    GR_noise[test_cond].copy(),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                GR_sim[test_cond] = sim_matched_GR[test_cond] + suffix_noise
            suffix_gr=GR_sim[test_cond].copy()
            finite_suffix_gr=np.isfinite(suffix_gr)
            if finite_suffix_gr.any():
                suffix_gr=_fill_nan_linear(suffix_gr)
            else:
                suffix_gr=np.zeros_like(suffix_gr,dtype=np.float32)
            suffix_gr[~np.isfinite(global_nan_source)]=np.nan
            GR_sim[test_cond]=suffix_gr
        if range_shift == 0.0:
            seen_cond=~test_cond
            GR_sim[seen_cond]=GR[seen_cond]

        # final updates
        well_data['horizontal']['Z']=Z_sim
        well_data['horizontal']['TVT']=TVT_sim
        well_data['horizontal']['TVT_input']=TVT_input_sim
        well_data['horizontal']['GR']=GR_sim
        return well_data

    def xy_shift(self,well_data,shift_bound=5):
        '''
        Apply a small row-preserving XY translation and adjust TVT by the
        local tangent-projected surface gradient d(TVT+Z)=a*dX+b*dY.
        '''
        well_id=well_data['well_id']
        TVT=well_data['horizontal']['TVT']
        TVT_input=well_data['horizontal']['TVT_input']
        X=well_data['horizontal']['X']
        Y=well_data['horizontal']['Y']
        
        a=self.xy_local_fits[well_id]['a']
        b=self.xy_local_fits[well_id]['b']
        x_shift,y_shift=np.random.uniform(-shift_bound,shift_bound,2)
        
        X_sim=X+x_shift
        Y_sim=Y+y_shift
        TVT_shift=a*x_shift+b*y_shift
        TVT_sim=TVT+TVT_shift
        TVT_input_sim=TVT_input.copy()
        finite_tvt_input=np.isfinite(TVT_input_sim)
        TVT_input_sim[finite_tvt_input]=TVT_input_sim[finite_tvt_input]+TVT_shift[finite_tvt_input]
        
        
        # create new GR by keep noise component unchanged
        GR=well_data['horizontal']['GR']
        v_GR=well_data['typewell']['GR']
        v_TVT=well_data['typewell']['TVT']
        matched_GR=np.interp(TVT, v_TVT, v_GR, left=np.nan, right=np.nan)
        sim_matched_GR=np.interp(TVT_sim, v_TVT, v_GR, left=np.nan, right=np.nan)
        GR_noise=(GR-matched_GR)
        GR_sim=sim_matched_GR+GR_noise
        # final updates
        well_data['horizontal']['X']=X_sim
        well_data['horizontal']['Y']=Y_sim
        well_data['horizontal']['TVT']=TVT_sim
        well_data['horizontal']['TVT_input']=TVT_input_sim
        well_data['horizontal']['GR']=GR_sim
        if 'geo_s_rel_prior_abs' in well_data['horizontal']:
            well_data['horizontal']['geo_s_rel_prior_abs'] = (
                well_data['horizontal']['geo_s_rel_prior_abs'] + TVT_shift
            ).astype('float32')
        return well_data

    def switch_typewell(self,well_data,noise_mode='copy',same_nan=True):
        '''
        Replace the paired Typewell with another cached Typewell, then rebuild
        horizontal GR by rematching the current TVT path and adding residual
        noise in the donor Typewell frame.
        '''
        horizontal=well_data['horizontal']
        TVT=horizontal['TVT']
        GR=horizontal['GR']
        old_typewell=well_data['typewell']
        new_typewell=self._sample_typewell_entry(well_data,TVT)
        if new_typewell is None:
            return well_data

        old_matched_GR=np.interp(
            TVT,
            old_typewell['TVT'],
            old_typewell['GR'],
            left=np.nan,
            right=np.nan,
        ).astype('float32')
        new_matched_GR=np.interp(
            TVT,
            new_typewell['TVT'],
            new_typewell['GR'],
            left=np.nan,
            right=np.nan,
        ).astype('float32')
        base_noise=(GR-old_matched_GR).astype('float32')

        if noise_mode=='copy':
            GR_noise=_fill_nan_linear(base_noise)
        elif noise_mode=='TVT_bias':
            GR_noise=self._sample_TVT_bias_noise(
                TVT,
                new_matched_GR,
                new_typewell['TVT'],
                new_typewell['GR'],
                cfg=self.sim_cfg['switch_typewell'],
            )
        else:
            raise ValueError(f"unknown switch_typewell noise_mode: {noise_mode!r}")

        GR_sim=(new_matched_GR+GR_noise).astype('float32')
        if same_nan:
            GR_sim[np.isnan(GR)]=np.nan

        well_data['typewell']={
            'TVT': new_typewell['TVT'],
            'GR': new_typewell['GR'],
        }
        horizontal['GR']=GR_sim
        return well_data

    def _sample_typewell_gr_jitter_mode(self, cfg):
        mode_weights = cfg.get("mode_weights", {"coherent_residual_copy": 1.0})
        if isinstance(mode_weights, str):
            mode_weights = {mode_weights: 1.0}
        return _sample_noise_mode(
            mode_weights,
            TYPEWELL_GR_JITTER_MODES,
            context="typewell_gr_jitter mode_weights",
        )

    @staticmethod
    def _typewell_window_rows_from_ft(typewell_tvt, window_ft):
        typewell_tvt = np.asarray(typewell_tvt, dtype=np.float32)
        if typewell_tvt.shape[0] < 2:
            return 1
        diffs = np.diff(typewell_tvt)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            return 1
        step = float(np.median(diffs))
        if step <= 0.0:
            return 1
        return max(1, int(round(float(window_ft) / step)))

    def _jitter_typewell_gr_curve(self, typewell, cfg):
        tvt = typewell["TVT"].astype(np.float32, copy=False)
        gr = typewell["GR"].astype(np.float32, copy=False)
        finite = np.isfinite(gr)
        if not finite.any():
            return gr.astype(np.float32, copy=True)

        center = np.float32(np.mean(gr[finite]))
        gain = np.float32(1.0 + np.random.normal(0.0, float(cfg.get("gain_std", 0.0))))
        gain_clip = cfg.get("gain_clip", (1.0, 1.0))
        gain = np.float32(np.clip(gain, float(gain_clip[0]), float(gain_clip[1])))
        offset = np.float32(np.random.normal(0.0, float(cfg.get("offset_std", 0.0))))
        out = (center + gain * (gr - center) + offset).astype(np.float32)

        drift_std = float(cfg.get("drift_std", 0.0))
        if drift_std > 0.0 and out.shape[0] > 1:
            drift_choices = tuple(cfg.get("drift_window_ft_choices", (1.0,)))
            window_ft = float(random.choice(drift_choices)) if drift_choices else 1.0
            window_rows = self._typewell_window_rows_from_ft(tvt, window_ft)
            drift = _sample_smooth_noise(out.shape[0], drift_std, (window_rows,))
            out = (out + drift).astype(np.float32)

        smooth_prob = float(cfg.get("smooth_prob", 0.0))
        if smooth_prob > 0.0 and out.shape[0] > 2 and random.random() < smooth_prob:
            window = int(random.choice(tuple(cfg.get("smooth_window_choices", (5,)))))
            smooth = _box_smooth_1d(out, window)
            blend = np.float32(_sample_range_uniform(cfg.get("smooth_blend_range", (0.0, 0.0))))
            out = ((np.float32(1.0) - blend) * out + blend * smooth).astype(np.float32)

        return out.astype(np.float32)

    def typewell_gr_jitter(self, well_data, cfg=None):
        cfg = self.aug_cfg.get("typewell_gr_jitter", {}) if cfg is None else cfg
        mode = self._sample_typewell_gr_jitter_mode(cfg)
        horizontal = well_data["horizontal"]
        typewell = well_data["typewell"]
        old_tvt = typewell["TVT"].astype(np.float32, copy=False)
        old_gr = typewell["GR"].astype(np.float32, copy=False)
        new_gr = self._jitter_typewell_gr_curve(typewell, cfg)

        if mode == "coherent_residual_copy":
            TVT = horizontal["TVT"].astype(np.float32, copy=False)
            GR = horizontal["GR"].astype(np.float32, copy=False)
            old_matched = np.interp(TVT, old_tvt, old_gr, left=np.nan, right=np.nan).astype(np.float32)
            new_matched = np.interp(TVT, old_tvt, new_gr, left=np.nan, right=np.nan).astype(np.float32)
            residual = (GR - old_matched).astype(np.float32)
            finite_residual = np.isfinite(residual)
            if finite_residual.any():
                residual_filled = _fill_nan_linear(residual)
                GR_new = (new_matched + residual_filled).astype(np.float32)
                out_of_support = ~np.isfinite(new_matched)
                GR_new[out_of_support] = GR[out_of_support]
                raw_nan = horizontal.get("GR_isnan_mask")
                if raw_nan is None:
                    raw_nan = ~np.isfinite(GR)
                else:
                    raw_nan = np.asarray(raw_nan, dtype=bool)
                GR_new[raw_nan] = np.nan
                horizontal["GR"] = GR_new.astype(np.float32)
        elif mode == "reference_only":
            pass
        else:
            raise ValueError(f"unknown typewell_gr_jitter mode: {mode!r}")

        well_data["typewell"] = {
            "TVT": old_tvt.astype(np.float32, copy=True),
            "GR": new_gr.astype(np.float32, copy=True),
        }
        return well_data

    def typewell_mirror(self, well_data, skip_out_of_typewell=True):
        """
        Reflect TVT and S=TVT+Z around the last visible anchor while keeping
        horizontal GR values row-aligned. The emitted Typewell stays in the
        ordinary TVT coordinate frame and carries no downstream mirror metadata.
        """
        horizontal = well_data["horizontal"]
        last_seen_idx = _last_seen_idx(horizontal, well_data["well_id"])
        tvt0 = np.float32(horizontal["TVT_input"][last_seen_idx])

        TVT = horizontal["TVT"].astype(np.float32, copy=False)
        Z = horizontal["Z"].astype(np.float32, copy=False)
        S = (TVT + Z).astype(np.float32)
        TVT_mirror = (np.float32(2.0) * tvt0 - TVT).astype(np.float32)
        S_mirror = (np.float32(2.0) * np.float32(S[last_seen_idx]) - S).astype(np.float32)

        typewell = well_data["typewell"]
        typewell_tvt = typewell["TVT"].astype(np.float32, copy=False)
        typewell_gr = typewell["GR"].astype(np.float32, copy=False)
        source_min = typewell_tvt[0]
        source_max = typewell_tvt[-1]
        support_min = np.maximum(source_min, np.float32(2.0) * tvt0 - source_max)
        support_max = np.minimum(source_max, np.float32(2.0) * tvt0 - source_min)
        if skip_out_of_typewell:
            prefix_start = max(0, last_seen_idx + 1 - self.prefix_len)
            suffix_end = min(horizontal["MD"].shape[0], last_seen_idx + 1 + self.target_len)
            window = TVT_mirror[prefix_start:suffix_end]
            finite = np.isfinite(window)
            if finite.any() and (
                np.nanmin(window[finite]) < support_min
                or np.nanmax(window[finite]) > support_max
            ):
                return well_data

        horizontal["TVT"] = TVT_mirror
        horizontal["Z"] = (S_mirror - TVT_mirror).astype(np.float32)
        TVT_input = horizontal["TVT_input"].astype(np.float32, copy=True)
        finite_tvt_input = np.isfinite(TVT_input)
        TVT_input[finite_tvt_input] = (
            np.float32(2.0) * tvt0 - TVT_input[finite_tvt_input]
        ).astype(np.float32)
        horizontal["TVT_input"] = TVT_input
        for key in ("geo_s_rel_prior_abs", "geo_direct_s_rel_prior_abs"):
            if key in horizontal:
                geo_s = horizontal[key].astype(np.float32, copy=False)
                horizontal[key] = (
                    np.float32(2.0) * np.float32(S[last_seen_idx]) - geo_s
                ).astype(np.float32)

        typewell_keep = (typewell_tvt >= support_min) & (typewell_tvt <= support_max)
        reflected_typewell_tvt = (np.float32(2.0) * tvt0 - typewell_tvt).astype(np.float32)
        reflected_typewell_keep = (
            (reflected_typewell_tvt >= support_min)
            & (reflected_typewell_tvt <= support_max)
        )
        path_keep = (
            np.isfinite(TVT_mirror)
            & (TVT_mirror >= support_min)
            & (TVT_mirror <= support_max)
        )
        mirror_axis = np.concatenate(
            [
                typewell_tvt[typewell_keep],
                reflected_typewell_tvt[reflected_typewell_keep],
                TVT_mirror[path_keep],
                np.asarray([support_min, support_max], dtype=np.float32),
            ]
        ).astype(np.float32)
        mirror_axis = np.sort(mirror_axis)
        mirror_axis = mirror_axis[np.r_[True, np.diff(mirror_axis) > np.float32(1e-6)]]
        mirrored_gr = np.interp(
            np.float32(2.0) * tvt0 - mirror_axis,
            typewell_tvt,
            typewell_gr,
            left=np.nan,
            right=np.nan,
        ).astype(np.float32)
        finite = np.isfinite(mirrored_gr)
        well_data["typewell"] = {
            "TVT": mirror_axis[finite].astype(np.float32, copy=True),
            "GR": mirrored_gr[finite].astype(np.float32, copy=True),
        }
        if "pf_cache" in well_data:
            pf_cache = well_data["pf_cache"]
            out = dict(pf_cache)
            for prob_key in ("pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi"):
                if prob_key in pf_cache:
                    out[prob_key] = _normalize_prob_rows(
                        np.asarray(pf_cache[prob_key], dtype=np.float32)[:, ::-1].astype(np.float32, copy=True)
                    )
            for rel_key in ("pf_tvt_rel_pred", "pf_tvt_rel_pred_ffbsi", "window_tvt_rel"):
                if rel_key in pf_cache:
                    values = np.asarray(pf_cache[rel_key], dtype=np.float32)
                    out[rel_key] = np.where(np.isfinite(values), -values, values).astype(np.float32)
            for abs_key in ("pf_tvt_pred", "pf_tvt_pred_ffbsi", "window_tvt"):
                if abs_key in pf_cache:
                    values = np.asarray(pf_cache[abs_key], dtype=np.float32)
                    out[abs_key] = np.where(
                        np.isfinite(values),
                        np.float32(2.0) * tvt0 - values,
                        values,
                    ).astype(np.float32)
            well_data["pf_cache"] = out
        if "pf_sample_cache" in well_data:
            pf_sample_cache = well_data["pf_sample_cache"]
            out = dict(pf_sample_cache)
            values = np.asarray(pf_sample_cache["pf_tvt_samples"], dtype=np.float32)
            out["pf_tvt_samples"] = np.where(
                np.isfinite(values),
                np.float32(2.0) * tvt0 - values,
                values,
            ).astype(np.float32)
            well_data["pf_sample_cache"] = out
        return well_data

    def reverse_path(self,well_data,head_TVT_bound=10,sim_head_range=1024):
        '''
        simulate we dig from tail to head
        1.for rare short seq large sim_head_range can create seq with zero pred window
          this doesn't affect learning, but later process shall care about this
        '''
        # rm head region with large TVT slope
        TVT_input=well_data['horizontal']['TVT_input']
        TVT0=TVT_input[~np.isnan(TVT_input)][-1]
        start_idx=np.where(np.abs(TVT_input-TVT0)>head_TVT_bound)[0][-1]+1
        # reverse path order
        for k,v in well_data['horizontal'].items():
            v=v[start_idx:][::-1]
            well_data['horizontal'][k]=v
        # new TVT_input copy from TVT, only keep head region
        TVT_input=well_data['horizontal']['TVT'].copy()
        TVT_input[sim_head_range:]=np.nan
        well_data['horizontal']['TVT_input']=TVT_input
        # MD is axis
        well_data['horizontal']['MD']=np.arange(len(TVT_input),dtype=np.float32)
        return well_data

    def start_point_shift(self,well_data,left_bound=300,right_bound=2000,max_ratio=0.3):
        '''
        change TVT_input, which determines where prediction start
        '''
        TVT_input=well_data['horizontal']['TVT_input']
        TVT=well_data['horizontal']['TVT']
        test_cond=np.isnan(TVT_input)
        if test_cond.sum()==0: # this can happen due to other aug/sim
            return well_data
        pred_start_idx=np.where(test_cond)[0][0]
        pred_window_len=len(TVT_input)-pred_start_idx
        max_shift_len=int(max_ratio*pred_window_len)
        lb=-min(left_bound,max_shift_len)
        ub=min(right_bound,max_shift_len)
        shift=int(np.random.uniform(lb,ub))
        if shift<0:
            TVT_input[pred_start_idx+shift:pred_start_idx]=np.nan
        else:
            TVT_input[pred_start_idx:pred_start_idx+shift]=TVT[pred_start_idx:pred_start_idx+shift]
        well_data['horizontal']['TVT_input']=TVT_input
        return well_data

    def GR_noise_shift(self,well_data,shift_range_ratio=0.3,noise_mode='copy'):
        '''
        sampling new noise, with nan pattern generated by cyclic shift
        '''
        TVT_input=well_data['horizontal']['TVT_input']
        TVT=well_data['horizontal']['TVT']
        GR=well_data['horizontal']['GR']
        v_GR=well_data['typewell']['GR']
        v_TVT=well_data['typewell']['TVT']
        
        test_cond=np.isnan(TVT_input)
        TVT_test=TVT[test_cond]
        GR_matched=np.interp(TVT_test, v_TVT, v_GR, left=np.nan, right=np.nan).astype('float32')
        GR_test=GR[test_cond]
        # nan contained in GR noise
        GR_noise=GR_test-GR_matched
        L=len(GR_test)
        if L == 0:
            return well_data
        if noise_mode=='global_pool':
            GR_noise=self._sample_global_pool_noise(L)
        else:
            shift=np.random.uniform(-L*shift_range_ratio,L*shift_range_ratio)
            shift=int(shift)
            index=np.arange(L)
            index=(index+shift)%L
            cyclic_GR_noise=GR_noise[index]
            if noise_mode=='copy':
                GR_noise=cyclic_GR_noise
            elif noise_mode=='TVT_bias':
                GR_noise=self._sample_TVT_bias_noise(
                    TVT_test,
                    GR_matched,
                    v_TVT,
                    v_GR,
                    cfg=self.aug_cfg['GR_noise_shift'],
                )
                GR_noise[~np.isfinite(cyclic_GR_noise)]=np.nan
            else:
                raise ValueError(f"unknown GR_noise_shift noise_mode: {noise_mode!r}")
        GR[test_cond]=GR_matched+GR_noise
        well_data['horizontal']['GR']=GR
        return well_data

    def MD_streching(self,well_data,strech_ratio=(0.7,1.3),apply_to_MD=True):
        '''
        Resample the horizontal path with a random MD sampling-rate factor.
        '''
        if np.isscalar(strech_ratio):
            ratio = float(strech_ratio)
            stretch_low, stretch_high = 1.0 - ratio, 1.0 + ratio
        else:
            if len(strech_ratio) != 2:
                raise ValueError(
                    "MD_streching strech_ratio must be a scalar or a two-value range"
                )
            stretch_low, stretch_high = (float(value) for value in strech_ratio)
        if (
            not np.isfinite(stretch_low)
            or not np.isfinite(stretch_high)
            or stretch_low <= 0.0
            or stretch_high < stretch_low
        ):
            raise ValueError(
                "MD_streching stretch factors must be finite, positive, and ordered"
            )
        r=np.random.uniform(stretch_low,stretch_high)
        MD=well_data['horizontal']['MD']
        MD_sim=np.linspace(MD[0],MD[-1],int(len(MD)*r))
        for k,v in well_data['horizontal'].items():
            # GR nan rate after interpolate will be larger
            v_sim=np.interp(MD_sim,MD,v,left=np.nan, right=np.nan).astype(np.float32)
            well_data['horizontal'][k]=v_sim
        if apply_to_MD:
            # if MD works as time
            well_data['horizontal']['MD']=MD_sim
        else:
            # if MD works as index
            well_data['horizontal']['MD']=np.arange(len(MD_sim),dtype=np.float32)
        return well_data

    def tail_cut(self, well_data, max_cut_ratio=0.5):
        """Remove a uniformly sampled fraction of the prediction suffix."""
        horizontal = well_data["horizontal"]
        last_seen_idx = _last_seen_idx(horizontal, well_data["well_id"])
        suffix_len = horizontal["MD"].shape[0] - last_seen_idx - 1
        if suffix_len <= 1 or max_cut_ratio <= 0.0:
            return well_data

        cut_ratio = float(np.random.uniform(0.0, max_cut_ratio))
        cut_len = min(int(suffix_len * cut_ratio), suffix_len - 1)
        if cut_len == 0:
            return well_data

        keep_len = horizontal["MD"].shape[0] - cut_len
        # Geo-prior arrays and orig_index are horizontal sidecars. Slicing all
        # rows keeps geo inputs aligned and makes the fixed PF cache remap stop
        # at the same retained original-row endpoint.
        for name, values in horizontal.items():
            horizontal[name] = values[:keep_len]
        return well_data


    def horizontal_gr_blur(self, well_data, flat_region_max_ratio=0.5):
        horizontal = well_data["horizontal"]
        suffix_start = _last_seen_idx(horizontal, well_data["well_id"]) + 1
        gr = horizontal["GR"].astype(np.float32, copy=False)
        out = gr.astype(np.float32, copy=True)
        out[suffix_start:] = _blur_low_variance_gr_region(
            gr[suffix_start:],
            flat_region_max_ratio,
        )
        horizontal["GR"] = out
        return well_data


    def GR_trf(self, well_data, cfg=None):
        cfg = self.aug_cfg.get("GR_trf", {}) if cfg is None else cfg
        gr = well_data["typewell"]["GR"].astype(np.float32, copy=False)
        out = gr.astype(np.float32, copy=True)
        p_std = float(cfg.get("p_std", 0.0))
        if p_std > 0.0:
            finite = np.isfinite(out)
            if finite.any():
                if np.any(out[finite] <= 0.0):
                    raise ValueError("GR_trf power transform requires positive finite Typewell GR values")
                original_mean = float(np.mean(out[finite].astype(np.float64, copy=False)))
                p = float(np.random.normal(1.0, p_std))
                powered = np.power(out[finite].astype(np.float64, copy=False), p)
                powered_mean = float(np.mean(powered))
                if not np.isfinite(powered_mean) or powered_mean <= 0.0:
                    raise ValueError("GR_trf power transform produced invalid Typewell GR mean")
                out[finite] = (powered * (original_mean / powered_mean)).astype(np.float32)
        a_std = float(cfg.get("a_std", 0.015))
        b_std = float(cfg.get("b_std", 1.0))
        ab_window = cfg.get("ab_window")
        if ab_window is None or int(ab_window) > out.shape[0]:
            a = np.float32(np.random.normal(1.0, a_std))
            b = np.float32(np.random.normal(0.0, b_std))
        else:
            ab_window = int(ab_window)
            a = _sample_boxcar_normal_curve(out.shape[0], 1.0, a_std, ab_window)
            b = _sample_boxcar_normal_curve(out.shape[0], 0.0, b_std, ab_window)
        if bool(cfg.get("a_keep_mean", False)):
            x_mean = np.float32(np.nanmean(out, dtype=np.float64))
            out = (x_mean + a * (out - x_mean) + b).astype(np.float32)
        else:
            out = (a * out + b).astype(np.float32)

        smooth_apply_prob = float(cfg.get("smooth_apply_prob", 0.0))
        sharpen_apply_prob = float(cfg.get("sharpen_apply_prob", 0.0))
        texture_apply_prob = smooth_apply_prob + sharpen_apply_prob
        if texture_apply_prob > 0.0:
            texture_draw = random.random()
            smooth_window = int(cfg.get("smooth_window", 5))
            if texture_draw < smooth_apply_prob:
                out = _box_smooth_1d(out, smooth_window)
            elif texture_draw < texture_apply_prob:
                gr_smooth = _box_smooth_1d(out, smooth_window)
                sharpen_alpha = np.float32(
                    _sample_range_uniform(cfg.get("sharpen_alpha_range", (0.0, 0.3)))
                )
                out = (out + sharpen_alpha * (out - gr_smooth)).astype(np.float32)

        center_mirror_prob = float(cfg.get("center_mirror_prob", 0.0))
        if center_mirror_prob > 0.0 and random.random() < center_mirror_prob:
            horizontal = well_data["horizontal"]
            target_mask = np.isnan(horizontal["TVT_input"])
            horizontal_gr = horizontal["GR"].astype(np.float32, copy=False)
            target_gr = horizontal_gr[target_mask]
            finite_target_gr = target_gr[np.isfinite(target_gr)]
            if finite_target_gr.size > 0:
                gr_mean = np.float32(np.mean(finite_target_gr.astype(np.float64, copy=False)))
                horizontal["GR"] = np.maximum(np.float32(2.0) * gr_mean - horizontal_gr, np.float32(0.0)).astype(np.float32)
                out = np.maximum(np.float32(2.0) * gr_mean - out, np.float32(0.0)).astype(np.float32)
        well_data["typewell"]["GR"] = out
        return well_data

    def typewell_drift(self, well_data, cfg=None):
        cfg = self.aug_cfg.get("typewell_drift", {}) if cfg is None else cfg
        shift = np.float32(np.random.normal(0.0, float(cfg.get("shift_std", 1.0))))
        typewell_tvt = well_data["typewell"]["TVT"].astype(np.float32, copy=False)
        well_data["typewell"]["TVT"] = (typewell_tvt + shift).astype(np.float32)
        return well_data

    def keep_ori_path(self,well_data):
        return well_data

    def process(self,well_data):
        '''
        well_id: well_id
        horizontal: k:array for k in ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]
        typewell: k:array for k in ["GR", "TVT"]
        '''
        well_data = deepcopy(well_data)
        if random.random()<self.sim_cfg['keep_ori_path']['apply_prob']:
            return self.keep_ori_path(well_data)

        typewell_mirror_cfg = self.sim_cfg.get("typewell_mirror", {})
        typewell_mirror_apply_prob = float(typewell_mirror_cfg.get("apply_prob", 0.0))
        if typewell_mirror_apply_prob>0.0 and random.random()<typewell_mirror_apply_prob:
            well_data=self.typewell_mirror(
                well_data,
                skip_out_of_typewell=typewell_mirror_cfg.get("skip_out_of_typewell", True),
            )
        if random.random()<self.sim_cfg['z_shift']['apply_prob']:
            z_noise_mode = _sample_noise_mode(
                self.sim_cfg['z_shift'].get('noise_mode', {'TVT_bias': 1.0}),
                Z_SHIFT_NOISE_MODES,
                context='z_shift noise_mode',
            )
            well_data=self.z_shift(
                well_data,
                S_jump_thr=self.sim_cfg['z_shift']['S_jump_thr'],
                z_shift_range=self.sim_cfg['z_shift'].get('z_shift_range',0.0),
                noise_mode=z_noise_mode,
                nan_mode=self._resolve_z_nan_mode(self.sim_cfg['z_shift']),
                drift_ratio=self.sim_cfg['z_shift']['drift_ratio'],
                fault_ratio=self.sim_cfg['z_shift'].get('fault_ratio',0.0),
                fault_max_num=self.sim_cfg['z_shift'].get('fault_max_num',0),
                fault_std=self.sim_cfg['z_shift'].get('fault_std',0.0),
            )
        if random.random()<self.sim_cfg['xy_shift']['apply_prob']:
            well_data=self.xy_shift(
                well_data,
                shift_bound=self.sim_cfg['xy_shift']['shift_bound'],
            )
        if random.random()<self.sim_cfg.get('switch_typewell',{}).get('apply_prob',0.0):
            well_data=self.switch_typewell(
                well_data,
                noise_mode=self.sim_cfg['switch_typewell'].get('noise_mode','copy'),
                same_nan=self.sim_cfg['switch_typewell'].get('same_nan',True),
            )
        typewell_gr_jitter_cfg = self.aug_cfg.get('typewell_gr_jitter', {})
        if random.random()<float(typewell_gr_jitter_cfg.get('apply_prob',0.0)):
            well_data=self.typewell_gr_jitter(
                well_data,
                cfg=typewell_gr_jitter_cfg,
            )

        if random.random()<self.aug_cfg['reverse_path']['apply_prob']:
            well_data=self.reverse_path(
                well_data,
                head_TVT_bound=self.aug_cfg['reverse_path']['head_TVT_bound'],
                sim_head_range=self.aug_cfg['reverse_path']['sim_head_range'],
            )
        # start_point_shift stays disabled for the PF-channel path because it
        # changes the visible-prefix boundary used to build fixed split caches.
        if random.random()<self.aug_cfg['start_point_shift']['apply_prob']:
            well_data=self.start_point_shift(
                well_data,
                left_bound=self.aug_cfg['start_point_shift']['left_bound'],
                right_bound=self.aug_cfg['start_point_shift']['right_bound'],
                max_ratio=self.aug_cfg['start_point_shift']['max_ratio'],
            )
        if random.random()<self.aug_cfg['GR_noise_shift']['apply_prob']:
            well_data=self.GR_noise_shift(
                well_data,
                shift_range_ratio=self.aug_cfg['GR_noise_shift']['shift_range_ratio'],
                noise_mode=self.aug_cfg['GR_noise_shift'].get('noise_mode','copy'),
            )
        horizontal_gr_blur_cfg = self.aug_cfg.get("horizontal_gr_blur", {})
        horizontal_gr_blur_apply_prob = float(horizontal_gr_blur_cfg.get("apply_prob", 0.0))
        if horizontal_gr_blur_apply_prob > 0.0 and random.random() < horizontal_gr_blur_apply_prob:
            well_data = self.horizontal_gr_blur(
                well_data,
                flat_region_max_ratio=float(horizontal_gr_blur_cfg.get("flat_region_max_ratio", 0.5)),
            )
        gr_trf_cfg = self.aug_cfg.get("GR_trf", {})
        if random.random() < float(gr_trf_cfg.get("apply_prob", 0.0)):
            well_data = self.GR_trf(well_data, cfg=gr_trf_cfg)

        typewell_drift_cfg = self.aug_cfg.get("typewell_drift", {})
        if random.random() < float(typewell_drift_cfg.get("apply_prob", 0.0)):
            well_data = self.typewell_drift(well_data, cfg=typewell_drift_cfg)
            
        if random.random()<self.aug_cfg['MD_streching']['apply_prob']:
            well_data=self.MD_streching(
                well_data,
                strech_ratio=self.aug_cfg['MD_streching']['strech_ratio'],
                apply_to_MD=self.aug_cfg['MD_streching']['apply_to_MD'],
            )

        tail_cut_cfg = self.aug_cfg.get("tail_cut", {})
        if random.random() < float(tail_cut_cfg.get("apply_prob", 0.0)):
            well_data = self.tail_cut(
                well_data,
                max_cut_ratio=float(tail_cut_cfg.get("max_cut_ratio", 0.5)),
            )
            
        return well_data
    
def GR_calibration(h_df, v_df, typewell_power=1.0, blend_weight=0.7):
    """Blend a visible-prefix pseudo curve with a calibrated raw Typewell.

    The optional power changes raw Typewell contrast while preserving its mean
    inside the same ``TVT0 +/- 100 ft`` neighborhood used to build the pseudo
    curve. ``blend_weight`` is the pseudo-curve weight.
    """
    typewell_power = float(typewell_power)
    blend_weight = float(blend_weight)
    if not np.isfinite(typewell_power) or typewell_power <= 0.0:
        raise ValueError(
            f"typewell_power must be finite and positive, got {typewell_power}"
        )
    if not np.isfinite(blend_weight) or not 0.0 <= blend_weight <= 1.0:
        raise ValueError(
            f"blend_weight must be finite and in [0, 1], got {blend_weight}"
        )

    h_df = h_df.copy()
    v_df = v_df.copy()
    target_cond = h_df["TVT_input"].isna()
    seen_h_df = h_df[~target_cond].copy()

    tvt0 = seen_h_df["TVT_input"].iloc[-1]
    refer_cond = seen_h_df["TVT_input"].between(tvt0 - 100, tvt0 + 100)
    refer_cond &= ~seen_h_df["GR"].isna()
    refer_df = seen_h_df[refer_cond][["TVT_input", "GR"]]
    refer_df["TVT"] = (refer_df["TVT_input"] * 4).round(0) / 4
    refer_df = refer_df.groupby("TVT")["GR"].mean().reset_index().sort_values("TVT")

    raw_gr = v_df["GR"].copy()
    if typewell_power == 1.0:
        calibrated_raw_gr = raw_gr.copy()
    else:
        if raw_gr.dropna().le(0.0).any():
            raise ValueError("typewell_power requires positive Typewell GR")
        local_cond = v_df["TVT"].between(tvt0 - 100, tvt0 + 100) & raw_gr.notna()
        if not local_cond.any():
            raise ValueError("typewell power calibration window has no Typewell rows")
        powered_raw_gr = raw_gr.pow(typewell_power)
        local_scale = (
            raw_gr.loc[local_cond].mean()
            / powered_raw_gr.loc[local_cond].mean()
        )
        calibrated_raw_gr = local_scale * powered_raw_gr

    v_df["GR_cali"] = calibrated_raw_gr
    cond = v_df["TVT"].between(refer_df["TVT"].min(), refer_df["TVT"].max())
    v_df.loc[cond, "GR_cali"] = np.interp(
        v_df.loc[cond, "TVT"].values,
        refer_df["TVT"].values,
        refer_df["GR"].values,
        left=np.nan,
        right=np.nan,
    )
    if blend_weight == 0.7:
        # Keep the legacy default arithmetic bit-for-bit unchanged.
        raw_weight = 0.3
    else:
        raw_weight = 1.0 - blend_weight
    v_df.loc[cond, "GR_cali"] = (
        blend_weight * v_df.loc[cond, "GR_cali"]
        + raw_weight * calibrated_raw_gr.loc[cond]
    )
    nan_cond = v_df["GR_cali"].isna()
    v_df.loc[nan_cond, "GR_cali"] = calibrated_raw_gr.loc[nan_cond]
    return v_df["GR_cali"].values
        
class SeqUNetDataset(Dataset):
    def __init__(
        self,
        path,
        well_ids,
        cfg,
        training=True,
        simulation=False,
        pf_cache_dir=None,
        pf_sample_cache_dir=None,
        geo_prior=None,
    ):
        self.path = Path(path)
        self.well_ids = list(well_ids)
        self.cfg = cfg
        self.training = training
        self.simulation = simulation
        self.use_pf_cache = uses_pf_heatmap_channels(cfg)
        self.use_pf_sample_cache = bool(simulation) and uses_pf_sample_trend(cfg)
        self.pf_cache_dir = Path(pf_cache_dir) if pf_cache_dir is not None else None
        self.pf_sample_cache_dir = Path(pf_sample_cache_dir) if pf_sample_cache_dir is not None else None
        self.geo_prior = {} if geo_prior is None else {str(k): v for k, v in geo_prior.items()}
        if self.use_pf_cache and self.pf_cache_dir is None:
            raise ValueError("PF U-Net channels requested but SeqUNetDataset received no pf_cache_dir")
        if self.use_pf_sample_cache and self.pf_sample_cache_dir is None:
            raise ValueError("PF-sample-backed trend requested but SeqUNetDataset received no pf_sample_cache_dir")
        self.well_data = [self._load_well_data(well_id) for well_id in self.well_ids]
        self.sim = Simulator(self.well_data, sim_cfg=cfg.sim_cfg ,aug_cfg=cfg.aug_cfg, cfg=cfg) if simulation else None

    def _load_well_data(self, well_id):
        h_cols_needed = {"MD", "X", "Y", "Z", "GR", "TVT_input", "TVT",
                         'ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA'}
        h_df = pd.read_csv(self.path / f"{well_id}__horizontal_well.csv", usecols=lambda col: col in h_cols_needed)
        h_cols = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
        if "TVT" in h_df.columns:
            h_cols.append("TVT")
        horizontal = {col: h_df[col].to_numpy(dtype="float32", copy=True) for col in h_cols}
        horizontal["orig_index"] = np.arange(len(h_df), dtype=np.float32)
        if str(well_id) not in self.geo_prior:
            raise ValueError(f"{well_id}: missing fold-safe geo prior")
        geo_prior_item = self.geo_prior[str(well_id)]
        if isinstance(geo_prior_item, dict):
            geo_s_rel = np.asarray(geo_prior_item["S_rel_prior"], dtype=np.float32)
            geo_diag = {
                name: np.asarray(geo_prior_item[name], dtype=np.float32)
                for name in ("geo_nbr_std", "geo_nbr_dS_std", "geo_radial_extrap_score")
                if name in geo_prior_item
            }
        elif hasattr(geo_prior_item, "s_rel"):
            geo_s_rel = np.asarray(geo_prior_item.s_rel, dtype=np.float32)
            geo_diag = {
                "geo_nbr_std": np.asarray(geo_prior_item.geo_nbr_std, dtype=np.float32),
                "geo_nbr_dS_std": np.asarray(geo_prior_item.geo_nbr_dS_std, dtype=np.float32),
                "geo_radial_extrap_score": np.asarray(geo_prior_item.geo_radial_extrap_score, dtype=np.float32),
            }
        else:
            geo_s_rel = np.asarray(geo_prior_item, dtype=np.float32)
            geo_diag = {}
        if geo_s_rel.shape[0] != len(h_df):
            raise ValueError(f"{well_id}: geo prior length {geo_s_rel.shape[0]} != horizontal length {len(h_df)}")
        last_seen_idx = _last_seen_idx(horizontal, well_id)
        s0 = np.float32(horizontal["TVT_input"][last_seen_idx] + horizontal["Z"][last_seen_idx])
        horizontal["geo_s_rel_prior_abs"] = (s0 + geo_s_rel).astype("float32")
        for name, values in geo_diag.items():
            if values.shape[0] != len(h_df):
                raise ValueError(f"{well_id}: {name} length {values.shape[0]} != horizontal length {len(h_df)}")
            horizontal[name] = values.astype("float32", copy=False)
        # S_avg=[['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']].mean(axis=1) 
        # S_avg=(TVT+Z)+C and has better resolution
        if "ANCC" in h_df.columns:
            S_avg=h_df[['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']].ffill().bfill().mean(axis=1).values
            horizontal['S_avg']=S_avg.astype('float32')

        typewell_df = pd.read_csv(self.path / f"{well_id}__typewell.csv", usecols=["TVT", "GR"])
        gr_normalize_local_std = None
        if (
            _cfg_mode(self.cfg, "gr_normalize_std_mode", {"global", "local"}) == "local"
            and not _gr_interpolate_enabled(self.cfg)
        ):
            raw_order = np.argsort(typewell_df["TVT"].to_numpy(dtype="float32"))
            raw_typewell = {
                "TVT": typewell_df["TVT"].to_numpy(dtype="float32", copy=True)[raw_order],
                "GR": typewell_df["GR"].to_numpy(dtype="float32", copy=True)[raw_order],
            }
            gr_normalize_local_std = _prefix_direct_gr_rmse_after_downsample(horizontal, raw_typewell, self.cfg)
        ### GR calibration 
        if bool(getattr(self.cfg, "typewell_calibration_with_seen", False)):
            typewell_df["GR"] = GR_calibration(
                h_df,
                typewell_df,
                typewell_power=float(
                    getattr(self.cfg, "typewell_calibration_power", 1.0)
                ),
                blend_weight=float(
                    getattr(self.cfg, "typewell_calibration_blend_weight", 0.7)
                ),
            )
        order = np.argsort(typewell_df["TVT"].to_numpy(dtype="float32"))
        typewell = {
            "TVT": typewell_df["TVT"].to_numpy(dtype="float32", copy=True)[order],
            "GR": typewell_df["GR"].to_numpy(dtype="float32", copy=True)[order],
        }
        well_data = {
            "well_id": well_id,
            "horizontal": horizontal,
            "typewell": typewell,
        }
        if gr_normalize_local_std is not None:
            well_data["gr_normalize_local_std"] = gr_normalize_local_std
        if self.use_pf_cache:
            well_data["pf_cache"] = load_pf_heatmap_cache(self.pf_cache_dir, well_id)
        if self.use_pf_sample_cache:
            well_data["pf_sample_cache"] = load_pf_sample_cache(self.pf_sample_cache_dir, well_id)
        _, matched_gr_metrics = _make_static_candidate_sidecars(
            horizontal,
            typewell,
            well_data.get("pf_cache"),
            self.cfg,
        )
        if matched_gr_metrics:
            well_data["matched_gr_metrics"] = matched_gr_metrics
        return well_data

    def __len__(self):
        return len(self.well_ids)

    def __getitem__(self, idx):
        well_data = self.well_data[idx]
        if self.simulation:
            well_data = self.sim.process(well_data)
        item = make_unet_item(well_data, self.cfg, self.training)
        if self.training and self.simulation:
            mixup_prob = float(getattr(self.cfg, "mixup_prob", 0.0))
            mix_applied = False
            if mixup_prob > 0.0 and len(self.well_data) > 1 and random.random() < mixup_prob:
                # Mix only the common valid suffix bins; item-1 context and
                # metadata remain the outer sample identity.
                mix_idx = random.randrange(len(self.well_data) - 1)
                if mix_idx >= idx:
                    mix_idx += 1
                mix_well_data = self.sim.process(self.well_data[mix_idx])
                mix_item = make_unet_item(mix_well_data, self.cfg, self.training)
                item = apply_unet_common_mixup(item, mix_item, self.cfg)
                mix_applied = True
            cutmix_prob = float(getattr(self.cfg, "cutmix_prob", 0.0))
            if (not mix_applied) and cutmix_prob > 0.0 and len(self.well_data) > 1 and random.random() < cutmix_prob:
                # Cutmix transplants one contiguous suffix-bin span. The span
                # uses the same suffix-progress offset in both wells because
                # early and tail bins have different target/reliability priors.
                mix_idx = random.randrange(len(self.well_data) - 1)
                if mix_idx >= idx:
                    mix_idx += 1
                mix_well_data = self.sim.process(self.well_data[mix_idx])
                mix_item = make_unet_item(mix_well_data, self.cfg, self.training)
                item = apply_unet_common_cutmix(item, mix_item, self.cfg)
            item = apply_unet_item_augmentations(item, self.cfg)
        return item


def collate_seq_batch(batch):
    tensor_keys = [
        "geo_s_rel_prior",
        "unet_static",
        "typewell_aux",
        "z_rel",
        "bin_count",
        "z_diff",
        "target",
        "typewell_target_probs",
        "target_mask",
        "row_mask",
        "suffix_mask",
    ]
    out = {}
    for key in tensor_keys:
        arr = np.stack([item[key] for item in batch])
        if arr.dtype == bool:
            out[key] = torch.from_numpy(arr)
        else:
            out[key] = torch.from_numpy(arr.astype("float32"))
    aux_keys = sorted(batch[0]["aux_targets"])
    out["aux_targets"] = {
        key: torch.from_numpy(np.stack([item["aux_targets"][key] for item in batch]).astype("float32"))
        for key in aux_keys
    }
    out["meta"] = [item["meta"] for item in batch]
    return out
