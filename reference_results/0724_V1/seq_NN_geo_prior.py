#!/usr/bin/env python3
"""Fold-safe XY IDW geo prior for the sequence NN pipeline.

The production prior is the robust anisotropic ``idw_dS_xy_wls`` SOTA from
``xy_surface_interpolation_research.py``, narrowed to the sequence pipeline
contract. The older tangent ``idw_dS_xy`` mode remains available for ablation:

* support is built from target-bearing train wells only;
* train-fold labeling excludes the queried well's own suffix;
* validation/test labeling uses all fold/full train support respectively;
* the queried well's visible prefix may be used as same-well support;
* output is original-row ``S_rel_prior = S_prior - S0`` plus IDW neighbor
  diagnostics for every query row.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from seq_NN_cfg import GeoPriorConfig


_LEGACY_GEO_PRIOR_FIELDS = {
    "point_neighbor_metric": "euclidean",
    "anisotropy_angle_deg": 0.0,
    "anisotropy_ratio": 1.0,
    "query_upsample_mode": "repeat",
    "wls_normalize_dxy": False,
    "wls_robust_mode": "none",
    "wls_robust_scale": 0.0,
    "wls_robust_iters": 0,
}


POST_TRAIN_GEO_DIAGNOSTIC_NAMES = (
    "geo_nbr_distance",
    "geo_radial_extrap_score",
    "geo_nbr_path_alignment",
)


@dataclass(frozen=True)
class GeoWell:
    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tvt: np.ndarray | None
    tvt_input: np.ndarray
    suffix_start: int
    centroid_x: float
    centroid_y: float

    @property
    def has_target(self) -> bool:
        return self.tvt is not None

    @property
    def s(self) -> np.ndarray:
        if self.tvt is None:
            raise ValueError(f"{self.well_id}: target TVT is required for support/scoring")
        return self.tvt + self.z

    @property
    def anchor_idx(self) -> int:
        return int(self.suffix_start - 1)

    @property
    def s0(self) -> float:
        return float(self.tvt_input[self.anchor_idx] + self.z[self.anchor_idx])

    @property
    def tvt0(self) -> float:
        return float(self.tvt_input[self.anchor_idx])

    @property
    def suffix_len(self) -> int:
        return int(self.md.shape[0] - self.suffix_start)


@dataclass(frozen=True)
class Support:
    xy: np.ndarray
    y: np.ndarray
    path_dxy: np.ndarray


@dataclass(frozen=True)
class GeoPriorResult:
    s_rel: np.ndarray
    geo_nbr_std: np.ndarray
    geo_nbr_dS_std: np.ndarray
    geo_radial_extrap_score: np.ndarray
    geo_nbr_distance: np.ndarray
    geo_nbr_path_alignment: np.ndarray


def _cfg_value(cfg, name):
    if isinstance(cfg, dict):
        return cfg[name]
    return getattr(cfg, name)


def resolve_geo_prior_cfg(cfg) -> GeoPriorConfig:
    raw = _cfg_value(cfg, "geo_prior_cfg")
    if isinstance(raw, GeoPriorConfig):
        state = vars(raw)
        if "point_neighbor_metric" in state:
            return raw
        values = {
            field: state[field] if field in state else meta.default
            for field, meta in GeoPriorConfig.__dataclass_fields__.items()
        }
        values.update(_LEGACY_GEO_PRIOR_FIELDS)
        return GeoPriorConfig(**values)
    if isinstance(raw, dict):
        return GeoPriorConfig(**raw)
    values = {}
    for field, meta in GeoPriorConfig.__dataclass_fields__.items():
        values[field] = getattr(raw, field, meta.default)
    return GeoPriorConfig(**values)


def _is_wls_method(method: str) -> bool:
    return method == "idw_dS_xy_wls"


def _is_supported_method(method: str) -> bool:
    return method in {"idw_dS_xy", "idw_dS_xy_wls"}


def _anisotropy_transform_xy(xy: np.ndarray, angle_deg: float, ratio: float) -> np.ndarray:
    ratio = float(ratio)
    if ratio <= 0.0:
        raise ValueError(f"anisotropy_ratio must be positive, got {ratio}")
    arr = xy.astype(np.float64, copy=False)
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    along = arr[:, 0] * c + arr[:, 1] * s
    cross = -arr[:, 0] * s + arr[:, 1] * c
    return np.column_stack([along / ratio, cross])


def _point_metric_xy(xy: np.ndarray, cfg: GeoPriorConfig) -> np.ndarray:
    if cfg.point_neighbor_metric == "euclidean":
        return xy.astype(np.float64, copy=False)
    if cfg.point_neighbor_metric == "anisotropic":
        return _anisotropy_transform_xy(
            xy,
            angle_deg=float(cfg.anisotropy_angle_deg),
            ratio=float(cfg.anisotropy_ratio),
        )
    raise ValueError(f"unsupported point_neighbor_metric={cfg.point_neighbor_metric!r}")


def _read_horizontal(path: Path, well_id: str) -> GeoWell:
    dtypes = {name: "float32" for name in ("MD", "X", "Y", "Z", "TVT", "TVT_input")}
    df = pd.read_csv(
        path / f"{well_id}__horizontal_well.csv",
        usecols=lambda col: col in dtypes,
        dtype=dtypes,
    )
    tvt_input = df["TVT_input"].to_numpy(dtype=np.float32, copy=True)
    suffix_rows = np.flatnonzero(~np.isfinite(tvt_input))
    if suffix_rows.size == 0:
        raise ValueError(f"{well_id}: no TVT_input NaN suffix")
    suffix_start = int(suffix_rows[0])
    expected = np.arange(suffix_start, len(df), dtype=np.int64)
    if suffix_start == 0 or suffix_rows.size != expected.size or not np.array_equal(suffix_rows, expected):
        raise ValueError(f"{well_id}: TVT_input missing rows are not one contiguous anchored suffix")
    tvt = df["TVT"].to_numpy(dtype=np.float32, copy=True) if "TVT" in df.columns else None
    return GeoWell(
        well_id=str(well_id),
        md=df["MD"].to_numpy(dtype=np.float32, copy=True),
        x=df["X"].to_numpy(dtype=np.float32, copy=True),
        y=df["Y"].to_numpy(dtype=np.float32, copy=True),
        z=df["Z"].to_numpy(dtype=np.float32, copy=True),
        tvt=tvt,
        tvt_input=tvt_input,
        suffix_start=suffix_start,
        centroid_x=float(df["X"].mean()),
        centroid_y=float(df["Y"].mean()),
    )


def load_geo_wells(path: str | Path, well_ids) -> dict[str, GeoWell]:
    path = Path(path)
    return {str(well_id): _read_horizontal(path, str(well_id)) for well_id in well_ids}


def _mean_bins(values: np.ndarray, valid: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return np.empty((0, values.shape[1]), dtype=np.float32)
    bin_id = valid_idx // int(bin_size)
    n_bins = int(bin_id.max()) + 1
    counts = np.bincount(bin_id, minlength=n_bins).astype(np.float64)
    out = np.zeros((n_bins, values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        out[:, col] = np.bincount(bin_id, weights=values[valid_idx, col], minlength=n_bins)
    keep = counts > 0
    out[keep] /= counts[keep, None]
    return out[keep].astype(np.float32)


def _build_dsxy_support(
    well: GeoWell,
    row_mask: np.ndarray,
    bin_size: int,
    clip_ds_xy: float,
    filter_ds_thr: float,
    wls_normalize_dxy: bool,
    method: str,
    s_values: np.ndarray,
) -> Support:
    s = np.asarray(s_values, dtype=np.float32)
    if s.shape[0] != well.md.shape[0]:
        raise ValueError(f"{well.well_id}: S support length mismatch")
    ds = np.full(s.shape[0], np.nan, dtype=np.float32)
    dx = np.full(s.shape[0], np.nan, dtype=np.float32)
    dy = np.full(s.shape[0], np.nan, dtype=np.float32)
    ds[1:] = s[1:] - s[:-1]
    dx[1:] = well.x[1:] - well.x[:-1]
    dy[1:] = well.y[1:] - well.y[:-1]
    if filter_ds_thr > 0.0:
        row_mask = row_mask & (np.abs(ds) <= np.float32(filter_ds_thr))
    denom = dx * dx + dy * dy
    step_norm = np.sqrt(denom)
    path_dx = np.full_like(dx, np.nan)
    path_dy = np.full_like(dy, np.nan)
    np.divide(dx, step_norm, out=path_dx, where=denom > 1e-10)
    np.divide(dy, step_norm, out=path_dy, where=denom > 1e-10)
    gx = np.full_like(ds, np.nan)
    gy = np.full_like(ds, np.nan)
    np.divide(ds * dx, denom, out=gx, where=denom > 1e-10)
    np.divide(ds * dy, denom, out=gy, where=denom > 1e-10)
    if clip_ds_xy > 0.0:
        gx = np.clip(gx, -float(clip_ds_xy), float(clip_ds_xy))
        gy = np.clip(gy, -float(clip_ds_xy), float(clip_ds_xy))
    valid = (
        row_mask
        & np.r_[False, np.ones(s.shape[0] - 1, dtype=bool)]
        & np.isfinite(well.x)
        & np.isfinite(well.y)
        & np.isfinite(s)
        & np.isfinite(gx)
        & np.isfinite(gy)
        & (denom > 1e-10)
    )
    if _is_wls_method(method):
        if wls_normalize_dxy:
            eq_dx = path_dx
            eq_dy = path_dy
            eq_ds = np.full_like(ds, np.nan)
            np.divide(ds, step_norm, out=eq_ds, where=denom > 1e-10)
        else:
            eq_dx = dx
            eq_dy = dy
            eq_ds = ds
        values = np.column_stack(
            [well.x, well.y, eq_dx, eq_dy, eq_ds, gx, gy, path_dx, path_dy]
        ).astype(
            np.float32,
            copy=False,
        )
        y_dim = 5
    elif method == "idw_dS_xy":
        values = np.column_stack([well.x, well.y, gx, gy, path_dx, path_dy]).astype(
            np.float32,
            copy=False,
        )
        y_dim = 2
    else:
        raise ValueError(f"unsupported geo prior method={method!r}")
    merged = _mean_bins(
        values,
        valid,
        bin_size,
    )
    if merged.shape[0] == 0:
        return Support(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, y_dim), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    return Support(
        merged[:, :2],
        merged[:, 2 : 2 + y_dim],
        merged[:, 2 + y_dim : 4 + y_dim],
    )


def build_suffix_support(well: GeoWell, cfg: GeoPriorConfig) -> Support:
    if not _is_supported_method(cfg.method):
        raise ValueError(f"unsupported geo prior method={cfg.method!r}")
    if cfg.support_region != "suffix":
        raise ValueError(f"unsupported support_region={cfg.support_region!r}; expected 'suffix'")
    row_mask = np.zeros(well.md.shape[0], dtype=bool)
    row_mask[well.suffix_start :] = True
    return _build_dsxy_support(
        well,
        row_mask,
        int(cfg.train_bin_size),
        float(cfg.clip_ds_xy),
        float(cfg.filter_ds_thr),
        bool(cfg.wls_normalize_dxy),
        str(cfg.method),
        well.s,
    )


def build_prefix_support(well: GeoWell, cfg: GeoPriorConfig) -> Support:
    rows = int(cfg.prefix_support_rows)
    if rows <= 0:
        y_dim = 5 if _is_wls_method(str(cfg.method)) else 2
        return Support(
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, y_dim), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    start = max(0, well.suffix_start - rows)
    row_mask = np.zeros(well.md.shape[0], dtype=bool)
    row_mask[start : well.suffix_start] = True
    prefix_s = well.tvt_input + well.z
    return _build_dsxy_support(
        well,
        row_mask,
        int(cfg.prefix_bin_size),
        float(cfg.clip_ds_xy),
        float(cfg.filter_ds_thr),
        bool(cfg.wls_normalize_dxy),
        str(cfg.method),
        prefix_s,
    )


def _query_bin_xy(
    well: GeoWell,
    cfg: GeoPriorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_idx = np.arange(well.suffix_start, well.md.shape[0], dtype=np.int64)
    if raw_idx.size == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            raw_idx,
            np.empty((0, 2), dtype=np.float32),
        )
    rel = raw_idx - well.suffix_start
    bin_id = rel // int(cfg.query_bin_size)
    n_bins = int(bin_id.max()) + 1
    sums_x = np.bincount(bin_id, weights=well.x[raw_idx], minlength=n_bins)
    sums_y = np.bincount(bin_id, weights=well.y[raw_idx], minlength=n_bins)
    sums_pos = np.bincount(bin_id, weights=raw_idx, minlength=n_bins)
    prev_idx = raw_idx - 1
    sums_dx = np.bincount(bin_id, weights=well.x[raw_idx] - well.x[prev_idx], minlength=n_bins)
    sums_dy = np.bincount(bin_id, weights=well.y[raw_idx] - well.y[prev_idx], minlength=n_bins)
    counts = np.bincount(bin_id, minlength=n_bins).astype(np.float64)
    xy = np.column_stack([sums_x / counts, sums_y / counts]).astype(np.float32)
    bin_row_pos = (sums_pos / counts).astype(np.float32)
    dxy = np.column_stack([sums_dx / counts, sums_dy / counts]).astype(np.float32)
    return xy, bin_row_pos, raw_idx, dxy


def _upsample_bin_values(
    raw_idx: np.ndarray,
    bin_row_pos: np.ndarray,
    bin_values: np.ndarray,
    cfg: GeoPriorConfig,
) -> np.ndarray:
    if raw_idx.size == 0:
        if bin_values.ndim == 1:
            return np.empty(0, dtype=np.float32)
        return np.empty((0, bin_values.shape[1]), dtype=np.float32)
    values = np.asarray(bin_values)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if cfg.query_upsample_mode == "repeat":
        rel = raw_idx - raw_idx[0]
        bin_id = np.minimum(rel // int(cfg.query_bin_size), values.shape[0] - 1)
        out = values[bin_id].astype(np.float32, copy=False)
        return out[:, 0] if squeeze else out
    if cfg.query_upsample_mode != "linear":
        raise ValueError(f"unsupported query_upsample_mode={cfg.query_upsample_mode!r}")
    out = np.empty((raw_idx.shape[0], values.shape[1]), dtype=np.float32)
    raw_pos = raw_idx.astype(np.float32, copy=False)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(raw_pos, bin_row_pos, values[:, col]).astype(np.float32)
    return out[:, 0] if squeeze else out


def _solve_weighted_dS_xy_wls_once(
    weights: np.ndarray,
    dxy: np.ndarray,
    ds: np.ndarray,
    fallback_grad: np.ndarray,
    cfg: GeoPriorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = weights.astype(np.float64, copy=False)
    weights = weights / weights.sum(axis=1, keepdims=True)
    fallback = (fallback_grad * weights[:, :, None]).sum(axis=1)

    a00 = (weights * dxy[:, :, 0] * dxy[:, :, 0]).sum(axis=1)
    a01 = (weights * dxy[:, :, 0] * dxy[:, :, 1]).sum(axis=1)
    a11 = (weights * dxy[:, :, 1] * dxy[:, :, 1]).sum(axis=1)
    b0 = (weights * dxy[:, :, 0] * ds).sum(axis=1)
    b1 = (weights * dxy[:, :, 1] * ds).sum(axis=1)

    trace = a00 + a11
    ridge = np.maximum(trace * float(cfg.wls_ridge_frac), 1e-12)
    m00 = a00 + ridge
    m11 = a11 + ridge
    det = m00 * m11 - a01 * a01
    beta = fallback.copy()
    solvable = (trace > 1e-12) & (np.abs(det) > 1e-18)
    if solvable.any():
        beta[solvable, 0] = (m11[solvable] * b0[solvable] - a01[solvable] * b1[solvable]) / det[solvable]
        beta[solvable, 1] = (-a01[solvable] * b0[solvable] + m00[solvable] * b1[solvable]) / det[solvable]

    disc = np.sqrt(np.maximum((a00 - a11) * (a00 - a11) + 4.0 * a01 * a01, 0.0))
    eig_max = 0.5 * (trace + disc)
    eig_min = 0.5 * (trace - disc)
    eig_ratio = eig_min / np.maximum(eig_max, 1e-12)
    max_grad = float(cfg.wls_max_grad)
    if max_grad > 0.0:
        grad_ok = (np.abs(beta[:, 0]) <= max_grad) & (np.abs(beta[:, 1]) <= max_grad)
    else:
        grad_ok = np.ones(beta.shape[0], dtype=bool)
    use_wls = solvable & (eig_ratio >= float(cfg.wls_min_eig_ratio)) & grad_ok
    beta[~use_wls] = fallback[~use_wls]
    return beta, fallback, use_wls


def _robust_wls_factor(residual: np.ndarray, scale: float, mode: str) -> np.ndarray:
    if scale <= 0.0 or mode == "none":
        return np.ones_like(residual, dtype=np.float64)
    abs_residual = np.abs(residual)
    if mode == "huber":
        return np.minimum(1.0, scale / np.maximum(abs_residual, 1e-12))
    if mode == "tukey":
        u = abs_residual / scale
        factor = np.zeros_like(residual, dtype=np.float64)
        keep = u < 1.0
        factor[keep] = np.square(1.0 - np.square(u[keep]))
        return factor
    raise ValueError(f"unsupported wls_robust_mode={mode!r}")


def _solve_weighted_dS_xy_wls(
    weights: np.ndarray,
    dxy: np.ndarray,
    ds: np.ndarray,
    fallback_grad: np.ndarray,
    cfg: GeoPriorConfig,
) -> np.ndarray:
    weights = weights.astype(np.float64, copy=False)
    weights = weights / weights.sum(axis=1, keepdims=True)
    beta, _, _ = _solve_weighted_dS_xy_wls_once(weights, dxy, ds, fallback_grad, cfg)
    robust_mode = str(cfg.wls_robust_mode)
    robust_scale = float(cfg.wls_robust_scale)
    robust_iters = int(cfg.wls_robust_iters)
    if robust_mode == "none" or robust_scale <= 0.0 or robust_iters <= 0:
        return beta

    for _ in range(robust_iters):
        pred_ds = beta[:, None, 0] * dxy[:, :, 0] + beta[:, None, 1] * dxy[:, :, 1]
        robust_weights = weights * _robust_wls_factor(ds - pred_ds, robust_scale, robust_mode)
        empty = robust_weights.sum(axis=1) <= 1e-300
        if empty.any():
            robust_weights[empty] = weights[empty]
        robust_weights /= robust_weights.sum(axis=1, keepdims=True)
        beta, _, _ = _solve_weighted_dS_xy_wls_once(
            robust_weights,
            dxy,
            ds,
            fallback_grad,
            cfg,
        )
    return beta


def _predict_idw(
    train_xy: np.ndarray,
    train_y: np.ndarray,
    train_path_dxy: np.ndarray,
    query_xy: np.ndarray,
    query_dxy: np.ndarray,
    cfg: GeoPriorConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if query_xy.shape[0] == 0:
        empty = np.empty(0, dtype=np.float32)
        return np.empty((0, 2), dtype=np.float32), {
            "geo_nbr_std": empty,
            "geo_nbr_dS_std": empty,
            "geo_radial_extrap_score": empty,
            "geo_nbr_distance": empty,
            "geo_nbr_path_alignment": empty,
        }
    if train_xy.shape[0] == 0:
        raise ValueError("geo prior support has no points")
    k_eff = min(int(cfg.point_neighbors), train_xy.shape[0])
    train_metric_xy = _point_metric_xy(train_xy, cfg)
    query_metric_xy = _point_metric_xy(query_xy, cfg)
    tree = cKDTree(train_metric_xy)
    dist, idx = tree.query(query_metric_xy, k=k_eff, workers=-1)
    if k_eff == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = 1.0 / np.maximum(dist, float(cfg.distance_eps)) ** float(cfg.idw_power)
    exact = dist <= float(cfg.distance_eps)
    if exact.any():
        weights[exact] = 1e18
    weights /= weights.sum(axis=1, keepdims=True)
    neighbor_xy = train_xy[idx].astype(np.float64, copy=False)
    neighbor_y = train_y[idx].astype(np.float64, copy=False)
    neighbor_path_dxy = train_path_dxy[idx].astype(np.float64, copy=False)
    if _is_wls_method(str(cfg.method)):
        if neighbor_y.shape[2] != 5:
            raise ValueError(f"{cfg.method}: expected 5 support columns, got {neighbor_y.shape[2]}")
        neighbor_dxy = neighbor_y[:, :, :2]
        neighbor_ds = neighbor_y[:, :, 2]
        neighbor_grad = neighbor_y[:, :, 3:5]
        mean_grad = _solve_weighted_dS_xy_wls(weights, neighbor_dxy, neighbor_ds, neighbor_grad, cfg)
    else:
        if neighbor_y.shape[2] != 2:
            raise ValueError(f"{cfg.method}: expected 2 support columns, got {neighbor_y.shape[2]}")
        neighbor_grad = neighbor_y
        mean_grad = (neighbor_grad * weights[:, :, None]).sum(axis=1)
    mean_xy = (neighbor_xy * weights[:, :, None]).sum(axis=1)
    query_neighbor_offset = query_xy.astype(np.float64, copy=False)[:, None, :] - neighbor_xy
    nbr_distance = np.sqrt(
        (weights * np.sum(query_neighbor_offset * query_neighbor_offset, axis=2)).sum(axis=1)
    )
    centered_xy = neighbor_xy - mean_xy[:, None, :]
    nbr_var = (weights * np.sum(centered_xy * centered_xy, axis=2)).sum(axis=1)
    query_offset = query_xy.astype(np.float64, copy=False) - mean_xy
    query_center_dist = np.sqrt(np.sum(query_offset * query_offset, axis=1))
    nbr_std = np.sqrt(np.maximum(nbr_var, 0.0))
    radial_score = query_center_dist / np.maximum(nbr_std, float(cfg.distance_eps))
    query_path_dxy = query_dxy.astype(np.float64, copy=False)
    path_dot = np.sum(neighbor_path_dxy * query_path_dxy[:, None, :], axis=2)
    path_norm = np.linalg.norm(neighbor_path_dxy, axis=2)
    path_norm *= np.linalg.norm(query_path_dxy, axis=1)[:, None]
    # Opposite drilling directions describe the same local path axis.
    path_cosine = np.abs(path_dot) / np.maximum(path_norm, float(cfg.distance_eps))
    path_alignment = (weights * np.clip(path_cosine, 0.0, 1.0)).sum(axis=1)
    # Use the same IDW neighbor weights as the gradient average. This measures
    # disagreement in each neighbor's implied dS for the query-bin trajectory step.
    neighbor_implied_ds = (
        neighbor_grad[:, :, 0] * query_dxy[:, None, 0]
        + neighbor_grad[:, :, 1] * query_dxy[:, None, 1]
    )
    mean_ds = (neighbor_implied_ds * weights).sum(axis=1)
    ds_var = (weights * np.square(neighbor_implied_ds - mean_ds[:, None])).sum(axis=1)
    diagnostics = {
        "geo_nbr_std": nbr_std.astype(np.float32),
        "geo_nbr_dS_std": np.sqrt(np.maximum(ds_var, 0.0)).astype(np.float32),
        "geo_radial_extrap_score": radial_score.astype(np.float32),
        "geo_nbr_distance": nbr_distance.astype(np.float32),
        "geo_nbr_path_alignment": path_alignment.astype(np.float32),
    }
    return mean_grad.astype(np.float32), diagnostics


def _concat_support(
    support_cache: dict[str, Support],
    selected_wells: np.ndarray,
    query_xy: np.ndarray,
    cfg: GeoPriorConfig,
    extra_support: Support | None,
) -> Support:
    xy_parts = []
    y_parts = []
    path_dxy_parts = []
    if extra_support is not None and extra_support.xy.shape[0] > 0:
        xy_parts.append(extra_support.xy)
        y_parts.append(extra_support.y)
        path_dxy_parts.append(extra_support.path_dxy)
    for well_id in selected_wells:
        support = support_cache[str(well_id)]
        if support.xy.shape[0] == 0:
            continue
        xy_parts.append(support.xy)
        y_parts.append(support.y)
        path_dxy_parts.append(support.path_dxy)
    if not xy_parts:
        raise ValueError("selected geo-prior neighbor wells have no support points")
    xy = np.vstack(xy_parts).astype(np.float32, copy=False)
    y = np.vstack(y_parts).astype(np.float32, copy=False)
    path_dxy = np.vstack(path_dxy_parts).astype(np.float32, copy=False)
    max_points = int(cfg.max_train_points)
    if max_points > 0 and xy.shape[0] > max_points:
        qtree = cKDTree(query_xy.astype(np.float64, copy=False))
        dist, _ = qtree.query(xy.astype(np.float64, copy=False), k=1, workers=-1)
        keep = np.argpartition(dist, max_points - 1)[:max_points]
        keep.sort()
        xy = xy[keep]
        y = y[keep]
        path_dxy = path_dxy[keep]
    return Support(xy, y, path_dxy)


def _build_prior_for_query_well(
    query_well: GeoWell,
    support_cache: dict[str, Support],
    support_wells: list[str],
    support_centroids: np.ndarray,
    cfg: GeoPriorConfig,
) -> GeoPriorResult:
    out = np.full(query_well.md.shape[0], np.nan, dtype=np.float32)
    geo_nbr_std = np.full(query_well.md.shape[0], np.nan, dtype=np.float32)
    geo_nbr_dS_std = np.full(query_well.md.shape[0], np.nan, dtype=np.float32)
    geo_radial_extrap_score = np.full(query_well.md.shape[0], np.nan, dtype=np.float32)
    geo_nbr_distance = np.full(query_well.md.shape[0], np.nan, dtype=np.float32)
    geo_nbr_path_alignment = np.full(query_well.md.shape[0], np.nan, dtype=np.float32)
    prefix_s = query_well.tvt_input[: query_well.suffix_start] + query_well.z[: query_well.suffix_start]
    out[: query_well.suffix_start] = (prefix_s - np.float32(query_well.s0)).astype(np.float32)
    geo_nbr_std[: query_well.suffix_start] = 0.0
    geo_nbr_dS_std[: query_well.suffix_start] = 0.0
    geo_radial_extrap_score[: query_well.suffix_start] = 0.0
    geo_nbr_distance[: query_well.suffix_start] = 0.0
    geo_nbr_path_alignment[: query_well.suffix_start] = 0.0
    query_xy, bin_row_pos, raw_idx, query_dxy = _query_bin_xy(query_well, cfg)
    if raw_idx.size == 0:
        return GeoPriorResult(
            out,
            geo_nbr_std,
            geo_nbr_dS_std,
            geo_radial_extrap_score,
            geo_nbr_distance,
            geo_nbr_path_alignment,
        )
    k_paths = min(int(cfg.neighbor_wells), len(support_wells))
    if k_paths <= 0:
        raise ValueError("geo prior needs at least one support well")
    path_tree = cKDTree(support_centroids.astype(np.float64, copy=False))
    _, local_idx = path_tree.query(
        np.asarray([[query_well.centroid_x, query_well.centroid_y]], dtype=np.float64),
        k=k_paths,
    )
    local_idx = np.asarray(local_idx).reshape(-1)
    selected_wells = np.asarray(support_wells, dtype=str)[local_idx]
    own_prefix_support = build_prefix_support(query_well, cfg)
    support = _concat_support(
        support_cache,
        selected_wells,
        query_xy,
        cfg,
        extra_support=own_prefix_support,
    )
    bin_grad, bin_diagnostics = _predict_idw(
        support.xy,
        support.y,
        support.path_dxy,
        query_xy,
        query_dxy,
        cfg,
    )
    raw_grad = _upsample_bin_values(raw_idx, bin_row_pos, bin_grad, cfg)
    geo_nbr_std[raw_idx] = _upsample_bin_values(
        raw_idx,
        bin_row_pos,
        bin_diagnostics["geo_nbr_std"],
        cfg,
    )
    geo_nbr_dS_std[raw_idx] = _upsample_bin_values(
        raw_idx,
        bin_row_pos,
        bin_diagnostics["geo_nbr_dS_std"],
        cfg,
    )
    geo_radial_extrap_score[raw_idx] = _upsample_bin_values(
        raw_idx,
        bin_row_pos,
        bin_diagnostics["geo_radial_extrap_score"],
        cfg,
    )
    geo_nbr_distance[raw_idx] = _upsample_bin_values(
        raw_idx,
        bin_row_pos,
        bin_diagnostics["geo_nbr_distance"],
        cfg,
    )
    geo_nbr_path_alignment[raw_idx] = _upsample_bin_values(
        raw_idx,
        bin_row_pos,
        bin_diagnostics["geo_nbr_path_alignment"],
        cfg,
    )
    prev_idx = raw_idx - 1
    ds = raw_grad[:, 0] * (query_well.x[raw_idx] - query_well.x[prev_idx])
    ds += raw_grad[:, 1] * (query_well.y[raw_idx] - query_well.y[prev_idx])
    out[raw_idx] = np.cumsum(ds, dtype=np.float64).astype(np.float32)
    return GeoPriorResult(
        out,
        geo_nbr_std,
        geo_nbr_dS_std,
        geo_radial_extrap_score,
        geo_nbr_distance,
        geo_nbr_path_alignment,
    )


def _prior_s_rel_array(value) -> np.ndarray:
    if isinstance(value, GeoPriorResult):
        return value.s_rel
    if isinstance(value, dict):
        return np.asarray(value["S_rel_prior"], dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _score_prior(wells: dict[str, GeoWell], prior: dict[str, GeoPriorResult], well_ids: list[str]) -> float | None:
    sse = 0.0
    n = 0
    for well_id in well_ids:
        well = wells[str(well_id)]
        if well.tvt is None:
            return None
        pred_s = well.s0 + _prior_s_rel_array(prior[str(well_id)])[well.suffix_start :]
        err = pred_s.astype(np.float64) - well.s[well.suffix_start :].astype(np.float64)
        sse += float(np.dot(err, err))
        n += int(err.size)
    if n == 0:
        return None
    return float(np.sqrt(sse / n))


def make_geo_prior_for_wells(
    support_path: str | Path,
    query_path: str | Path,
    support_wells,
    query_wells,
    cfg: GeoPriorConfig,
    *,
    exclude_query_from_support: bool,
) -> tuple[dict[str, GeoPriorResult], dict]:
    support_wells = [str(well_id) for well_id in support_wells]
    query_wells = [str(well_id) for well_id in query_wells]
    support_data = load_geo_wells(support_path, support_wells)
    same_path = Path(support_path).resolve() == Path(query_path).resolve()
    if same_path:
        missing_query_wells = [well_id for well_id in query_wells if well_id not in support_data]
        if missing_query_wells:
            support_data.update(load_geo_wells(query_path, missing_query_wells))
        query_data = {well_id: support_data[well_id] for well_id in query_wells}
    else:
        query_data = load_geo_wells(query_path, query_wells)

    support_cache = {well_id: build_suffix_support(support_data[well_id], cfg) for well_id in support_wells}
    support_centroids_all = {
        well_id: (support_data[well_id].centroid_x, support_data[well_id].centroid_y)
        for well_id in support_wells
    }

    prior = {}
    started = time.perf_counter()
    for query_well_id in query_wells:
        if exclude_query_from_support:
            active_support_wells = [well_id for well_id in support_wells if well_id != query_well_id]
        else:
            active_support_wells = support_wells
        active_centroids = np.asarray([support_centroids_all[well_id] for well_id in active_support_wells], dtype=np.float64)
        prior[query_well_id] = _build_prior_for_query_well(
            query_data[query_well_id],
            support_cache,
            active_support_wells,
            active_centroids,
            cfg,
        )
    elapsed = time.perf_counter() - started
    rmse = _score_prior(query_data, prior, query_wells)
    summary = {
        "method": cfg.method,
        "config": cfg.to_dict(),
        "support_path": str(support_path),
        "query_path": str(query_path),
        "support_wells": len(support_wells),
        "query_wells": len(query_wells),
        "exclude_query_from_support": bool(exclude_query_from_support),
        "elapsed_sec": float(elapsed),
        "rmse": None if rmse is None else float(rmse),
    }
    return prior, summary
