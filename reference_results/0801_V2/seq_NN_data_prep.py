import hashlib
import json
import math
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from numba import njit
except ImportError:
    njit = None

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None


PF_HEATMAP_CHANNELS = {
    "pf_prob",
    "pf_particle_density_prob",
    "pf_mean_abs_diff",
    "pf_prob_ffbsi",
    "pf_mean_abs_diff_ffbsi",
    "pf_tvt",
    "pf_tvt_ffbsi",
    "pf_tvt_matched_gr",
    "pf_tvt_ffbsi_matched_gr",
    "pf_tvt_matched_gr_rmse",
    "pf_tvt_matched_gr_corr",
    "pf_tvt_ffbsi_matched_gr_rmse",
    "pf_tvt_ffbsi_matched_gr_corr",
    "pf_entropy_ffbsi",
    "pf_ess_frac_ffbsi",
    "pf_max_prob_ffbsi",
    "pf_anchor_mass_5",
    "pf_anchor_mass_10",
    "pf_anchor_mass_20",
    "pf_geo_abs_diff",
    "pf_anchor_abs_diff",
    "pf_filtered_ffbsi_abs_diff",
}


PF_PROFILE_CFG_KEY_MAP = {
    "PF_heatmap_lik_scale": "lik_scale",
    "PF_heatmap_ref_grid_step": "ref_grid_step",
    "PF_heatmap_query_gr_mode": "query_gr_mode",
    "PF_heatmap_seen_blend_weight": "seen_blend_weight",
    "PF_heatmap_typewell_gr_calibration": "typewell_gr_calibration",
    "PF_heatmap_raw_ref_likelihood_weight": "raw_ref_likelihood_weight",
    "PF_heatmap_gr_ambiguity_power": "gr_ambiguity_power",
    "PF_heatmap_gr_ambiguity_min_power": "gr_ambiguity_min_power",
    "PF_heatmap_gr_ambiguity_contrast": "gr_ambiguity_contrast",
    "PF_heatmap_gr_ambiguity_ref_mode": "gr_ambiguity_ref_mode",
    "PF_heatmap_gr_information_power": "gr_information_power",
    "PF_heatmap_gr_information_center": "gr_information_center",
    "PF_heatmap_gr_information_min_multiplier": "gr_information_min_multiplier",
    "PF_heatmap_gr_information_max_multiplier": "gr_information_max_multiplier",
    "PF_heatmap_gr_information_slope_weight": "gr_information_slope_weight",
    "PF_heatmap_gr_information_ref_mode": "gr_information_ref_mode",
    "PF_heatmap_gr_shape_power": "gr_shape_power",
    "PF_heatmap_gr_shape_mode": "gr_shape_mode",
    "PF_heatmap_gr_shape_window": "gr_shape_window",
    "PF_heatmap_gr_shape_min_points": "gr_shape_min_points",
    "PF_heatmap_gr_shape_sigma_floor": "gr_shape_sigma_floor",
    "PF_heatmap_gr_shape_ref_mode": "gr_shape_ref_mode",
    "PF_heatmap_missing_likelihood_power": "missing_likelihood_power",
    "PF_heatmap_missing_gap_decay": "missing_gap_decay",
    "PF_heatmap_missing_min_power": "missing_min_power",
    "PF_heatmap_outlier_prob": "outlier_prob",
    "PF_heatmap_outlier_likelihood": "outlier_likelihood",
    "PF_heatmap_dynamic_sigma_alpha": "dynamic_sigma_alpha",
    "PF_heatmap_dynamic_sigma_threshold": "dynamic_sigma_threshold",
    "PF_heatmap_dynamic_sigma_power": "dynamic_sigma_power",
    "PF_heatmap_dynamic_sigma_min": "dynamic_sigma_min",
    "PF_heatmap_dynamic_sigma_max": "dynamic_sigma_max",
    "PF_heatmap_seed_path_weight_mode": "seed_path_weight_mode",
    "PF_heatmap_seed_prob_weight_mode": "seed_prob_weight_mode",
    "PF_heatmap_seed_local_weight_mode": "seed_local_weight_mode",
    "PF_heatmap_seed_local_score_source": "seed_local_score_source",
    "PF_heatmap_seed_local_lik_scale": "seed_local_lik_scale",
    "PF_heatmap_seed_local_half_life_blocks": "seed_local_half_life_blocks",
    "PF_heatmap_seed_local_global_mix": "seed_local_global_mix",
    "PF_heatmap_seed_local_min_power": "seed_local_min_power",
    "PF_heatmap_seed_local_combine_mode": "seed_local_combine_mode",
    "PF_heatmap_seed_local_residual_alpha": "seed_local_residual_alpha",
    "PF_heatmap_seed_local_residual_clip": "seed_local_residual_clip",
    "PF_heatmap_state_z_weight": "pf_state_z_weight",
    "PF_heatmap_momentum": "pf_momentum",
    "PF_heatmap_rate_noise": "pf_rate_noise",
    "PF_heatmap_pos_noise": "pf_pos_noise",
    "PF_heatmap_resample_threshold": "pf_resample_threshold",
    "PF_heatmap_resample_obs_power_adapt": "pf_resample_obs_power_adapt",
    "PF_heatmap_resample_min_threshold": "pf_resample_min_threshold",
    "PF_heatmap_rough_pos": "pf_rough_pos",
    "PF_heatmap_rough_rate": "pf_rough_rate",
    "PF_heatmap_rescue_frac": "pf_rescue_frac",
    "PF_heatmap_rescue_pos_sd": "pf_rescue_pos_sd",
    "PF_heatmap_rescue_rate_sd": "pf_rescue_rate_sd",
    "PF_heatmap_init_pos_sd": "pf_init_pos_sd",
    "PF_heatmap_init_rate_sd": "pf_init_rate_sd",
    "PF_heatmap_rate_mean_weight": "pf_rate_mean_weight",
    "PF_heatmap_jump_prob": "pf_jump_prob",
    "PF_heatmap_jump_sd": "pf_jump_sd",
    "PF_heatmap_jump_rate_sd": "pf_jump_rate_sd",
    "PF_heatmap_missing_jump_boost": "pf_missing_jump_boost",
    "PF_heatmap_jump_tail_prob": "pf_jump_tail_prob",
    "PF_heatmap_jump_tail_sd": "pf_jump_tail_sd",
    "PF_heatmap_jump_tail_rate_sd": "pf_jump_tail_rate_sd",
    "PF_heatmap_jump_tail_dist": "pf_jump_tail_dist",
    "PF_heatmap_jump_tail_clip": "pf_jump_tail_clip",
    "PF_heatmap_jump_tail_missing_boost": "pf_jump_tail_missing_boost",
    "PF_heatmap_anchor_sigma": "pf_anchor_sigma",
    "PF_heatmap_anchor_power": "pf_anchor_power",
    "PF_heatmap_correlated_rate_alpha": "pf_correlated_rate_alpha",
    "PF_heatmap_correlated_pos_alpha": "pf_correlated_pos_alpha",
    "PF_heatmap_lookahead_power": "pf_lookahead_power",
    "PF_heatmap_lookahead_steps": "pf_lookahead_steps",
    "PF_heatmap_lookahead_decay": "pf_lookahead_decay",
    "PF_heatmap_lookahead_max_gap": "pf_lookahead_max_gap",
    "PF_heatmap_lookahead_delta_power": "pf_lookahead_delta_power",
    "PF_heatmap_stratified_init": "pf_stratified_init",
    "PF_heatmap_finite_run_power_boost": "pf_finite_run_power_boost",
    "PF_heatmap_finite_run_power_cap": "pf_finite_run_power_cap",
    "PF_heatmap_finite_run_power_decay": "pf_finite_run_power_decay",
    "PF_heatmap_finite_run_power_floor": "pf_finite_run_power_floor",
    "PF_heatmap_conf_obs_power_decay": "pf_conf_obs_power_decay",
    "PF_heatmap_conf_obs_power_floor": "pf_conf_obs_power_floor",
    "PF_heatmap_conf_obs_power_power": "pf_conf_obs_power_power",
    "PF_heatmap_missing_noise_scale": "pf_missing_noise_scale",
    "PF_heatmap_missing_jump_scale": "pf_missing_jump_scale",
    "PF_heatmap_ess_jump_boost": "pf_ess_jump_boost",
    "PF_heatmap_ess_jump_power": "pf_ess_jump_power",
    "PF_heatmap_surprise_jump_boost": "pf_surprise_jump_boost",
    "PF_heatmap_surprise_jump_threshold": "pf_surprise_jump_threshold",
    "PF_heatmap_surprise_jump_power": "pf_surprise_jump_power",
    "PF_heatmap_ess_rough_boost": "pf_ess_rough_boost",
    "PF_heatmap_ess_rough_power": "pf_ess_rough_power",
    "PF_heatmap_prob_temperature": "pf_prob_temperature",
    "PF_heatmap_ffbsi_mode": "pf_ffbsi_mode",
    "PF_heatmap_ffbsi_n_paths": "pf_ffbsi_n_paths",
    "PF_heatmap_ffbsi_fallback_mode": "pf_ffbsi_fallback_mode",
    "PF_heatmap_ffbsi_max_active_bins": "pf_ffbsi_max_active_bins",
    "PF_heatmap_ffbsi_transition_model": "pf_ffbsi_transition_model",
    "PF_heatmap_ffbsi_transition_scale": "pf_ffbsi_transition_scale",
    "PF_heatmap_ffbsi_pos_floor": "pf_ffbsi_pos_floor",
    "PF_heatmap_ffbsi_rate_floor": "pf_ffbsi_rate_floor",
}


PF_PROFILE_ALLOWED_CACHE_KEYS = set(PF_PROFILE_CFG_KEY_MAP.values())


def _profile_cache_key(key):
    key = str(key)
    if key in PF_PROFILE_CFG_KEY_MAP:
        return PF_PROFILE_CFG_KEY_MAP[key]
    return key


def _json_clean_profile_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_clean_profile_value(v) for v in value]
    if isinstance(value, list):
        return [_json_clean_profile_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_clean_profile_value(v) for k, v in value.items()}
    return value


def _normalize_profile_mixture_spec(raw_spec):
    if raw_spec is None or raw_spec == "":
        return []
    if isinstance(raw_spec, str):
        raw_spec = json.loads(raw_spec)
    if not isinstance(raw_spec, (list, tuple)):
        raise ValueError("PF_heatmap_profile_mixture_spec must be a list of profile dictionaries")

    profiles = []
    for idx, raw_profile in enumerate(raw_spec):
        if not isinstance(raw_profile, dict):
            raise ValueError("each PF profile must be a dictionary")
        name = str(raw_profile.get("name", f"profile{idx:02d}"))
        weight = float(raw_profile.get("weight", 1.0))
        if weight <= 0.0 or not np.isfinite(weight):
            raise ValueError(f"PF profile {name!r} has invalid weight={weight!r}")
        raw_overrides = raw_profile.get("overrides", {})
        if not isinstance(raw_overrides, dict):
            raise ValueError(f"PF profile {name!r} overrides must be a dictionary")
        overrides = {}
        for key, value in raw_overrides.items():
            cache_key = _profile_cache_key(key)
            if cache_key not in PF_PROFILE_ALLOWED_CACHE_KEYS:
                raise ValueError(f"PF profile {name!r} override key {key!r} is not supported")
            overrides[cache_key] = _json_clean_profile_value(value)
        profiles.append(
            {
                "name": name,
                "weight": weight,
                "overrides": dict(sorted(overrides.items())),
            }
        )

    total = sum(profile["weight"] for profile in profiles)
    if total <= 0.0:
        return []
    for profile in profiles:
        profile["weight"] = float(profile["weight"] / total)
    return profiles


def _profile_indices_for_seed_count(profiles, n_seeds, shuffle_seed=0):
    n_seeds = int(n_seeds)
    if n_seeds <= 0 or not profiles:
        return np.zeros(max(n_seeds, 0), dtype=np.int64)

    weights = np.asarray([float(profile["weight"]) for profile in profiles], dtype=np.float64)
    weights /= weights.sum()
    counts = np.zeros(len(profiles), dtype=np.int64)
    positive = np.flatnonzero(weights > 0.0)
    if n_seeds >= len(positive):
        counts[positive] = 1
        remaining = n_seeds - int(counts.sum())
        base = weights * remaining
    else:
        remaining = n_seeds
        base = weights * remaining

    add = np.floor(base).astype(np.int64)
    counts += add
    left = n_seeds - int(counts.sum())
    if left > 0:
        frac = base - np.floor(base)
        order = np.argsort(frac)[::-1]
        for idx in order[:left]:
            counts[idx] += 1

    # In very small seed budgets, the guarantee-one branch can over-allocate
    # after rounding edge cases. Trim from the lowest-prior profiles first.
    while int(counts.sum()) > n_seeds:
        removable = np.flatnonzero(counts > 0)
        idx = removable[np.argmin(weights[removable])]
        counts[idx] -= 1

    profile_indices = np.repeat(np.arange(len(profiles), dtype=np.int64), counts)
    if profile_indices.size != n_seeds:
        raise RuntimeError("PF profile allocation failed to match n_seeds")
    rng = np.random.default_rng(int(shuffle_seed) + 100003 * int(n_seeds))
    rng.shuffle(profile_indices)
    return profile_indices


def _apply_profile_to_cache_cfg(cache_cfg, profile):
    if not profile:
        return cache_cfg
    overrides = profile.get("overrides", {})
    if not overrides:
        return cache_cfg
    out = dict(cache_cfg)
    out.update(overrides)
    return out


def _interpolate_horizontal_gr(gr_series, fallback):
    gr = gr_series.astype(float).interpolate(limit_direction="both")
    return gr.fillna(float(fallback))


def _prefix_gr_for_pf_sigma(gr_series):
    # Match the strong reference PF: missing prefix GR inflates the likelihood
    # scale instead of being interpolated into an overconfident GR matcher.
    return gr_series.fillna(0.0).to_numpy(dtype=float)


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    """Conservative PF. Returns (predictions_array, total_log_likelihood)."""
    tw_s   = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    tw_mean_gr = float(np.nanmean(tw_gr)) if np.isfinite(tw_gr).any() else 0.0
    gr_interp = _interpolate_horizontal_gr(hw["GR"], tw_mean_gr)

    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0:
        return hw['TVT_input'].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last['TVT_input'])
    last_Z   = float(last['Z'])
    last_MD  = float(last['MD'])

    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(_prefix_gr_for_pf_sigma(kn['GR']) - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + 3.0 * rng.standard_normal(N)  # wider init spread helps wells with abrupt TVT shift at PS
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5

    md_v = ev['MD'].values.astype(float)
    z_v  = ev['Z'].values.astype(float)
    # Interpolate GR gaps before tracking; high-NaN wells should still advance
    # with a continuous GR observation stream, matching the existing data-prep
    # pattern used elsewhere in this sequence pipeline.
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw['TVT_input'].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos  = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=32, scale=5.0):
    """
    128-seed lik-weighted PF ensemble.
    More seeds give better coverage of the TVT exploration space.
    """
    preds = []
    liks  = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)

    liks   = np.array(liks)
    liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale)
    weights /= weights.sum()

    return (weights[:, None] * np.stack(preds, 0)).sum(0)


def uses_pf_heatmap_channels(cfg):
    static_channels = getattr(cfg, "unet_static_channels", getattr(cfg, "stage_unet_static_channels", ()))
    return any(name in PF_HEATMAP_CHANNELS for name in static_channels)


def _z_shift_sample_model(cfg):
    z_shift_cfg = getattr(cfg, "sim_cfg", {}).get("z_shift", {})
    return str(z_shift_cfg.get("sample_model", z_shift_cfg.get("trend_sample_model", "target")))


def uses_pf_sample_trend(cfg):
    return _z_shift_sample_model(cfg).lower() in {"pf_sample", "mixed"}


def pf_heatmap_split_dir(cfg, split):
    return Path(getattr(cfg, "PF_heatmap_cache_dir", Path("PF_cache"))) / split


def pf_sample_split_dir(cfg, split):
    cache_dir = getattr(cfg, "PF_sample_cache_dir", None)
    if cache_dir is None:
        cache_dir = Path(getattr(cfg, "PF_heatmap_cache_dir", Path("PF_cache"))) / "PF_sample"
    return Path(cache_dir) / split


def _pf_typewell_gr_calibration_mode(cfg):
    if hasattr(cfg, "PF_heatmap_typewell_gr_calibration"):
        return str(getattr(cfg, "PF_heatmap_typewell_gr_calibration"))
    apply_gr_calibration = bool(getattr(cfg, "PF_heatmap_apply_gr_calibration", True))
    return "seen_blend" if apply_gr_calibration else "none"


def pf_heatmap_cache_config(cfg):
    typewell_gr_calibration = _pf_typewell_gr_calibration_mode(cfg)
    profile_mixture_spec = _normalize_profile_mixture_spec(
        getattr(cfg, "PF_heatmap_profile_mixture_spec", None)
    )
    suffix_merge_likelihood_mode = str(getattr(cfg, "PF_heatmap_suffix_merge_likelihood_mode", "block"))
    suffix_merge_likelihood_mode_norm = suffix_merge_likelihood_mode.lower()
    suffix_merge_default_alpha = 1.0 if suffix_merge_likelihood_mode_norm == "block" else 0.5
    suffix_merge_default_adjust = False if suffix_merge_likelihood_mode_norm == "block" else True
    return {
        "version": int(getattr(cfg, "PF_heatmap_version", 1)),
        "n_particles": int(getattr(cfg, "PF_heatmap_n_particles", 500)),
        "n_seeds": int(getattr(cfg, "PF_heatmap_n_seeds", 16)),
        "base_seed": int(getattr(cfg, "PF_heatmap_base_seed", 202605)),
        "profile_mixture_mode": "profiles" if profile_mixture_spec else "fixed",
        "profile_mixture_spec": profile_mixture_spec,
        "lik_scale": float(getattr(cfg, "PF_heatmap_lik_scale", 5.0)),
        "axis_sigma": float(getattr(cfg, "PF_heatmap_axis_sigma", 0.0)),
        "ref_grid_step": float(getattr(cfg, "PF_heatmap_ref_grid_step", 0.2)),
        "query_gr_mode": str(getattr(cfg, "PF_heatmap_query_gr_mode", "interp")),
        "apply_gr_calibration": typewell_gr_calibration != "none",
        "typewell_gr_calibration": typewell_gr_calibration,
        "seen_blend_weight": float(getattr(cfg, "PF_heatmap_seen_blend_weight", 0.7)),
        "raw_ref_likelihood_weight": float(getattr(cfg, "PF_heatmap_raw_ref_likelihood_weight", 0.0)),
        "gr_ambiguity_power": float(getattr(cfg, "PF_heatmap_gr_ambiguity_power", 0.0)),
        "gr_ambiguity_min_power": float(getattr(cfg, "PF_heatmap_gr_ambiguity_min_power", 0.35)),
        "gr_ambiguity_contrast": float(getattr(cfg, "PF_heatmap_gr_ambiguity_contrast", 1.0)),
        "gr_ambiguity_ref_mode": str(getattr(cfg, "PF_heatmap_gr_ambiguity_ref_mode", "primary")),
        "gr_information_power": float(getattr(cfg, "PF_heatmap_gr_information_power", 0.0)),
        "gr_information_center": float(getattr(cfg, "PF_heatmap_gr_information_center", 0.75)),
        "gr_information_min_multiplier": float(getattr(cfg, "PF_heatmap_gr_information_min_multiplier", 0.75)),
        "gr_information_max_multiplier": float(getattr(cfg, "PF_heatmap_gr_information_max_multiplier", 1.25)),
        "gr_information_slope_weight": float(getattr(cfg, "PF_heatmap_gr_information_slope_weight", 0.0)),
        "gr_information_ref_mode": str(getattr(cfg, "PF_heatmap_gr_information_ref_mode", "primary")),
        "gr_shape_power": float(getattr(cfg, "PF_heatmap_gr_shape_power", 0.0)),
        "gr_shape_mode": str(getattr(cfg, "PF_heatmap_gr_shape_mode", "resid_corr")),
        "gr_shape_window": int(getattr(cfg, "PF_heatmap_gr_shape_window", 15)),
        "gr_shape_min_points": int(getattr(cfg, "PF_heatmap_gr_shape_min_points", 7)),
        "gr_shape_sigma_floor": float(getattr(cfg, "PF_heatmap_gr_shape_sigma_floor", 0.35)),
        "gr_shape_ref_mode": str(getattr(cfg, "PF_heatmap_gr_shape_ref_mode", "primary")),
        "missing_likelihood_power": float(getattr(cfg, "PF_heatmap_missing_likelihood_power", 1.0)),
        "missing_gap_decay": float(getattr(cfg, "PF_heatmap_missing_gap_decay", 0.0)),
        "missing_min_power": float(getattr(cfg, "PF_heatmap_missing_min_power", 0.05)),
        "outlier_prob": float(getattr(cfg, "PF_heatmap_outlier_prob", 0.0)),
        "outlier_likelihood": float(getattr(cfg, "PF_heatmap_outlier_likelihood", 0.05)),
        "dynamic_sigma_alpha": float(getattr(cfg, "PF_heatmap_dynamic_sigma_alpha", 0.0)),
        "dynamic_sigma_threshold": float(getattr(cfg, "PF_heatmap_dynamic_sigma_threshold", 1.25)),
        "dynamic_sigma_power": float(getattr(cfg, "PF_heatmap_dynamic_sigma_power", 1.0)),
        "dynamic_sigma_min": float(getattr(cfg, "PF_heatmap_dynamic_sigma_min", 0.85)),
        "dynamic_sigma_max": float(getattr(cfg, "PF_heatmap_dynamic_sigma_max", 2.0)),
        "seed_path_weight_mode": str(getattr(cfg, "PF_heatmap_seed_path_weight_mode", "likelihood")),
        "seed_prob_weight_mode": str(getattr(cfg, "PF_heatmap_seed_prob_weight_mode", "equal")),
        "seed_local_weight_mode": str(getattr(cfg, "PF_heatmap_seed_local_weight_mode", "off")),
        "seed_local_score_source": str(getattr(cfg, "PF_heatmap_seed_local_score_source", "pf_likelihood")),
        "seed_local_lik_scale": float(getattr(cfg, "PF_heatmap_seed_local_lik_scale", 1.0)),
        "seed_local_half_life_blocks": float(getattr(cfg, "PF_heatmap_seed_local_half_life_blocks", 8.0)),
        "seed_local_global_mix": float(getattr(cfg, "PF_heatmap_seed_local_global_mix", 0.0)),
        "seed_local_min_power": float(getattr(cfg, "PF_heatmap_seed_local_min_power", 1e-6)),
        "seed_local_combine_mode": str(getattr(cfg, "PF_heatmap_seed_local_combine_mode", "replace")),
        "seed_local_residual_alpha": float(getattr(cfg, "PF_heatmap_seed_local_residual_alpha", 0.0)),
        "seed_local_residual_clip": float(getattr(cfg, "PF_heatmap_seed_local_residual_clip", 5.0)),
        "pf_state_z_weight": float(getattr(cfg, "PF_heatmap_state_z_weight", 1.0)),
        "pf_momentum": float(getattr(cfg, "PF_heatmap_momentum", 0.998)),
        "pf_rate_noise": float(getattr(cfg, "PF_heatmap_rate_noise", 0.002)),
        "pf_pos_noise": float(getattr(cfg, "PF_heatmap_pos_noise", 0.005)),
        "pf_resample_threshold": float(getattr(cfg, "PF_heatmap_resample_threshold", 0.5)),
        "pf_resample_obs_power_adapt": float(getattr(cfg, "PF_heatmap_resample_obs_power_adapt", 0.0)),
        "pf_resample_min_threshold": float(getattr(cfg, "PF_heatmap_resample_min_threshold", 0.0)),
        "pf_rough_pos": float(getattr(cfg, "PF_heatmap_rough_pos", 0.1)),
        "pf_rough_rate": float(getattr(cfg, "PF_heatmap_rough_rate", 0.001)),
        "pf_rescue_frac": float(getattr(cfg, "PF_heatmap_rescue_frac", 0.0)),
        "pf_rescue_pos_sd": float(getattr(cfg, "PF_heatmap_rescue_pos_sd", 0.0)),
        "pf_rescue_rate_sd": float(getattr(cfg, "PF_heatmap_rescue_rate_sd", 0.0)),
        "pf_init_pos_sd": float(getattr(cfg, "PF_heatmap_init_pos_sd", 3.0)),
        "pf_init_rate_sd": float(getattr(cfg, "PF_heatmap_init_rate_sd", 0.01)),
        "pf_rate_mean_weight": float(getattr(cfg, "PF_heatmap_rate_mean_weight", 0.0)),
        "pf_jump_prob": float(getattr(cfg, "PF_heatmap_jump_prob", 0.0)),
        "pf_jump_sd": float(getattr(cfg, "PF_heatmap_jump_sd", 0.0)),
        "pf_jump_rate_sd": float(getattr(cfg, "PF_heatmap_jump_rate_sd", 0.0)),
        "pf_missing_jump_boost": float(getattr(cfg, "PF_heatmap_missing_jump_boost", 0.0)),
        "pf_jump_tail_prob": float(getattr(cfg, "PF_heatmap_jump_tail_prob", 0.0)),
        "pf_jump_tail_sd": float(getattr(cfg, "PF_heatmap_jump_tail_sd", 0.0)),
        "pf_jump_tail_rate_sd": float(getattr(cfg, "PF_heatmap_jump_tail_rate_sd", 0.0)),
        "pf_jump_tail_dist": int(getattr(cfg, "PF_heatmap_jump_tail_dist", 1)),
        "pf_jump_tail_clip": float(getattr(cfg, "PF_heatmap_jump_tail_clip", 0.0)),
        "pf_jump_tail_missing_boost": float(getattr(cfg, "PF_heatmap_jump_tail_missing_boost", 0.0)),
        "pf_anchor_sigma": float(getattr(cfg, "PF_heatmap_anchor_sigma", 0.0)),
        "pf_anchor_power": float(getattr(cfg, "PF_heatmap_anchor_power", 1.0)),
        "pf_correlated_rate_alpha": float(getattr(cfg, "PF_heatmap_correlated_rate_alpha", 0.0)),
        "pf_correlated_pos_alpha": float(getattr(cfg, "PF_heatmap_correlated_pos_alpha", 0.0)),
        "pf_lookahead_power": float(getattr(cfg, "PF_heatmap_lookahead_power", 0.0)),
        "pf_lookahead_steps": int(getattr(cfg, "PF_heatmap_lookahead_steps", 1)),
        "pf_lookahead_decay": float(getattr(cfg, "PF_heatmap_lookahead_decay", 0.5)),
        "pf_lookahead_max_gap": float(getattr(cfg, "PF_heatmap_lookahead_max_gap", 256.0)),
        "pf_lookahead_delta_power": float(getattr(cfg, "PF_heatmap_lookahead_delta_power", 0.0)),
        "pf_stratified_init": int(getattr(cfg, "PF_heatmap_stratified_init", 0)),
        "pf_finite_run_power_boost": float(getattr(cfg, "PF_heatmap_finite_run_power_boost", 0.0)),
        "pf_finite_run_power_cap": float(getattr(cfg, "PF_heatmap_finite_run_power_cap", 1.0)),
        "pf_finite_run_power_decay": float(getattr(cfg, "PF_heatmap_finite_run_power_decay", 0.0)),
        "pf_finite_run_power_floor": float(getattr(cfg, "PF_heatmap_finite_run_power_floor", 0.35)),
        "pf_conf_obs_power_decay": float(getattr(cfg, "PF_heatmap_conf_obs_power_decay", 0.0)),
        "pf_conf_obs_power_floor": float(getattr(cfg, "PF_heatmap_conf_obs_power_floor", 0.55)),
        "pf_conf_obs_power_power": float(getattr(cfg, "PF_heatmap_conf_obs_power_power", 1.0)),
        "pf_missing_noise_scale": float(getattr(cfg, "PF_heatmap_missing_noise_scale", 1.0)),
        "pf_missing_jump_scale": float(getattr(cfg, "PF_heatmap_missing_jump_scale", 1.0)),
        "pf_ess_jump_boost": float(getattr(cfg, "PF_heatmap_ess_jump_boost", 0.0)),
        "pf_ess_jump_power": float(getattr(cfg, "PF_heatmap_ess_jump_power", 1.0)),
        "pf_surprise_jump_boost": float(getattr(cfg, "PF_heatmap_surprise_jump_boost", 0.0)),
        "pf_surprise_jump_threshold": float(getattr(cfg, "PF_heatmap_surprise_jump_threshold", 1.5)),
        "pf_surprise_jump_power": float(getattr(cfg, "PF_heatmap_surprise_jump_power", 1.0)),
        "pf_ess_rough_boost": float(getattr(cfg, "PF_heatmap_ess_rough_boost", 0.0)),
        "pf_ess_rough_power": float(getattr(cfg, "PF_heatmap_ess_rough_power", 1.0)),
        "pf_prob_temperature": float(getattr(cfg, "PF_heatmap_prob_temperature", 1.0)),
        "pf_ffbsi_mode": str(getattr(cfg, "PF_heatmap_ffbsi_mode", "off")),
        "pf_ffbsi_n_paths": int(getattr(cfg, "PF_heatmap_ffbsi_n_paths", 0)),
        "pf_ffbsi_fallback_mode": str(getattr(cfg, "PF_heatmap_ffbsi_fallback_mode", "filtered")),
        "pf_ffbsi_max_active_bins": int(getattr(cfg, "PF_heatmap_ffbsi_max_active_bins", 512)),
        "pf_ffbsi_transition_model": str(getattr(cfg, "PF_heatmap_ffbsi_transition_model", "gaussian")),
        "pf_ffbsi_transition_scale": float(getattr(cfg, "PF_heatmap_ffbsi_transition_scale", 1.5)),
        "pf_ffbsi_pos_floor": float(getattr(cfg, "PF_heatmap_ffbsi_pos_floor", 0.35)),
        "pf_ffbsi_rate_floor": float(getattr(cfg, "PF_heatmap_ffbsi_rate_floor", 0.0025)),
        "suffix_merge_k": int(getattr(cfg, "PF_heatmap_suffix_merge_k", 1)),
        "suffix_merge_obs_power_alpha": float(getattr(cfg, "PF_heatmap_suffix_merge_obs_power_alpha", suffix_merge_default_alpha)),
        "suffix_merge_adjust_dynamics": bool(getattr(cfg, "PF_heatmap_suffix_merge_adjust_dynamics", suffix_merge_default_adjust)),
        "suffix_merge_likelihood_mode": suffix_merge_likelihood_mode,
        "suffix_merge_state_mode": str(getattr(cfg, "PF_heatmap_suffix_merge_state_mode", "start")),
        "typewell_window": float(cfg.typewell_window),
        "typewell_len": int(cfg.typewell_len),
        "downsample": int(cfg.downsample),
        "prefix_len": int(cfg.prefix_len),
        "target_len": int(cfg.target_len),
        "raw_len": int(cfg.raw_len),
        "num_bins": int(cfg.num_bins),
    }


def pf_heatmap_cache_digest(cache_cfg):
    payload = json.dumps(cache_cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def pf_sample_cache_config(cfg):
    cache_cfg = dict(pf_heatmap_cache_config(cfg))
    pf_heatmap_version = int(cache_cfg["version"])
    sample_count = int(getattr(cfg, "PF_sample_count", 10))
    sample_seeds = int(getattr(cfg, "PF_sample_seeds", 8))
    if sample_count <= 0:
        raise ValueError(f"PF_sample_count must be positive, got {sample_count}")
    if sample_seeds <= 0:
        raise ValueError(f"PF_sample_seeds must be positive, got {sample_seeds}")

    cache_cfg.update(
        {
            "cache_kind": "PF_sample",
            "version": int(getattr(cfg, "PF_sample_version", 1)),
            "pf_heatmap_version": pf_heatmap_version,
            "sample_count": sample_count,
            "sample_seeds": sample_seeds,
            "sample_base_seed": int(
                getattr(cfg, "PF_sample_base_seed", int(cache_cfg["base_seed"]) + 911093)
            ),
            "path_resolution": "row_suffix",
            "path_source": "filtered_pf_mean",
            "n_seeds": sample_seeds,
        }
    )
    return cache_cfg


def pf_sample_cache_digest(cache_cfg):
    return pf_heatmap_cache_digest(cache_cfg)


def pf_heatmap_file(split_dir, well_id):
    return Path(split_dir) / f"{well_id}.npz"


def pf_sample_file(split_dir, well_id):
    return Path(split_dir) / f"{well_id}.npz"


def load_pf_heatmap_cache(cache_dir, well_id):
    path = pf_heatmap_file(cache_dir, well_id)
    if not path.exists():
        raise FileNotFoundError(f"missing PF heatmap cache for {well_id}: {path}")
    with np.load(path) as data:
        # Original visible-prefix PF rows are painted from known TVT_input.
        # Sanitize them at load time before simulation/augmentation remaps.
        last_seen_idx = int(data["last_seen_idx"])
        pf_prob = data["pf_prob"].astype("float32")
        if "pf_particle_density_prob" in data:
            pf_particle_density_prob = data["pf_particle_density_prob"].astype("float32")
        else:
            pf_particle_density_prob = pf_prob.copy()
        pf_tvt_pred = data["pf_tvt_pred"].astype("float32")
        pf_tvt_rel_pred = data["pf_tvt_rel_pred"].astype("float32")
        if "pf_prob_ffbsi" in data:
            pf_prob_ffbsi = data["pf_prob_ffbsi"].astype("float32")
            pf_tvt_pred_ffbsi = data["pf_tvt_pred_ffbsi"].astype("float32")
            pf_tvt_rel_pred_ffbsi = data["pf_tvt_rel_pred_ffbsi"].astype("float32")
        else:
            pf_prob_ffbsi = pf_prob.copy()
            pf_tvt_pred_ffbsi = pf_tvt_pred.copy()
            pf_tvt_rel_pred_ffbsi = pf_tvt_rel_pred.copy()
        window_orig_index = data["window_orig_index"].astype("float32")
        original_prefix = (
            np.isfinite(window_orig_index)
            & (window_orig_index >= 0.0)
            & (window_orig_index <= np.float32(last_seen_idx))
        )
        pf_prob[original_prefix, :] = 0.0
        pf_particle_density_prob[original_prefix, :] = 0.0
        pf_tvt_pred[original_prefix] = np.nan
        pf_tvt_rel_pred[original_prefix] = np.nan
        pf_prob_ffbsi[original_prefix, :] = 0.0
        pf_tvt_pred_ffbsi[original_prefix] = np.nan
        pf_tvt_rel_pred_ffbsi[original_prefix] = np.nan
        return {
            "well_id": str(data["well_id"]),
            "cache_digest": str(data["cache_digest"]),
            "last_seen_idx": last_seen_idx,
            "tvt0": float(data["tvt0"]),
            "prefix_start": int(data["prefix_start"]),
            "suffix_len": int(data["suffix_len"]),
            "pf_prob": pf_prob,
            "pf_particle_density_prob": pf_particle_density_prob,
            "pf_tvt_pred": pf_tvt_pred,
            "pf_tvt_rel_pred": pf_tvt_rel_pred,
            "pf_prob_ffbsi": pf_prob_ffbsi,
            "pf_tvt_pred_ffbsi": pf_tvt_pred_ffbsi,
            "pf_tvt_rel_pred_ffbsi": pf_tvt_rel_pred_ffbsi,
            "window_tvt": data["window_tvt"].astype("float32"),
            "window_tvt_rel": data["window_tvt_rel"].astype("float32"),
            "window_has_tvt": data["window_has_tvt"].astype(bool),
            "window_orig_index": window_orig_index,
            "target_mask": data["target_mask"].astype(bool),
        }


def load_pf_sample_cache(cache_dir, well_id):
    path = pf_sample_file(cache_dir, well_id)
    if not path.exists():
        raise FileNotFoundError(f"missing PF sample cache for {well_id}: {path}")
    with np.load(path) as data:
        return {
            "well_id": str(data["well_id"]),
            "cache_digest": str(data["cache_digest"]),
            "last_seen_idx": int(data["last_seen_idx"]),
            "tvt0": float(data["tvt0"]),
            "suffix_len": int(data["suffix_len"]),
            "kept_suffix_len": int(data["kept_suffix_len"]),
            "pf_tvt_samples": data["pf_tvt_samples"].astype("float32"),
        }


def ensure_pf_heatmap_cache(split, data_path, well_ids, cfg, log=None):
    if not uses_pf_heatmap_channels(cfg):
        return None
    if njit is None:
        raise ImportError("numba is required for PF heatmap cache generation")

    split = str(split)
    data_path = Path(data_path)
    well_ids = list(well_ids)
    split_dir = pf_heatmap_split_dir(cfg, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    cache_cfg = pf_heatmap_cache_config(cfg)
    cache_digest = pf_heatmap_cache_digest(cache_cfg)
    config_path = split_dir / "config.json"
    requested_files = [pf_heatmap_file(split_dir, well_id) for well_id in well_ids]
    requested_well_count = len(well_ids)
    existing_file_count = len(list(split_dir.glob("*.npz")))

    existing_cfg = None
    if config_path.exists():
        with config_path.open("r") as f:
            existing_cfg = json.load(f)
    requested_files_present = all(path.exists() for path in requested_files)
    config_matches = (
        existing_cfg is not None
        and existing_cfg.get("cache_digest") == cache_digest
        and existing_cfg.get("cache_cfg") == cache_cfg
    )
    cache_ready = (
        config_matches
        and existing_file_count == requested_well_count
        and requested_files_present
    )
    if cache_ready:
        _log(log, f"PF heatmap cache reuse split={split}: {split_dir} wells={len(well_ids):,} digest={cache_digest}")
        return split_dir

    if existing_cfg is not None:
        if not config_matches:
            reason = "config changed"
        elif existing_file_count != requested_well_count:
            reason = f"well count changed (files={existing_file_count}, requested={requested_well_count})"
        elif not requested_files_present:
            reason = "requested well files missing"
        else:
            reason = "cache incomplete"
        _log(log, f"PF heatmap cache {reason} for split={split}; clearing {split_dir}")
        for path in split_dir.glob("*.npz"):
            path.unlink()
        summary_path = split_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        diag_dir = split_dir / "diagnostics"
        if diag_dir.exists():
            shutil.rmtree(diag_dir)
    elif existing_file_count > 0:
        _log(log, f"PF heatmap cache has no config for split={split}; clearing {split_dir}")
        for path in split_dir.glob("*.npz"):
            path.unlink()
        summary_path = split_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        diag_dir = split_dir / "diagnostics"
        if diag_dir.exists():
            shutil.rmtree(diag_dir)

    config_payload = {
        "cache_digest": cache_digest,
        "cache_cfg": cache_cfg,
    }
    with config_path.open("w") as f:
        json.dump(config_payload, f, indent=2, sort_keys=True)

    _warmup_pf_heatmap_numba()
    if split == "train" and len(well_ids) > 0:
        _benchmark_numba_vs_python(data_path, well_ids[: min(2, len(well_ids))], cfg, log)

    started = time.perf_counter()
    worker_args = [
        (
            data_path,
            well_id,
            split_dir,
            cache_cfg,
            cache_digest,
            split == "train",
        )
        for well_id in well_ids
    ]
    num_workers = min(int(getattr(cfg, "PF_heatmap_num_workers", 4)), len(worker_args))
    num_workers = max(num_workers, 1)
    _log(
        log,
        "generating PF heatmap cache "
        f"split={split} wells={len(well_ids):,} workers={num_workers} "
        f"N={cache_cfg['n_particles']} seeds={cache_cfg['n_seeds']} "
        f"shape=({cache_cfg['num_bins']},{cache_cfg['typewell_len']}) digest={cache_digest}",
    )
    if num_workers == 1:
        results = [
            _generate_pf_heatmap_cache_one(args)
            for args in tqdm(worker_args, desc=f"PF {split}", dynamic_ncols=True)
        ]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_generate_pf_heatmap_cache_one, worker_args),
                    total=len(worker_args),
                    desc=f"PF {split}",
                    dynamic_ncols=True,
                )
            )

    elapsed = time.perf_counter() - started
    _write_pf_cache_summary(split_dir, split, results, elapsed, cache_digest, cache_cfg, log)
    return split_dir


def ensure_pf_sample_cache(split, data_path, well_ids, cfg, log=None):
    if not uses_pf_sample_trend(cfg):
        return None
    if njit is None:
        raise ImportError("numba is required for PF sample cache generation")

    split = str(split)
    data_path = Path(data_path)
    well_ids = list(well_ids)
    split_dir = pf_sample_split_dir(cfg, split)
    split_dir.mkdir(parents=True, exist_ok=True)
    cache_cfg = pf_sample_cache_config(cfg)
    cache_digest = pf_sample_cache_digest(cache_cfg)
    config_path = split_dir / "config.json"
    requested_well_count = len(well_ids)
    existing_file_count = len(list(split_dir.glob("*.npz")))

    existing_cfg = None
    if config_path.exists():
        with config_path.open("r") as f:
            existing_cfg = json.load(f)

    config_matches = (
        existing_cfg is not None
        and existing_cfg.get("cache_digest") == cache_digest
        and existing_cfg.get("cache_cfg") == cache_cfg
    )
    requested_files = [pf_sample_file(split_dir, well_id) for well_id in well_ids]
    missing_wells = [well_id for well_id, path in zip(well_ids, requested_files) if not path.exists()]
    rewrite_config = existing_cfg is None or not config_matches
    if (
        config_matches
        and existing_file_count == requested_well_count
        and not missing_wells
    ):
        _log(
            log,
            f"PF sample cache reuse split={split}: {split_dir} wells={len(well_ids):,} digest={cache_digest}",
        )
        return split_dir

    if existing_cfg is not None and (
        not config_matches
        or existing_file_count != requested_well_count
    ):
        if not config_matches:
            reason = "config changed"
        else:
            reason = f"well count changed (files={existing_file_count}, requested={requested_well_count})"
        _log(log, f"PF sample cache {reason} for split={split}; clearing {split_dir}")
        for path in split_dir.glob("*.npz"):
            path.unlink()
        summary_path = split_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        missing_wells = well_ids
        rewrite_config = True
    elif existing_cfg is None and existing_file_count > 0:
        _log(log, f"PF sample cache has no config for split={split}; clearing {split_dir}")
        for path in split_dir.glob("*.npz"):
            path.unlink()
        summary_path = split_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        missing_wells = well_ids
        rewrite_config = True

    if rewrite_config:
        config_payload = {
            "cache_digest": cache_digest,
            "cache_cfg": cache_cfg,
        }
        with config_path.open("w") as f:
            json.dump(config_payload, f, indent=2, sort_keys=True)

    if not missing_wells:
        return split_dir

    _warmup_pf_heatmap_numba()
    started = time.perf_counter()
    worker_args = [
        (
            data_path,
            well_id,
            split_dir,
            cache_cfg,
            cache_digest,
            split == "train",
        )
        for well_id in missing_wells
    ]
    num_workers = min(int(getattr(cfg, "PF_heatmap_num_workers", 4)), len(worker_args))
    num_workers = max(num_workers, 1)
    _log(
        log,
        "generating PF sample cache "
        f"split={split} missing={len(missing_wells):,}/{len(well_ids):,} workers={num_workers} "
        f"N={cache_cfg['n_particles']} sample_count={cache_cfg['sample_count']} "
        f"sample_seeds={cache_cfg['sample_seeds']} digest={cache_digest}",
    )
    if num_workers == 1:
        results = [
            _generate_pf_sample_cache_one(args)
            for args in tqdm(worker_args, desc=f"PF sample {split}", dynamic_ncols=True)
        ]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_generate_pf_sample_cache_one, worker_args),
                    total=len(worker_args),
                    desc=f"PF sample {split}",
                    dynamic_ncols=True,
                )
            )

    elapsed = time.perf_counter() - started
    _write_pf_sample_cache_summary(split_dir, split, results, elapsed, cache_digest, cache_cfg, log)
    return split_dir


def _log(log, message):
    if log is None:
        print(message, flush=True)
    else:
        log(message)


def _write_pf_cache_summary(split_dir, split, results, elapsed, cache_digest, cache_cfg, log):
    tvt_rmse_values = np.asarray([r["tvt_rmse"] for r in results if np.isfinite(r["tvt_rmse"])], dtype=float)
    tvt_rmse_values_ffbsi = np.asarray([r["tvt_rmse_ffbsi"] for r in results if np.isfinite(r["tvt_rmse_ffbsi"])], dtype=float)
    row_counts = np.asarray([r["target_count"] for r in results], dtype=float)
    row_sum_min = np.asarray([r["prob_row_sum_min"] for r in results if np.isfinite(r["prob_row_sum_min"])], dtype=float)
    row_sum_max = np.asarray([r["prob_row_sum_max"] for r in results if np.isfinite(r["prob_row_sum_max"])], dtype=float)
    row_sum_min_ffbsi = np.asarray([r["prob_row_sum_min_ffbsi"] for r in results if np.isfinite(r["prob_row_sum_min_ffbsi"])], dtype=float)
    row_sum_max_ffbsi = np.asarray([r["prob_row_sum_max_ffbsi"] for r in results if np.isfinite(r["prob_row_sum_max_ffbsi"])], dtype=float)
    weighted_sse = float(sum(r["tvt_sse"] for r in results if np.isfinite(r["tvt_sse"])))
    weighted_count = float(sum(r["target_count"] for r in results))
    weighted_rmse = math.sqrt(weighted_sse / weighted_count) if weighted_count > 0 else math.nan
    weighted_sse_ffbsi = float(sum(r["tvt_sse_ffbsi"] for r in results if np.isfinite(r["tvt_sse_ffbsi"])))
    weighted_count_ffbsi = float(sum(r["target_count_ffbsi"] for r in results))
    weighted_rmse_ffbsi = math.sqrt(weighted_sse_ffbsi / weighted_count_ffbsi) if weighted_count_ffbsi > 0 else math.nan
    summary = {
        "split": split,
        "cache_digest": cache_digest,
        "cache_cfg": cache_cfg,
        "elapsed_sec": elapsed,
        "sec_per_well": elapsed / max(len(results), 1),
        "well_count": len(results),
        "row_weighted_tvt_rmse": weighted_rmse,
        "row_weighted_tvt_rmse_ffbsi": weighted_rmse_ffbsi,
        "target_rows": int(weighted_count),
        "target_rows_ffbsi": int(weighted_count_ffbsi),
        "well_tvt_rmse_mean": float(np.mean(tvt_rmse_values)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_median": float(np.median(tvt_rmse_values)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_q75": float(np.quantile(tvt_rmse_values, 0.75)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_q95": float(np.quantile(tvt_rmse_values, 0.95)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_q99": float(np.quantile(tvt_rmse_values, 0.99)) if tvt_rmse_values.size else math.nan,
        "well_tvt_rmse_mean_ffbsi": float(np.mean(tvt_rmse_values_ffbsi)) if tvt_rmse_values_ffbsi.size else math.nan,
        "well_tvt_rmse_median_ffbsi": float(np.median(tvt_rmse_values_ffbsi)) if tvt_rmse_values_ffbsi.size else math.nan,
        "well_tvt_rmse_q75_ffbsi": float(np.quantile(tvt_rmse_values_ffbsi, 0.75)) if tvt_rmse_values_ffbsi.size else math.nan,
        "well_tvt_rmse_q95_ffbsi": float(np.quantile(tvt_rmse_values_ffbsi, 0.95)) if tvt_rmse_values_ffbsi.size else math.nan,
        "well_tvt_rmse_q99_ffbsi": float(np.quantile(tvt_rmse_values_ffbsi, 0.99)) if tvt_rmse_values_ffbsi.size else math.nan,
        "target_row_count_mean": float(np.mean(row_counts)) if row_counts.size else math.nan,
        "prob_active_row_sum_min": float(np.min(row_sum_min)) if row_sum_min.size else math.nan,
        "prob_active_row_sum_max": float(np.max(row_sum_max)) if row_sum_max.size else math.nan,
        "prob_active_row_sum_min_ffbsi": float(np.min(row_sum_min_ffbsi)) if row_sum_min_ffbsi.size else math.nan,
        "prob_active_row_sum_max_ffbsi": float(np.max(row_sum_max_ffbsi)) if row_sum_max_ffbsi.size else math.nan,
    }
    with (Path(split_dir) / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    msg = (
        f"PF heatmap cache generated split={split}: wells={len(results):,}, "
        f"elapsed={elapsed:.2f}s, sec/well={elapsed / max(len(results), 1):.3f}"
    )
    if np.isfinite(weighted_rmse):
        ffbsi_rmse_text = f"{weighted_rmse_ffbsi:.4f}" if np.isfinite(weighted_rmse_ffbsi) else "nan"
        msg += (
            f", TVT_RMSE filtered={weighted_rmse:.4f}/ffbsi={ffbsi_rmse_text}, "
            f"well_rmse mean/median/q75/q95/q99="
            f"{summary['well_tvt_rmse_mean']:.4f}/"
            f"{summary['well_tvt_rmse_median']:.4f}/"
            f"{summary['well_tvt_rmse_q75']:.4f}/"
            f"{summary['well_tvt_rmse_q95']:.4f}/"
            f"{summary['well_tvt_rmse_q99']:.4f}, "
            f"prob_row_sum={summary['prob_active_row_sum_min']:.6f}.."
            f"{summary['prob_active_row_sum_max']:.6f}, "
            f"prob_row_sum_ffbsi={summary['prob_active_row_sum_min_ffbsi']:.6f}.."
            f"{summary['prob_active_row_sum_max_ffbsi']:.6f}"
        )
    _log(log, msg)


def _write_pf_sample_cache_summary(split_dir, split, results, elapsed, cache_digest, cache_cfg, log):
    sample_rmse_values = np.asarray(
        [r["sample_tvt_rmse_mean"] for r in results if np.isfinite(r["sample_tvt_rmse_mean"])],
        dtype=float,
    )
    suffix_counts = np.asarray([r["kept_suffix_len"] for r in results], dtype=float)
    summary = {
        "split": split,
        "cache_digest": cache_digest,
        "cache_cfg": cache_cfg,
        "elapsed_sec": elapsed,
        "sec_per_well": elapsed / max(len(results), 1),
        "well_count_generated": len(results),
        "sample_count": int(cache_cfg["sample_count"]),
        "sample_seeds": int(cache_cfg["sample_seeds"]),
        "suffix_len_mean": float(np.mean(suffix_counts)) if suffix_counts.size else math.nan,
        "sample_tvt_rmse_mean": float(np.mean(sample_rmse_values)) if sample_rmse_values.size else math.nan,
        "sample_tvt_rmse_median": float(np.median(sample_rmse_values)) if sample_rmse_values.size else math.nan,
    }
    with (Path(split_dir) / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    msg = (
        f"PF sample cache generated split={split}: wells={len(results):,}, "
        f"elapsed={elapsed:.2f}s, sec/well={elapsed / max(len(results), 1):.3f}"
    )
    if sample_rmse_values.size:
        msg += (
            f", sample_TVT_RMSE mean/median="
            f"{summary['sample_tvt_rmse_mean']:.4f}/{summary['sample_tvt_rmse_median']:.4f}"
        )
    _log(log, msg)


def _read_well_frames(data_path, well_id):
    h_cols = ["MD", "Z", "GR", "TVT_input"]
    h_path = Path(data_path) / f"{well_id}__horizontal_well.csv"
    h_df = pd.read_csv(h_path)
    h_df = h_df[[col for col in h_cols + ["TVT"] if col in h_df.columns]]
    tw_df = pd.read_csv(Path(data_path) / f"{well_id}__typewell.csv", usecols=["TVT", "GR"])
    return h_df, tw_df


def _gr_calibration_seen_blend(h_df, tw_df, blend_weight=0.7):
    """Production PF copy of the local seen-prefix Typewell GR blend."""
    target_cond = h_df["TVT_input"].isna()
    seen_h_df = h_df[~target_cond].copy()
    blend_weight = float(blend_weight)

    tvt0 = seen_h_df["TVT_input"].iloc[-1]
    refer_cond = seen_h_df["TVT_input"].between(tvt0 - 100, tvt0 + 100)
    refer_cond &= ~seen_h_df["GR"].isna()
    refer_df = seen_h_df[refer_cond][["TVT_input", "GR"]].copy()
    refer_df["TVT"] = (refer_df["TVT_input"] * 4).round(0) / 4
    refer_df = refer_df.groupby("TVT")["GR"].mean().reset_index().sort_values("TVT")

    out = tw_df["GR"].to_numpy(dtype=np.float64, copy=True)
    cond = tw_df["TVT"].between(refer_df["TVT"].min(), refer_df["TVT"].max()).to_numpy()
    if cond.any():
        interp_gr = np.interp(
            tw_df.loc[cond, "TVT"].to_numpy(dtype=np.float64),
            refer_df["TVT"].to_numpy(dtype=np.float64),
            refer_df["GR"].to_numpy(dtype=np.float64),
            left=np.nan,
            right=np.nan,
        )
        raw_gr = tw_df.loc[cond, "GR"].to_numpy(dtype=np.float64)
        blended = blend_weight * interp_gr + (1.0 - blend_weight) * raw_gr
        out[np.flatnonzero(cond)] = np.where(np.isfinite(blended), blended, raw_gr)
    return out


def _apply_typewell_gr_calibration(h_df, tw_df, cache_cfg):
    mode = str(cache_cfg.get("typewell_gr_calibration", "none"))
    if mode in {"none", "", "raw"}:
        return tw_df
    out = tw_df.copy()
    if mode == "seen_blend":
        out["GR"] = _gr_calibration_seen_blend(
            h_df,
            tw_df,
            blend_weight=float(cache_cfg.get("seen_blend_weight", 0.7)),
        )
        return out
    raise ValueError(f"unknown PF typewell_gr_calibration={mode!r}")


def _typewell_arrays_for_cache_cfg(h_df, raw_tw_df, cache_cfg):
    tw_df = _apply_typewell_gr_calibration(h_df, raw_tw_df, cache_cfg)
    typewell = tw_df.sort_values("TVT")
    tw_tvt = typewell["TVT"].to_numpy(dtype=np.float64)
    tw_gr = typewell["GR"].fillna(typewell["GR"].mean()).to_numpy(dtype=np.float64)
    return tw_tvt, tw_gr


def _stable_well_seed_offset(well_id):
    digest = hashlib.sha1(str(well_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _pf_sample_seed_grid(cache_cfg, well_id):
    sample_count = int(cache_cfg["sample_count"])
    sample_seeds = int(cache_cfg["sample_seeds"])
    max_seed = int(np.iinfo(np.int32).max)
    used = set()
    seed_grid = np.empty((sample_count, sample_seeds), dtype=np.int64)
    base_seed = int(cache_cfg["sample_base_seed"])
    well_offset = _stable_well_seed_offset(well_id)
    for sample_idx in range(sample_count):
        for inner_idx in range(sample_seeds):
            counter = 0
            while True:
                payload = f"{base_seed}:{well_offset}:{well_id}:{sample_idx}:{inner_idx}:{counter}".encode("utf-8")
                seed = int(hashlib.sha1(payload).hexdigest()[:8], 16) % max_seed
                if seed not in used:
                    used.add(seed)
                    seed_grid[sample_idx, inner_idx] = seed
                    break
                counter += 1
    return seed_grid


def _generate_pf_sample_cache_one(args):
    data_path, well_id, split_dir, cache_cfg, cache_digest, has_target = args
    h_df, tw_df = _read_well_frames(data_path, well_id)
    item = _build_pf_sample_item(h_df, tw_df, cache_cfg, well_id=well_id, has_target=has_target)
    path = pf_sample_file(split_dir, well_id)
    np.savez_compressed(
        path,
        well_id=np.asarray(well_id),
        cache_digest=np.asarray(cache_digest),
        last_seen_idx=np.asarray(item["last_seen_idx"], dtype=np.int64),
        tvt0=np.asarray(item["tvt0"], dtype=np.float32),
        suffix_len=np.asarray(item["suffix_len"], dtype=np.int64),
        kept_suffix_len=np.asarray(item["kept_suffix_len"], dtype=np.int64),
        pf_tvt_samples=item["pf_tvt_samples"].astype("float32"),
    )
    return {
        "well_id": well_id,
        "suffix_len": int(item["suffix_len"]),
        "kept_suffix_len": int(item["kept_suffix_len"]),
        "sample_tvt_rmse_mean": float(item["sample_tvt_rmse_mean"]),
    }


def _build_pf_sample_item(h_df, tw_df, cache_cfg, well_id, has_target):
    raw_tw_df = tw_df.copy()
    tw_tvt_default, tw_gr_default = _typewell_arrays_for_cache_cfg(h_df, raw_tw_df, cache_cfg)
    tvt_input = h_df["TVT_input"].to_numpy(dtype=np.float64)
    finite_seen = np.flatnonzero(np.isfinite(tvt_input))
    if finite_seen.size == 0:
        raise ValueError(f"{well_id}: no finite TVT_input anchor rows")
    last_seen_idx = int(finite_seen[-1])
    tvt0 = float(tvt_input[last_seen_idx])
    suffix_start = last_seen_idx + 1
    suffix_len = len(h_df) - suffix_start
    kept_suffix_len = suffix_len

    sample_count = int(cache_cfg["sample_count"])
    sample_seeds = int(cache_cfg["sample_seeds"])
    pf_tvt_samples = np.full((sample_count, kept_suffix_len), np.nan, dtype=np.float32)
    sample_rmses = []
    if kept_suffix_len <= 0:
        return {
            "last_seen_idx": last_seen_idx,
            "tvt0": tvt0,
            "suffix_len": suffix_len,
            "kept_suffix_len": kept_suffix_len,
            "pf_tvt_samples": pf_tvt_samples,
            "sample_tvt_rmse_mean": math.nan,
        }

    query_md = h_df["MD"].to_numpy(dtype=np.float64)[suffix_start:]
    query_z = h_df["Z"].to_numpy(dtype=np.float64)[suffix_start:]
    tw_mean_gr = float(np.nanmean(tw_gr_default))
    if not np.isfinite(tw_mean_gr):
        tw_mean_gr = 0.0
    raw_gr = h_df["GR"].to_numpy(dtype=np.float64)
    gr_interp = _interpolate_horizontal_gr(h_df["GR"], tw_mean_gr)
    query_gr_mode = str(cache_cfg.get("query_gr_mode", "interp"))
    if query_gr_mode not in {"interp", "soft_interp", "skip"}:
        raise ValueError(f"unknown PF query_gr_mode={query_gr_mode!r}")
    query_raw_gr = raw_gr[suffix_start:]
    query_gr = gr_interp.fillna(tw_mean_gr).to_numpy(dtype=np.float64)[suffix_start:]
    finite_query_gr = np.isfinite(query_raw_gr)
    if query_gr_mode == "interp":
        query_gr_power = np.ones_like(query_gr, dtype=np.float64)
    elif query_gr_mode == "skip":
        query_gr_power = finite_query_gr.astype(np.float64)
    else:
        missing_power = float(cache_cfg.get("missing_likelihood_power", 1.0))
        query_gr_power = np.where(finite_query_gr, 1.0, missing_power).astype(np.float64)
    gap_decay = float(cache_cfg.get("missing_gap_decay", 0.0))
    if gap_decay > 0.0 and (~finite_query_gr).any():
        missing_min_power = float(cache_cfg.get("missing_min_power", 0.05))
        missing_min_power = float(np.clip(missing_min_power, 0.0, 1.0))
        gap_len = 0
        for i in range(len(query_gr_power)):
            if finite_query_gr[i]:
                gap_len = 0
            else:
                gap_len += 1
                decayed = math.exp(-float(gap_len) / gap_decay)
                query_gr_power[i] *= max(missing_min_power, decayed)

    known_md = h_df["MD"].to_numpy(dtype=np.float64)[:suffix_start]
    known_z = h_df["Z"].to_numpy(dtype=np.float64)[:suffix_start]
    known_gr_for_sigma = _prefix_gr_for_pf_sigma(h_df["GR"].iloc[:suffix_start])
    known_tvt = tvt_input[:suffix_start]
    raw_typewell = raw_tw_df.sort_values("TVT")
    raw_tw_tvt = raw_typewell["TVT"].to_numpy(dtype=np.float64)
    raw_tw_gr = raw_typewell["GR"].fillna(raw_typewell["GR"].mean()).to_numpy(dtype=np.float64)
    bin_idx = np.arange(kept_suffix_len, dtype=np.int64)
    profiles = list(cache_cfg.get("profile_mixture_spec", []))
    seed_grid = _pf_sample_seed_grid(cache_cfg, well_id)

    for sample_idx in range(sample_count):
        seeds = seed_grid[sample_idx]
        profile_indices = _profile_indices_for_seed_count(
            profiles,
            len(seeds),
            shuffle_seed=int(seeds[0]),
        )
        seed_paths = []
        seed_liks = []
        for seed_idx, seed in enumerate(seeds):
            seed_cache_cfg = cache_cfg
            seed_tw_tvt = tw_tvt_default
            seed_tw_gr = tw_gr_default
            if profiles:
                seed_cache_cfg = _apply_profile_to_cache_cfg(
                    cache_cfg,
                    profiles[int(profile_indices[seed_idx])],
                )
                if (
                    seed_cache_cfg.get("typewell_gr_calibration") != cache_cfg.get("typewell_gr_calibration")
                    or abs(
                        float(seed_cache_cfg.get("seen_blend_weight", 0.7))
                        - float(cache_cfg.get("seen_blend_weight", 0.7))
                    )
                    > 1e-12
                ):
                    seed_tw_tvt, seed_tw_gr = _typewell_arrays_for_cache_cfg(
                        h_df,
                        raw_tw_df,
                        seed_cache_cfg,
                    )
            one = _run_numba_pf_heatmap_bins(
                query_md=query_md,
                query_z=query_z,
                query_gr=query_gr,
                query_gr_power=query_gr_power,
                query_gr_finite=finite_query_gr.astype(np.float64),
                known_md=known_md,
                known_z=known_z,
                known_gr_for_sigma=known_gr_for_sigma,
                known_tvt=known_tvt,
                tw_tvt=seed_tw_tvt,
                tw_gr=seed_tw_gr,
                raw_tw_tvt=raw_tw_tvt,
                raw_tw_gr=raw_tw_gr,
                tvt0=tvt0,
                bin_idx=bin_idx,
                num_bins=kept_suffix_len,
                typewell_window=float(cache_cfg["typewell_window"]),
                typewell_len=1,
                n_particles=int(cache_cfg["n_particles"]),
                seed=int(seed),
                cache_cfg=seed_cache_cfg,
                obs_gr=query_gr[:, None].astype(np.float64, copy=False),
                obs_power=query_gr_power[:, None].astype(np.float64, copy=False),
                obs_finite=finite_query_gr[:, None].astype(np.float64, copy=False),
                obs_md_delta=np.zeros((len(query_md), 1), dtype=np.float64),
                obs_z=query_z[:, None].astype(np.float64, copy=False),
                obs_count=np.ones(len(query_md), dtype=np.int64),
                obs_likelihood_mode=0,
                shape_gr=np.zeros((1, 1), dtype=np.float64),
                shape_md_delta=np.zeros((1, 1), dtype=np.float64),
                shape_z=np.zeros((1, 1), dtype=np.float64),
                shape_count=np.zeros(1, dtype=np.int64),
            )
            seed_paths.append(one["pf_tvt_pred"])
            seed_liks.append(one["log_lik"])

        seed_liks = np.asarray(seed_liks, dtype=np.float64)
        path_stack = np.stack(seed_paths, axis=0)
        if profiles:
            path_weights = _profile_seed_ensemble_weights(
                seed_liks,
                profile_indices,
                profiles,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
            sample_path = _combine_seed_paths_with_weights(path_stack, path_weights)
        else:
            sample_path = _combine_seed_paths(
                path_stack,
                seed_liks,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
        pf_tvt_samples[sample_idx] = sample_path.astype(np.float32)

    if has_target and "TVT" in h_df.columns and kept_suffix_len > 0:
        target_tvt = h_df["TVT"].to_numpy(dtype=np.float32)[suffix_start:]
        for sample_path in pf_tvt_samples:
            finite = np.isfinite(sample_path) & np.isfinite(target_tvt)
            if finite.any():
                err = sample_path[finite] - target_tvt[finite]
                sample_rmses.append(math.sqrt(float(np.mean(np.square(err, dtype=np.float64)))))

    return {
        "last_seen_idx": last_seen_idx,
        "tvt0": tvt0,
        "suffix_len": suffix_len,
        "kept_suffix_len": kept_suffix_len,
        "pf_tvt_samples": pf_tvt_samples,
        "sample_tvt_rmse_mean": float(np.mean(sample_rmses)) if sample_rmses else math.nan,
    }


def _generate_pf_heatmap_cache_one(args):
    data_path, well_id, split_dir, cache_cfg, cache_digest, has_target = args
    h_df, tw_df = _read_well_frames(data_path, well_id)
    item = _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target)
    path = pf_heatmap_file(split_dir, well_id)
    np.savez_compressed(
        path,
        well_id=np.asarray(well_id),
        cache_digest=np.asarray(cache_digest),
        last_seen_idx=np.asarray(item["last_seen_idx"], dtype=np.int64),
        tvt0=np.asarray(item["tvt0"], dtype=np.float32),
        prefix_start=np.asarray(item["prefix_start"], dtype=np.int64),
        suffix_len=np.asarray(item["suffix_len"], dtype=np.int64),
        pf_prob=item["pf_prob"].astype("float16"),
        pf_particle_density_prob=item["pf_particle_density_prob"].astype("float16"),
        pf_tvt_pred=item["pf_tvt_pred"].astype("float32"),
        pf_tvt_rel_pred=item["pf_tvt_rel_pred"].astype("float32"),
        pf_prob_ffbsi=item["pf_prob_ffbsi"].astype("float16"),
        pf_tvt_pred_ffbsi=item["pf_tvt_pred_ffbsi"].astype("float32"),
        pf_tvt_rel_pred_ffbsi=item["pf_tvt_rel_pred_ffbsi"].astype("float32"),
        window_tvt=item["window_tvt"].astype("float32"),
        window_tvt_rel=item["window_tvt_rel"].astype("float32"),
        window_has_tvt=item["window_has_tvt"].astype(bool),
        window_orig_index=item["window_orig_index"].astype("float32"),
        target_mask=item["target_mask"].astype(bool),
    )
    return {
        "well_id": well_id,
        "target_count": int(item["target_count"]),
        "tvt_sse": float(item["tvt_sse"]),
        "tvt_rmse": float(item["tvt_rmse"]),
        "prob_row_sum_min": float(item["prob_row_sum_min"]),
        "prob_row_sum_max": float(item["prob_row_sum_max"]),
        "target_count_ffbsi": int(item["target_count_ffbsi"]),
        "tvt_sse_ffbsi": float(item["tvt_sse_ffbsi"]),
        "tvt_rmse_ffbsi": float(item["tvt_rmse_ffbsi"]),
        "prob_row_sum_min_ffbsi": float(item["prob_row_sum_min_ffbsi"]),
        "prob_row_sum_max_ffbsi": float(item["prob_row_sum_max_ffbsi"]),
    }


def _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target):
    raw_tw_df = tw_df.copy()
    tw_tvt_default, tw_gr_default = _typewell_arrays_for_cache_cfg(h_df, raw_tw_df, cache_cfg)
    tvt_input = h_df["TVT_input"].to_numpy(dtype=np.float64)
    finite_seen = np.flatnonzero(np.isfinite(tvt_input))
    if finite_seen.size == 0:
        raise ValueError("no finite TVT_input anchor rows")
    last_seen_idx = int(finite_seen[-1])
    tvt0 = float(tvt_input[last_seen_idx])
    suffix_start = last_seen_idx + 1
    row_count = len(h_df)
    suffix_len = row_count - suffix_start
    kept_suffix_len = min(suffix_len, int(cache_cfg["target_len"]))

    prefix_len = int(cache_cfg["prefix_len"])
    raw_len = int(cache_cfg["raw_len"])
    num_bins = int(cache_cfg["num_bins"])
    downsample = int(cache_cfg["downsample"])
    typewell_len = int(cache_cfg["typewell_len"])
    typewell_window = float(cache_cfg["typewell_window"])
    merge_k = max(1, int(cache_cfg.get("suffix_merge_k", 1)))
    merge_likelihood_mode = str(cache_cfg.get("suffix_merge_likelihood_mode", "block")).lower()
    if merge_likelihood_mode not in {"mean", "block"}:
        raise ValueError(f"unknown PF suffix merge likelihood mode={merge_likelihood_mode!r}")
    merge_default_alpha = 1.0 if merge_likelihood_mode == "block" else 0.5
    merge_power_alpha = float(cache_cfg.get("suffix_merge_obs_power_alpha", merge_default_alpha))
    merge_state_mode = str(cache_cfg.get("suffix_merge_state_mode", "start")).lower()
    if merge_state_mode not in {"start", "end"}:
        raise ValueError(f"unknown PF suffix merge state mode={merge_state_mode!r}")

    window_tvt = np.full(num_bins, np.nan, dtype=np.float32)
    window_tvt_rel = np.full(num_bins, np.nan, dtype=np.float32)
    window_has_tvt = np.zeros(num_bins, dtype=bool)
    window_orig_index = np.full(num_bins, np.nan, dtype=np.float32)
    target_mask = np.zeros(num_bins, dtype=bool)

    prefix_start = max(0, last_seen_idx + 1 - prefix_len)
    _fill_window_tvt_bins(
        h_df=h_df,
        last_seen_idx=last_seen_idx,
        prefix_start=prefix_start,
        tvt0=tvt0,
        raw_len=raw_len,
        prefix_len=prefix_len,
        target_len=int(cache_cfg["target_len"]),
        downsample=downsample,
        num_bins=num_bins,
        window_tvt=window_tvt,
        window_tvt_rel=window_tvt_rel,
        window_has_tvt=window_has_tvt,
        window_orig_index=window_orig_index,
        target_mask=target_mask,
        has_target=has_target,
    )

    pf_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    pf_particle_density_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    pf_tvt_pred = np.zeros(num_bins, dtype=np.float32)
    pf_tvt_rel_pred = np.zeros(num_bins, dtype=np.float32)
    pf_prob_ffbsi = np.zeros((num_bins, typewell_len), dtype=np.float32)
    pf_tvt_pred_ffbsi = np.zeros(num_bins, dtype=np.float32)
    pf_tvt_rel_pred_ffbsi = np.zeros(num_bins, dtype=np.float32)
    if kept_suffix_len > 0:
        query_md = h_df["MD"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
        query_z = h_df["Z"].to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
        tw_mean_gr = float(np.nanmean(tw_gr_default))
        if not np.isfinite(tw_mean_gr):
            tw_mean_gr = 0.0
        raw_gr = h_df["GR"].to_numpy(dtype=np.float64)
        gr_interp = _interpolate_horizontal_gr(h_df["GR"], tw_mean_gr)
        query_gr_mode = str(cache_cfg.get("query_gr_mode", "interp"))
        if query_gr_mode not in {"interp", "soft_interp", "skip"}:
            raise ValueError(f"unknown PF query_gr_mode={query_gr_mode!r}")
        query_raw_gr = raw_gr[suffix_start : suffix_start + kept_suffix_len]
        query_gr = gr_interp.fillna(tw_mean_gr).to_numpy(dtype=np.float64)[suffix_start : suffix_start + kept_suffix_len]
        finite_query_gr = np.isfinite(query_raw_gr)
        if query_gr_mode == "interp":
            query_gr_power = np.ones_like(query_gr, dtype=np.float64)
        elif query_gr_mode == "skip":
            query_gr_power = finite_query_gr.astype(np.float64)
        else:
            missing_power = float(cache_cfg.get("missing_likelihood_power", 1.0))
            query_gr_power = np.where(finite_query_gr, 1.0, missing_power).astype(np.float64)
        gap_decay = float(cache_cfg.get("missing_gap_decay", 0.0))
        if gap_decay > 0.0 and (~finite_query_gr).any():
            missing_min_power = float(cache_cfg.get("missing_min_power", 0.05))
            missing_min_power = float(np.clip(missing_min_power, 0.0, 1.0))
            gap_len = 0
            for i in range(len(query_gr_power)):
                if finite_query_gr[i]:
                    gap_len = 0
                else:
                    gap_len += 1
                    decayed = math.exp(-float(gap_len) / gap_decay)
                    query_gr_power[i] *= max(missing_min_power, decayed)
        if merge_k > 1:
            (
                query_md_pf,
                query_z_pf,
                query_gr_pf,
                query_gr_power_pf,
                query_gr_finite_pf,
                query_obs_gr_pf,
                query_obs_power_pf,
                query_obs_finite_pf,
                query_obs_md_delta_pf,
                query_obs_z_pf,
                query_obs_count_pf,
                merge_block_starts,
                merge_block_ends,
            ) = _merge_suffix_query_blocks(
                query_md=query_md,
                query_z=query_z,
                query_gr=query_gr,
                query_gr_power=query_gr_power,
                query_gr_finite=finite_query_gr.astype(np.float64),
                merge_k=merge_k,
                power_alpha=merge_power_alpha,
                likelihood_mode=merge_likelihood_mode,
                state_mode=merge_state_mode,
            )
            seed_bin_idx = np.arange(len(query_md_pf), dtype=np.int64)
            seed_num_bins = int(len(query_md_pf))
        else:
            query_md_pf = query_md
            query_z_pf = query_z
            query_gr_pf = query_gr
            query_gr_power_pf = query_gr_power
            query_gr_finite_pf = finite_query_gr.astype(np.float64)
            query_obs_gr_pf = query_gr[:, None].astype(np.float64, copy=False)
            query_obs_power_pf = query_gr_power[:, None].astype(np.float64, copy=False)
            query_obs_finite_pf = finite_query_gr[:, None].astype(np.float64, copy=False)
            query_obs_md_delta_pf = np.zeros((len(query_md), 1), dtype=np.float64)
            query_obs_z_pf = query_z[:, None].astype(np.float64, copy=False)
            query_obs_count_pf = np.ones(len(query_md), dtype=np.int64)
            merge_block_starts = np.arange(kept_suffix_len, dtype=np.int64)
            merge_block_ends = merge_block_starts + 1
            seed_bin_idx = (prefix_len + np.arange(kept_suffix_len, dtype=np.int64)) // downsample
            seed_num_bins = int(num_bins)
        # Shape correlation is applied at seed-ensemble time, not inside the
        # per-particle emission update. Passing neutral arrays keeps the older
        # experimental kernel signature stable while avoiding row-level motif
        # over-chasing and the large runtime cost of per-particle windows.
        shape_gr = np.zeros((1, 1), dtype=np.float64)
        shape_md_delta = np.zeros((1, 1), dtype=np.float64)
        shape_z = np.zeros((1, 1), dtype=np.float64)
        shape_count = np.zeros(1, dtype=np.int64)
        known_md = h_df["MD"].to_numpy(dtype=np.float64)[: suffix_start]
        known_z = h_df["Z"].to_numpy(dtype=np.float64)[: suffix_start]
        known_gr_for_sigma = _prefix_gr_for_pf_sigma(h_df["GR"].iloc[:suffix_start])
        known_tvt = tvt_input[: suffix_start]
        raw_typewell = raw_tw_df.sort_values("TVT")
        raw_tw_tvt = raw_typewell["TVT"].to_numpy(dtype=np.float64)
        raw_tw_gr = raw_typewell["GR"].fillna(raw_typewell["GR"].mean()).to_numpy(dtype=np.float64)
        seeds = _sample_pf_seeds(int(cache_cfg["base_seed"]), int(cache_cfg["n_seeds"]))
        profiles = list(cache_cfg.get("profile_mixture_spec", []))
        profile_indices = _profile_indices_for_seed_count(
            profiles,
            len(seeds),
            shuffle_seed=int(cache_cfg["base_seed"]),
        )
        needs_shape_bonus = float(cache_cfg.get("gr_shape_power", 0.0)) > 0.0
        if not needs_shape_bonus and profiles:
            for profile in profiles:
                profile_cfg = _apply_profile_to_cache_cfg(cache_cfg, profile)
                if float(profile_cfg.get("gr_shape_power", 0.0)) > 0.0:
                    needs_shape_bonus = True
                    break
        if needs_shape_bonus:
            query_gr_bin, query_gr_bin_count = _build_gr_shape_bin_series(
                query_gr_pf,
                query_gr_finite_pf,
                seed_bin_idx,
                seed_num_bins,
            )
        else:
            query_gr_bin = np.empty(0, dtype=np.float64)
            query_gr_bin_count = np.empty(0, dtype=np.int64)
        seed_paths = []
        seed_liks = []
        seed_probs = []
        seed_particle_density_probs = []
        seed_paths_ffbsi = []
        seed_probs_ffbsi = []
        has_ffbsi_seed = False
        for seed_idx, seed in enumerate(seeds):
            seed_cache_cfg = cache_cfg
            seed_tw_tvt = tw_tvt_default
            seed_tw_gr = tw_gr_default
            if profiles:
                seed_cache_cfg = _apply_profile_to_cache_cfg(
                    cache_cfg,
                    profiles[int(profile_indices[seed_idx])],
                )
                if (
                    seed_cache_cfg.get("typewell_gr_calibration") != cache_cfg.get("typewell_gr_calibration")
                    or abs(float(seed_cache_cfg.get("seen_blend_weight", 0.7)) - float(cache_cfg.get("seen_blend_weight", 0.7))) > 1e-12
                ):
                    seed_tw_tvt, seed_tw_gr = _typewell_arrays_for_cache_cfg(
                        h_df,
                        raw_tw_df,
                        seed_cache_cfg,
                    )
            seed_cache_cfg = _effective_pf_cache_cfg_for_merge(seed_cache_cfg)
            one = _run_numba_pf_heatmap_bins(
                query_md=query_md_pf,
                query_z=query_z_pf,
                query_gr=query_gr_pf,
                query_gr_power=query_gr_power_pf,
                query_gr_finite=query_gr_finite_pf,
                obs_gr=query_obs_gr_pf,
                obs_power=query_obs_power_pf,
                obs_finite=query_obs_finite_pf,
                obs_md_delta=query_obs_md_delta_pf,
                obs_z=query_obs_z_pf,
                obs_count=query_obs_count_pf,
                obs_likelihood_mode=1 if (merge_k > 1 and merge_likelihood_mode == "block") else 0,
                shape_gr=shape_gr,
                shape_md_delta=shape_md_delta,
                shape_z=shape_z,
                shape_count=shape_count,
                known_md=known_md,
                known_z=known_z,
                known_gr_for_sigma=known_gr_for_sigma,
                known_tvt=known_tvt,
                tw_tvt=seed_tw_tvt,
                tw_gr=seed_tw_gr,
                raw_tw_tvt=raw_tw_tvt,
                raw_tw_gr=raw_tw_gr,
                tvt0=tvt0,
                bin_idx=seed_bin_idx,
                num_bins=seed_num_bins,
                typewell_window=typewell_window,
                typewell_len=typewell_len,
                n_particles=int(cache_cfg["n_particles"]),
                seed=int(seed),
                cache_cfg=seed_cache_cfg,
            )
            seed_paths.append(one["pf_tvt_pred"])
            seed_paths_ffbsi.append(one.get("pf_tvt_pred_ffbsi", one["pf_tvt_pred"]))
            seed_shape_bonus = 0.0
            if needs_shape_bonus:
                seed_shape_bonus = _pf_seed_gr_shape_bonus(
                    query_gr_bin=query_gr_bin,
                    query_gr_bin_count=query_gr_bin_count,
                    seed_tvt_pred=one["pf_tvt_pred"],
                    tw_tvt=seed_tw_tvt,
                    tw_gr=seed_tw_gr,
                    raw_tw_tvt=raw_tw_tvt,
                    raw_tw_gr=raw_tw_gr,
                    gr_shape_power=float(seed_cache_cfg.get("gr_shape_power", 0.0)),
                    gr_shape_mode=str(seed_cache_cfg.get("gr_shape_mode", "resid_corr")),
                    gr_shape_window=int(seed_cache_cfg.get("gr_shape_window", 15)),
                    gr_shape_min_points=int(seed_cache_cfg.get("gr_shape_min_points", 7)),
                    gr_shape_sigma_floor=float(seed_cache_cfg.get("gr_shape_sigma_floor", 0.35)),
                    gr_shape_ref_mode=str(seed_cache_cfg.get("gr_shape_ref_mode", "primary")),
                    raw_ref_weight=float(seed_cache_cfg.get("raw_ref_likelihood_weight", 0.0)),
                )
            seed_liks.append(float(one["log_lik"]) + seed_shape_bonus)
            seed_probs.append(one["pf_prob"])
            seed_particle_density_probs.append(one.get("pf_particle_density_prob", one["pf_prob"]))
            seed_probs_ffbsi.append(one.get("pf_prob_ffbsi", one["pf_prob"]))
            has_ffbsi_seed = has_ffbsi_seed or bool(one.get("ffbsi_ok", False))
        seed_liks = np.asarray(seed_liks, dtype=np.float64)
        prob_stack = np.stack(seed_probs, axis=0)
        particle_density_stack = np.stack(seed_particle_density_probs, axis=0)
        prob_stack_ffbsi = np.stack(seed_probs_ffbsi, axis=0)
        if profiles:
            prob_weights = _profile_seed_ensemble_weights(
                seed_liks,
                profile_indices,
                profiles,
                mode=str(cache_cfg.get("seed_prob_weight_mode", "equal")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
                mode_key="seed_prob_weight_mode",
            )
        else:
            prob_weights = _seed_ensemble_weights(
                seed_liks,
                mode=str(cache_cfg.get("seed_prob_weight_mode", "equal")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
        pf_prob = (prob_stack * prob_weights[:, None, None]).sum(axis=0).astype(np.float32)
        pf_prob = _normalize_prob_rows(pf_prob)
        density_weights = np.full(len(seed_particle_density_probs), 1.0 / len(seed_particle_density_probs), dtype=np.float32)
        pf_particle_density_prob = (
            particle_density_stack * density_weights[:, None, None]
        ).sum(axis=0).astype(np.float32)
        pf_particle_density_prob = _normalize_prob_rows(pf_particle_density_prob)
        pf_prob_ffbsi = (prob_stack_ffbsi * prob_weights[:, None, None]).sum(axis=0).astype(np.float32)
        pf_prob_ffbsi = _normalize_prob_rows(pf_prob_ffbsi)
        prob_temperature = float(cache_cfg.get("pf_prob_temperature", 1.0))
        if prob_temperature > 0.0 and abs(prob_temperature - 1.0) > 1e-12:
            pf_prob = np.power(pf_prob, np.float32(1.0 / prob_temperature)).astype(np.float32)
            pf_prob = _normalize_prob_rows(pf_prob)
            pf_prob_ffbsi = np.power(pf_prob_ffbsi, np.float32(1.0 / prob_temperature)).astype(np.float32)
            pf_prob_ffbsi = _normalize_prob_rows(pf_prob_ffbsi)
        axis_sigma = float(cache_cfg.get("axis_sigma", 0.0))
        if axis_sigma > 0.0:
            if gaussian_filter1d is None:
                raise ImportError("scipy.ndimage.gaussian_filter1d is required for PF_heatmap_axis_sigma > 0")
            pf_prob = gaussian_filter1d(pf_prob, sigma=axis_sigma, axis=1, mode="constant")
            pf_prob = _normalize_prob_rows(pf_prob)
            pf_prob_ffbsi = gaussian_filter1d(pf_prob_ffbsi, sigma=axis_sigma, axis=1, mode="constant")
            pf_prob_ffbsi = _normalize_prob_rows(pf_prob_ffbsi)

        path_stack = np.stack(seed_paths, axis=0)
        path_stack_ffbsi = np.stack(seed_paths_ffbsi, axis=0)
        if profiles:
            path_weights = _profile_seed_ensemble_weights(
                seed_liks,
                profile_indices,
                profiles,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
                mode_key="seed_path_weight_mode",
            )
            pf_tvt_pred = _combine_seed_paths_with_weights(path_stack, path_weights)
            pf_tvt_pred_ffbsi = _combine_seed_paths_with_weights(path_stack_ffbsi, path_weights)
        else:
            pf_tvt_pred = _combine_seed_paths(
                path_stack,
                seed_liks,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
            pf_tvt_pred_ffbsi = _combine_seed_paths(
                path_stack_ffbsi,
                seed_liks,
                mode=str(cache_cfg.get("seed_path_weight_mode", "likelihood")),
                lik_scale=float(cache_cfg.get("lik_scale", 5.0)),
            )
        if not has_ffbsi_seed:
            pf_prob_ffbsi = pf_prob.copy()
            pf_tvt_pred_ffbsi = pf_tvt_pred.copy()
        if merge_k > 1:
            pf_prob = _expand_merged_prob_to_bins(
                pf_prob,
                merge_block_starts,
                merge_block_ends,
                prefix_len,
                downsample,
                num_bins,
            )
            pf_particle_density_prob = _expand_merged_prob_to_bins(
                pf_particle_density_prob,
                merge_block_starts,
                merge_block_ends,
                prefix_len,
                downsample,
                num_bins,
            )
            pf_prob_ffbsi = _expand_merged_prob_to_bins(
                pf_prob_ffbsi,
                merge_block_starts,
                merge_block_ends,
                prefix_len,
                downsample,
                num_bins,
            )
            pf_tvt_pred = _downsample_suffix_values(
                _row_path_from_merged_steps(
                    pf_tvt_pred,
                    tvt0,
                    merge_block_starts,
                    merge_block_ends,
                    kept_suffix_len,
                ),
                prefix_len,
                downsample,
                num_bins,
            )
            pf_tvt_pred_ffbsi = _downsample_suffix_values(
                _row_path_from_merged_steps(
                    pf_tvt_pred_ffbsi,
                    tvt0,
                    merge_block_starts,
                    merge_block_ends,
                    kept_suffix_len,
                ),
                prefix_len,
                downsample,
                num_bins,
            )
        pf_tvt_rel_pred = (pf_tvt_pred - np.float32(tvt0)).astype(np.float32)
        pf_tvt_rel_pred_ffbsi = (pf_tvt_pred_ffbsi - np.float32(tvt0)).astype(np.float32)
        pf_tvt_pred = np.nan_to_num(pf_tvt_pred, nan=0.0).astype(np.float32)
        pf_tvt_rel_pred = np.nan_to_num(pf_tvt_rel_pred, nan=0.0).astype(np.float32)
        pf_tvt_pred_ffbsi = np.nan_to_num(pf_tvt_pred_ffbsi, nan=0.0).astype(np.float32)
        pf_tvt_rel_pred_ffbsi = np.nan_to_num(pf_tvt_rel_pred_ffbsi, nan=0.0).astype(np.float32)

    fixed_path_bins = window_has_tvt & ~target_mask
    if fixed_path_bins.any():
        _paint_fixed_path_density(
            pf_prob=pf_prob,
            pf_tvt_pred=pf_tvt_pred,
            pf_tvt_rel_pred=pf_tvt_rel_pred,
            tvt0=tvt0,
            tvt_rel=window_tvt_rel,
            valid=fixed_path_bins,
            typewell_window=typewell_window,
            typewell_len=typewell_len,
        )
        _paint_fixed_path_density(
            pf_prob=pf_particle_density_prob,
            pf_tvt_pred=np.zeros_like(pf_tvt_pred),
            pf_tvt_rel_pred=np.zeros_like(pf_tvt_rel_pred),
            tvt0=tvt0,
            tvt_rel=window_tvt_rel,
            valid=fixed_path_bins,
            typewell_window=typewell_window,
            typewell_len=typewell_len,
        )
        _paint_fixed_path_density(
            pf_prob=pf_prob_ffbsi,
            pf_tvt_pred=pf_tvt_pred_ffbsi,
            pf_tvt_rel_pred=pf_tvt_rel_pred_ffbsi,
            tvt0=tvt0,
            tvt_rel=window_tvt_rel,
            valid=fixed_path_bins,
            typewell_window=typewell_window,
            typewell_len=typewell_len,
        )

    valid_rows = target_mask & np.isfinite(window_tvt) & np.isfinite(pf_tvt_pred) & (pf_tvt_pred != 0.0)
    if has_target and valid_rows.any():
        err = pf_tvt_pred[valid_rows] - window_tvt[valid_rows]
        tvt_sse = float(np.square(err).sum())
        target_count = int(valid_rows.sum())
        tvt_rmse = math.sqrt(tvt_sse / target_count)
    else:
        tvt_sse = math.nan
        target_count = 0
        tvt_rmse = math.nan
    valid_rows_ffbsi = target_mask & np.isfinite(window_tvt) & np.isfinite(pf_tvt_pred_ffbsi) & (pf_tvt_pred_ffbsi != 0.0)
    if has_target and valid_rows_ffbsi.any():
        err_ffbsi = pf_tvt_pred_ffbsi[valid_rows_ffbsi] - window_tvt[valid_rows_ffbsi]
        tvt_sse_ffbsi = float(np.square(err_ffbsi).sum())
        target_count_ffbsi = int(valid_rows_ffbsi.sum())
        tvt_rmse_ffbsi = math.sqrt(tvt_sse_ffbsi / target_count_ffbsi)
    else:
        tvt_sse_ffbsi = math.nan
        target_count_ffbsi = 0
        tvt_rmse_ffbsi = math.nan
    prob_sums = pf_prob.sum(axis=1)
    active_prob = prob_sums[prob_sums > 0]
    prob_sums_ffbsi = pf_prob_ffbsi.sum(axis=1)
    active_prob_ffbsi = prob_sums_ffbsi[prob_sums_ffbsi > 0]
    return {
        "last_seen_idx": last_seen_idx,
        "tvt0": tvt0,
        "prefix_start": prefix_start,
        "suffix_len": suffix_len,
        "pf_prob": pf_prob,
        "pf_particle_density_prob": pf_particle_density_prob,
        "pf_tvt_pred": pf_tvt_pred,
        "pf_tvt_rel_pred": pf_tvt_rel_pred,
        "pf_prob_ffbsi": pf_prob_ffbsi,
        "pf_tvt_pred_ffbsi": pf_tvt_pred_ffbsi,
        "pf_tvt_rel_pred_ffbsi": pf_tvt_rel_pred_ffbsi,
        "window_tvt": np.nan_to_num(window_tvt, nan=0.0).astype(np.float32),
        "window_tvt_rel": np.nan_to_num(window_tvt_rel, nan=0.0).astype(np.float32),
        "window_has_tvt": window_has_tvt,
        "window_orig_index": np.nan_to_num(window_orig_index, nan=-1.0).astype(np.float32),
        "target_mask": target_mask,
        "target_count": target_count,
        "tvt_sse": tvt_sse,
        "tvt_rmse": tvt_rmse,
        "prob_row_sum_min": float(active_prob.min()) if active_prob.size else math.nan,
        "prob_row_sum_max": float(active_prob.max()) if active_prob.size else math.nan,
        "target_count_ffbsi": target_count_ffbsi,
        "tvt_sse_ffbsi": tvt_sse_ffbsi,
        "tvt_rmse_ffbsi": tvt_rmse_ffbsi,
        "prob_row_sum_min_ffbsi": float(active_prob_ffbsi.min()) if active_prob_ffbsi.size else math.nan,
        "prob_row_sum_max_ffbsi": float(active_prob_ffbsi.max()) if active_prob_ffbsi.size else math.nan,
    }


def _fill_window_tvt_bins(
    h_df,
    last_seen_idx,
    prefix_start,
    tvt0,
    raw_len,
    prefix_len,
    target_len,
    downsample,
    num_bins,
    window_tvt,
    window_tvt_rel,
    window_has_tvt,
    window_orig_index,
    target_mask,
    has_target,
):
    raw_orig_index = np.full(raw_len, np.nan, dtype=np.float32)
    row_orig_index = np.arange(len(h_df), dtype=np.float32)
    raw_tvt = np.full(raw_len, np.nan, dtype=np.float32)
    raw_target = np.zeros(raw_len, dtype=bool)
    prefix_size = last_seen_idx + 1 - prefix_start
    prefix_dest_start = prefix_len - prefix_size
    tvt_input = h_df["TVT_input"].to_numpy(dtype=np.float32)
    if prefix_size > 0:
        raw_tvt[prefix_dest_start:prefix_len] = tvt_input[prefix_start : last_seen_idx + 1]
        raw_orig_index[prefix_dest_start:prefix_len] = row_orig_index[prefix_start : last_seen_idx + 1]
    suffix_size = min(len(h_df) - (last_seen_idx + 1), target_len)
    if suffix_size > 0:
        dst = slice(prefix_len, prefix_len + suffix_size)
        raw_orig_index[dst] = row_orig_index[last_seen_idx + 1 : last_seen_idx + 1 + suffix_size]
        if has_target and "TVT" in h_df.columns:
            tvt = h_df["TVT"].to_numpy(dtype=np.float32)
            raw_tvt[dst] = tvt[last_seen_idx + 1 : last_seen_idx + 1 + suffix_size]
            raw_target[dst] = True
    raw_has_tvt = np.isfinite(raw_tvt)
    bin_idx = np.arange(raw_len, dtype=np.int64) // int(downsample)
    valid = raw_has_tvt & (bin_idx < num_bins)
    if valid.any():
        sums = np.bincount(bin_idx[valid], weights=raw_tvt[valid], minlength=num_bins)
        counts = np.bincount(bin_idx[valid], minlength=num_bins)
        filled = counts > 0
        window_tvt[filled] = (sums[filled] / counts[filled]).astype(np.float32)
        window_tvt_rel[filled] = window_tvt[filled] - np.float32(tvt0)
        window_has_tvt[filled] = True
    target_counts = np.bincount(bin_idx[raw_target & (bin_idx < num_bins)], minlength=num_bins)
    target_mask[:] = target_counts > 0
    _fill_orig_index_bins(raw_orig_index, downsample, num_bins, window_orig_index)


def _paint_fixed_path_density(
    pf_prob,
    pf_tvt_pred,
    pf_tvt_rel_pred,
    tvt0,
    tvt_rel,
    valid,
    typewell_window,
    typewell_len,
):
    axis0 = -float(typewell_window) + float(typewell_window) / int(typewell_len)
    axis_step = 2.0 * float(typewell_window) / int(typewell_len)
    rows = np.flatnonzero(valid & np.isfinite(tvt_rel))
    for row in rows:
        rel = float(tvt_rel[row])
        grid_pos = (rel - axis0) / axis_step
        left = int(np.floor(grid_pos))
        frac = grid_pos - left
        pf_prob[row, :] = 0.0
        if 0 <= left < int(typewell_len):
            pf_prob[row, left] += np.float32(1.0 - frac)
        right = left + 1
        if 0 <= right < int(typewell_len):
            pf_prob[row, right] += np.float32(frac)
        row_sum = float(pf_prob[row].sum())
        if row_sum > 0.0:
            pf_prob[row] /= np.float32(row_sum)
            pf_tvt_rel_pred[row] = np.float32(rel)
            pf_tvt_pred[row] = np.float32(float(tvt0) + rel)


def _fill_orig_index_bins(raw_orig_index, downsample, num_bins, window_orig_index):
    bin_idx = np.arange(len(raw_orig_index), dtype=np.int64) // int(downsample)
    valid = np.isfinite(raw_orig_index) & (bin_idx < int(num_bins))
    if not valid.any():
        return
    sums = np.bincount(bin_idx[valid], weights=raw_orig_index[valid], minlength=int(num_bins))
    counts = np.bincount(bin_idx[valid], minlength=int(num_bins))
    filled = counts > 0
    window_orig_index[filled] = (sums[filled] / counts[filled]).astype(np.float32)


def _sample_pf_seeds(base_seed, n_seeds):
    rng = np.random.default_rng(int(base_seed))
    return rng.integers(0, np.iinfo(np.int32).max, size=int(n_seeds), dtype=np.int64)


def _single_pf_seed(base_seed):
    return int(_sample_pf_seeds(base_seed, 1)[0])


def _normalize_prob_rows(prob):
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    prob[prob < 0.0] = 0.0
    row_sum = prob.sum(axis=1, keepdims=True)
    np.divide(prob, row_sum, out=prob, where=row_sum > 0.0)
    return prob


def _downsample_suffix_values(values, prefix_len, downsample, num_bins):
    out = np.full(int(num_bins), np.nan, dtype=np.float32)
    if len(values) == 0:
        return out
    bin_idx = (int(prefix_len) + np.arange(len(values), dtype=np.int64)) // int(downsample)
    valid = bin_idx < int(num_bins)
    sums = np.bincount(
        bin_idx[valid],
        weights=np.asarray(values, dtype=np.float32)[valid],
        minlength=int(num_bins),
    )
    counts = np.bincount(bin_idx[valid], minlength=int(num_bins))
    filled = counts > 0
    out[filled] = (sums[filled] / counts[filled]).astype(np.float32)
    return out


def _combine_seed_prob_stack(seed_probs, prob_weights, cache_cfg):
    pf_prob = (np.stack(seed_probs, axis=0) * prob_weights[:, None, None]).sum(axis=0).astype(np.float32)
    pf_prob = _normalize_prob_rows(pf_prob)
    prob_temperature = float(cache_cfg.get("pf_prob_temperature", 1.0))
    if prob_temperature > 0.0 and abs(prob_temperature - 1.0) > 1e-12:
        pf_prob = np.power(pf_prob, np.float32(1.0 / prob_temperature)).astype(np.float32)
        pf_prob = _normalize_prob_rows(pf_prob)
    axis_sigma = float(cache_cfg.get("axis_sigma", 0.0))
    if axis_sigma > 0.0:
        if gaussian_filter1d is None:
            raise ImportError("scipy.ndimage.gaussian_filter1d is required for PF_heatmap_axis_sigma > 0")
        pf_prob = gaussian_filter1d(pf_prob, sigma=axis_sigma, axis=1, mode="constant")
        pf_prob = _normalize_prob_rows(pf_prob)
    return pf_prob


def _pf_arrays_from_hist(hist_bins, path_sums, count_bins, num_bins, typewell_len):
    pf_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    valid_bins = count_bins > 0
    pf_prob[valid_bins] = (hist_bins[valid_bins] / count_bins[valid_bins, None]).astype(np.float32)
    pf_prob = _normalize_prob_rows(pf_prob)
    pf_tvt_pred = np.full(num_bins, np.nan, dtype=np.float32)
    pf_tvt_pred[valid_bins] = (path_sums[valid_bins] / count_bins[valid_bins]).astype(np.float32)
    return pf_prob, pf_tvt_pred


def _pf_prob_from_hist(hist_bins, count_bins, num_bins, typewell_len):
    pf_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    valid_bins = count_bins > 0
    pf_prob[valid_bins] = (hist_bins[valid_bins] / count_bins[valid_bins, None]).astype(np.float32)
    return _normalize_prob_rows(pf_prob)


def _local_gr_ambiguity_power(
    query_gr,
    query_gr_finite,
    tvt0,
    tw_tvt,
    tw_gr,
    gr_sigma,
    typewell_window,
    ambiguity_power,
    min_power,
    contrast,
):
    out = np.ones(len(query_gr), dtype=np.float64)
    ambiguity_power = float(ambiguity_power)
    if ambiguity_power <= 0.0 or len(query_gr) == 0:
        return out
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        return out

    local = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    local &= (tw_tvt >= float(tvt0) - float(typewell_window))
    local &= (tw_tvt <= float(tvt0) + float(typewell_window))
    if local.sum() < 4:
        return out

    local_gr = tw_gr[local].astype(np.float64)
    min_power = float(np.clip(min_power, 0.0, 1.0))
    contrast = max(float(contrast), 1e-6)
    inv_sigma = 1.0 / float(gr_sigma)
    for i in range(len(query_gr)):
        if query_gr_finite[i] <= 0.0 or not np.isfinite(query_gr[i]):
            continue
        d = (float(query_gr[i]) - local_gr) * inv_sigma
        logits = -0.5 * d * d
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            continue
        probs /= prob_sum
        max_prob = float(np.max(probs))
        ess = 1.0 / max(float(np.sum(probs * probs)), 1e-12)
        unique = max(max_prob, 1.0 / ess)
        unique = float(np.clip(unique * contrast, 0.0, 1.0))
        out[i] = min_power + (1.0 - min_power) * (unique ** ambiguity_power)
    return out


def _local_gr_information_multiplier(
    query_gr,
    query_gr_finite,
    tvt0,
    tw_tvt,
    tw_gr,
    gr_sigma,
    typewell_window,
    information_power,
    center,
    min_multiplier,
    max_multiplier,
    slope_weight,
):
    out = np.ones(len(query_gr), dtype=np.float64)
    information_power = float(information_power)
    if information_power <= 0.0 or len(query_gr) == 0:
        return out
    if not np.isfinite(gr_sigma) or gr_sigma <= 0.0:
        return out

    local = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    local &= (tw_tvt >= float(tvt0) - float(typewell_window))
    local &= (tw_tvt <= float(tvt0) + float(typewell_window))
    if local.sum() < 4:
        return out

    local_tvt = tw_tvt[local].astype(np.float64)
    local_gr = tw_gr[local].astype(np.float64)
    order = np.argsort(local_tvt)
    local_tvt = local_tvt[order]
    local_gr = local_gr[order]
    min_multiplier = float(np.clip(min_multiplier, 0.05, 10.0))
    max_multiplier = float(np.clip(max_multiplier, min_multiplier, 10.0))
    center = max(float(center), 1e-6)
    slope_weight = float(np.clip(slope_weight, 0.0, 1.0))
    inv_sigma = 1.0 / float(gr_sigma)
    log_n = math.log(float(len(local_gr)))
    if slope_weight > 0.0:
        local_slope = np.abs(np.gradient(local_gr, local_tvt))
    else:
        local_slope = np.zeros_like(local_gr)

    for i in range(len(query_gr)):
        if query_gr_finite[i] <= 0.0 or not np.isfinite(query_gr[i]):
            continue
        d = (float(query_gr[i]) - local_gr) * inv_sigma
        logits = -0.5 * d * d
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        prob_sum = float(probs.sum())
        if prob_sum <= 0.0 or not np.isfinite(prob_sum):
            continue
        probs /= prob_sum
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-300)))) / max(log_n, 1e-12)
        entropy_info = float(np.clip(1.0 - entropy, 0.0, 1.0))
        if slope_weight > 0.0:
            slope_raw = float(np.sum(probs * local_slope)) * inv_sigma
            slope_info = slope_raw / (1.0 + slope_raw)
            info = (1.0 - slope_weight) * entropy_info + slope_weight * slope_info
        else:
            info = entropy_info
        multiplier = math.pow(max(info / center, 1e-6), information_power)
        out[i] = float(np.clip(multiplier, min_multiplier, max_multiplier))
    return out


def _gr_shape_mode_code(mode):
    mode = str(mode).lower()
    if mode in {"off", "none", "0"}:
        return 0
    if mode in {"resid_corr", "residual_corr", "centered", "center"}:
        return 1
    if mode in {"ncc", "pearson", "corr"}:
        return 2
    raise ValueError(f"unknown PF_heatmap_gr_shape_mode={mode!r}")


def _gr_shape_ref_mode_code(mode):
    mode = str(mode).lower()
    if mode in {"primary", "seen", "seen_blend"}:
        return 0
    if mode == "raw":
        return 1
    if mode in {"mixed", "point"}:
        return 2
    raise ValueError(f"unknown PF_heatmap_gr_shape_ref_mode={mode!r}")


def _build_gr_shape_windows(md, z, gr, gr_finite, window):
    window = int(window)
    if window <= 1:
        window = 1
    if window % 2 == 0:
        window += 1
    n = len(gr)
    shape_gr = np.zeros((n, window), dtype=np.float64)
    shape_md_delta = np.zeros((n, window), dtype=np.float64)
    shape_z = np.zeros((n, window), dtype=np.float64)
    shape_count = np.zeros(n, dtype=np.int64)
    if n == 0:
        return shape_gr, shape_md_delta, shape_z, shape_count
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    gr = np.asarray(gr, dtype=np.float64)
    finite = np.asarray(gr_finite, dtype=np.float64) > 0.0
    for i in range(n):
        lo = max(0, i - window + 1)
        hi = i + 1
        slot = 0
        for row in range(lo, hi):
            if not finite[row] or not np.isfinite(gr[row]):
                continue
            if slot >= window:
                break
            shape_gr[i, slot] = gr[row]
            shape_md_delta[i, slot] = md[row] - md[i]
            shape_z[i, slot] = z[row]
            slot += 1
        shape_count[i] = slot
    return shape_gr, shape_md_delta, shape_z, shape_count


def _build_gr_shape_bin_series(gr, gr_finite, bin_idx, num_bins):
    gr = np.asarray(gr, dtype=np.float64)
    gr_finite = np.asarray(gr_finite, dtype=np.float64) > 0.0
    bin_idx = np.asarray(bin_idx, dtype=np.int64)
    num_bins = int(num_bins)
    out = np.full(num_bins, np.nan, dtype=np.float64)
    count = np.zeros(num_bins, dtype=np.int64)
    valid = gr_finite & np.isfinite(gr) & (bin_idx >= 0) & (bin_idx < num_bins)
    if valid.any():
        sums = np.bincount(bin_idx[valid], weights=gr[valid], minlength=num_bins)
        count = np.bincount(bin_idx[valid], minlength=num_bins).astype(np.int64, copy=False)
        filled = count > 0
        out[filled] = sums[filled] / count[filled]
    return out, count


def _pf_seed_gr_shape_bonus(
    query_gr_bin,
    query_gr_bin_count,
    seed_tvt_pred,
    tw_tvt,
    tw_gr,
    raw_tw_tvt,
    raw_tw_gr,
    gr_shape_power,
    gr_shape_mode,
    gr_shape_window,
    gr_shape_min_points,
    gr_shape_sigma_floor,
    gr_shape_ref_mode,
    raw_ref_weight,
):
    gr_shape_power = float(gr_shape_power)
    if gr_shape_power <= 0.0:
        return 0.0

    query_gr_bin = np.asarray(query_gr_bin, dtype=np.float64)
    query_gr_bin_count = np.asarray(query_gr_bin_count, dtype=np.int64)
    seed_tvt_pred = np.asarray(seed_tvt_pred, dtype=np.float64)
    valid = (
        np.isfinite(query_gr_bin)
        & np.isfinite(seed_tvt_pred)
        & np.isfinite(query_gr_bin_count)
        & (query_gr_bin_count > 0)
    )
    if not valid.any():
        return 0.0
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < int(gr_shape_min_points):
        return 0.0

    gr_shape_window = int(gr_shape_window)
    if gr_shape_window > 0 and valid_idx.size > gr_shape_window:
        valid_idx = valid_idx[-gr_shape_window:]

    obs = query_gr_bin[valid_idx].astype(np.float64, copy=False)
    pred = np.interp(seed_tvt_pred[valid_idx], tw_tvt, tw_gr).astype(np.float64, copy=False)
    shape_ref_mode = str(gr_shape_ref_mode).lower()
    if shape_ref_mode == "raw" and raw_tw_tvt.size >= 3:
        pred = np.interp(seed_tvt_pred[valid_idx], raw_tw_tvt, raw_tw_gr).astype(np.float64, copy=False)
    elif shape_ref_mode in {"mixed", "point"} and raw_tw_tvt.size >= 3 and raw_ref_weight > 0.0:
        raw_pred = np.interp(seed_tvt_pred[valid_idx], raw_tw_tvt, raw_tw_gr).astype(np.float64, copy=False)
        pred = (1.0 - float(raw_ref_weight)) * pred + float(raw_ref_weight) * raw_pred

    shape_mode = str(gr_shape_mode).lower()
    if shape_mode == "resid_corr":
        if obs.size < 3 or pred.size < 3:
            return 0.0
        obs = np.diff(obs)
        pred = np.diff(pred)
        min_corr_points = max(2, int(gr_shape_min_points) - 1)
    elif shape_mode not in {"ncc", "pearson", "corr"}:
        raise ValueError(f"unknown PF_heatmap_gr_shape_mode={gr_shape_mode!r}")
    else:
        min_corr_points = int(gr_shape_min_points)

    if obs.size < min_corr_points or pred.size < min_corr_points:
        return 0.0

    obs = obs - float(np.mean(obs))
    pred = pred - float(np.mean(pred))
    obs_var = float(np.mean(obs * obs))
    pred_var = float(np.mean(pred * pred))
    if obs_var <= float(gr_shape_sigma_floor) ** 2 or pred_var <= float(gr_shape_sigma_floor) ** 2:
        return 0.0
    corr = float(np.mean(obs * pred) / math.sqrt(max(obs_var * pred_var, 1e-18)))
    corr = float(np.clip(corr, -1.0, 1.0))
    bonus = max(corr, 0.0)
    return float(gr_shape_power) * float(valid_idx.size) * bonus


def _seed_ensemble_weights(seed_liks, mode="likelihood", lik_scale=5.0):
    seed_liks = np.asarray(seed_liks, dtype=np.float64)
    if seed_liks.size == 0:
        return seed_liks
    finite = np.isfinite(seed_liks)
    if mode == "equal" or not finite.any():
        return np.ones(seed_liks.size, dtype=np.float64) / float(seed_liks.size)
    if mode == "winner":
        weights = np.zeros(seed_liks.size, dtype=np.float64)
        weights[int(np.nanargmax(seed_liks))] = 1.0
        return weights
    if mode == "rank":
        order = np.argsort(np.where(finite, seed_liks, -np.inf))[::-1]
        weights = np.zeros(seed_liks.size, dtype=np.float64)
        ranks = np.arange(seed_liks.size, 0, -1, dtype=np.float64)
        weights[order] = ranks
        return weights / weights.sum()
    if mode != "likelihood":
        raise ValueError(f"unknown seed ensemble weight mode={mode!r}")
    scale = max(float(lik_scale), 1e-6)
    shifted = np.where(finite, seed_liks, -np.inf) - np.max(seed_liks[finite])
    weights = np.exp(shifted / scale)
    weight_sum = weights.sum()
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.ones(seed_liks.size, dtype=np.float64) / float(seed_liks.size)
    return weights / weight_sum


def _profile_seed_ensemble_weights(
    seed_liks,
    profile_indices,
    profiles,
    mode="likelihood",
    lik_scale=5.0,
    mode_key=None,
):
    seed_liks = np.asarray(seed_liks, dtype=np.float64)
    profile_indices = np.asarray(profile_indices, dtype=np.int64)
    if not profiles or seed_liks.size == 0:
        return _seed_ensemble_weights(seed_liks, mode=mode, lik_scale=lik_scale)
    if profile_indices.size != seed_liks.size:
        raise ValueError("profile_indices must have one entry per seed likelihood")

    out = np.zeros(seed_liks.size, dtype=np.float64)
    profile_weights = np.asarray([float(profile["weight"]) for profile in profiles], dtype=np.float64)
    profile_weights /= profile_weights.sum()
    for profile_idx, profile_weight in enumerate(profile_weights):
        seed_mask = profile_indices == profile_idx
        if not seed_mask.any():
            continue
        profile_overrides = profiles[profile_idx].get("overrides", {})
        local_mode = str(profile_overrides.get(mode_key, mode)) if mode_key is not None else str(mode)
        local_lik_scale = float(profile_overrides.get("lik_scale", lik_scale))
        local_weights = _seed_ensemble_weights(
            seed_liks[seed_mask],
            mode=local_mode,
            lik_scale=local_lik_scale,
        )
        out[seed_mask] = profile_weight * local_weights

    total = out.sum()
    if total <= 0.0 or not np.isfinite(total):
        return np.ones(seed_liks.size, dtype=np.float64) / float(seed_liks.size)
    return out / total


def _combine_seed_paths(path_stack, seed_liks, mode="likelihood", lik_scale=5.0):
    path_stack = np.asarray(path_stack, dtype=np.float64)
    mode = str(mode)
    if mode == "median":
        return _nanmedian_seed_path(path_stack).astype(np.float32)
    if mode == "trimmed":
        return _trimmed_seed_path_mean(path_stack, trim_frac=0.10).astype(np.float32)
    if mode == "rank_trimmed":
        weights = _seed_ensemble_weights(seed_liks, mode="rank", lik_scale=lik_scale)
        return _weighted_trimmed_seed_path_mean(path_stack, weights, trim_frac=0.10).astype(np.float32)
    weights = _seed_ensemble_weights(seed_liks, mode=mode, lik_scale=lik_scale)
    valid_path = np.isfinite(path_stack)
    weighted = np.where(valid_path, path_stack, 0.0) * weights[:, None]
    denom = valid_path.astype(np.float64).T @ weights
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float32)
    np.divide(weighted.sum(axis=0), denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _combine_seed_paths_with_weights(path_stack, weights):
    path_stack = np.asarray(path_stack, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if path_stack.ndim != 2:
        raise ValueError("path_stack must be a 2D seed x bin array")
    if weights.size != path_stack.shape[0]:
        raise ValueError("weights must have one value per seed path")
    valid_path = np.isfinite(path_stack)
    weighted = np.where(valid_path, path_stack, 0.0) * weights[:, None]
    denom = valid_path.astype(np.float64).T @ weights
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float32)
    np.divide(weighted.sum(axis=0), denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _trimmed_seed_path_mean(path_stack, trim_frac=0.10):
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float64)
    trim_frac = float(np.clip(trim_frac, 0.0, 0.45))
    for col in range(path_stack.shape[1]):
        vals = path_stack[:, col]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        vals.sort()
        k = int(math.floor(vals.size * trim_frac))
        if 2 * k < vals.size:
            vals = vals[k : vals.size - k]
        out[col] = float(np.mean(vals))
    return out


def _weighted_trimmed_seed_path_mean(path_stack, weights, trim_frac=0.10):
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    trim_frac = float(np.clip(trim_frac, 0.0, 0.45))
    for col in range(path_stack.shape[1]):
        vals = path_stack[:, col]
        valid = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        vals = vals[valid]
        w = weights[valid]
        if vals.size == 0:
            continue
        order = np.argsort(vals)
        vals = vals[order]
        w = w[order]
        if trim_frac > 0.0 and vals.size >= 5:
            cdf = np.cumsum(w) / np.sum(w)
            keep = (cdf >= trim_frac) & (cdf <= 1.0 - trim_frac)
            if keep.any():
                vals = vals[keep]
                w = w[keep]
        out[col] = float(np.sum(vals * w) / np.sum(w))
    return out


def _nanmedian_seed_path(path_stack):
    out = np.full(path_stack.shape[1], np.nan, dtype=np.float64)
    for col in range(path_stack.shape[1]):
        vals = path_stack[:, col]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[col] = float(np.median(vals))
    return out



def _prepare_ref_grid(tw_tvt, tw_gr, step=0.2):
    valid = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    tw_tvt = tw_tvt[valid]
    tw_gr = tw_gr[valid]
    order = np.argsort(tw_tvt)
    tw_tvt = tw_tvt[order]
    tw_gr = tw_gr[order]
    grid_min = float(tw_tvt[0])
    grid_max = float(tw_tvt[-1])
    grid_tvt = np.arange(grid_min, grid_max + float(step), float(step), dtype=np.float64)
    return np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float64), grid_min, float(step)


def _run_numba_pf_heatmap_bins(
    query_md,
    query_z,
    query_gr,
    query_gr_power,
    query_gr_finite,
    obs_gr,
    obs_power,
    obs_finite,
    obs_md_delta,
    obs_z,
    obs_count,
    obs_likelihood_mode,
    shape_gr,
    shape_md_delta,
    shape_z,
    shape_count,
    known_md,
    known_z,
    known_gr_for_sigma,
    known_tvt,
    tw_tvt,
    tw_gr,
    raw_tw_tvt,
    raw_tw_gr,
    tvt0,
    bin_idx,
    num_bins,
    typewell_window,
    typewell_len,
    n_particles,
    seed,
    cache_cfg=None,
):
    known_mask = np.isfinite(known_md) & np.isfinite(known_z) & np.isfinite(known_tvt)
    known_md = known_md[known_mask]
    known_z = known_z[known_mask]
    known_tvt = known_tvt[known_mask]
    known_gr_for_sigma = known_gr_for_sigma[known_mask]
    if len(query_md) == 0 or len(known_tvt) < 2 or len(tw_tvt) < 3:
        return {
            "pf_prob": np.zeros((num_bins, typewell_len), dtype=np.float32),
            "pf_particle_density_prob": np.zeros((num_bins, typewell_len), dtype=np.float32),
            "pf_tvt_pred": np.full(num_bins, np.nan, dtype=np.float32),
            "log_lik": -np.inf,
        }

    if cache_cfg is None:
        cache_cfg = {}
    shape_power = 0.0
    shape_mode_code = 0
    shape_ref_mode_code = 0
    shape_min_points = 7
    shape_sigma_floor = 0.35
    ref_grid, grid_min, grid_step = _prepare_ref_grid(tw_tvt, tw_gr, step=float(cache_cfg.get("ref_grid_step", 0.2)))
    tw_at_k = np.interp(known_tvt, tw_tvt, tw_gr)
    sigma = float(np.clip(np.nanstd(known_gr_for_sigma - tw_at_k), 10.0, 60.0))
    raw_ref_weight = float(cache_cfg.get("raw_ref_likelihood_weight", 0.0))
    need_raw_ref_grid = raw_ref_weight > 0.0
    if need_raw_ref_grid and len(raw_tw_tvt) >= 3:
        raw_ref_grid, raw_grid_min, raw_grid_step = _prepare_ref_grid(
            raw_tw_tvt,
            raw_tw_gr,
            step=float(cache_cfg.get("ref_grid_step", 0.2)),
        )
        raw_ref_weight = float(np.clip(raw_ref_weight, 0.0, 1.0))
        if shape_ref_mode_code == 1 and raw_ref_weight <= 0.0:
            raw_ref_weight = 0.0
    else:
        raw_ref_grid = np.asarray([0.0], dtype=np.float64)
        raw_grid_min = 0.0
        raw_grid_step = 1.0
        raw_ref_weight = 0.0
    ambiguity_power = float(cache_cfg.get("gr_ambiguity_power", 0.0))
    if ambiguity_power > 0.0:
        amb_mode = str(cache_cfg.get("gr_ambiguity_ref_mode", "primary"))
        if amb_mode == "raw" and len(raw_tw_tvt) >= 3:
            amb_tvt = raw_tw_tvt
            amb_gr = raw_tw_gr
        else:
            amb_tvt = tw_tvt
            amb_gr = tw_gr
        query_gr_power = query_gr_power * _local_gr_ambiguity_power(
            query_gr=query_gr,
            query_gr_finite=query_gr_finite,
            tvt0=tvt0,
            tw_tvt=amb_tvt,
            tw_gr=amb_gr,
            gr_sigma=sigma,
            typewell_window=typewell_window,
            ambiguity_power=ambiguity_power,
            min_power=float(cache_cfg.get("gr_ambiguity_min_power", 0.35)),
            contrast=float(cache_cfg.get("gr_ambiguity_contrast", 1.0)),
        )
    information_power = float(cache_cfg.get("gr_information_power", 0.0))
    if information_power > 0.0:
        info_mode = str(cache_cfg.get("gr_information_ref_mode", "primary"))
        if info_mode == "raw" and len(raw_tw_tvt) >= 3:
            info_tvt = raw_tw_tvt
            info_gr = raw_tw_gr
        else:
            info_tvt = tw_tvt
            info_gr = tw_gr
        query_gr_power = query_gr_power * _local_gr_information_multiplier(
            query_gr=query_gr,
            query_gr_finite=query_gr_finite,
            tvt0=tvt0,
            tw_tvt=info_tvt,
            tw_gr=info_gr,
            gr_sigma=sigma,
            typewell_window=typewell_window,
            information_power=information_power,
            center=float(cache_cfg.get("gr_information_center", 0.75)),
            min_multiplier=float(cache_cfg.get("gr_information_min_multiplier", 0.75)),
            max_multiplier=float(cache_cfg.get("gr_information_max_multiplier", 1.25)),
            slope_weight=float(cache_cfg.get("gr_information_slope_weight", 0.0)),
        )
    tail_start = max(0, len(known_tvt) - 30)
    dt = np.diff(known_tvt[tail_start:])
    dz = np.diff(known_z[tail_start:])
    dm = np.diff(known_md[tail_start:])
    move = dm > 0
    state_z_weight = float(cache_cfg.get("pf_state_z_weight", 1.0))
    init_rate = float(np.median((dt + state_z_weight * dz)[move] / dm[move])) if move.sum() >= 3 else 0.0
    init_pos = float(known_tvt[-1] + state_z_weight * known_z[-1])
    initial_prev_md = float(known_md[-1])
    tvt_min = float(tw_tvt[0] - 100.0)
    tvt_max = float(tw_tvt[-1] + 100.0)
    ffbsi_mode = str(cache_cfg.get("pf_ffbsi_mode", "off")).lower()
    ffbsi_n_paths = int(cache_cfg.get("pf_ffbsi_n_paths", 0))
    ffbsi_fallback_mode = str(cache_cfg.get("pf_ffbsi_fallback_mode", "filtered")).lower()
    if ffbsi_mode in {"on", "ffbsi", "fallback"} and ffbsi_n_paths > 0:
        try:
            (
                filtered_hist_bins,
                filtered_path_sums,
                filtered_count_bins,
                filtered_particle_hist_bins,
                ffbsi_hist_bins,
                ffbsi_path_sums,
                ffbsi_count_bins,
                log_lik,
                ffbsi_ok,
            ) = _pf_heatmap_ffbsi_kernel(
                query_md.astype(np.float64),
                query_z.astype(np.float64),
                query_gr.astype(np.float64),
                query_gr_power.astype(np.float64),
                query_gr_finite.astype(np.float64),
                obs_gr.astype(np.float64),
                obs_power.astype(np.float64),
                obs_finite.astype(np.float64),
                obs_md_delta.astype(np.float64),
                obs_z.astype(np.float64),
                obs_count.astype(np.int64),
                int(obs_likelihood_mode),
                shape_gr.astype(np.float64),
                shape_md_delta.astype(np.float64),
                shape_z.astype(np.float64),
                shape_count.astype(np.int64),
                ref_grid,
                grid_min,
                grid_step,
                raw_ref_grid,
                raw_grid_min,
                raw_grid_step,
                raw_ref_weight,
                sigma,
                float(cache_cfg.get("outlier_prob", 0.0)),
                float(cache_cfg.get("outlier_likelihood", 0.05)),
                float(cache_cfg.get("dynamic_sigma_alpha", 0.0)),
                float(cache_cfg.get("dynamic_sigma_threshold", 1.25)),
                float(cache_cfg.get("dynamic_sigma_power", 1.0)),
                float(cache_cfg.get("dynamic_sigma_min", 0.85)),
                float(cache_cfg.get("dynamic_sigma_max", 2.0)),
                float(shape_power),
                int(shape_mode_code),
                int(shape_min_points),
                float(shape_sigma_floor),
                int(shape_ref_mode_code),
                init_pos,
                init_rate,
                initial_prev_md,
                float(tvt0),
                bin_idx.astype(np.int64),
                int(num_bins),
                float(typewell_window),
                int(typewell_len),
                int(n_particles),
                tvt_min,
                tvt_max,
                int(seed),
                float(cache_cfg.get("pf_momentum", 0.998)),
                float(cache_cfg.get("pf_rate_noise", 0.002)),
                float(cache_cfg.get("pf_pos_noise", 0.005)),
                float(cache_cfg.get("pf_resample_threshold", 0.5)),
                float(cache_cfg.get("pf_resample_obs_power_adapt", 0.0)),
                float(cache_cfg.get("pf_resample_min_threshold", 0.0)),
                float(cache_cfg.get("pf_rough_pos", 0.1)),
                float(cache_cfg.get("pf_rough_rate", 0.001)),
                float(cache_cfg.get("pf_rescue_frac", 0.0)),
                float(cache_cfg.get("pf_rescue_pos_sd", 0.0)),
                float(cache_cfg.get("pf_rescue_rate_sd", 0.0)),
                float(cache_cfg.get("pf_init_pos_sd", 3.0)),
                float(cache_cfg.get("pf_init_rate_sd", 0.01)),
                float(cache_cfg.get("pf_rate_mean_weight", 0.0)),
                float(cache_cfg.get("pf_jump_prob", 0.0)),
                float(cache_cfg.get("pf_jump_sd", 0.0)),
                float(cache_cfg.get("pf_jump_rate_sd", 0.0)),
                float(cache_cfg.get("pf_missing_jump_boost", 0.0)),
                float(cache_cfg.get("pf_jump_tail_prob", 0.0)),
                float(cache_cfg.get("pf_jump_tail_sd", 0.0)),
                float(cache_cfg.get("pf_jump_tail_rate_sd", 0.0)),
                int(cache_cfg.get("pf_jump_tail_dist", 1)),
                float(cache_cfg.get("pf_jump_tail_clip", 0.0)),
                float(cache_cfg.get("pf_jump_tail_missing_boost", 0.0)),
                float(cache_cfg.get("pf_anchor_sigma", 0.0)),
                float(cache_cfg.get("pf_anchor_power", 1.0)),
                float(cache_cfg.get("pf_correlated_rate_alpha", 0.0)),
                float(cache_cfg.get("pf_correlated_pos_alpha", 0.0)),
                float(cache_cfg.get("pf_lookahead_power", 0.0)),
                int(cache_cfg.get("pf_lookahead_steps", 1)),
                float(cache_cfg.get("pf_lookahead_decay", 0.5)),
                float(cache_cfg.get("pf_lookahead_max_gap", 256.0)),
                float(cache_cfg.get("pf_lookahead_delta_power", 0.0)),
                int(cache_cfg.get("pf_stratified_init", 0)),
                float(cache_cfg.get("pf_finite_run_power_boost", 0.0)),
                float(cache_cfg.get("pf_finite_run_power_cap", 1.0)),
                float(cache_cfg.get("pf_finite_run_power_decay", 0.0)),
                float(cache_cfg.get("pf_finite_run_power_floor", 0.35)),
                float(cache_cfg.get("pf_conf_obs_power_decay", 0.0)),
                float(cache_cfg.get("pf_conf_obs_power_floor", 0.55)),
                float(cache_cfg.get("pf_conf_obs_power_power", 1.0)),
                float(cache_cfg.get("pf_missing_noise_scale", 1.0)),
                float(cache_cfg.get("pf_missing_jump_scale", 1.0)),
                float(cache_cfg.get("pf_ess_jump_boost", 0.0)),
                float(cache_cfg.get("pf_ess_jump_power", 1.0)),
                float(cache_cfg.get("pf_surprise_jump_boost", 0.0)),
                float(cache_cfg.get("pf_surprise_jump_threshold", 1.5)),
                float(cache_cfg.get("pf_surprise_jump_power", 1.0)),
                float(cache_cfg.get("pf_ess_rough_boost", 0.0)),
                float(cache_cfg.get("pf_ess_rough_power", 1.0)),
                state_z_weight,
                ffbsi_n_paths,
                int(cache_cfg.get("pf_ffbsi_max_active_bins", 512)),
                float(cache_cfg.get("pf_ffbsi_transition_scale", 1.5)),
                float(cache_cfg.get("pf_ffbsi_pos_floor", 0.35)),
                float(cache_cfg.get("pf_ffbsi_rate_floor", 0.0025)),
            )
            if ffbsi_ok > 0:
                pf_prob, pf_tvt_pred = _pf_arrays_from_hist(
                    filtered_hist_bins,
                    filtered_path_sums,
                    filtered_count_bins,
                    num_bins,
                    typewell_len,
                )
                pf_particle_density_prob = _pf_prob_from_hist(
                    filtered_particle_hist_bins,
                    filtered_count_bins,
                    num_bins,
                    typewell_len,
                )
                pf_prob_ffbsi, pf_tvt_pred_ffbsi = _pf_arrays_from_hist(
                    ffbsi_hist_bins,
                    ffbsi_path_sums,
                    ffbsi_count_bins,
                    num_bins,
                    typewell_len,
                )
                return {
                    "pf_prob": pf_prob,
                    "pf_particle_density_prob": pf_particle_density_prob,
                    "pf_tvt_pred": pf_tvt_pred,
                    "pf_prob_ffbsi": pf_prob_ffbsi,
                    "pf_tvt_pred_ffbsi": pf_tvt_pred_ffbsi,
                    "log_lik": float(log_lik),
                    "ffbsi_ok": True,
                }
            if ffbsi_fallback_mode not in {"filtered", "on", "true", "1"}:
                raise RuntimeError("FFBSi heatmap kernel could not run for this well")
        except Exception:
            if ffbsi_fallback_mode not in {"filtered", "on", "true", "1"}:
                raise
    hist_bins, path_sums, count_bins, particle_hist_bins, log_lik = _pf_heatmap_kernel(
        query_md.astype(np.float64),
        query_z.astype(np.float64),
        query_gr.astype(np.float64),
        query_gr_power.astype(np.float64),
        query_gr_finite.astype(np.float64),
        obs_gr.astype(np.float64),
        obs_power.astype(np.float64),
        obs_finite.astype(np.float64),
        obs_md_delta.astype(np.float64),
        obs_z.astype(np.float64),
        obs_count.astype(np.int64),
        int(obs_likelihood_mode),
        shape_gr.astype(np.float64),
        shape_md_delta.astype(np.float64),
        shape_z.astype(np.float64),
        shape_count.astype(np.int64),
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        sigma,
        float(cache_cfg.get("outlier_prob", 0.0)),
        float(cache_cfg.get("outlier_likelihood", 0.05)),
        float(cache_cfg.get("dynamic_sigma_alpha", 0.0)),
        float(cache_cfg.get("dynamic_sigma_threshold", 1.25)),
        float(cache_cfg.get("dynamic_sigma_power", 1.0)),
        float(cache_cfg.get("dynamic_sigma_min", 0.85)),
        float(cache_cfg.get("dynamic_sigma_max", 2.0)),
        float(shape_power),
        int(shape_mode_code),
        int(shape_min_points),
        float(shape_sigma_floor),
        int(shape_ref_mode_code),
        init_pos,
        init_rate,
        initial_prev_md,
        float(tvt0),
        bin_idx.astype(np.int64),
        int(num_bins),
        float(typewell_window),
        int(typewell_len),
        int(n_particles),
        tvt_min,
        tvt_max,
        int(seed),
        float(cache_cfg.get("pf_momentum", 0.998)),
        float(cache_cfg.get("pf_rate_noise", 0.002)),
        float(cache_cfg.get("pf_pos_noise", 0.005)),
        float(cache_cfg.get("pf_resample_threshold", 0.5)),
        float(cache_cfg.get("pf_resample_obs_power_adapt", 0.0)),
        float(cache_cfg.get("pf_resample_min_threshold", 0.0)),
        float(cache_cfg.get("pf_rough_pos", 0.1)),
        float(cache_cfg.get("pf_rough_rate", 0.001)),
        float(cache_cfg.get("pf_rescue_frac", 0.0)),
        float(cache_cfg.get("pf_rescue_pos_sd", 0.0)),
        float(cache_cfg.get("pf_rescue_rate_sd", 0.0)),
        float(cache_cfg.get("pf_init_pos_sd", 3.0)),
        float(cache_cfg.get("pf_init_rate_sd", 0.01)),
        float(cache_cfg.get("pf_rate_mean_weight", 0.0)),
        float(cache_cfg.get("pf_jump_prob", 0.0)),
        float(cache_cfg.get("pf_jump_sd", 0.0)),
        float(cache_cfg.get("pf_jump_rate_sd", 0.0)),
        float(cache_cfg.get("pf_missing_jump_boost", 0.0)),
        float(cache_cfg.get("pf_jump_tail_prob", 0.0)),
        float(cache_cfg.get("pf_jump_tail_sd", 0.0)),
        float(cache_cfg.get("pf_jump_tail_rate_sd", 0.0)),
        int(cache_cfg.get("pf_jump_tail_dist", 1)),
        float(cache_cfg.get("pf_jump_tail_clip", 0.0)),
        float(cache_cfg.get("pf_jump_tail_missing_boost", 0.0)),
        float(cache_cfg.get("pf_anchor_sigma", 0.0)),
        float(cache_cfg.get("pf_anchor_power", 1.0)),
        float(cache_cfg.get("pf_correlated_rate_alpha", 0.0)),
        float(cache_cfg.get("pf_correlated_pos_alpha", 0.0)),
        float(cache_cfg.get("pf_lookahead_power", 0.0)),
        int(cache_cfg.get("pf_lookahead_steps", 1)),
        float(cache_cfg.get("pf_lookahead_decay", 0.5)),
        float(cache_cfg.get("pf_lookahead_max_gap", 256.0)),
        float(cache_cfg.get("pf_lookahead_delta_power", 0.0)),
        int(cache_cfg.get("pf_stratified_init", 0)),
        float(cache_cfg.get("pf_finite_run_power_boost", 0.0)),
        float(cache_cfg.get("pf_finite_run_power_cap", 1.0)),
        float(cache_cfg.get("pf_finite_run_power_decay", 0.0)),
        float(cache_cfg.get("pf_finite_run_power_floor", 0.35)),
        float(cache_cfg.get("pf_conf_obs_power_decay", 0.0)),
        float(cache_cfg.get("pf_conf_obs_power_floor", 0.55)),
        float(cache_cfg.get("pf_conf_obs_power_power", 1.0)),
        float(cache_cfg.get("pf_missing_noise_scale", 1.0)),
        float(cache_cfg.get("pf_missing_jump_scale", 1.0)),
        float(cache_cfg.get("pf_ess_jump_boost", 0.0)),
        float(cache_cfg.get("pf_ess_jump_power", 1.0)),
        float(cache_cfg.get("pf_surprise_jump_boost", 0.0)),
        float(cache_cfg.get("pf_surprise_jump_threshold", 1.5)),
        float(cache_cfg.get("pf_surprise_jump_power", 1.0)),
        float(cache_cfg.get("pf_ess_rough_boost", 0.0)),
        float(cache_cfg.get("pf_ess_rough_power", 1.0)),
        state_z_weight,
    )
    pf_prob = np.zeros((num_bins, typewell_len), dtype=np.float32)
    valid_bins = count_bins > 0
    pf_prob[valid_bins] = (hist_bins[valid_bins] / count_bins[valid_bins, None]).astype(np.float32)
    pf_prob = _normalize_prob_rows(pf_prob)
    pf_particle_density_prob = _pf_prob_from_hist(particle_hist_bins, count_bins, num_bins, typewell_len)
    pf_tvt_pred = np.full(num_bins, np.nan, dtype=np.float32)
    pf_tvt_pred[valid_bins] = (path_sums[valid_bins] / count_bins[valid_bins]).astype(np.float32)
    return {
        "pf_prob": pf_prob,
        "pf_particle_density_prob": pf_particle_density_prob,
        "pf_tvt_pred": pf_tvt_pred,
        "log_lik": float(log_lik),
        "ffbsi_ok": False,
    }


def _merge_suffix_query_blocks(
    query_md,
    query_z,
    query_gr,
    query_gr_power,
    query_gr_finite,
    merge_k,
    power_alpha,
    likelihood_mode,
    state_mode,
):
    query_md = np.asarray(query_md, dtype=np.float64)
    query_z = np.asarray(query_z, dtype=np.float64)
    query_gr = np.asarray(query_gr, dtype=np.float64)
    query_gr_power = np.asarray(query_gr_power, dtype=np.float64)
    query_gr_finite = np.asarray(query_gr_finite, dtype=np.float64)
    merge_k = max(1, int(merge_k))
    power_alpha = float(power_alpha)
    likelihood_mode = str(likelihood_mode).lower()
    state_mode = str(state_mode).lower()

    n_rows = int(query_md.size)
    if merge_k <= 1 or n_rows <= 1:
        starts = np.arange(n_rows, dtype=np.int64)
        ends = starts + 1
        return (
            query_md.copy(),
            query_z.copy(),
            query_gr.copy(),
            query_gr_power.copy(),
            query_gr_finite.copy(),
            query_gr[:, None].copy(),
            query_gr_power[:, None].copy(),
            query_gr_finite[:, None].copy(),
            np.zeros((n_rows, 1), dtype=np.float64),
            query_z[:, None].copy(),
            np.ones(n_rows, dtype=np.int64),
            starts,
            ends,
        )

    starts = np.arange(0, n_rows, merge_k, dtype=np.int64)
    ends = np.minimum(starts + merge_k, n_rows).astype(np.int64, copy=False)
    n_blocks = int(starts.size)
    merged_md = np.zeros(n_blocks, dtype=np.float64)
    merged_z = np.zeros(n_blocks, dtype=np.float64)
    merged_gr = np.zeros(n_blocks, dtype=np.float64)
    merged_power = np.zeros(n_blocks, dtype=np.float64)
    merged_finite = np.zeros(n_blocks, dtype=np.float64)
    obs_width = int(merge_k)
    obs_gr = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_power = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_finite = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_md_delta = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_z = np.zeros((n_blocks, obs_width), dtype=np.float64)
    obs_count = np.zeros(n_blocks, dtype=np.int64)

    for block_idx, (start, end) in enumerate(zip(starts, ends)):
        if state_mode == "end":
            state_row = end - 1
        else:
            state_row = start
        merged_md[block_idx] = query_md[state_row]
        merged_z[block_idx] = query_z[state_row]
        block_gr = query_gr[start:end]
        block_power = np.clip(query_gr_power[start:end], 0.0, None)
        block_finite = np.clip(query_gr_finite[start:end], 0.0, 1.0)
        block_len = int(end - start)
        obs_count[block_idx] = block_len
        obs_gr[block_idx, :block_len] = block_gr
        obs_power[block_idx, :block_len] = block_power
        obs_finite[block_idx, :block_len] = block_finite
        obs_md_delta[block_idx, :block_len] = query_md[start:end] - merged_md[block_idx]
        obs_z[block_idx, :block_len] = query_z[start:end]
        weight_sum = float(np.sum(block_power))
        if weight_sum > 0.0 and np.isfinite(weight_sum):
            merged_gr[block_idx] = float(np.sum(block_gr * block_power) / weight_sum)
        else:
            merged_gr[block_idx] = float(np.mean(block_gr)) if block_gr.size else 0.0
        if weight_sum > 0.0:
            merged_power[block_idx] = float(math.pow(max(weight_sum, 0.0), power_alpha))
            if likelihood_mode == "block":
                obs_power[block_idx, :block_len] *= merged_power[block_idx] / weight_sum
        else:
            merged_power[block_idx] = 0.0
        merged_finite[block_idx] = float(np.mean(block_finite)) if block_finite.size else 0.0

    return (
        merged_md,
        merged_z,
        merged_gr,
        merged_power,
        merged_finite,
        obs_gr,
        obs_power,
        obs_finite,
        obs_md_delta,
        obs_z,
        obs_count,
        starts,
        ends,
    )


def _row_path_from_merged_steps(step_tvt_pred, tvt0, block_starts, block_ends, total_len):
    total_len = int(total_len)
    out = np.full(total_len, np.nan, dtype=np.float32)
    if total_len <= 0:
        return out
    prev_tvt = float(tvt0)
    for step_tvt, start, end in zip(step_tvt_pred, block_starts, block_ends):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        cur_tvt = float(step_tvt) if np.isfinite(step_tvt) else prev_tvt
        span = end - start
        delta = (cur_tvt - prev_tvt) / float(span)
        for offset in range(span):
            out[start + offset] = np.float32(prev_tvt + delta * float(offset + 1))
        prev_tvt = cur_tvt
    last_filled = np.flatnonzero(np.isfinite(out))
    if last_filled.size > 0 and int(last_filled[-1]) + 1 < total_len:
        out[int(last_filled[-1]) + 1 :] = out[int(last_filled[-1])]
    return out


def _expand_merged_prob_to_bins(step_prob, block_starts, block_ends, prefix_len, downsample, num_bins):
    step_prob = np.asarray(step_prob, dtype=np.float32)
    out = np.zeros((int(num_bins), int(step_prob.shape[1])), dtype=np.float32)
    counts = np.zeros(int(num_bins), dtype=np.float32)
    for prob_row, start, end in zip(step_prob, block_starts, block_ends):
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        row_bins = (int(prefix_len) + np.arange(start, end, dtype=np.int64)) // int(downsample)
        row_bins = row_bins[(row_bins >= 0) & (row_bins < int(num_bins))]
        if row_bins.size == 0:
            continue
        uniq, uniq_counts = np.unique(row_bins, return_counts=True)
        for bin_idx, bin_count in zip(uniq, uniq_counts):
            out[int(bin_idx)] += np.float32(bin_count) * prob_row
            counts[int(bin_idx)] += np.float32(bin_count)
    valid = counts > 0.0
    out[valid] /= counts[valid, None]
    return _normalize_prob_rows(out)


def _effective_pf_cache_cfg_for_merge(cache_cfg):
    merge_k = max(1, int(cache_cfg.get("suffix_merge_k", 1)))
    if merge_k <= 1:
        return dict(cache_cfg)
    out = dict(cache_cfg)
    if not bool(cache_cfg.get("suffix_merge_adjust_dynamics", False)):
        return out

    row_momentum = float(np.clip(out.get("pf_momentum", 0.998), 0.0, 0.999999999))
    out["pf_momentum"] = float(np.clip(math.pow(row_momentum, merge_k), 0.0, 0.999999999))

    innovation_scale = math.sqrt(
        float(sum(math.pow(row_momentum, 2 * step_idx) for step_idx in range(merge_k)))
    )
    if innovation_scale > 0.0:
        out["pf_rate_noise"] = float(out.get("pf_rate_noise", 0.0)) * innovation_scale
    out["pf_pos_noise"] = float(out.get("pf_pos_noise", 0.0)) * math.sqrt(float(merge_k))

    for key in ("pf_correlated_rate_alpha", "pf_correlated_pos_alpha"):
        alpha = float(np.clip(out.get(key, 0.0), 0.0, 0.999999999))
        if alpha > 0.0:
            out[key] = float(math.pow(alpha, merge_k))

    rate_mean_weight = float(np.clip(out.get("pf_rate_mean_weight", 0.0), 0.0, 1.0))
    if rate_mean_weight > 0.0:
        out["pf_rate_mean_weight"] = float(1.0 - math.pow(1.0 - rate_mean_weight, merge_k))

    return out


def _warmup_pf_heatmap_numba():
    if njit is None:
        return
    _run_numba_pf_heatmap_bins(
        query_md=np.asarray([1.0, 2.0], dtype=np.float64),
        query_z=np.asarray([-9000.0, -9000.1], dtype=np.float64),
        query_gr=np.asarray([80.0, 81.0], dtype=np.float64),
        query_gr_power=np.asarray([1.0, 1.0], dtype=np.float64),
        query_gr_finite=np.asarray([1.0, 1.0], dtype=np.float64),
        obs_gr=np.asarray([[80.0], [81.0]], dtype=np.float64),
        obs_power=np.asarray([[1.0], [1.0]], dtype=np.float64),
        obs_finite=np.asarray([[1.0], [1.0]], dtype=np.float64),
        obs_md_delta=np.zeros((2, 1), dtype=np.float64),
        obs_z=np.asarray([[-9000.0], [-9000.1]], dtype=np.float64),
        obs_count=np.ones(2, dtype=np.int64),
        obs_likelihood_mode=0,
        shape_gr=np.zeros((2, 1), dtype=np.float64),
        shape_md_delta=np.zeros((2, 1), dtype=np.float64),
        shape_z=np.zeros((2, 1), dtype=np.float64),
        shape_count=np.zeros(2, dtype=np.int64),
        known_md=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        known_z=np.asarray([-9000.0, -9000.05, -9000.1], dtype=np.float64),
        known_gr_for_sigma=np.asarray([80.0, 81.0, 82.0], dtype=np.float64),
        known_tvt=np.asarray([11000.0, 11000.0, 11000.1], dtype=np.float64),
        tw_tvt=np.asarray([10999.0, 11000.0, 11001.0], dtype=np.float64),
        tw_gr=np.asarray([79.0, 80.0, 82.0], dtype=np.float64),
        raw_tw_tvt=np.asarray([10999.0, 11000.0, 11001.0], dtype=np.float64),
        raw_tw_gr=np.asarray([79.0, 80.0, 82.0], dtype=np.float64),
        tvt0=11000.1,
        bin_idx=np.asarray([0, 0], dtype=np.int64),
        num_bins=1,
        typewell_window=10.0,
        typewell_len=20,
        n_particles=4,
        seed=0,
        cache_cfg={},
    )


def _benchmark_numba_vs_python(data_path, well_ids, cfg, log):
    bench_n = min(128, int(getattr(cfg, "PF_heatmap_n_particles", 500)))
    if bench_n <= 0:
        return
    rows = []
    for well_id in well_ids:
        h_df, tw_df = _read_well_frames(data_path, well_id)
        if "TVT" not in h_df.columns:
            continue
        bench_seed = _single_pf_seed(int(getattr(cfg, "PF_heatmap_base_seed", 202605)))
        t0 = time.perf_counter()
        slow_pred, _ = run_particle_filter(h_df, tw_df, n_particles=bench_n, seed=bench_seed)
        slow_sec = time.perf_counter() - t0
        cache_cfg = pf_heatmap_cache_config(cfg)
        cache_cfg["n_particles"] = bench_n
        cache_cfg["n_seeds"] = 1
        t1 = time.perf_counter()
        fast = _build_pf_heatmap_item(h_df, tw_df, cache_cfg, has_target=True)
        fast_sec = time.perf_counter() - t1
        last_seen_idx = int(np.flatnonzero(np.isfinite(h_df["TVT_input"].to_numpy(dtype=np.float64)))[-1])
        suffix_start = last_seen_idx + 1
        kept_len = min(len(h_df) - suffix_start, int(cfg.target_len))
        slow_bin = _downsample_suffix_values_for_benchmark(
            slow_pred[suffix_start : suffix_start + kept_len],
            cfg,
        )
        target_bin = fast["window_tvt"]
        mask = fast["target_mask"] & np.isfinite(target_bin) & np.isfinite(slow_bin) & (fast["pf_tvt_pred"] != 0.0)
        if mask.any():
            slow_target = float(np.sqrt(np.mean(np.square(slow_bin[mask] - target_bin[mask]))))
            fast_target = float(np.sqrt(np.mean(np.square(fast["pf_tvt_pred"][mask] - target_bin[mask]))))
            path_gap = float(np.sqrt(np.mean(np.square(slow_bin[mask] - fast["pf_tvt_pred"][mask]))))
        else:
            slow_target = fast_target = path_gap = math.nan
        rows.append((slow_sec, fast_sec, slow_target, fast_target, path_gap))
    if not rows:
        return
    arr = np.asarray(rows, dtype=float)
    _log(
        log,
        "PF numba smoke: "
        f"slow_sec/well={np.nanmean(arr[:, 0]):.3f}, "
        f"numba_sec/well={np.nanmean(arr[:, 1]):.3f}, "
        f"speedup={np.nanmean(arr[:, 0]) / max(np.nanmean(arr[:, 1]), 1e-9):.2f}x, "
        f"slow_TVT_RMSE={np.nanmean(arr[:, 2]):.4f}, "
        f"numba_TVT_RMSE={np.nanmean(arr[:, 3]):.4f}, "
        f"path_gap_RMSE={np.nanmean(arr[:, 4]):.4f}",
    )


def _downsample_suffix_values_for_benchmark(values, cfg):
    out = np.full(int(cfg.num_bins), np.nan, dtype=np.float32)
    if len(values) == 0:
        return out
    bin_idx = (int(cfg.prefix_len) + np.arange(len(values), dtype=np.int64)) // int(cfg.downsample)
    valid = bin_idx < int(cfg.num_bins)
    sums = np.bincount(bin_idx[valid], weights=np.asarray(values, dtype=np.float32)[valid], minlength=int(cfg.num_bins))
    counts = np.bincount(bin_idx[valid], minlength=int(cfg.num_bins))
    filled = counts > 0
    out[filled] = (sums[filled] / counts[filled]).astype(np.float32)
    return out


if njit is not None:

    @njit(cache=False)
    def _interp_grid_numba(grid, tvt, grid_min, grid_step):
        idx = int((tvt - grid_min) / grid_step)
        if idx < 0:
            return grid[0]
        last = len(grid) - 1
        if idx >= last:
            return grid[last]
        frac = (tvt - grid_min) / grid_step - idx
        return grid[idx] * (1.0 - frac) + grid[idx + 1] * frac

    @njit(cache=False)
    def _pf_gr_shape_likelihood_numba(
        row,
        tvt,
        rate,
        z_row,
        state_z_weight,
        gr_center,
        sigma_eff,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        shape_gr,
        shape_md_delta,
        shape_z,
        shape_count,
        shape_power,
        shape_mode,
        shape_min_points,
        shape_sigma_floor,
        shape_ref_mode,
    ):
        if shape_power <= 0.0 or shape_mode <= 0:
            return 1.0
        n = shape_count[row]
        if n < shape_min_points or n <= 1:
            return 1.0
        shape_width = shape_gr.shape[1]
        if n > shape_width:
            n = shape_width
        if n < shape_min_points or n <= 1:
            return 1.0
        if sigma_eff <= 1e-6:
            sigma_eff = 1e-6
        sigma_floor = shape_sigma_floor * sigma_eff
        if sigma_floor <= 1e-6:
            sigma_floor = 1e-6

        center_pred = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
        if shape_ref_mode == 1:
            center_pred = _interp_grid_numba(raw_ref_grid, tvt, raw_grid_min, raw_grid_step)
        elif shape_ref_mode == 2 and raw_ref_weight > 0.0:
            raw_center_pred = _interp_grid_numba(raw_ref_grid, tvt, raw_grid_min, raw_grid_step)
            center_pred = (1.0 - raw_ref_weight) * center_pred + raw_ref_weight * raw_center_pred

        obs_mean = 0.0
        pred_mean = 0.0
        for k in range(n):
            pred_tvt = (tvt + rate * shape_md_delta[row, k]) + state_z_weight * (z_row - shape_z[row, k])
            pred = _interp_grid_numba(ref_grid, pred_tvt, grid_min, grid_step)
            if shape_ref_mode == 1:
                pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
            elif shape_ref_mode == 2 and raw_ref_weight > 0.0:
                raw_pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
                pred = (1.0 - raw_ref_weight) * pred + raw_ref_weight * raw_pred
            if shape_mode == 1:
                obs_mean += 0.0
                pred_mean += 0.0
            else:
                obs_mean += shape_gr[row, k]
                pred_mean += pred
        if shape_mode != 1:
            obs_mean /= n
            pred_mean /= n

        obs_var = 0.0
        pred_var = 0.0
        cross = 0.0
        for k in range(n):
            pred_tvt = (tvt + rate * shape_md_delta[row, k]) + state_z_weight * (z_row - shape_z[row, k])
            pred = _interp_grid_numba(ref_grid, pred_tvt, grid_min, grid_step)
            if shape_ref_mode == 1:
                pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
            elif shape_ref_mode == 2 and raw_ref_weight > 0.0:
                raw_pred = _interp_grid_numba(raw_ref_grid, pred_tvt, raw_grid_min, raw_grid_step)
                pred = (1.0 - raw_ref_weight) * pred + raw_ref_weight * raw_pred
            if shape_mode == 1:
                obs = shape_gr[row, k] - gr_center
                pred = pred - center_pred
            else:
                obs = shape_gr[row, k]
            obs_d = obs - obs_mean
            pred_d = pred - pred_mean
            obs_var += obs_d * obs_d
            pred_var += pred_d * pred_d
            cross += obs_d * pred_d
        if obs_var <= sigma_floor * sigma_floor or pred_var <= sigma_floor * sigma_floor:
            return 1.0
        corr = cross / math.sqrt(obs_var * pred_var)
        if corr > 1.0:
            corr = 1.0
        elif corr < -1.0:
            corr = -1.0
        bonus = max(corr, 0.0)
        if shape_mode == 2:
            bonus = 0.5 * (1.0 + corr)
        lk = 1.0 + shape_power * bonus
        if lk < 1e-300:
            lk = 1e-300
        return lk

    @njit(cache=False)
    def _pf_gr_point_likelihood_numba(
        obs_gr,
        tvt,
        obs_power,
        obs_finite,
        sigma_eff,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        outlier_prob,
        outlier_likelihood,
    ):
        if obs_power <= 0.0:
            return 1.0
        expected_gr = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
        d = (obs_gr - expected_gr) / sigma_eff
        d2 = d * d
        if d2 > 600.0:
            d2 = 600.0
        primary_lk = math.exp(-0.5 * d2 * obs_power)
        if raw_ref_weight > 0.0:
            raw_expected_gr = _interp_grid_numba(raw_ref_grid, tvt, raw_grid_min, raw_grid_step)
            raw_d = (obs_gr - raw_expected_gr) / sigma_eff
            raw_d2 = raw_d * raw_d
            if raw_d2 > 600.0:
                raw_d2 = 600.0
            raw_lk = math.exp(-0.5 * raw_d2 * obs_power)
            primary_lk = (1.0 - raw_ref_weight) * primary_lk + raw_ref_weight * raw_lk
        if outlier_prob > 0.0 and obs_finite > 0.0:
            primary_lk = (1.0 - outlier_prob) * primary_lk + outlier_prob * outlier_likelihood
        if primary_lk < 1e-300:
            primary_lk = 1e-300
        return primary_lk

    @njit(cache=False)
    def _pf_resample_numba(pos, rate, weight, rough_pos, rough_rate):
        n = len(pos)
        cum = np.empty(n)
        total = 0.0
        for j in range(n):
            total += weight[j]
            cum[j] = total
        start = np.random.uniform(0.0, 1.0 / n)
        new_pos = np.empty(n)
        new_rate = np.empty(n)
        cursor = 0
        for j in range(n):
            u = start + j / n
            while cursor < n - 1 and cum[cursor] < u:
                cursor += 1
            new_pos[j] = pos[cursor] + rough_pos * np.random.randn()
            new_rate[j] = rate[cursor] + rough_rate * np.random.randn()
        return new_pos, new_rate

    @njit(cache=False)
    def _pf_jump_noise_numba(sd, dist, clip_abs):
        if sd <= 0.0:
            return 0.0
        if dist == 1:
            # Laplace noise parameterized by standard deviation.
            u = np.random.random() - 0.5
            scale = sd * 0.7071067811865475
            if u >= 0.0:
                x = -scale * math.log(max(1e-12, 1.0 - 2.0 * u))
            else:
                x = scale * math.log(max(1e-12, 1.0 + 2.0 * u))
        elif dist == 2:
            # Clipped Cauchy-like tail. Use only with a finite clip.
            x = sd * 0.25 * math.tan(math.pi * (np.random.random() - 0.5))
        else:
            x = sd * np.random.randn()
        if clip_abs > 0.0:
            if x > clip_abs:
                x = clip_abs
            elif x < -clip_abs:
                x = -clip_abs
        return x

    @njit(cache=False)
    def _pf_sample_weight_index_numba(weight, n):
        total = 0.0
        for j in range(n):
            if weight[j] > 0.0 and np.isfinite(weight[j]):
                total += weight[j]
        if total <= 0.0:
            return int(np.random.randint(0, n))
        u = np.random.random() * total
        c = 0.0
        for j in range(n):
            if weight[j] > 0.0 and np.isfinite(weight[j]):
                c += weight[j]
                if c >= u:
                    return j
        return n - 1

    @njit(cache=False)
    def _pf_sample_log_score_index_numba(log_score, n):
        best = 0
        best_score = -1e300
        for j in range(n):
            if log_score[j] > best_score:
                best_score = log_score[j]
                best = j
        if best_score <= -1e290:
            return best
        total = 0.0
        for j in range(n):
            total += math.exp(log_score[j] - best_score)
        if total <= 0.0 or not np.isfinite(total):
            return best
        u = np.random.random() * total
        c = 0.0
        for j in range(n):
            c += math.exp(log_score[j] - best_score)
            if c >= u:
                return j
        return best

    @njit(cache=False)
    def _pf_transition_scale_numba(
        dmd,
        gr_power,
        gr_finite,
        rate_noise,
        pos_noise,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_missing_boost,
        missing_noise_scale,
        missing_jump_scale,
        transition_scale,
        pos_floor,
        rate_floor,
    ):
        missing_frac = 1.0 - gr_finite
        if missing_frac < 0.0:
            missing_frac = 0.0
        elif missing_frac > 1.0:
            missing_frac = 1.0
        local_rate_noise = rate_noise
        local_pos_noise = pos_noise
        if missing_frac > 0.0 and missing_noise_scale != 1.0:
            noise_scale = 1.0 + (missing_noise_scale - 1.0) * missing_frac
            local_rate_noise *= noise_scale
            local_pos_noise *= noise_scale
        step_jump_prob = jump_prob * dmd
        if missing_frac > 0.0 and missing_jump_scale != 1.0:
            step_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
        if gr_power < 1.0 and missing_jump_boost > 0.0:
            step_jump_prob += missing_jump_boost * (1.0 - gr_power) * dmd
        if step_jump_prob > 1.0:
            step_jump_prob = 1.0
        elif step_jump_prob < 0.0:
            step_jump_prob = 0.0

        step_tail_jump_prob = jump_tail_prob * dmd
        if missing_frac > 0.0 and missing_jump_scale != 1.0:
            step_tail_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
        if gr_power < 1.0 and jump_tail_missing_boost > 0.0:
            step_tail_jump_prob += jump_tail_missing_boost * (1.0 - gr_power) * dmd
        if step_tail_jump_prob > 1.0:
            step_tail_jump_prob = 1.0
        elif step_tail_jump_prob < 0.0:
            step_tail_jump_prob = 0.0

        rate_var = local_rate_noise * local_rate_noise
        pos_var = local_pos_noise * local_pos_noise + dmd * dmd * local_rate_noise * local_rate_noise
        if step_jump_prob > 0.0:
            pos_var += step_jump_prob * jump_sd * jump_sd
            rate_var += step_jump_prob * jump_rate_sd * jump_rate_sd
        if step_tail_jump_prob > 0.0:
            pos_var += step_tail_jump_prob * jump_tail_sd * jump_tail_sd
            rate_var += step_tail_jump_prob * jump_tail_rate_sd * jump_tail_rate_sd
        pos_sd = math.sqrt(max(pos_var, 1e-18)) * transition_scale
        rate_sd = math.sqrt(max(rate_var, 1e-18)) * transition_scale
        if pos_sd < pos_floor:
            pos_sd = pos_floor
        if rate_sd < rate_floor:
            rate_sd = rate_floor
        return pos_sd, rate_sd

    @njit(cache=False)
    def _pf_heatmap_kernel(
        md,
        z,
        gr,
        gr_power,
        gr_finite,
        block_obs_gr,
        block_obs_power,
        block_obs_finite,
        block_obs_md_delta,
        block_obs_z,
        block_obs_count,
        obs_likelihood_mode,
        shape_gr,
        shape_md_delta,
        shape_z,
        shape_count,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        gr_sigma,
        outlier_prob,
        outlier_likelihood,
        dynamic_sigma_alpha,
        dynamic_sigma_threshold,
        dynamic_sigma_power,
        dynamic_sigma_min,
        dynamic_sigma_max,
        shape_power,
        shape_mode,
        shape_min_points,
        shape_sigma_floor,
        shape_ref_mode,
        init_pos,
        init_rate,
        initial_prev_md,
        tvt0,
        bin_idx,
        num_bins,
        typewell_window,
        typewell_len,
        n_particles,
        tvt_min,
        tvt_max,
        seed,
        momentum,
        rate_noise,
        pos_noise,
        resample_threshold,
        resample_obs_power_adapt,
        resample_min_threshold,
        rough_pos,
        rough_rate,
        rescue_frac,
        rescue_pos_sd,
        rescue_rate_sd,
        init_pos_sd,
        init_rate_sd,
        rate_mean_weight,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_dist,
        jump_tail_clip,
        jump_tail_missing_boost,
        anchor_sigma,
        anchor_power,
        correlated_rate_alpha,
        correlated_pos_alpha,
        lookahead_power,
        lookahead_steps,
        lookahead_decay,
        lookahead_max_gap,
        lookahead_delta_power,
        stratified_init,
        finite_run_power_boost,
        finite_run_power_cap,
        finite_run_power_decay,
        finite_run_power_floor,
        conf_obs_power_decay,
        conf_obs_power_floor,
        conf_obs_power_power,
        missing_noise_scale,
        missing_jump_scale,
        ess_jump_boost,
        ess_jump_power,
        surprise_jump_boost,
        surprise_jump_threshold,
        surprise_jump_power,
        ess_rough_boost,
        ess_rough_power,
        state_z_weight,
    ):
        np.random.seed(seed)
        hist_bins = np.zeros((num_bins, typewell_len))
        particle_hist_bins = np.zeros((num_bins, typewell_len))
        path_sums = np.zeros(num_bins)
        count_bins = np.zeros(num_bins)
        pos = np.empty(n_particles)
        rate = np.empty(n_particles)
        weight = np.ones(n_particles) / n_particles
        for j in range(n_particles):
            if stratified_init != 0:
                u = (j + np.random.random()) / n_particles
                pos[j] = init_pos + init_pos_sd * (2.0 * u - 1.0) * 1.7320508075688772
                u_rate = ((j * 37) % n_particles + np.random.random()) / n_particles
                rate[j] = init_rate + init_rate_sd * (2.0 * u_rate - 1.0) * 1.7320508075688772
            else:
                pos[j] = init_pos + init_pos_sd * np.random.randn()
                rate[j] = init_rate + init_rate_sd * np.random.randn()

        axis0 = -typewell_window + typewell_window / typewell_len
        axis_step = 2.0 * typewell_window / typewell_len
        prev_md = initial_prev_md
        anchor_z = z[0]
        if abs(state_z_weight) > 1e-12:
            anchor_z = (init_pos - tvt0) / state_z_weight
        log_lik = 0.0
        rate_shock = 0.0
        pos_shock = 0.0
        finite_run = 0.0
        sigma_mult = 1.0
        if outlier_prob < 0.0:
            outlier_prob = 0.0
        elif outlier_prob > 1.0:
            outlier_prob = 1.0
        if outlier_likelihood < 1e-300:
            outlier_likelihood = 1e-300
        elif outlier_likelihood > 1.0:
            outlier_likelihood = 1.0
        if dynamic_sigma_min <= 0.0:
            dynamic_sigma_min = 1e-6
        if dynamic_sigma_max < dynamic_sigma_min:
            dynamic_sigma_max = dynamic_sigma_min
        if dynamic_sigma_threshold <= 0.0:
            dynamic_sigma_threshold = 1.0
        if dynamic_sigma_power <= 0.0:
            dynamic_sigma_power = 1.0
        if jump_tail_prob < 0.0:
            jump_tail_prob = 0.0
        if jump_tail_sd < 0.0:
            jump_tail_sd = 0.0
        if jump_tail_rate_sd < 0.0:
            jump_tail_rate_sd = 0.0
        if jump_tail_sd <= 0.0 and jump_tail_rate_sd <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_prob <= 0.0 and jump_tail_missing_boost <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_dist < 0 or jump_tail_dist > 2:
            jump_tail_dist = 1
        if jump_tail_clip < 0.0:
            jump_tail_clip = 0.0
        if jump_tail_dist == 2 and jump_tail_clip <= 0.0 and jump_tail_sd > 0.0:
            jump_tail_clip = 6.0 * jump_tail_sd
        if lookahead_steps < 1:
            lookahead_steps = 1
        if lookahead_decay <= 0.0:
            lookahead_decay = 1.0
        if finite_run_power_decay < 0.0:
            finite_run_power_decay = 0.0
        if finite_run_power_floor < 0.0:
            finite_run_power_floor = 0.0
        elif finite_run_power_floor > 1.0:
            finite_run_power_floor = 1.0
        if conf_obs_power_decay < 0.0:
            conf_obs_power_decay = 0.0
        if conf_obs_power_floor < 0.0:
            conf_obs_power_floor = 0.0
        elif conf_obs_power_floor > 1.0:
            conf_obs_power_floor = 1.0
        if conf_obs_power_power <= 0.0:
            conf_obs_power_power = 1.0
        if ess_jump_boost < 0.0:
            ess_jump_boost = 0.0
        if ess_jump_power <= 0.0:
            ess_jump_power = 1.0
        if surprise_jump_boost < 0.0:
            surprise_jump_boost = 0.0
        if surprise_jump_threshold <= 0.0:
            surprise_jump_threshold = 1.0
        if surprise_jump_power <= 0.0:
            surprise_jump_power = 1.0
        prev_ess_frac = 1.0
        prev_surprise = 0.0
        obs_width = block_obs_power.shape[1]
        local_obs_power_arr = np.empty(obs_width)
        for i in range(len(md)):
            dmd = md[i] - prev_md
            if dmd < 1.0:
                dmd = 1.0
            missing_frac = 1.0 - gr_finite[i]
            if missing_frac < 0.0:
                missing_frac = 0.0
            elif missing_frac > 1.0:
                missing_frac = 1.0
            local_rate_noise = rate_noise
            local_pos_noise = pos_noise
            if missing_frac > 0.0 and missing_noise_scale != 1.0:
                noise_scale = 1.0 + (missing_noise_scale - 1.0) * missing_frac
                local_rate_noise *= noise_scale
                local_pos_noise *= noise_scale
            step_jump_prob = jump_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and missing_jump_boost > 0.0:
                step_jump_prob += missing_jump_boost * (1.0 - gr_power[i]) * dmd
            if ess_jump_boost > 0.0:
                ess_deficit = 1.0 - prev_ess_frac
                if ess_deficit > 0.0:
                    if ess_jump_power != 1.0:
                        ess_deficit = math.pow(ess_deficit, ess_jump_power)
                    step_jump_prob += ess_jump_boost * ess_deficit * dmd
            if surprise_jump_boost > 0.0 and prev_surprise > surprise_jump_threshold:
                surprise_excess = prev_surprise / surprise_jump_threshold - 1.0
                if surprise_excess > 0.0:
                    if surprise_jump_power != 1.0:
                        surprise_excess = math.pow(surprise_excess, surprise_jump_power)
                    step_jump_prob += surprise_jump_boost * surprise_excess * dmd
            if step_jump_prob > 1.0:
                step_jump_prob = 1.0
            step_tail_jump_prob = jump_tail_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_tail_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and jump_tail_missing_boost > 0.0:
                step_tail_jump_prob += jump_tail_missing_boost * (1.0 - gr_power[i]) * dmd
            if step_tail_jump_prob > 1.0:
                step_tail_jump_prob = 1.0
            if correlated_rate_alpha > 0.0:
                rate_shock = correlated_rate_alpha * rate_shock + math.sqrt(max(0.0, 1.0 - correlated_rate_alpha * correlated_rate_alpha)) * np.random.randn()
            if correlated_pos_alpha > 0.0:
                pos_shock = correlated_pos_alpha * pos_shock + math.sqrt(max(0.0, 1.0 - correlated_pos_alpha * correlated_pos_alpha)) * np.random.randn()
            for j in range(n_particles):
                rate_eps = np.random.randn()
                if correlated_rate_alpha > 0.0:
                    rate_eps = 0.5 * rate_eps + 0.8660254037844386 * rate_shock
                rate[j] = momentum * rate[j] + local_rate_noise * rate_eps
                if rate_mean_weight > 0.0:
                    rate[j] += rate_mean_weight * (init_rate - rate[j])
                pos_eps = np.random.randn()
                if correlated_pos_alpha > 0.0:
                    pos_eps = 0.5 * pos_eps + 0.8660254037844386 * pos_shock
                pos[j] = pos[j] + rate[j] * dmd + local_pos_noise * pos_eps
                if step_jump_prob > 0.0 and np.random.random() < step_jump_prob:
                    pos[j] += jump_sd * np.random.randn()
                    rate[j] += jump_rate_sd * np.random.randn()
                if step_tail_jump_prob > 0.0 and np.random.random() < step_tail_jump_prob:
                    pos[j] += _pf_jump_noise_numba(jump_tail_sd, jump_tail_dist, jump_tail_clip)
                    rate_clip = 0.0
                    if jump_tail_clip > 0.0 and jump_tail_sd > 0.0 and jump_tail_rate_sd > 0.0:
                        rate_clip = jump_tail_clip * jump_tail_rate_sd / jump_tail_sd
                    elif jump_tail_dist == 2 and jump_tail_rate_sd > 0.0:
                        rate_clip = 6.0 * jump_tail_rate_sd
                    rate[j] += _pf_jump_noise_numba(jump_tail_rate_sd, jump_tail_dist, rate_clip)
                tvt = pos[j] - state_z_weight * z[i]
                if tvt < tvt_min:
                    tvt = tvt_min
                elif tvt > tvt_max:
                    tvt = tvt_max
                pos[j] = tvt + state_z_weight * z[i]

            weight_sum = 0.0
            avg_lk = 0.0
            obs_power = gr_power[i]
            if obs_power < 0.0:
                obs_power = 0.0
            obs_count_i = block_obs_count[i]
            if obs_count_i < 1:
                obs_count_i = 1
            if obs_count_i > obs_width:
                obs_count_i = obs_width
            total_local_obs_power = 0.0
            if obs_likelihood_mode == 1:
                block_obs_power_sum = 0.0
                block_obs_power_adjusted_sum = 0.0
                for obs_idx in range(obs_count_i):
                    local_power = block_obs_power[i, obs_idx]
                    if local_power < 0.0:
                        local_power = 0.0
                    block_obs_power_sum += local_power
                    if block_obs_finite[i, obs_idx] > 0.0:
                        finite_run += 1.0
                    else:
                        finite_run = 0.0
                    if finite_run_power_boost > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        local_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                        if finite_run_power_cap > 0.0 and local_power > finite_run_power_cap:
                            local_power = finite_run_power_cap
                    if finite_run_power_decay > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                        if run_mult < finite_run_power_floor:
                            run_mult = finite_run_power_floor
                        local_power *= run_mult
                    local_obs_power_arr[obs_idx] = local_power
                    block_obs_power_adjusted_sum += local_power
                obs_power = block_obs_power_adjusted_sum
                if block_obs_power_adjusted_sum > 0.0 and block_obs_power_sum > 0.0:
                    scale_to_merged_power = gr_power[i] / block_obs_power_sum
                    if scale_to_merged_power < 0.0:
                        scale_to_merged_power = 0.0
                    obs_power = block_obs_power_adjusted_sum * scale_to_merged_power
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= scale_to_merged_power
                    total_local_obs_power = obs_power
            else:
                if gr_finite[i] > 0.0:
                    finite_run += 1.0
                else:
                    finite_run = 0.0
                if finite_run_power_boost > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    obs_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                    if finite_run_power_cap > 0.0 and obs_power > finite_run_power_cap:
                        obs_power = finite_run_power_cap
                if finite_run_power_decay > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                    if run_mult < finite_run_power_floor:
                        run_mult = finite_run_power_floor
                    obs_power *= run_mult
                local_obs_power_arr[0] = obs_power
                total_local_obs_power = obs_power
            if conf_obs_power_decay > 0.0 and obs_power > 0.0 and prev_ess_frac < 1.0:
                conf_mult = 1.0 / (1.0 + conf_obs_power_decay * math.pow(1.0 - prev_ess_frac, conf_obs_power_power))
                if conf_mult < conf_obs_power_floor:
                    conf_mult = conf_obs_power_floor
                obs_power *= conf_mult
                if total_local_obs_power > 0.0:
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= conf_mult
            sigma_eff = gr_sigma * sigma_mult
            if sigma_eff <= 1e-6:
                sigma_eff = 1e-6
            pred_gr_mean = 0.0
            pred_gr_weight = 0.0
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0:
                for j in range(n_particles):
                    tvt_dyn = pos[j] - state_z_weight * z[i]
                    expected_gr_dyn = _interp_grid_numba(ref_grid, tvt_dyn, grid_min, grid_step)
                    if raw_ref_weight > 0.0:
                        raw_expected_gr_dyn = _interp_grid_numba(raw_ref_grid, tvt_dyn, raw_grid_min, raw_grid_step)
                        expected_gr_dyn = (1.0 - raw_ref_weight) * expected_gr_dyn + raw_ref_weight * raw_expected_gr_dyn
                    pred_gr_mean += weight[j] * expected_gr_dyn
                    pred_gr_weight += weight[j]
                if pred_gr_weight > 0.0:
                    pred_gr_mean /= pred_gr_weight
            for j in range(n_particles):
                tvt = pos[j] - state_z_weight * z[i]
                lk = 1.0
                if obs_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_power = local_obs_power_arr[obs_idx]
                            if local_power <= 0.0:
                                continue
                            local_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            lk *= _pf_gr_point_likelihood_numba(
                                block_obs_gr[i, obs_idx],
                                local_tvt,
                                local_power,
                                block_obs_finite[i, obs_idx],
                                sigma_eff,
                                ref_grid,
                                grid_min,
                                grid_step,
                                raw_ref_grid,
                                raw_grid_min,
                                raw_grid_step,
                                raw_ref_weight,
                                outlier_prob,
                                outlier_likelihood,
                            )
                    else:
                        lk *= _pf_gr_point_likelihood_numba(
                            gr[i],
                            tvt,
                            obs_power,
                            gr_finite[i],
                            sigma_eff,
                            ref_grid,
                            grid_min,
                            grid_step,
                            raw_ref_grid,
                            raw_grid_min,
                            raw_grid_step,
                            raw_ref_weight,
                            outlier_prob,
                            outlier_likelihood,
                        )
                if shape_power > 0.0 and shape_mode > 0:
                    lk *= _pf_gr_shape_likelihood_numba(
                        i,
                        tvt,
                        rate[j],
                        z[i],
                        state_z_weight,
                        gr[i],
                        sigma_eff,
                        ref_grid,
                        grid_min,
                        grid_step,
                        raw_ref_grid,
                        raw_grid_min,
                        raw_grid_step,
                        raw_ref_weight,
                        shape_gr,
                        shape_md_delta,
                        shape_z,
                        shape_count,
                        shape_power,
                        shape_mode,
                        shape_min_points,
                        shape_sigma_floor,
                        shape_ref_mode,
                    )
                if lookahead_power > 0.0:
                    next_i = i + 1
                    found = 0
                    lookahead_weight = lookahead_power
                    while next_i < len(md) and found < lookahead_steps:
                        if gr_power[next_i] > 0.0 and gr_finite[next_i] > 0.0:
                            gap = md[next_i] - md[i]
                            if gap < 1.0:
                                gap = 1.0
                            if lookahead_max_gap > 0.0 and gap > lookahead_max_gap:
                                break
                            if lookahead_max_gap <= 0.0 or gap <= lookahead_max_gap:
                                future_tvt = pos[j] + rate[j] * gap - state_z_weight * z[next_i]
                                expected_gr_next = _interp_grid_numba(ref_grid, future_tvt, grid_min, grid_step)
                                d_next = (gr[next_i] - expected_gr_next) / sigma_eff
                                d2_next = d_next * d_next
                                if d2_next > 600.0:
                                    d2_next = 600.0
                                next_power = lookahead_weight * gr_power[next_i]
                                if lookahead_delta_power > 0.0 and gr_finite[i] > 0.0:
                                    expected_gr_now = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
                                    obs_delta = gr[next_i] - gr[i]
                                    pred_delta = expected_gr_next - expected_gr_now
                                    delta_d = (obs_delta - pred_delta) / sigma_eff
                                    delta_d2 = delta_d * delta_d
                                    if delta_d2 > 600.0:
                                        delta_d2 = 600.0
                                    d2_next = (1.0 - lookahead_delta_power) * d2_next + lookahead_delta_power * delta_d2
                                next_lk = math.exp(-0.5 * d2_next * next_power)
                                if raw_ref_weight > 0.0:
                                    raw_expected_gr_next = _interp_grid_numba(raw_ref_grid, future_tvt, raw_grid_min, raw_grid_step)
                                    raw_d_next = (gr[next_i] - raw_expected_gr_next) / sigma_eff
                                    raw_d2_next = raw_d_next * raw_d_next
                                    if raw_d2_next > 600.0:
                                        raw_d2_next = 600.0
                                    raw_next_lk = math.exp(-0.5 * raw_d2_next * next_power)
                                    next_lk = (1.0 - raw_ref_weight) * next_lk + raw_ref_weight * raw_next_lk
                                if outlier_prob > 0.0:
                                    next_lk = (1.0 - outlier_prob) * next_lk + outlier_prob * outlier_likelihood
                                lk *= next_lk
                                found += 1
                                lookahead_weight *= lookahead_decay
                                if lookahead_weight <= 0.0:
                                    break
                        next_i += 1
                if anchor_sigma > 0.0 and anchor_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_anchor_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            da = (local_anchor_tvt - tvt0) / anchor_sigma
                            da2 = da * da
                            if da2 > 600.0:
                                da2 = 600.0
                            lk *= math.exp(-0.5 * da2 * anchor_power)
                    else:
                        da = (tvt - tvt0) / anchor_sigma
                        da2 = da * da
                        if da2 > 600.0:
                            da2 = 600.0
                        lk *= math.exp(-0.5 * da2 * anchor_power)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += weight[j] * lk
                weight[j] *= lk
                weight_sum += weight[j]
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += math.log(avg_lk)
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0 and pred_gr_weight > 0.0:
                surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
                target_mult = math.pow(max(surprise / dynamic_sigma_threshold, 1e-6), dynamic_sigma_power)
                if target_mult < dynamic_sigma_min:
                    target_mult = dynamic_sigma_min
                elif target_mult > dynamic_sigma_max:
                    target_mult = dynamic_sigma_max
                sigma_mult = (1.0 - dynamic_sigma_alpha) * sigma_mult + dynamic_sigma_alpha * target_mult
                if sigma_mult < dynamic_sigma_min:
                    sigma_mult = dynamic_sigma_min
                elif sigma_mult > dynamic_sigma_max:
                    sigma_mult = dynamic_sigma_max
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weight[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weight[j] = 1.0 / n_particles

            b = bin_idx[i]
            if 0 <= b < num_bins:
                output_obs_idx = obs_count_i - 1
                if output_obs_idx < 0:
                    output_obs_idx = 0
                elif output_obs_idx >= obs_width:
                    output_obs_idx = obs_width - 1
                tvt_mean = 0.0
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if obs_likelihood_mode == 1:
                        tvt = tvt + rate[j] * block_obs_md_delta[i, output_obs_idx] + state_z_weight * (z[i] - block_obs_z[i, output_obs_idx])
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    tvt_mean += weight[j] * tvt
                    rel = tvt - tvt0
                    grid_pos = (rel - axis0) / axis_step
                    left = math.floor(grid_pos)
                    frac = grid_pos - left
                    if left >= 0 and left < typewell_len:
                        hist_bins[b, left] += weight[j] * (1.0 - frac)
                        particle_hist_bins[b, left] += 1.0 - frac
                    right = left + 1
                    if right >= 0 and right < typewell_len:
                        hist_bins[b, right] += weight[j] * frac
                        particle_hist_bins[b, right] += frac
                path_sums[b] += tvt_mean
                count_bins[b] += 1.0

            effective_n_inv = 0.0
            for j in range(n_particles):
                effective_n_inv += weight[j] * weight[j]
            if effective_n_inv > 0.0:
                prev_ess_frac = 1.0 / (effective_n_inv * n_particles)
                if prev_ess_frac < 0.0:
                    prev_ess_frac = 0.0
                elif prev_ess_frac > 1.0:
                    prev_ess_frac = 1.0
            else:
                prev_ess_frac = 0.0
            if gr_finite[i] > 0.0 and pred_gr_weight > 0.0 and gr_sigma > 1e-6:
                prev_surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
            elif gr_finite[i] <= 0.0:
                prev_surprise *= 0.85
            local_resample_threshold = resample_threshold
            if resample_obs_power_adapt > 0.0 and local_resample_threshold > 0.0:
                adapt_scale = 1.0 / (1.0 + resample_obs_power_adapt * max(obs_power, 0.0))
                local_resample_threshold *= adapt_scale
                if resample_min_threshold > 0.0 and local_resample_threshold < resample_min_threshold:
                    local_resample_threshold = resample_min_threshold
            if local_resample_threshold > 0.0 and 1.0 / effective_n_inv < local_resample_threshold * n_particles:
                local_rough_pos = rough_pos
                local_rough_rate = rough_rate
                if ess_rough_boost > 0.0:
                    ess_frac = 1.0 / (effective_n_inv * n_particles)
                    deficit = local_resample_threshold - ess_frac
                    if deficit > 0.0:
                        if ess_rough_power != 1.0:
                            deficit = math.pow(deficit / max(local_resample_threshold, 1e-12), ess_rough_power) * local_resample_threshold
                        local_rough_pos *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                        local_rough_rate *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                pos, rate = _pf_resample_numba(pos, rate, weight, local_rough_pos, local_rough_rate)
                if rescue_frac > 0.0 and rescue_pos_sd > 0.0 and rescue_rate_sd > 0.0:
                    n_rescue = int(math.floor(rescue_frac * n_particles))
                    if n_rescue > 0:
                        rescue_center = init_pos + init_rate * (md[i] - md[0])
                        for j in range(n_rescue):
                            idx = n_particles - 1 - j
                            pos[idx] = rescue_center + rescue_pos_sd * np.random.randn()
                            rate[idx] = init_rate + rescue_rate_sd * np.random.randn()
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    pos[j] = tvt + state_z_weight * z[i]
                    weight[j] = 1.0 / n_particles

            prev_md = md[i]
        return hist_bins, path_sums, count_bins, particle_hist_bins, log_lik

    @njit(cache=False)
    def _pf_heatmap_ffbsi_kernel(
        md,
        z,
        gr,
        gr_power,
        gr_finite,
        block_obs_gr,
        block_obs_power,
        block_obs_finite,
        block_obs_md_delta,
        block_obs_z,
        block_obs_count,
        obs_likelihood_mode,
        shape_gr,
        shape_md_delta,
        shape_z,
        shape_count,
        ref_grid,
        grid_min,
        grid_step,
        raw_ref_grid,
        raw_grid_min,
        raw_grid_step,
        raw_ref_weight,
        gr_sigma,
        outlier_prob,
        outlier_likelihood,
        dynamic_sigma_alpha,
        dynamic_sigma_threshold,
        dynamic_sigma_power,
        dynamic_sigma_min,
        dynamic_sigma_max,
        shape_power,
        shape_mode,
        shape_min_points,
        shape_sigma_floor,
        shape_ref_mode,
        init_pos,
        init_rate,
        initial_prev_md,
        tvt0,
        bin_idx,
        num_bins,
        typewell_window,
        typewell_len,
        n_particles,
        tvt_min,
        tvt_max,
        seed,
        momentum,
        rate_noise,
        pos_noise,
        resample_threshold,
        resample_obs_power_adapt,
        resample_min_threshold,
        rough_pos,
        rough_rate,
        rescue_frac,
        rescue_pos_sd,
        rescue_rate_sd,
        init_pos_sd,
        init_rate_sd,
        rate_mean_weight,
        jump_prob,
        jump_sd,
        jump_rate_sd,
        missing_jump_boost,
        jump_tail_prob,
        jump_tail_sd,
        jump_tail_rate_sd,
        jump_tail_dist,
        jump_tail_clip,
        jump_tail_missing_boost,
        anchor_sigma,
        anchor_power,
        correlated_rate_alpha,
        correlated_pos_alpha,
        lookahead_power,
        lookahead_steps,
        lookahead_decay,
        lookahead_max_gap,
        lookahead_delta_power,
        stratified_init,
        finite_run_power_boost,
        finite_run_power_cap,
        finite_run_power_decay,
        finite_run_power_floor,
        conf_obs_power_decay,
        conf_obs_power_floor,
        conf_obs_power_power,
        missing_noise_scale,
        missing_jump_scale,
        ess_jump_boost,
        ess_jump_power,
        surprise_jump_boost,
        surprise_jump_threshold,
        surprise_jump_power,
        ess_rough_boost,
        ess_rough_power,
        state_z_weight,
        n_paths,
        max_active_bins,
        transition_scale,
        pos_floor,
        rate_floor,
    ):
        np.random.seed(seed)
        hist_bins = np.zeros((num_bins, typewell_len))
        path_sums = np.zeros(num_bins)
        count_bins = np.zeros(num_bins)
        filtered_hist = np.zeros((num_bins, typewell_len))
        filtered_particle_hist = np.zeros((num_bins, typewell_len))
        filtered_path_sums = np.zeros(num_bins)
        filtered_count_bins = np.zeros(num_bins)
        pos = np.empty(n_particles)
        rate = np.empty(n_particles)
        weight = np.ones(n_particles) / n_particles

        bin_map = -np.ones(num_bins, dtype=np.int64)
        active_bins = np.empty(num_bins, dtype=np.int64)
        active_row_start = np.empty(num_bins, dtype=np.int64)
        active_row_end = np.empty(num_bins, dtype=np.int64)
        active_count = 0
        prev_bin = -1
        for i0 in range(len(md)):
            b0 = bin_idx[i0]
            if b0 >= 0 and b0 < num_bins:
                if b0 != prev_bin:
                    if active_count >= max_active_bins:
                        return (
                            filtered_hist,
                            filtered_path_sums,
                            filtered_count_bins,
                            filtered_particle_hist,
                            hist_bins,
                            path_sums,
                            count_bins,
                            -1e300,
                            0,
                        )
                    bin_map[b0] = active_count
                    active_bins[active_count] = b0
                    active_row_start[active_count] = i0
                    active_row_end[active_count] = i0
                    active_count += 1
                    prev_bin = b0
                else:
                    snap_idx0 = bin_map[b0]
                    if snap_idx0 >= 0:
                        active_row_end[snap_idx0] = i0
        if active_count <= 0 or n_paths <= 0:
            return (
                filtered_hist,
                filtered_path_sums,
                filtered_count_bins,
                filtered_particle_hist,
                hist_bins,
                path_sums,
                count_bins,
                -1e300,
                0,
            )

        snap_pos = np.empty((active_count, n_particles))
        snap_rate = np.empty((active_count, n_particles))
        snap_weight = np.empty((active_count, n_particles))
        snap_md = np.empty(active_count)
        snap_z = np.empty(active_count)
        snap_output_md_delta = np.zeros(active_count)
        snap_output_z = np.empty(active_count)
        snap_gr_power = np.empty(active_count)
        snap_gr_finite = np.empty(active_count)
        row_to_snap = -np.ones(len(md), dtype=np.int64)
        for a0 in range(active_count):
            row_to_snap[(active_row_start[a0] + active_row_end[a0]) // 2] = a0

        for j in range(n_particles):
            if stratified_init != 0:
                u = (j + np.random.random()) / n_particles
                pos[j] = init_pos + init_pos_sd * (2.0 * u - 1.0) * 1.7320508075688772
                u_rate = ((j * 37) % n_particles + np.random.random()) / n_particles
                rate[j] = init_rate + init_rate_sd * (2.0 * u_rate - 1.0) * 1.7320508075688772
            else:
                pos[j] = init_pos + init_pos_sd * np.random.randn()
                rate[j] = init_rate + init_rate_sd * np.random.randn()

        axis0 = -typewell_window + typewell_window / typewell_len
        axis_step = 2.0 * typewell_window / typewell_len
        prev_md = initial_prev_md
        anchor_z = z[0]
        if abs(state_z_weight) > 1e-12:
            anchor_z = (init_pos - tvt0) / state_z_weight
        log_lik = 0.0
        rate_shock = 0.0
        pos_shock = 0.0
        finite_run = 0.0
        sigma_mult = 1.0
        if outlier_prob < 0.0:
            outlier_prob = 0.0
        elif outlier_prob > 1.0:
            outlier_prob = 1.0
        if outlier_likelihood < 1e-300:
            outlier_likelihood = 1e-300
        elif outlier_likelihood > 1.0:
            outlier_likelihood = 1.0
        if dynamic_sigma_min <= 0.0:
            dynamic_sigma_min = 1e-6
        if dynamic_sigma_max < dynamic_sigma_min:
            dynamic_sigma_max = dynamic_sigma_min
        if dynamic_sigma_threshold <= 0.0:
            dynamic_sigma_threshold = 1.0
        if dynamic_sigma_power <= 0.0:
            dynamic_sigma_power = 1.0
        if jump_tail_prob < 0.0:
            jump_tail_prob = 0.0
        if jump_tail_sd < 0.0:
            jump_tail_sd = 0.0
        if jump_tail_rate_sd < 0.0:
            jump_tail_rate_sd = 0.0
        if jump_tail_sd <= 0.0 and jump_tail_rate_sd <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_prob <= 0.0 and jump_tail_missing_boost <= 0.0:
            jump_tail_prob = 0.0
            jump_tail_missing_boost = 0.0
        if jump_tail_dist < 0 or jump_tail_dist > 2:
            jump_tail_dist = 1
        if jump_tail_clip < 0.0:
            jump_tail_clip = 0.0
        if jump_tail_dist == 2 and jump_tail_clip <= 0.0 and jump_tail_sd > 0.0:
            jump_tail_clip = 6.0 * jump_tail_sd
        if lookahead_steps < 1:
            lookahead_steps = 1
        if lookahead_decay <= 0.0:
            lookahead_decay = 1.0
        if finite_run_power_decay < 0.0:
            finite_run_power_decay = 0.0
        if finite_run_power_floor < 0.0:
            finite_run_power_floor = 0.0
        elif finite_run_power_floor > 1.0:
            finite_run_power_floor = 1.0
        if conf_obs_power_decay < 0.0:
            conf_obs_power_decay = 0.0
        if conf_obs_power_floor < 0.0:
            conf_obs_power_floor = 0.0
        elif conf_obs_power_floor > 1.0:
            conf_obs_power_floor = 1.0
        if conf_obs_power_power <= 0.0:
            conf_obs_power_power = 1.0
        if ess_jump_boost < 0.0:
            ess_jump_boost = 0.0
        if ess_jump_power <= 0.0:
            ess_jump_power = 1.0
        if surprise_jump_boost < 0.0:
            surprise_jump_boost = 0.0
        if surprise_jump_threshold <= 0.0:
            surprise_jump_threshold = 1.0
        if surprise_jump_power <= 0.0:
            surprise_jump_power = 1.0
        prev_ess_frac = 1.0
        prev_surprise = 0.0
        obs_width = block_obs_power.shape[1]
        local_obs_power_arr = np.empty(obs_width)

        for i in range(len(md)):
            dmd = md[i] - prev_md
            if dmd < 1.0:
                dmd = 1.0
            missing_frac = 1.0 - gr_finite[i]
            if missing_frac < 0.0:
                missing_frac = 0.0
            elif missing_frac > 1.0:
                missing_frac = 1.0
            local_rate_noise = rate_noise
            local_pos_noise = pos_noise
            if missing_frac > 0.0 and missing_noise_scale != 1.0:
                noise_scale = 1.0 + (missing_noise_scale - 1.0) * missing_frac
                local_rate_noise *= noise_scale
                local_pos_noise *= noise_scale
            step_jump_prob = jump_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and missing_jump_boost > 0.0:
                step_jump_prob += missing_jump_boost * (1.0 - gr_power[i]) * dmd
            if ess_jump_boost > 0.0:
                ess_deficit = 1.0 - prev_ess_frac
                if ess_deficit > 0.0:
                    if ess_jump_power != 1.0:
                        ess_deficit = math.pow(ess_deficit, ess_jump_power)
                    step_jump_prob += ess_jump_boost * ess_deficit * dmd
            if surprise_jump_boost > 0.0 and prev_surprise > surprise_jump_threshold:
                surprise_excess = prev_surprise / surprise_jump_threshold - 1.0
                if surprise_excess > 0.0:
                    if surprise_jump_power != 1.0:
                        surprise_excess = math.pow(surprise_excess, surprise_jump_power)
                    step_jump_prob += surprise_jump_boost * surprise_excess * dmd
            if step_jump_prob > 1.0:
                step_jump_prob = 1.0
            step_tail_jump_prob = jump_tail_prob * dmd
            if missing_frac > 0.0 and missing_jump_scale != 1.0:
                step_tail_jump_prob *= 1.0 + (missing_jump_scale - 1.0) * missing_frac
            if gr_power[i] < 1.0 and jump_tail_missing_boost > 0.0:
                step_tail_jump_prob += jump_tail_missing_boost * (1.0 - gr_power[i]) * dmd
            if step_tail_jump_prob > 1.0:
                step_tail_jump_prob = 1.0
            if correlated_rate_alpha > 0.0:
                rate_shock = correlated_rate_alpha * rate_shock + math.sqrt(max(0.0, 1.0 - correlated_rate_alpha * correlated_rate_alpha)) * np.random.randn()
            if correlated_pos_alpha > 0.0:
                pos_shock = correlated_pos_alpha * pos_shock + math.sqrt(max(0.0, 1.0 - correlated_pos_alpha * correlated_pos_alpha)) * np.random.randn()
            for j in range(n_particles):
                rate_eps = np.random.randn()
                if correlated_rate_alpha > 0.0:
                    rate_eps = 0.5 * rate_eps + 0.8660254037844386 * rate_shock
                rate[j] = momentum * rate[j] + local_rate_noise * rate_eps
                if rate_mean_weight > 0.0:
                    rate[j] += rate_mean_weight * (init_rate - rate[j])
                pos_eps = np.random.randn()
                if correlated_pos_alpha > 0.0:
                    pos_eps = 0.5 * pos_eps + 0.8660254037844386 * pos_shock
                pos[j] = pos[j] + rate[j] * dmd + local_pos_noise * pos_eps
                if step_jump_prob > 0.0 and np.random.random() < step_jump_prob:
                    pos[j] += jump_sd * np.random.randn()
                    rate[j] += jump_rate_sd * np.random.randn()
                if step_tail_jump_prob > 0.0 and np.random.random() < step_tail_jump_prob:
                    pos[j] += _pf_jump_noise_numba(jump_tail_sd, jump_tail_dist, jump_tail_clip)
                    rate_clip = 0.0
                    if jump_tail_clip > 0.0 and jump_tail_sd > 0.0 and jump_tail_rate_sd > 0.0:
                        rate_clip = jump_tail_clip * jump_tail_rate_sd / jump_tail_sd
                    elif jump_tail_dist == 2 and jump_tail_rate_sd > 0.0:
                        rate_clip = 6.0 * jump_tail_rate_sd
                    rate[j] += _pf_jump_noise_numba(jump_tail_rate_sd, jump_tail_dist, rate_clip)
                tvt = pos[j] - state_z_weight * z[i]
                if tvt < tvt_min:
                    tvt = tvt_min
                elif tvt > tvt_max:
                    tvt = tvt_max
                pos[j] = tvt + state_z_weight * z[i]

            weight_sum = 0.0
            avg_lk = 0.0
            obs_power = gr_power[i]
            if obs_power < 0.0:
                obs_power = 0.0
            obs_count_i = block_obs_count[i]
            if obs_count_i < 1:
                obs_count_i = 1
            if obs_count_i > obs_width:
                obs_count_i = obs_width
            total_local_obs_power = 0.0
            if obs_likelihood_mode == 1:
                block_obs_power_sum = 0.0
                block_obs_power_adjusted_sum = 0.0
                for obs_idx in range(obs_count_i):
                    local_power = block_obs_power[i, obs_idx]
                    if local_power < 0.0:
                        local_power = 0.0
                    block_obs_power_sum += local_power
                    if block_obs_finite[i, obs_idx] > 0.0:
                        finite_run += 1.0
                    else:
                        finite_run = 0.0
                    if finite_run_power_boost > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        local_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                        if finite_run_power_cap > 0.0 and local_power > finite_run_power_cap:
                            local_power = finite_run_power_cap
                    if finite_run_power_decay > 0.0 and local_power > 0.0 and finite_run > 1.0:
                        run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                        if run_mult < finite_run_power_floor:
                            run_mult = finite_run_power_floor
                        local_power *= run_mult
                    local_obs_power_arr[obs_idx] = local_power
                    block_obs_power_adjusted_sum += local_power
                obs_power = block_obs_power_adjusted_sum
                if block_obs_power_adjusted_sum > 0.0 and block_obs_power_sum > 0.0:
                    scale_to_merged_power = gr_power[i] / block_obs_power_sum
                    if scale_to_merged_power < 0.0:
                        scale_to_merged_power = 0.0
                    obs_power = block_obs_power_adjusted_sum * scale_to_merged_power
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= scale_to_merged_power
                    total_local_obs_power = obs_power
            else:
                if gr_finite[i] > 0.0:
                    finite_run += 1.0
                else:
                    finite_run = 0.0
                if finite_run_power_boost > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    obs_power *= 1.0 + finite_run_power_boost * math.log(1.0 + finite_run)
                    if finite_run_power_cap > 0.0 and obs_power > finite_run_power_cap:
                        obs_power = finite_run_power_cap
                if finite_run_power_decay > 0.0 and obs_power > 0.0 and finite_run > 1.0:
                    run_mult = 1.0 / math.sqrt(1.0 + finite_run_power_decay * (finite_run - 1.0))
                    if run_mult < finite_run_power_floor:
                        run_mult = finite_run_power_floor
                    obs_power *= run_mult
                local_obs_power_arr[0] = obs_power
                total_local_obs_power = obs_power
            if conf_obs_power_decay > 0.0 and obs_power > 0.0 and prev_ess_frac < 1.0:
                conf_mult = 1.0 / (1.0 + conf_obs_power_decay * math.pow(1.0 - prev_ess_frac, conf_obs_power_power))
                if conf_mult < conf_obs_power_floor:
                    conf_mult = conf_obs_power_floor
                obs_power *= conf_mult
                if total_local_obs_power > 0.0:
                    for obs_idx in range(obs_count_i):
                        local_obs_power_arr[obs_idx] *= conf_mult
            sigma_eff = gr_sigma * sigma_mult
            if sigma_eff <= 1e-6:
                sigma_eff = 1e-6
            pred_gr_mean = 0.0
            pred_gr_weight = 0.0
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0:
                for j in range(n_particles):
                    tvt_dyn = pos[j] - state_z_weight * z[i]
                    expected_gr_dyn = _interp_grid_numba(ref_grid, tvt_dyn, grid_min, grid_step)
                    if raw_ref_weight > 0.0:
                        raw_expected_gr_dyn = _interp_grid_numba(raw_ref_grid, tvt_dyn, raw_grid_min, raw_grid_step)
                        expected_gr_dyn = (1.0 - raw_ref_weight) * expected_gr_dyn + raw_ref_weight * raw_expected_gr_dyn
                    pred_gr_mean += weight[j] * expected_gr_dyn
                    pred_gr_weight += weight[j]
                if pred_gr_weight > 0.0:
                    pred_gr_mean /= pred_gr_weight
            for j in range(n_particles):
                tvt = pos[j] - state_z_weight * z[i]
                lk = 1.0
                if obs_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_power = local_obs_power_arr[obs_idx]
                            if local_power <= 0.0:
                                continue
                            local_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            lk *= _pf_gr_point_likelihood_numba(
                                block_obs_gr[i, obs_idx],
                                local_tvt,
                                local_power,
                                block_obs_finite[i, obs_idx],
                                sigma_eff,
                                ref_grid,
                                grid_min,
                                grid_step,
                                raw_ref_grid,
                                raw_grid_min,
                                raw_grid_step,
                                raw_ref_weight,
                                outlier_prob,
                                outlier_likelihood,
                            )
                    else:
                        lk *= _pf_gr_point_likelihood_numba(
                            gr[i],
                            tvt,
                            obs_power,
                            gr_finite[i],
                            sigma_eff,
                            ref_grid,
                            grid_min,
                            grid_step,
                            raw_ref_grid,
                            raw_grid_min,
                            raw_grid_step,
                            raw_ref_weight,
                            outlier_prob,
                            outlier_likelihood,
                        )
                if shape_power > 0.0 and shape_mode > 0:
                    lk *= _pf_gr_shape_likelihood_numba(
                        i,
                        tvt,
                        rate[j],
                        z[i],
                        state_z_weight,
                        gr[i],
                        sigma_eff,
                        ref_grid,
                        grid_min,
                        grid_step,
                        raw_ref_grid,
                        raw_grid_min,
                        raw_grid_step,
                        raw_ref_weight,
                        shape_gr,
                        shape_md_delta,
                        shape_z,
                        shape_count,
                        shape_power,
                        shape_mode,
                        shape_min_points,
                        shape_sigma_floor,
                        shape_ref_mode,
                    )
                if lookahead_power > 0.0:
                    next_i = i + 1
                    found = 0
                    lookahead_weight = lookahead_power
                    while next_i < len(md) and found < lookahead_steps:
                        if gr_power[next_i] > 0.0 and gr_finite[next_i] > 0.0:
                            gap = md[next_i] - md[i]
                            if gap < 1.0:
                                gap = 1.0
                            if lookahead_max_gap > 0.0 and gap > lookahead_max_gap:
                                break
                            if lookahead_max_gap <= 0.0 or gap <= lookahead_max_gap:
                                future_tvt = pos[j] + rate[j] * gap - state_z_weight * z[next_i]
                                expected_gr_next = _interp_grid_numba(ref_grid, future_tvt, grid_min, grid_step)
                                d_next = (gr[next_i] - expected_gr_next) / sigma_eff
                                d2_next = d_next * d_next
                                if d2_next > 600.0:
                                    d2_next = 600.0
                                next_power = lookahead_weight * gr_power[next_i]
                                if lookahead_delta_power > 0.0 and gr_finite[i] > 0.0:
                                    expected_gr_now = _interp_grid_numba(ref_grid, tvt, grid_min, grid_step)
                                    obs_delta = gr[next_i] - gr[i]
                                    pred_delta = expected_gr_next - expected_gr_now
                                    delta_d = (obs_delta - pred_delta) / sigma_eff
                                    delta_d2 = delta_d * delta_d
                                    if delta_d2 > 600.0:
                                        delta_d2 = 600.0
                                    d2_next = (1.0 - lookahead_delta_power) * d2_next + lookahead_delta_power * delta_d2
                                next_lk = math.exp(-0.5 * d2_next * next_power)
                                if raw_ref_weight > 0.0:
                                    raw_expected_gr_next = _interp_grid_numba(raw_ref_grid, future_tvt, raw_grid_min, raw_grid_step)
                                    raw_d_next = (gr[next_i] - raw_expected_gr_next) / sigma_eff
                                    raw_d2_next = raw_d_next * raw_d_next
                                    if raw_d2_next > 600.0:
                                        raw_d2_next = 600.0
                                    raw_next_lk = math.exp(-0.5 * raw_d2_next * next_power)
                                    next_lk = (1.0 - raw_ref_weight) * next_lk + raw_ref_weight * raw_next_lk
                                if outlier_prob > 0.0:
                                    next_lk = (1.0 - outlier_prob) * next_lk + outlier_prob * outlier_likelihood
                                lk *= next_lk
                                found += 1
                                lookahead_weight *= lookahead_decay
                                if lookahead_weight <= 0.0:
                                    break
                        next_i += 1
                if anchor_sigma > 0.0 and anchor_power > 0.0:
                    if obs_likelihood_mode == 1:
                        for obs_idx in range(obs_count_i):
                            local_anchor_tvt = tvt + rate[j] * block_obs_md_delta[i, obs_idx] + state_z_weight * (z[i] - block_obs_z[i, obs_idx])
                            da = (local_anchor_tvt - tvt0) / anchor_sigma
                            da2 = da * da
                            if da2 > 600.0:
                                da2 = 600.0
                            lk *= math.exp(-0.5 * da2 * anchor_power)
                    else:
                        da = (tvt - tvt0) / anchor_sigma
                        da2 = da * da
                        if da2 > 600.0:
                            da2 = 600.0
                        lk *= math.exp(-0.5 * da2 * anchor_power)
                if lk < 1e-300:
                    lk = 1e-300
                avg_lk += weight[j] * lk
                weight[j] *= lk
                weight_sum += weight[j]
            if avg_lk < 1e-300:
                avg_lk = 1e-300
            log_lik += math.log(avg_lk)
            if obs_power > 0.0 and dynamic_sigma_alpha > 0.0 and gr_finite[i] > 0.0 and pred_gr_weight > 0.0:
                surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
                target_mult = math.pow(max(surprise / dynamic_sigma_threshold, 1e-6), dynamic_sigma_power)
                if target_mult < dynamic_sigma_min:
                    target_mult = dynamic_sigma_min
                elif target_mult > dynamic_sigma_max:
                    target_mult = dynamic_sigma_max
                sigma_mult = (1.0 - dynamic_sigma_alpha) * sigma_mult + dynamic_sigma_alpha * target_mult
                if sigma_mult < dynamic_sigma_min:
                    sigma_mult = dynamic_sigma_min
                elif sigma_mult > dynamic_sigma_max:
                    sigma_mult = dynamic_sigma_max
            if weight_sum > 0.0:
                for j in range(n_particles):
                    weight[j] /= weight_sum
            else:
                for j in range(n_particles):
                    weight[j] = 1.0 / n_particles

            b = bin_idx[i]
            if 0 <= b < num_bins:
                output_obs_idx = obs_count_i - 1
                if output_obs_idx < 0:
                    output_obs_idx = 0
                elif output_obs_idx >= obs_width:
                    output_obs_idx = obs_width - 1
                tvt_mean = 0.0
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if obs_likelihood_mode == 1:
                        tvt = tvt + rate[j] * block_obs_md_delta[i, output_obs_idx] + state_z_weight * (z[i] - block_obs_z[i, output_obs_idx])
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    tvt_mean += weight[j] * tvt
                    rel = tvt - tvt0
                    grid_pos = (rel - axis0) / axis_step
                    left = math.floor(grid_pos)
                    frac = grid_pos - left
                    if left >= 0 and left < typewell_len:
                        filtered_hist[b, left] += weight[j] * (1.0 - frac)
                        filtered_particle_hist[b, left] += 1.0 - frac
                    right = left + 1
                    if right >= 0 and right < typewell_len:
                        filtered_hist[b, right] += weight[j] * frac
                        filtered_particle_hist[b, right] += frac
                filtered_path_sums[b] += tvt_mean
                filtered_count_bins[b] += 1.0
                snap_idx = row_to_snap[i]
                if snap_idx >= 0:
                    snap_md[snap_idx] = md[i]
                    snap_z[snap_idx] = z[i]
                    if obs_likelihood_mode == 1:
                        snap_output_md_delta[snap_idx] = block_obs_md_delta[i, output_obs_idx]
                        snap_output_z[snap_idx] = block_obs_z[i, output_obs_idx]
                    else:
                        snap_output_md_delta[snap_idx] = 0.0
                        snap_output_z[snap_idx] = z[i]
                    snap_gr_power[snap_idx] = gr_power[i]
                    snap_gr_finite[snap_idx] = gr_finite[i]
                    for j in range(n_particles):
                        snap_pos[snap_idx, j] = pos[j]
                        snap_rate[snap_idx, j] = rate[j]
                        snap_weight[snap_idx, j] = weight[j]

            effective_n_inv = 0.0
            for j in range(n_particles):
                effective_n_inv += weight[j] * weight[j]
            if effective_n_inv > 0.0:
                prev_ess_frac = 1.0 / (effective_n_inv * n_particles)
                if prev_ess_frac < 0.0:
                    prev_ess_frac = 0.0
                elif prev_ess_frac > 1.0:
                    prev_ess_frac = 1.0
            else:
                prev_ess_frac = 0.0
            if gr_finite[i] > 0.0 and pred_gr_weight > 0.0 and gr_sigma > 1e-6:
                prev_surprise = abs(gr[i] - pred_gr_mean) / gr_sigma
            elif gr_finite[i] <= 0.0:
                prev_surprise *= 0.85
            local_resample_threshold = resample_threshold
            if resample_obs_power_adapt > 0.0 and local_resample_threshold > 0.0:
                adapt_scale = 1.0 / (1.0 + resample_obs_power_adapt * max(obs_power, 0.0))
                local_resample_threshold *= adapt_scale
                if resample_min_threshold > 0.0 and local_resample_threshold < resample_min_threshold:
                    local_resample_threshold = resample_min_threshold
            if local_resample_threshold > 0.0 and 1.0 / effective_n_inv < local_resample_threshold * n_particles:
                local_rough_pos = rough_pos
                local_rough_rate = rough_rate
                if ess_rough_boost > 0.0:
                    ess_frac = 1.0 / (effective_n_inv * n_particles)
                    deficit = local_resample_threshold - ess_frac
                    if deficit > 0.0:
                        if ess_rough_power != 1.0:
                            deficit = math.pow(deficit / max(local_resample_threshold, 1e-12), ess_rough_power) * local_resample_threshold
                        local_rough_pos *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                        local_rough_rate *= 1.0 + ess_rough_boost * deficit / max(local_resample_threshold, 1e-12)
                pos, rate = _pf_resample_numba(pos, rate, weight, local_rough_pos, local_rough_rate)
                if rescue_frac > 0.0 and rescue_pos_sd > 0.0 and rescue_rate_sd > 0.0:
                    n_rescue = int(math.floor(rescue_frac * n_particles))
                    if n_rescue > 0:
                        rescue_center = init_pos + init_rate * (md[i] - md[0])
                        for j in range(n_rescue):
                            idx = n_particles - 1 - j
                            pos[idx] = rescue_center + rescue_pos_sd * np.random.randn()
                            rate[idx] = init_rate + rescue_rate_sd * np.random.randn()
                for j in range(n_particles):
                    tvt = pos[j] - state_z_weight * z[i]
                    if tvt < tvt_min:
                        tvt = tvt_min
                    elif tvt > tvt_max:
                        tvt = tvt_max
                    pos[j] = tvt + state_z_weight * z[i]
                    weight[j] = 1.0 / n_particles

            prev_md = md[i]

        if active_count <= 1:
            return (
                filtered_hist,
                filtered_path_sums,
                filtered_count_bins,
                filtered_particle_hist,
                hist_bins,
                path_sums,
                count_bins,
                log_lik,
                0,
            )

        log_score = np.empty(n_particles)
        for s in range(n_paths):
            next_idx = _pf_sample_weight_index_numba(snap_weight[active_count - 1], n_particles)
            next_pos = snap_pos[active_count - 1, next_idx]
            next_rate = snap_rate[active_count - 1, next_idx]
            b_last = active_bins[active_count - 1]
            tvt_last = next_pos - state_z_weight * snap_z[active_count - 1]
            if obs_likelihood_mode == 1:
                tvt_last = tvt_last + next_rate * snap_output_md_delta[active_count - 1] + state_z_weight * (snap_z[active_count - 1] - snap_output_z[active_count - 1])
            if tvt_last < tvt_min:
                tvt_last = tvt_min
            elif tvt_last > tvt_max:
                tvt_last = tvt_max
            rel_last = tvt_last - tvt0
            grid_pos_last = (rel_last - axis0) / axis_step
            left_last = math.floor(grid_pos_last)
            frac_last = grid_pos_last - left_last
            if left_last >= 0 and left_last < typewell_len:
                hist_bins[b_last, left_last] += 1.0 - frac_last
            right_last = left_last + 1
            if right_last >= 0 and right_last < typewell_len:
                hist_bins[b_last, right_last] += frac_last
            path_sums[b_last] += tvt_last
            count_bins[b_last] += 1.0

            for a in range(active_count - 2, -1, -1):
                dmd = snap_md[a + 1] - snap_md[a]
                if dmd < 1.0:
                    dmd = 1.0
                pos_sd, rate_sd = _pf_transition_scale_numba(
                    dmd,
                    snap_gr_power[a + 1],
                    snap_gr_finite[a + 1],
                    rate_noise,
                    pos_noise,
                    jump_prob,
                    jump_sd,
                    jump_rate_sd,
                    missing_jump_boost,
                    jump_tail_prob,
                    jump_tail_sd,
                    jump_tail_rate_sd,
                    jump_tail_missing_boost,
                    missing_noise_scale,
                    missing_jump_scale,
                    transition_scale,
                    pos_floor,
                    rate_floor,
                )
                for j in range(n_particles):
                    wj = snap_weight[a, j]
                    if wj <= 0.0 or not np.isfinite(wj):
                        log_score[j] = -1e300
                        continue
                    pred_rate = momentum * snap_rate[a, j]
                    if rate_mean_weight > 0.0:
                        pred_rate += rate_mean_weight * (init_rate - pred_rate)
                    pred_pos = snap_pos[a, j] + pred_rate * dmd
                    rate_d = (next_rate - pred_rate) / rate_sd
                    pos_d = (next_pos - pred_pos) / pos_sd
                    rate_log_lk = -0.5 * rate_d * rate_d
                    if rate_log_lk < -300.0:
                        rate_log_lk = -300.0
                    pos_log_lk = -0.5 * pos_d * pos_d
                    if pos_log_lk < -300.0:
                        pos_log_lk = -300.0
                    log_score[j] = math.log(max(wj, 1e-300)) + rate_log_lk + pos_log_lk
                idx = _pf_sample_log_score_index_numba(log_score, n_particles)
                next_idx = idx
                next_pos = snap_pos[a, idx]
                next_rate = snap_rate[a, idx]
                b = active_bins[a]
                tvt = next_pos - state_z_weight * snap_z[a]
                if obs_likelihood_mode == 1:
                    tvt = tvt + next_rate * snap_output_md_delta[a] + state_z_weight * (snap_z[a] - snap_output_z[a])
                if tvt < tvt_min:
                    tvt = tvt_min
                elif tvt > tvt_max:
                    tvt = tvt_max
                rel = tvt - tvt0
                grid_pos = (rel - axis0) / axis_step
                left = math.floor(grid_pos)
                frac = grid_pos - left
                if left >= 0 and left < typewell_len:
                    hist_bins[b, left] += 1.0 - frac
                right = left + 1
                if right >= 0 and right < typewell_len:
                    hist_bins[b, right] += frac
                path_sums[b] += tvt
                count_bins[b] += 1.0
        return (
            filtered_hist,
            filtered_path_sums,
            filtered_count_bins,
            filtered_particle_hist,
            hist_bins,
            path_sums,
            count_bins,
            log_lik,
            1,
        )
