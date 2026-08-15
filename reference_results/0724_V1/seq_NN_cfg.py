import random
import math
from dataclasses import dataclass, asdict
from copy import deepcopy
from pathlib import Path


UNET_MODE = "unet"
TWO_STAGE_UNET_MODE = "two_stage_unet"
LEGACY_STAGE_UNET_MODE = "stage_unet"
MODEL_MODES = (UNET_MODE, TWO_STAGE_UNET_MODE)
SEQ_TARGET_MODES = (UNET_MODE,)
TYPEWELL_GR_GRID_TYPES = ("interpolate", "avg")


def _ordered_unique(*groups):
    out = []
    seen = set()
    for group in groups:
        for name in group:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return tuple(out)


def _channels_except(channels, *excluded_groups):
    excluded = set()
    for group in excluded_groups:
        excluded.update(group)
    return tuple(name for name in channels if name not in excluded)


@dataclass(frozen=True)
class GeoPriorConfig:
    method: str = "idw_dS_xy_wls"
    neighbor_wells: int = 12
    point_neighbors: int = 112
    idw_power: float = 2.0
    point_neighbor_metric: str = "anisotropic"
    anisotropy_angle_deg: float = 50.0
    anisotropy_ratio: float = 1.3
    train_bin_size: int = 64
    query_bin_size: int = 32
    query_upsample_mode: str = "linear"
    prefix_bin_size: int = 32
    prefix_support_rows: int = 512
    clip_ds_xy: float = 0.0
    filter_ds_thr: float = 0.15
    wls_normalize_dxy: bool = True
    wls_ridge_frac: float = 0.01
    wls_min_eig_ratio: float = 0.05
    wls_max_grad: float = 0.25
    wls_robust_mode: str = "tukey"
    wls_robust_scale: float = 0.10
    wls_robust_iters: int = 1
    max_train_points: int = 12000
    support_region: str = "suffix"
    distance_eps: float = 1e-6

    def to_dict(self):
        return asdict(self)


class CFG:
    project_root = Path(__file__).resolve().parents[1]
    local_data_dir = project_root / "data"
    kaggle_data_dir = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    data_dir = local_data_dir if local_data_dir.exists() else kaggle_data_dir
    train_path = data_dir / "train"
    test_path = data_dir / "test"
    output_dir = project_root / "outputs"
    f = 0
    device = "cuda"
    well_limit = None
    submit_mode = False

    # Sequence construction.
    downsample = 32
    prefix_len = 1024
    target_len = 10000
    raw_len = prefix_len + target_len
    num_bins = raw_len // downsample + 1
    # Pipeline-level canonicalization switch.  When disabled, sequence
    # construction and inference keep the historical fixed-window behavior.
    strech_to_full_size = False
    # In full-stretch mode, restore original-row weighting for the sequence
    # loss by scaling each well by original_suffix_len / target_len.
    strech_to_full_size_match_seq_weight = False
    typewell_window = 100.0
    typewell_len = 400
    tw_gr_grid_type = "interpolate"
    geo_prior_cfg = GeoPriorConfig()
    PF_heatmap_cache_dir = project_root / "PF_cache"
    PF_sample_cache_dir = project_root / "PF_sample_cache"
    PF_sample_count = 10
    PF_sample_seeds = 8
    PF_sample_version = 1
    PF_heatmap_n_particles = 500
    PF_heatmap_n_seeds = 256
    PF_heatmap_base_seed = 20265667
    PF_heatmap_lik_scale = 6.0
    PF_heatmap_axis_sigma = 0.0
    PF_heatmap_ref_grid_step = 0.05
    PF_heatmap_query_gr_mode = "interp"
    PF_heatmap_apply_gr_calibration = True
    PF_heatmap_typewell_gr_calibration = "seen_blend"
    PF_heatmap_seen_blend_weight = 0.75
    PF_heatmap_raw_ref_likelihood_weight = 0.075
    PF_heatmap_gr_ambiguity_power = 0.0
    PF_heatmap_gr_ambiguity_min_power = 0.35
    PF_heatmap_gr_ambiguity_contrast = 1.0
    PF_heatmap_gr_ambiguity_ref_mode = "primary"
    PF_heatmap_missing_likelihood_power = 1.0
    PF_heatmap_missing_gap_decay = 0.0
    PF_heatmap_missing_min_power = 0.05
    PF_heatmap_seed_path_weight_mode = "likelihood"
    PF_heatmap_seed_prob_weight_mode = "likelihood"
    PF_heatmap_seed_local_weight_mode = "off"
    PF_heatmap_seed_local_score_source = "pf_likelihood"
    PF_heatmap_seed_local_lik_scale = 1.0
    PF_heatmap_seed_local_half_life_blocks = 8.0
    PF_heatmap_seed_local_global_mix = 0.0
    PF_heatmap_seed_local_min_power = 1e-6
    PF_heatmap_seed_local_combine_mode = "replace"
    PF_heatmap_seed_local_residual_alpha = 0.0
    PF_heatmap_seed_local_residual_clip = 5.0
    PF_heatmap_state_z_weight = 1.0
    PF_heatmap_momentum = 0.9995
    PF_heatmap_rate_noise = 0.0009
    PF_heatmap_pos_noise = 0.0025
    PF_heatmap_resample_threshold = 0.30
    PF_heatmap_resample_obs_power_adapt = 0.45
    PF_heatmap_resample_min_threshold = 0.30
    PF_heatmap_rough_pos = 0.08
    PF_heatmap_rough_rate = 0.006
    PF_heatmap_init_pos_sd = 2.0
    PF_heatmap_init_rate_sd = 0.02
    PF_heatmap_rate_mean_weight = 0.00100
    PF_heatmap_jump_prob = 0.000075
    PF_heatmap_jump_sd = 7.0
    PF_heatmap_jump_rate_sd = 0.00030
    PF_heatmap_missing_jump_boost = 0.00020
    PF_heatmap_anchor_sigma = 80.0
    PF_heatmap_anchor_power = 0.002
    PF_heatmap_correlated_rate_alpha = 0.0
    PF_heatmap_correlated_pos_alpha = 0.2
    PF_heatmap_lookahead_power = 0.0
    PF_heatmap_stratified_init = 1
    PF_heatmap_finite_run_power_boost = 0.06
    PF_heatmap_finite_run_power_cap = 1.0
    PF_heatmap_finite_run_power_decay = 1.0
    PF_heatmap_finite_run_power_floor = 0.60
    PF_heatmap_missing_noise_scale = 1.0
    PF_heatmap_missing_jump_scale = 1.0
    PF_heatmap_jump_tail_prob = 0.0
    PF_heatmap_jump_tail_sd = 0.0
    PF_heatmap_jump_tail_rate_sd = 0.0
    PF_heatmap_jump_tail_dist = 1
    PF_heatmap_jump_tail_clip = 0.0
    PF_heatmap_jump_tail_missing_boost = 0.0
    PF_heatmap_outlier_prob = 0.0
    PF_heatmap_outlier_likelihood = 0.05
    PF_heatmap_dynamic_sigma_alpha = 0.0
    PF_heatmap_dynamic_sigma_threshold = 1.25
    PF_heatmap_dynamic_sigma_power = 1.0
    PF_heatmap_dynamic_sigma_min = 0.85
    PF_heatmap_dynamic_sigma_max = 2.0
    PF_heatmap_gr_information_power = 0.0
    PF_heatmap_gr_information_center = 0.75
    PF_heatmap_gr_information_min_multiplier = 0.75
    PF_heatmap_gr_information_max_multiplier = 1.25
    PF_heatmap_gr_information_slope_weight = 0.0
    PF_heatmap_gr_information_ref_mode = "primary"
    PF_heatmap_lookahead_steps = 1
    PF_heatmap_lookahead_decay = 0.5
    PF_heatmap_lookahead_max_gap = 256.0
    PF_heatmap_lookahead_delta_power = 0.0
    PF_heatmap_ess_rough_boost = 0.0
    PF_heatmap_ess_rough_power = 1.0
    PF_heatmap_prob_temperature = 1.0
    PF_heatmap_ffbsi_mode = "fallback" #"fallback"
    PF_heatmap_ffbsi_n_paths = 64
    PF_heatmap_ffbsi_fallback_mode = "filtered"
    PF_heatmap_ffbsi_max_active_bins = 512
    PF_heatmap_ffbsi_transition_model = "gaussian"
    PF_heatmap_ffbsi_transition_scale = 1.35
    PF_heatmap_ffbsi_pos_floor = 0.15
    PF_heatmap_ffbsi_rate_floor = 0.0025
    PF_heatmap_profile_mixture_spec = [
        {"name": "base", "weight": 0.6514002115702939, "overrides": {}},
        {
            "name": "wide_init",
            "weight": 0.19545809861466074,
            "overrides": {
                "pf_init_pos_sd": 2.5,
                "pf_init_rate_sd": 0.025,
                "pf_resample_threshold": 0.35,
                "pf_rough_pos": 0.10,
                "pf_rough_rate": 0.006,
            },
        },
        {
            "name": "smooth_stiff",
            "weight": 0.053120440286232196,
            "overrides": {
                "pf_jump_prob": 0.000075,
                "pf_jump_rate_sd": 0.00030,
                "pf_jump_sd": 9.0,
                "pf_momentum": 0.9995,
                "pf_pos_noise": 0.0030,
                "pf_rate_mean_weight": 0.00150,
                "pf_rate_noise": 0.0008,
                "pf_rough_pos": 0.06,
                "pf_rough_rate": 0.0045,
            },
        },
        {
            "name": "jump_rare_big",
            "weight": 0.07370294173329474,
            "overrides": {
                "pf_jump_prob": 0.000075,
                "pf_jump_rate_sd": 0.00060,
                "pf_jump_sd": 10.0,
                "pf_missing_jump_boost": 0.000075,
                "pf_resample_threshold": 0.35,
            },
        },
        {
            "name": "tight_init",
            "weight": 0.02631830779551842,
            "overrides": {
                "pf_init_pos_sd": 1.0,
                "pf_init_rate_sd": 0.015,
                "pf_pos_noise": 0.0035,
                "pf_rough_pos": 0.06,
            },
        },
    ]
    PF_heatmap_suffix_merge_k = 64
    PF_heatmap_suffix_merge_likelihood_mode = "block"
    PF_heatmap_suffix_merge_state_mode = "start"
    PF_heatmap_suffix_merge_obs_power_alpha = 1.05
    PF_heatmap_suffix_merge_adjust_dynamics = False
    PF_heatmap_num_workers = 8
    PF_heatmap_overwrite_cache = False
    PF_heatmap_version = 15

    # simulation and augmentation
    typewell_calibration_with_seen = True
    typewell_calibration_power = 1.0
    typewell_calibration_blend_weight = 0.7
    gr_normalize_mean_mode = "global" # "global"/"local"
    gr_normalize_std_mode = "global" # "global"/"local"
    gr_normalize_local_std_floor = 3.0
    gr_interpolate = False
    out_of_range_GR_mask = False
    sim_cfg={
        'keep_ori_path':{
            'apply_prob':0.0,
        },
        'z_shift':{
            'apply_prob':0.75,
            'S_jump_thr':0.15,
            'sample_model':'diff_block_bootstrap', # 'target'/'target_with_rules'/'diff_block_bootstrap'/'PF_sample'/'mixed'
            'sample_from_same_well':False,
            'rule_apply_prob':1.0,
            'rule_reverse_prob':0.5,
            'rule_clip_prob':0.5,
            'rule_clip_min_len':3000,
            'rule_stretch_range':(0.9,1.1),
            'rule_scale_range':(0.9,1.1),
            'diff_block_lengths':(1024,2048,4096),
            'diff_block_weights':(0.45,0.40,0.15),
            'diff_block_join_width':16,
            'diff_block_reverse_prob':0.0,
            'diff_block_candidate_count':1,
            'pf_sample_sg_window':127,
            'pf_sample_sg_polyorder':2,
            'z_shift_range':40,
            'drift_ratio':0.0,
            'fault_ratio':0.05,
            'fault_max_num':3,
            'fault_std':1.75,
            # Mapping values are relative sampling weights. Omitted modes have
            # zero probability; a legacy string is still accepted as weight 1.
            'noise_mode':{'TVT_bias_v2':1.0}, # 'copy'/'real_block'/'structured_random'/'TVT_bias'/'TVT_bias_v2'/'mixed_V1'/'conditional_GR'/'decomposed_GR'/'global_pool'
            'nan_mode':{'same':1.0}, # 'same'/'none'/'shift'/'global' or weighted mapping
            'TVT_bias_std':0.95,
            'TVT_bias_window':1280,
            'TVT_bias_v2_tvt_std':0.95,
            'TVT_bias_v2_windows':(1536,256,2),
            'TVT_bias_v2_weights':(0.68,0.20,0.12),
            'TVT_bias_v2_gr_std_ranges':((5.6,8.4),(8.4,12.2),(12.2,17.5)),
            'TVT_bias_v2_gr_std_weights':(0.24,0.51,0.25),
            'TVT_bias_v2_scale_mode':'std',
            'TVT_bias_v2_gr_offset_std':2.0,
            'TVT_bias_v2_clip_abs':36.0,
            'mixed_V1_tvt_weight':0.4,
            'mixed_V1_target_std':10.7,
            'mixed_V1_offset_std':1.1,
            'mixed_V1_windows':(320,96,24),
            'mixed_V1_weights':(0.45,0.25,0.22,0.08),
            'mixed_V1_tail_df':6.0,
            'mixed_V1_clip_abs':45.0,
            'mixed_V1_slope_clip_ref':0.0,
            'mixed_V1_slope_clip_power':1.0,
            'mixed_V1_slope_smooth_window':9,
            'conditional_GR_bins':28,
            'conditional_GR_min_count':512,
            'conditional_GR_mean_shrink':0.45,
            'conditional_GR_scale_blend':0.85,
            'conditional_GR_windows':(1024,256,32),
            'conditional_GR_weights':(0.48,0.25,0.20,0.07),
            'conditional_GR_tail_df':6.0,
            'conditional_GR_texture_weight':0.25,
            'conditional_GR_tvt_std':0.75,
            'conditional_GR_offset_std':1.2,
            'conditional_GR_std_ranges':((5.6,8.4),(8.4,12.2),(12.2,17.5)),
            'conditional_GR_std_weights':(0.24,0.51,0.25),
            'conditional_GR_clip_abs':45.0,
            'decomposed_GR_mean_mode':'power',
            'decomposed_GR_power':0.9308,
            'decomposed_GR_mean_shrink':0.70,
            'decomposed_GR_scale_blend':0.55,
            'decomposed_GR_random_windows':(256,64,8),
            'decomposed_GR_random_weights':(0.34,0.28,0.22,0.16),
            'decomposed_GR_tail_df':6.0,
            'decomposed_GR_tvt_weight':0.26,
            'decomposed_GR_tvt_base_std':0.26,
            'decomposed_GR_tvt_base_windows':(512,96,8),
            'decomposed_GR_tvt_base_weights':(0.50,0.32,0.18),
            'decomposed_GR_tvt_event_prob':0.50,
            'decomposed_GR_tvt_event_max':3,
            'decomposed_GR_tvt_event_min_len':24,
            'decomposed_GR_tvt_event_max_len':160,
            'decomposed_GR_tvt_event_std':1.25,
            'decomposed_GR_tvt_event_smooth':9,
            'decomposed_GR_target_rmse':10.7,
            'decomposed_GR_offset_std':0.9,
            'decomposed_GR_std_ranges':((5.6,8.4),(8.4,12.2),(12.2,17.5)),
            'decomposed_GR_std_weights':(0.24,0.51,0.25),
            'decomposed_GR_clip_abs':45.0,
            'real_block_len':512,
            'real_block_alpha_range':(0.65,0.90),
            'real_block_scale_clip':(0.6,1.6),
            'structured_alpha_range':(0.25,0.65),
            'structured_windows':(160,64,16),
            'structured_weights':(0.50,0.18,0.25,0.07),
            'structured_scale_clip':(0.6,1.6),
            'structured_clip_abs':45.0,
        },
        'xy_shift':{
            # PF heatmap channels are remapped by original row index plus the
            # induced TVT-axis shift, so this row-preserving augmentation is
            # compatible with cached PF U-Net features.
            'apply_prob': 0.0,
            'ddS_jump_thr': 0.005,
            'min_window':10,
            'shift_bound':10,
            'a_bound':0.06,
            'b_bound':0.08,
        },
        'typewell_mirror':{
            # Source-level reflection around the last visible TVT anchor.
            # It mirrors TVT/S paths plus Typewell/PF TVT-axis sidecars while
            # keeping horizontal GR row values fixed.
            'apply_prob':0.0,
            'skip_out_of_typewell':True,
        },
        'switch_typewell':{
            'apply_prob':0.0,
            'noise_mode':'TVT_bias', # 'copy'/'TVT_bias'
            'same_nan':True,
            'TVT_bias_std':0.95,
            'TVT_bias_window':1280,
        },
    }

    aug_cfg={
        'reverse_path':{
            'apply_prob':0.25,
            'head_TVT_bound':10,
            'sim_head_range':256,
        },
        'start_point_shift':{
            # Disabled for PF heatmap training because changing the visible
            # prefix/target boundary invalidates the fixed train/test cache
            # split semantics.
            'apply_prob':0.0,
            'left_bound':0.0,
            'right_bound':2000,
            'max_ratio':0.3,
        },
        'GR_noise_shift':{
            'apply_prob':0.0,
            'shift_range_ratio':1.0,
            'noise_mode':'copy', # 'copy'/'TVT_bias'/'global_pool'
            'TVT_bias_std':0.95,
            'TVT_bias_window':1280,
        },
        'horizontal_gr_blur':{
            # Blur one low-variance row-contiguous interval in the masked suffix.
            'apply_prob':0.0,
            'flat_region_max_ratio':0.5,
        },
        'MD_streching':{
            'apply_prob':0.4,
            'strech_ratio':(0.7,1.3),
            'apply_to_MD':True,
        },
        'tail_cut':{
            # Source-level variable-horizon augmentation. It removes only the
            # prediction suffix tail and keeps geo/PF sidecars row-aligned.
            'apply_prob':0.2,
            'max_cut_ratio':0.5,
        },
        'channel_mask_2d':{
            'apply_prob':0.7,
            'mask_prob':0.1,
            'PF_mask_prob':0.05,
        },
        'PF_rotate_shift':{
            # Post-item PF-prior corruption. It shifts already-built PF heatmap,
            # PF path, and PF-axis-distance channels by a target-trend-shaped
            # vector, so the normal cache orig-index/TVT-delta remap cannot
            # correct it.
            'apply_prob':0.1,
            'ratio_jitter':0.25,
            'max_abs_shift':20.0,
            'shift_mode':'absolute', # 'absolute' matches k*r*(TVT_rel-TVT_rel0); 'delta' uses k*(r-1).
            'center_mode':'first_valid',
            'k_clip':0.0,
            'min_fit_denom':1e-6,
        },
        'prior_corrupt_2d':{
            # Post-item corruption for candidate sidecars. Disabled by default;
            # enable as an ablation to train reliability under wrong geo/PF priors.
            'apply_prob':0.0,
            'geo_apply_prob':0.75,
            'pf_apply_prob':0.60,
            'geo_bias_std':1.5,
            'geo_ramp_std':3.0,
            'geo_smooth_noise_std':2.5,
            'geo_smooth_window_choices':(9,17,33),
            'geo_jump_prob':0.05,
            'geo_jump_std':4.0,
            'geo_max_abs_shift':12.0,
            'pf_path_noise_std':1.5,
            'pf_path_smooth_window_choices':(5,11,23),
            'pf_path_max_abs_shift':8.0,
            'pf_temperature_range':(0.85,1.35),
            'pf_row_dropout_prob':0.02,
            'pf_wrong_mode_prob':0.04,
            'pf_wrong_mode_shift_range':(6.0,14.0),
            'geo_pf_disagreement_prob':0.0,
            'corrupt_metric_channels':False,
        },
        'seq_mask':{
            'apply_prob':0.1,
            'mask_ratio':0.3,
        },
        'hflip':{
            'apply_prob':0.0,
        },
        'typewell_gr_jitter':{
            # Source-level Typewell GR calibration/texture jitter. It runs after
            # switch_typewell and before GR_noise_shift when enabled.
            'apply_prob':0.0,
            'mode_weights':{
                'coherent_residual_copy':0.75,
                'reference_only':0.25,
            },
            'offset_std':2.0,
            'gain_std':0.04,
            'gain_clip':(0.90,1.10),
            'drift_std':1.5,
            'drift_window_ft_choices':(15.0,30.0,60.0),
            'smooth_prob':0.10,
            'smooth_window_choices':(5,11,21),
            'smooth_blend_range':(0.10,0.35),
        },
        'GR_trf':{
            # Source-level GR transform. The optional power/affine steps affect
            # Typewell GR; center_mirror additionally mirrors both horizontal
            # and Typewell GR around the finite target-region horizontal GR mean.
            'apply_prob':0.6,
            'p_std':0.0,
            'a_std':0.03,
            'a_keep_mean':False,
            'b_std':2.0,
            'ab_window':None,
            'center_mirror_prob':0.0,
            # Smooth and sharpen are mutually exclusive conditional branches.
            'smooth_apply_prob':0.1,
            'sharpen_apply_prob':0.1,
            'smooth_window':5,
            'sharpen_alpha_range':(0.0,0.3),
        },
        'typewell_drift':{
            # Shift the complete Typewell TVT coordinate axis by one shared draw.
            'apply_prob':0.0,
            'shift_std':1.0,
        },
    }
    mixup_prob = 0.0
    mixup_alpha = 0.2
    cutmix_prob = 0.0
    cutmix_alpha = 1.0

    # Train-only constants computed on the fixed seq representation from all
    # 773 local train wells on 2026-05-24. Anchor-relative quantities use their
    # own stats, not the raw-value stats used by the tabular pipeline.
    horizontal_stats = {
        "tvt_input_rel": (-18.08, 48.21),
        "seen_S_rel": (-2.56, 20.68),
        "seen_S_diff": (0.004918, 0.050903),
        "gr": (87.43, 21.09),
        "x_rel": (-131.08, 1527.46),
        "y_rel": (237.93, 2401.99),
        "z_rel": (13.55, 100.51),
        "x_diff": (-0.062839, 0.527670),
        "y_diff": (0.110370, 0.832883),
        "z_diff": (-0.016892, 0.106500),
        "md_rel": (2098.93, 1948.90),
    }
    # Train-global mean/std on the exact fixed-window model-bin representation.
    # These derived motion features operate after x_diff/y_diff downsampling, so
    # their scale is resolution-dependent.
    xy_diff_derived_stats = {
        16: {
            "x_twice_diff": (0.000007185043, 0.012188078570),
            "y_twice_diff": (0.000011960563, 0.006090105006),
            "xy_diff_ac": (0.999936116739, 0.000148069079),
        },
        32: {
            "x_twice_diff": (0.000023421102, 0.010835373293),
            "y_twice_diff": (0.000021618158, 0.007536659599),
            "xy_diff_ac": (0.999933189900, 0.000191574017),
        },
        64: {
            "x_twice_diff": (0.000051624575, 0.016827531475),
            "y_twice_diff": (0.000034859021, 0.013101391696),
            "xy_diff_ac": (0.999822787219, 0.000485642321),
        },
    }
    typewell_stats = {
        "tvt_rel": (0.00, 57.73),
        "gr": (78.62, 28.81),
    }
    global_stats = {
        "X0": (2967133.88, 46774.97),
        "Y0": (1075363.82, 35467.73),
        "Z0": (-9447.93, 611.16),
        "TVT0": (11551.06, 619.50),
    }
    xy_stats = {
        "X": (2965999.12, 47749.98),
        "Y": (1074747.73, 35603.54),
        "Z": (-9390.81, 634.48),
    }
    # U-Net final target is TVT - TVT0. The independent geo prior is
    # generated before dataset construction and then downsampled by the dataset.
    seq_target_mode = UNET_MODE
    alignment_target_mode = "exp_smooth" # "nearest2"/"exp_smooth"
    alignment_exp_smooth_sigma = 1.25
    
    unet_loss_weights = {
        "regression": 0.05/3,
        "offset": 0.0,
        "alignment": 0.05*4/3,
        "GR_penalty": 0.05/3,
        "dS_penalty": 0.0,
        "unet_GR_RMSE": 0.0,
    }
    dS_windows = (1, 7, 15, 31)
    dS_weights = tuple([0.25]*4)
    target_stats = {
        UNET_MODE: (1.591691, 15.853182),
        LEGACY_STAGE_UNET_MODE: (1.591691, 15.853182),
        "d(TVT+Z)": (0.005535, 0.037389),
    }
    surface_stats = {
        "S_rel": (14.711614, 107.048849),
    }
    matched_gr_metric_stats = {
        "geo_tvt_matched_gr_rmse": (18.814210, 5.746824),
        "geo_tvt_matched_gr_corr": (0.074979, 0.187799),
        "pf_tvt_matched_gr_rmse": (14.576291, 4.609135),
        "pf_tvt_matched_gr_corr": (0.355683, 0.258646),
        "pf_tvt_ffbsi_matched_gr_rmse": (13.257146, 4.927424),
        "pf_tvt_ffbsi_matched_gr_corr": (0.455824, 0.296240),
    }
    gr_diff_std = horizontal_stats["gr"][1]
    GR_penalty_clip = 25.0
    gr_slope_norm_clip = 8.0
    gr_tw_slope_norm_clip = 8.0
    # Mean/std after first normalizing GR amplitudes by horizontal GR mean/scale.
    # These full-train constants match the supported resolution sweep.
    gr_quadratic_stats = {
        16: {
            "gr_quadratic_a": (-0.000355310, 0.345196338),
            "gr_quadratic_b": (0.000113479, 0.278552245),
            "gr_quadratic_c": (-0.025152284, 1.039252233),
            "gr_quadratic_rmse": (0.166847150, 0.110077664),
        },
        32: {
            "gr_quadratic_a": (0.000585828, 0.382492540),
            "gr_quadratic_b": (0.001914565, 0.338615701),
            "gr_quadratic_c": (-0.006968186, 1.025529919),
            "gr_quadratic_rmse": (0.212819023, 0.131140903),
        },
        64: {
            "gr_quadratic_a": (-0.001034686, 0.496664219),
            "gr_quadratic_b": (0.004367155, 0.425087849),
            "gr_quadratic_c": (-0.003270647, 1.011349970),
            "gr_quadratic_rmse": (0.256740339, 0.151208789),
        },
    }
    gr_quadratic_norm_clip = 8.0
    xy_diff_derived_norm_clip = 8.0
    gr_compression_power = 0.9308
    gr_ratio_floor = 1.0
    tvt_diff_clip = 100.0

    # Training.
    fold_count = 5
    cv_repeats = 3
    cv_split_mode = "geo_skfold" # "kfold"/"geo_skfold"
    cv_geo_map_path = None
    cv_kmeans_clusters = 20
    cv_kmeans_seed = 0
    cv_split_seed = 33
    train_rm_wells = [
        # rm ones with TVT matched GR inconsistent with true GR and also large oof error
        # based on oof stats is a bit risky, but they should be identifyable only with training loss
        #'48c1f673','4ed93db6','1b1c5372','7a5660b0','9a8ae0d6','000d7d20',
    ]
    model_down_weight_wells = []
    model_down_weight = 0.0

    seed = 7
    epochs = 300
    log_freq = 5
    batch_size = 16
    val_batch_size = 16
    num_workers = 4
    prefetch_factor = 1
    persistent_workers = True
    pin_memory = True
    train_drop_last = True
    optimizer_betas = (0.9, 0.999)
    optimizer_eps = 1e-8
    muon_momentum = 0.95
    muon_ns_steps = 5
    muon_nesterov = True
    grad_accum_steps = 1
    grad_clip_norm = 10.0
    huber_delta = 1.0
    min_seq_weight = 1.0
    seq_noise_weight = False
    seq_noise_smooth = 3.0
    seq_noise_power = 1.0
    min_epochs = 170
    early_stopping_rounds = 30
    onecycle_pct_start = 0.1
    onecycle_div_factor = 10.0
    onecycle_final_div_factor = 100.0

    finetune_epochs = None
    finetune_log_freq = 1
    finetune_lr_reduce = 0.3
    optimizer_cfg = {
        "optimizer": "adamw",
        "lr": 4e-4/2,
        "min_lr": 1e-6/2,
        "weight_decay": 2e-2,
        "scheduler": "cos",
        "constant_ratio": 0.0,
    }
    
    amp_dtype = "bfloat16"
    ema_enabled = True
    ema_decay = 0.99
    # Seeds are always fixed in seq_NN_train.py. This flag additionally enables
    # strict deterministic CUDA kernels, which is reproducible but much slower
    # for the 2D ConvNet.
    deterministic = False
    channels_last_2d = True

    # Model.
    model_name = "unet" # "unet"/"two_stage_unet"
    unet_static_channels = (
        "tw_gr",
        #"tw_gr_m0p25",
        #"tw_gr_p0p25",
        "gr",
        "gr_abs_diff",
        #"gr_abs_diff_tw_m0p25",
        #"gr_abs_diff_tw_p0p25",
        "gr_isnan_rate",
        "gr_std",
        "gr_first_last_delta",
        "gr_slope",
        "gr_quadratic_a",
        "gr_quadratic_b",
        "gr_quadratic_c",
        "gr_quadratic_rmse",
        #"md_rel",
        "tw_gr_is_nan",
        "tw_tvt_rel",
        "seen_tvt_rel",
        "tw_seen_tvt_abs_diff",
        'z_diff',
        #"geo_s_rel",
        #"geo_tvt_rel",
        #"geo_tvt_abs_diff",
        #"geo_dS",
        "geo_tvt_diff",
        #"delta_geo_trend_rbf_5",
        #"geo_tvt_matched_gr_rmse",
        #"geo_tvt_matched_gr_corr",
        #"geo_nbr_std",
        #"geo_nbr_dS_std",
        #"geo_radial_extrap_score",
        #"pf_prob",
        #"pf_particle_density_prob",
        #"pf_mean_abs_diff",
        #"pf_prob_ffbsi",
        #"pf_mean_abs_diff_ffbsi",
        #"pf_tvt",
        #"pf_tvt_ffbsi",
        #"pf_tvt_matched_gr",
        #"pf_tvt_ffbsi_matched_gr",
        #"pf_tvt_matched_gr_rmse",
        #"pf_tvt_matched_gr_corr",
        #"pf_tvt_ffbsi_matched_gr_rmse",
        #"pf_tvt_ffbsi_matched_gr_corr",
        #"x_diff",
        #"y_diff",
        # "x_twice_diff",
        # "y_twice_diff",
        #"xy_diff_ac",
        # "x_abs_sin_0",
        # "x_abs_cos_0",
        # "y_abs_sin_0",
        # "y_abs_cos_0",
        # "x_abs_sin_1",
        # "x_abs_cos_1",
        # "y_abs_sin_1",
        # "y_abs_cos_1",
        # "x_abs_sin_2",
        # "x_abs_cos_2",
        # "y_abs_sin_2",
        # "y_abs_cos_2",
        # "x_abs_sin_3",
        # "x_abs_cos_3",
        # "y_abs_sin_3",
        # "y_abs_cos_3",
        # "x_abs_sin_4",
        # "x_abs_cos_4",
        # "y_abs_sin_4",
        # "y_abs_cos_4",
        #"gr_grad_mean",
        #"gr_grad_std",
        #"gr_diff_mean",
        #"gr_diff_std",
    )
    two_stage_tw_channels = (
        "tw_gr",
        "tw_gr_is_nan",
    )
    two_stage_h_channels = (
        "gr",
        "gr_isnan_rate",
        "gr_std",
        "gr_first_last_delta",
        "gr_slope",
    )
    two_stage_shared_channels = _channels_except(
        unet_static_channels,
        [],#two_stage_tw_channels,
        [],#two_stage_h_channels,
    )
    unet_cfg = {
        "unet_arch": "convnext_small", # "unet"/"convnext_small"
        "unet_pretrained": None,
        "unet_timm_model_name": "hf_hub:timm/convnext_small.in12k_ft_in1k_384",
        "unet_stem_stride": (2, 4),
        "unet_stem_down_mode": "pool",
        "unet_convnext_norm_replace": "BN",  # supports "BN" / "IN" / "LN" (keep native)
        "unet_convnext_bn_eps": 1e-4,
        "unet_convnext_bn_momentum": 0.1,
        "unet_convnext_bn_track_running_stats": True,
        "unet_emb_dim": 32,
        "unet_mults": (1, 2, 4, 4),
        "unet_stacks": 1,
        "res_kernel_size": 3,
        "resblock_act": "silu",  # supports "silu" / "relu"
        "resblock_norm": "BN",  # supports "BN" / "IN" / "LN"
        "unet_dropout": 0.05,
        "head_hidden_mult": 2,
        "updown_mode_2d": "non-param",
        "regression_head_mode": "logits",
        "share_logits_head": True,
        "offset_detach_logits_head_in_loss": False,
        "unet_channel_mask_prob": 0.0,
        "unet_channel_mask_all_prob": 0.0,
        "unet_channel_mask_value": 0.0,
    }
    two_stage_unet_cfg = {
        # ### encoder cfg
        # "tw_channels": two_stage_tw_channels,
        # "h_channels": two_stage_h_channels,
        # "shared_channels": two_stage_shared_channels,
        # "encoder_emb_dim": 64,
        # "encoder_stages": 4,
        # "corr_matrix_dim": 8,
        # "encoder_kernel_size": 3,
        # "encoder_dropout": None,
        # ### decoder cfg
        # "unet_arch": "convnext_small", # "unet"/"convnext_small"
        # "unet_pretrained": None,
        # "unet_timm_model_name": "hf_hub:timm/convnext_small.in12k_ft_in1k_384",
        # "unet_stem_stride": (2, 4),
        # "unet_stem_down_mode": "pool",
        # "unet_convnext_norm_replace": "BN",  # supports "BN" / "IN" / "LN" (keep native)
        # "unet_convnext_bn_eps": 1e-4,
        # "unet_convnext_bn_momentum": 0.1,
        # "unet_convnext_bn_track_running_stats": True,
        # "unet_emb_dim": 32,
        # "unet_mults": (1, 2, 4, 4),
        # "unet_stacks": 1,
        # "res_kernel_size": 3,
        # "resblock_act": "silu",  # supports "silu" / "relu"
        # "resblock_norm": "BN",  # supports "BN" / "IN" / "LN"
        # "unet_dropout": 0.05,
        # "head_hidden_mult": 2,
        # "updown_mode_2d": "non-param",
        # "regression_head_mode": "logits",
        # "share_logits_head": True,
        # "offset_detach_logits_head_in_loss": False,
        # "unet_channel_mask_prob": 0.0,
        # "unet_channel_mask_all_prob": 0.0,
        # "unet_channel_mask_value": 0.0,
    }
    model_cfg = unet_cfg

    # Prediction postprocessing.
    pred_sg_smooth = True
    pred_sg_smooth_window = 1025
    pred_sg_smooth_polyorder = 1
    pred_sg_smooth_blend = 0.6828376753426914

    # Output artifacts.
    oof_filename = "oof_df.pqt"
    submission_filename = "submission.csv"
    submission_details_filename = "submission_details.pqt"
    model_filename = "models.pkl"
    cfg_filename = "cfg.pkl"
    log_filename = "seq_nn.log"

    def __init__(self, **overrides):
        for name, value in self.__class__.__dict__.items():
            if name.startswith("__") or callable(value):
                continue
            setattr(self, name, deepcopy(value))

        for name, value in overrides.items():
            setattr(self, name, value)

        self.refresh()

    def set_sequence_geometry(self):
        self.raw_len = self.prefix_len + self.target_len
        if bool(getattr(self, "strech_to_full_size", False)):
            # The canonical stretch fills the complete raw window.  Use the
            # real ceiling here so an exactly divisible raw window does not
            # acquire an unpopulated terminal bin.
            self.num_bins = (self.raw_len + self.downsample - 1) // self.downsample
        else:
            # Preserve the historical geometry for the native path.
            self.num_bins = self.raw_len // self.downsample + 1

    def set_model_name(self, model_name):
        if model_name == LEGACY_STAGE_UNET_MODE:
            model_name = UNET_MODE
        if model_name not in MODEL_MODES:
            raise ValueError(f"sequence NN model_name must be one of {MODEL_MODES}, got {model_name!r}")
        self.model_name = model_name
        if model_name == UNET_MODE:
            self.model_cfg = self.unet_cfg
        else:
            self.model_cfg = self.two_stage_unet_cfg

    def _migrate_legacy_names(self):
        instance_attrs = self.__dict__
        if "typewell_calibration_power" not in instance_attrs:
            self.typewell_calibration_power = 1.0
        if "typewell_calibration_blend_weight" not in instance_attrs:
            self.typewell_calibration_blend_weight = 0.7
        if "strech_to_full_size" not in instance_attrs:
            self.strech_to_full_size = False
        if "strech_to_full_size_match_seq_weight" not in instance_attrs:
            self.strech_to_full_size_match_seq_weight = True
        if "unet_cfg" not in instance_attrs and "stage_unet_cfg" in instance_attrs:
            self.unet_cfg = self.stage_unet_cfg
        if "unet_static_channels" not in instance_attrs and "stage_unet_static_channels" in instance_attrs:
            self.unet_static_channels = self.stage_unet_static_channels
        if "unet_loss_weights" not in instance_attrs and "stage_unet_loss_weights" in instance_attrs:
            self.unet_loss_weights = self.stage_unet_loss_weights
        if "alignment_target_mode" not in instance_attrs and "stage_alignment_target_mode" in instance_attrs:
            self.alignment_target_mode = self.stage_alignment_target_mode
        if "alignment_exp_smooth_sigma" not in instance_attrs and "stage_alignment_exp_smooth_sigma" in instance_attrs:
            self.alignment_exp_smooth_sigma = self.stage_alignment_exp_smooth_sigma
        if "tvt_diff_clip" not in instance_attrs and "stage_tvt_diff_clip" in instance_attrs:
            self.tvt_diff_clip = self.stage_tvt_diff_clip
        for model_cfg_name in ("unet_cfg", "two_stage_unet_cfg"):
            model_cfg = getattr(self, model_cfg_name, None)
            if not isinstance(model_cfg, dict):
                continue
            if "unet_convnext_norm_to_bn" in model_cfg:
                legacy_replace = model_cfg.pop("unet_convnext_norm_to_bn")
                model_cfg.setdefault(
                    "unet_convnext_norm_replace",
                    "BN" if bool(legacy_replace) else None,
                )
        if "aug_cfg" in instance_attrs:
            for aug_name in ("horizontal_gr_blur", "GR_trf", "typewell_drift"):
                if aug_name not in self.aug_cfg:
                    self.aug_cfg[aug_name] = deepcopy(CFG.aug_cfg[aug_name])
                else:
                    for key, value in CFG.aug_cfg[aug_name].items():
                        self.aug_cfg[aug_name].setdefault(key, deepcopy(value))

    def refresh(self):
        self._migrate_legacy_names()
        self.set_sequence_geometry()
        self.set_seq_target_mode(self.seq_target_mode)
        self.set_model_name(self.model_name)
        return self

    def set_seq_target_mode(self, seq_target_mode):
        if seq_target_mode == LEGACY_STAGE_UNET_MODE:
            seq_target_mode = UNET_MODE
        if seq_target_mode != UNET_MODE:
            raise ValueError("sequence NN now supports only seq_target_mode='unet'")
        self.seq_target_mode = UNET_MODE


def make_cfg(**overrides):
    return CFG(**overrides)


def make_unet_cfg(*, model_overrides=None, loss_weights=None, **overrides):
    cfg = CFG(model_name=UNET_MODE, seq_target_mode=UNET_MODE, **overrides)
    if model_overrides:
        cfg.unet_cfg.update(model_overrides)
        cfg.set_model_name(UNET_MODE)
    if loss_weights is not None:
        cfg.unet_loss_weights = {
            key: float(loss_weights.get(key, 0.0))
            for key in cfg.unet_loss_weights
        }
    return cfg.refresh()


def make_two_stage_unet_cfg(
    *,
    model_overrides=None,
    loss_weights=None,
    tw_channels=None,
    h_channels=None,
    shared_channels=None,
    **overrides,
):
    cfg = CFG(model_name=TWO_STAGE_UNET_MODE, seq_target_mode=UNET_MODE, **overrides)
    if tw_channels is not None:
        cfg.two_stage_tw_channels = tuple(tw_channels)
    if h_channels is not None:
        cfg.two_stage_h_channels = tuple(h_channels)
    if shared_channels is None:
        cfg.two_stage_shared_channels = _channels_except(
            cfg.unet_static_channels,
            cfg.two_stage_tw_channels,
            cfg.two_stage_h_channels,
        )
    else:
        cfg.two_stage_shared_channels = tuple(shared_channels)
    cfg.unet_static_channels = _ordered_unique(
        cfg.two_stage_tw_channels,
        cfg.two_stage_h_channels,
        cfg.two_stage_shared_channels,
    )
    cfg.two_stage_unet_cfg.update(
        {
            "tw_channels": cfg.two_stage_tw_channels,
            "h_channels": cfg.two_stage_h_channels,
            "shared_channels": cfg.two_stage_shared_channels,
        }
    )
    if model_overrides:
        cfg.two_stage_unet_cfg.update(model_overrides)
    cfg.set_model_name(TWO_STAGE_UNET_MODE)
    if loss_weights is not None:
        cfg.unet_loss_weights = {
            key: float(loss_weights.get(key, 0.0))
            for key in cfg.unet_loss_weights
        }
    return cfg.refresh()


def make_loss_ratio_cfg(model_weight, **overrides):
    model_weight = float(model_weight)
    return make_unet_cfg(
        loss_weights={
            "regression": round(model_weight * 0.3, 12),
            "alignment": round(model_weight * 0.7, 12),
        },
        **overrides,
    )


def make_optimizer_cfg(*, shared=None, **overrides):
    cfg = make_unet_cfg(**overrides)
    if shared is not None:
        cfg.optimizer_cfg.update(shared)
    return cfg.refresh()


def _without_features(features, removed):
    removed = set(removed)
    return tuple(name for name in features if name not in removed)


def _with_features(features, added):
    result = list(features)
    for name in added:
        if name not in result:
            result.append(name)
    return tuple(result)


def make_feature_remove_cfg(*, static_remove=(), **overrides):
    cfg = make_unet_cfg(**overrides)
    if static_remove:
        cfg.unet_static_channels = _without_features(cfg.unet_static_channels, static_remove)
    return cfg.refresh()


def make_static_channel_add_cfg(*, static_add=(), **overrides):
    cfg = make_unet_cfg(**overrides)
    if static_add:
        cfg.unet_static_channels = _with_features(cfg.unet_static_channels, static_add)
    return cfg.refresh()


def make_geo_prior_cfg(**geo_prior_overrides):
    cfg = make_unet_cfg()
    if geo_prior_overrides:
        valid_fields = set(GeoPriorConfig.__dataclass_fields__)
        unknown = sorted(set(geo_prior_overrides) - valid_fields)
        if unknown:
            raise ValueError(f"unknown GeoPriorConfig fields: {unknown}")
        values = cfg.geo_prior_cfg.to_dict()
        values.update(geo_prior_overrides)
        cfg.geo_prior_cfg = GeoPriorConfig(**values)
    return cfg.refresh()


def make_sim_cfg(
    *,
    keep_ori_path=None,
    z_shift=None,
    xy_shift=None,
    typewell_mirror=None,
    switch_typewell=None,
    **overrides,
):
    cfg = make_unet_cfg(**overrides)
    if keep_ori_path is not None:
        cfg.sim_cfg["keep_ori_path"].update(keep_ori_path)
    if z_shift is not None:
        z_shift = dict(z_shift)
        if "nan_mode" not in z_shift and "same_nan" in z_shift:
            z_shift["nan_mode"] = "same" if bool(z_shift.pop("same_nan")) else "none"
        cfg.sim_cfg["z_shift"].update(z_shift)
    if xy_shift is not None:
        cfg.sim_cfg["xy_shift"].update(xy_shift)
    if typewell_mirror is not None:
        cfg.sim_cfg["typewell_mirror"].update(typewell_mirror)
    if switch_typewell is not None:
        cfg.sim_cfg["switch_typewell"].update(switch_typewell)
    return cfg.refresh()


def make_aug_cfg(
    *,
    reverse_path=None,
    start_point_shift=None,
    GR_noise_shift=None,
    horizontal_gr_blur=None,
    MD_streching=None,
    tail_cut=None,
    channel_mask_2d=None,
    PF_rotate_shift=None,
    prior_corrupt_2d=None,
    seq_mask=None,
    hflip=None,
    typewell_gr_jitter=None,
    GR_trf=None,
    typewell_drift=None,
    **overrides,
):
    cfg = make_unet_cfg(**overrides)
    if reverse_path is not None:
        cfg.aug_cfg["reverse_path"].update(reverse_path)
    if start_point_shift is not None:
        cfg.aug_cfg["start_point_shift"].update(start_point_shift)
    if GR_noise_shift is not None:
        cfg.aug_cfg["GR_noise_shift"].update(GR_noise_shift)
    if horizontal_gr_blur is not None:
        cfg.aug_cfg["horizontal_gr_blur"].update(horizontal_gr_blur)
    if MD_streching is not None:
        cfg.aug_cfg["MD_streching"].update(MD_streching)
    if tail_cut is not None:
        cfg.aug_cfg["tail_cut"].update(tail_cut)
    if channel_mask_2d is not None:
        cfg.aug_cfg["channel_mask_2d"].update(channel_mask_2d)
    if PF_rotate_shift is not None:
        cfg.aug_cfg["PF_rotate_shift"].update(PF_rotate_shift)
    if prior_corrupt_2d is not None:
        cfg.aug_cfg["prior_corrupt_2d"].update(prior_corrupt_2d)
    if seq_mask is not None:
        cfg.aug_cfg["seq_mask"].update(seq_mask)
    if hflip is not None:
        cfg.aug_cfg["hflip"].update(hflip)
    if typewell_gr_jitter is not None:
        cfg.aug_cfg["typewell_gr_jitter"].update(typewell_gr_jitter)
    if GR_trf is not None:
        cfg.aug_cfg["GR_trf"].update(GR_trf)
    if typewell_drift is not None:
        cfg.aug_cfg["typewell_drift"].update(typewell_drift)
    return cfg.refresh()

# Sequence random-search space. Keep this as the single source of truth when
# the search space changes. Names match the implemented config keys; in
# particular, MD_streching and strech_ratio are historical runtime spellings.
SEQ_RANDOM_SEARCH_SPACE = {
    "keep_ori_path_apply_prob": (0.0, 0.025, 0.05),
    "z_shift_sample_model": ("diff_block_bootstrap", "target_with_rules"),
    "z_shift_noise_mode": (
        {"TVT_bias": 1.0},
        {"TVT_bias_v2": 1.0},
        {"mixed_V1": 1.0},
        {"conditional_GR": 1.0},
        {"decomposed_GR": 1.0},
        {"TVT_bias": 0.5, "decomposed_GR": 0.5},
        {"TVT_bias": 0.5, "conditional_GR": 0.5},
        {"TVT_bias": 0.5, "TVT_bias_v2": 0.5},
    ),
    "z_shift_nan_mode": ("same", "same", "same", "same", "global"),
    "typewell_mirror_apply_prob": (0.0, 0.05, 0.1),
    "typewell_mirror_skip_out_of_typewell": (True, False),
    "switch_typewell_apply_prob": (0.0, 0.05, 0.1, 0.2, 0.4),
    "reverse_path_apply_prob": (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4),
    "gr_noise_shift_apply_prob": (0.2, 0.3, 0.4),
    "md_streching_strech_ratio": (0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
    "channel_mask_2d_apply_prob": (0.2, 0.3, 0.4, 0.5),
    "channel_mask_2d_mask_prob": (0.05, 0.1, 0.15),
    "pf_rotate_shift_apply_prob": (0.0, 0.05, 0.1, 0.15, 0.2),
    "pf_rotate_shift_ratio_jitter": (0.1, 0.15, 0.25, 0.3, 0.4, 0.5),
    "prior_corrupt_2d_apply_prob": (0.05, 0.1, 0.15, 0.2),
    "seq_mask_apply_prob": (0.0, 0.05, 0.1, 0.15, 0.2),
    "typewell_gr_jitter_apply_prob": (0.0, 0.05, 0.1, 0.15),
    "gr_trf_apply_prob": (0.0, 0.05, 0.1, 0.2, 0.3),
    "alignment_exp_smooth_sigma": (1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4),
    "optimizer_lr": (3e-4, 4e-4, 5e-4, 6e-4, 7e-4),
    "optimizer_weight_decay": (1e-3, 5e-3, 1e-2, 2e-2, 5e-2),
    "unet_loss_weight_regression": (0.05 / 3.0, 0.02 / 3.0, 0.05 * 2.0 / 3.0),
    "unet_loss_weight_alignment": (0.05 * 4.0 / 3.0, 0.05 * 6.0 / 3.0, 0.05 * 8.0 / 3.0),
}

SEQ_RANDOM_SEARCH_DATE = 20260630
SEQ_RANDOM_SEARCH_POSTSEED_START = 0
SEQ_RANDOM_SEARCH_POSTSEED_WIDTH = 3
SEQ_RANDOM_SEARCH_COUNT = 16
SEQ_RANDOM_SEARCH_MAX_ATTEMPTS = 10000


def make_seq_random_search_seed(date, postseed, *, width=SEQ_RANDOM_SEARCH_POSTSEED_WIDTH):
    limit = 10 ** int(width)
    postseed = int(postseed)
    if postseed < 0 or postseed >= limit:
        raise ValueError(f"postseed must be in [0, {limit}), got {postseed}")
    return int(date) * limit + postseed


def make_random_search_cfg(
    *,
    keep_ori_path_apply_prob,
    z_shift_sample_model,
    z_shift_noise_mode,
    z_shift_nan_mode,
    typewell_mirror_apply_prob,
    typewell_mirror_skip_out_of_typewell,
    switch_typewell_apply_prob,
    reverse_path_apply_prob,
    gr_noise_shift_apply_prob,
    md_streching_strech_ratio,
    channel_mask_2d_apply_prob,
    channel_mask_2d_mask_prob,
    pf_rotate_shift_apply_prob,
    pf_rotate_shift_ratio_jitter,
    prior_corrupt_2d_apply_prob,
    seq_mask_apply_prob,
    typewell_gr_jitter_apply_prob,
    gr_trf_apply_prob,
    alignment_exp_smooth_sigma,
    optimizer_lr,
    optimizer_weight_decay,
    unet_loss_weight_regression,
    unet_loss_weight_alignment,
):
    cfg = make_unet_cfg(alignment_exp_smooth_sigma=float(alignment_exp_smooth_sigma))
    cfg.sim_cfg["keep_ori_path"].update({"apply_prob": keep_ori_path_apply_prob})
    cfg.sim_cfg["z_shift"].update(
        {
            "sample_model": z_shift_sample_model,
            "noise_mode": deepcopy(z_shift_noise_mode),
            "nan_mode": z_shift_nan_mode,
        }
    )
    cfg.sim_cfg["typewell_mirror"].update(
        {
            "apply_prob": typewell_mirror_apply_prob,
            "skip_out_of_typewell": bool(typewell_mirror_skip_out_of_typewell),
        }
    )
    cfg.sim_cfg["switch_typewell"].update({"apply_prob": switch_typewell_apply_prob})
    cfg.aug_cfg["reverse_path"].update({"apply_prob": reverse_path_apply_prob})
    cfg.aug_cfg["GR_noise_shift"].update(
        {
            "apply_prob": gr_noise_shift_apply_prob,
        }
    )
    cfg.aug_cfg["MD_streching"].update(
        {
            "strech_ratio": md_streching_strech_ratio,
        }
    )
    cfg.aug_cfg["channel_mask_2d"].update(
        {
            "apply_prob": channel_mask_2d_apply_prob,
            "mask_prob": channel_mask_2d_mask_prob,
            "PF_mask_prob": channel_mask_2d_mask_prob,
        }
    )
    cfg.aug_cfg["PF_rotate_shift"].update(
        {
            "apply_prob": pf_rotate_shift_apply_prob,
            "ratio_jitter": pf_rotate_shift_ratio_jitter,
        }
    )
    cfg.aug_cfg["prior_corrupt_2d"].update({"apply_prob": prior_corrupt_2d_apply_prob})
    cfg.aug_cfg["seq_mask"].update(
        {
            "apply_prob": seq_mask_apply_prob,
        }
    )
    cfg.aug_cfg["typewell_gr_jitter"].update({"apply_prob": typewell_gr_jitter_apply_prob})
    cfg.aug_cfg["GR_trf"].update({"apply_prob": gr_trf_apply_prob})
    cfg.optimizer_cfg["lr"] = optimizer_lr
    cfg.optimizer_cfg["weight_decay"] = optimizer_weight_decay
    cfg.unet_loss_weights["regression"] = float(unet_loss_weight_regression)
    cfg.unet_loss_weights["alignment"] = float(unet_loss_weight_alignment)
    return cfg.refresh()


def sample_seq_random_search_spec(seed, search_space=None):
    search_space = SEQ_RANDOM_SEARCH_SPACE if search_space is None else search_space
    rng = random.Random(int(seed))
    spec = {}
    for key, choices in search_space.items():
        if not choices:
            raise ValueError(f"empty random-search choices for {key}")
        spec[key] = rng.choice(tuple(choices))
    return _normalize_seq_random_search_spec(spec)


def _normalize_seq_random_search_spec(spec):
    spec = dict(spec)
    # Collapse inactive sub-knobs to defaults so dedup compares effective configs.
    if float(spec["typewell_mirror_apply_prob"]) == 0.0:
        spec["typewell_mirror_skip_out_of_typewell"] = CFG.sim_cfg["typewell_mirror"][
            "skip_out_of_typewell"
        ]
    if float(spec["pf_rotate_shift_apply_prob"]) == 0.0:
        spec["pf_rotate_shift_ratio_jitter"] = CFG.aug_cfg["PF_rotate_shift"]["ratio_jitter"]
    return spec


def _canonical_seq_random_search_value(value):
    if isinstance(value, dict):
        return tuple(
            (str(key), _canonical_seq_random_search_value(value[key]))
            for key in sorted(value)
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_seq_random_search_value(item) for item in value)
    return value


def _seq_random_search_spec_key(spec):
    spec = _normalize_seq_random_search_spec(spec)
    return tuple((key, _canonical_seq_random_search_value(spec[key])) for key in sorted(spec))


def deduplicate_seq_random_search_specs(named_specs):
    deduped = []
    seen = set()
    for name, spec in named_specs:
        key = _seq_random_search_spec_key(spec)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, _normalize_seq_random_search_spec(spec)))
    return deduped


def sample_seq_random_search_specs(
    count,
    *,
    date=SEQ_RANDOM_SEARCH_DATE,
    postseed_start=SEQ_RANDOM_SEARCH_POSTSEED_START,
    postseed_width=SEQ_RANDOM_SEARCH_POSTSEED_WIDTH,
    search_space=None,
    max_attempts=None,
):
    max_attempts = SEQ_RANDOM_SEARCH_MAX_ATTEMPTS if max_attempts is None else int(max_attempts)
    named_specs = []
    seen = set()
    postseed = int(postseed_start)
    postseed_limit = 10 ** int(postseed_width)
    if postseed < 0 or postseed >= postseed_limit:
        raise ValueError(f"postseed_start must be in [0, {postseed_limit}), got {postseed_start}")
    attempts = 0
    while len(named_specs) < int(count) and attempts < max_attempts:
        if postseed >= postseed_limit:
            raise RuntimeError(
                f"postseed width {postseed_width} exhausted while sampling seq random-search configs"
            )
        seed = make_seq_random_search_seed(date, postseed, width=postseed_width)
        spec = sample_seq_random_search_spec(seed, search_space=search_space)
        key = _seq_random_search_spec_key(spec)
        if key not in seen:
            seen.add(key)
            named_specs.append((str(seed), spec))
        postseed += 1
        attempts += 1
    if len(named_specs) != int(count):
        raise RuntimeError(
            f"sampled {len(named_specs)} unique seq random-search configs, "
            f"expected {count}; increase max_attempts or expand the search space"
        )
    return named_specs


def make_seq_random_search_cfgs(
    count=SEQ_RANDOM_SEARCH_COUNT,
    *,
    date=SEQ_RANDOM_SEARCH_DATE,
    postseed_start=SEQ_RANDOM_SEARCH_POSTSEED_START,
):
    named_specs = sample_seq_random_search_specs(
        count=count,
        date=date,
        postseed_start=postseed_start,
    )
    return _validate_seq_train_cfgs(
        [(name, make_random_search_cfg(**spec)) for name, spec in named_specs]
    )


def _fmt_float_name(value):
    return str(value).replace(".", "p")


def _make_cfg_name(prefix, value):
    if isinstance(value, bool):
        value_text = str(value)
    elif isinstance(value, (tuple, list)):
        value_text = "_".join(str(v) for v in value)
    else:
        value_text = str(value)
    safe = (
        value_text.replace(".", "p")
        .replace("-", "m")
        .replace("/", "_")
        .replace("+", "plus")
        .replace(" ", "")
    )
    return f"{prefix}_{safe}"


def _equal_weights(count):
    count = int(count)
    if count <= 0:
        raise ValueError(f"count must be positive: {count}")
    weight = 1.0 / count
    weights = [weight] * (count - 1)
    weights.append(1.0 - sum(weights))
    return tuple(weights)


def _ordered_intersection(first, second):
    second = set(second)
    return [well_id for well_id in first if well_id in second]


def _is_valid_abs_fourier_channel(channel):
    parts = str(channel).split("_")
    if len(parts) != 4:
        return False
    axis, abs_token, kind, frequency = parts
    if axis not in {"x", "y", "z"} or abs_token != "abs" or kind not in {"sin", "cos"}:
        return False
    try:
        return int(frequency) >= 0
    except ValueError:
        return False


def _validate_seq_train_cfgs(cfgs):
    cfg_names = [name for name, _ in cfgs]
    if len(cfg_names) != len(set(cfg_names)):
        duplicated = sorted({name for name in cfg_names if cfg_names.count(name) > 1})
        raise ValueError(f"duplicate SEQ_TRAIN_CFGS names: {duplicated}")

    valid_static_channels = {
        "tw_gr",
        "tw_gr_m0p25",
        "tw_gr_p0p25",
        "gr",
        "gr_abs_diff",
        "gr_abs_diff_tw_m0p25",
        "gr_abs_diff_tw_p0p25",
        "gr_signed_diff",
        "gr_compression_residual",
        "gr_ratio_diff",
        "gr_diff_x_tw_slope",
        "gr_isnan_rate",
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
        "md_rel",
        "x_diff",
        "y_diff",
        "x_twice_diff",
        "y_twice_diff",
        "xy_diff_ac",
        "z_rel",
        "z_diff",
        "tw_gr_is_nan",
        "tw_tvt_rel",
        "seen_tvt_rel",
        "tw_seen_tvt_abs_diff",
        "geo_s_rel",
        "geo_tvt_rel",
        "geo_tvt_abs_diff",
        "geo_dS",
        "geo_tvt_diff",
        "delta_geo_trend_rbf_5",
        "geo_nbr_std",
        "geo_nbr_dS_std",
        "geo_radial_extrap_score",
        "pf_prob",
        "pf_particle_density_prob",
        "pf_mean_abs_diff",
        "pf_prob_ffbsi",
        "pf_mean_abs_diff_ffbsi",
        "pf_entropy_ffbsi",
        "pf_ess_frac_ffbsi",
        "pf_max_prob_ffbsi",
        "pf_anchor_mass_5",
        "pf_anchor_mass_10",
        "pf_anchor_mass_20",
        "pf_geo_abs_diff",
        "pf_anchor_abs_diff",
        "pf_filtered_ffbsi_abs_diff",
        "pf_tvt",
        "pf_tvt_ffbsi",
        "pf_tvt_matched_gr",
        "pf_tvt_ffbsi_matched_gr",
        "geo_tvt_matched_gr_rmse",
        "geo_tvt_matched_gr_corr",
        "pf_tvt_matched_gr_rmse",
        "pf_tvt_matched_gr_corr",
        "pf_tvt_ffbsi_matched_gr_rmse",
        "pf_tvt_ffbsi_matched_gr_corr",
    }
    valid_z_sample_models = {"target", "target_with_rules", "diff_block_bootstrap", "block_bootstrap", "PF_sample", "mixed"}
    valid_z_noise_modes = {
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
    valid_z_nan_modes = {"same", "none", "shift", "global"}
    valid_unet_arches = {"unet", "convnext_small"}
    valid_resblock_acts = {"silu", "relu"}
    valid_resblock_norms = {"BN", "IN", "LN"}
    valid_convnext_norm_replacements = {"BN", "IN", "LN"}
    valid_model_names = set(MODEL_MODES)
    valid_typewell_gr_grid_types = set(TYPEWELL_GR_GRID_TYPES)
    valid_typewell_gr_jitter_modes = {"coherent_residual_copy", "reference_only"}
    valid_gr_normalize_modes = {"global", "local"}
    valid_optimizers = {"adamw", "radam", "muon"}
    valid_schedulers = {"cos", "cosine", "onecycle"}

    for name, cfg in cfgs:
        model_name = str(getattr(cfg, "model_name", UNET_MODE))
        if model_name == LEGACY_STAGE_UNET_MODE:
            model_name = UNET_MODE
        if model_name not in valid_model_names:
            raise ValueError(f"{name}: unknown model_name={model_name!r}")
        model_cfg = getattr(cfg, "model_cfg", None)
        if model_cfg is None:
            model_cfg = cfg.two_stage_unet_cfg if model_name == TWO_STAGE_UNET_MODE else cfg.unet_cfg

        for mode_attr in ("gr_normalize_mean_mode", "gr_normalize_std_mode"):
            normalize_mode = str(getattr(cfg, mode_attr, "global"))
            if normalize_mode not in valid_gr_normalize_modes:
                raise ValueError(
                    f"{name}: {mode_attr} must be one of {sorted(valid_gr_normalize_modes)}, "
                    f"got {normalize_mode!r}"
                )

        typewell_calibration_power = float(
            getattr(cfg, "typewell_calibration_power", 1.0)
        )
        if not math.isfinite(typewell_calibration_power) or typewell_calibration_power <= 0.0:
            raise ValueError(
                f"{name}: typewell_calibration_power must be finite and positive, "
                f"got {typewell_calibration_power}"
            )
        typewell_calibration_blend_weight = float(
            getattr(cfg, "typewell_calibration_blend_weight", 0.7)
        )
        if (
            not math.isfinite(typewell_calibration_blend_weight)
            or not 0.0 <= typewell_calibration_blend_weight <= 1.0
        ):
            raise ValueError(
                f"{name}: typewell_calibration_blend_weight must be finite and in [0, 1], "
                f"got {typewell_calibration_blend_weight}"
            )

        tw_gr_grid_type = str(getattr(cfg, "tw_gr_grid_type", "interpolate")).lower()
        if tw_gr_grid_type not in valid_typewell_gr_grid_types:
            raise ValueError(
                f"{name}: tw_gr_grid_type must be one of {sorted(valid_typewell_gr_grid_types)}, "
                f"got {tw_gr_grid_type!r}"
            )

        static_channels = set(cfg.unet_static_channels)
        unknown_static = sorted(
            channel
            for channel in static_channels - valid_static_channels
            if not _is_valid_abs_fourier_channel(channel)
        )
        if unknown_static:
            raise ValueError(f"{name}: unknown unet_static_channels={unknown_static}")

        for positive_attr in ("gr_compression_power", "gr_ratio_floor"):
            positive_value = float(getattr(cfg, positive_attr, 0.0))
            if not math.isfinite(positive_value) or positive_value <= 0.0:
                raise ValueError(f"{name}: {positive_attr} must be finite and positive, got {positive_value}")
        gr_tw_slope_norm_clip = float(getattr(cfg, "gr_tw_slope_norm_clip", 8.0))
        if not math.isfinite(gr_tw_slope_norm_clip) or gr_tw_slope_norm_clip < 0.0:
            raise ValueError(
                f"{name}: gr_tw_slope_norm_clip must be finite and non-negative, "
                f"got {gr_tw_slope_norm_clip}"
            )
        xy_diff_derived_norm_clip = float(getattr(cfg, "xy_diff_derived_norm_clip", 8.0))
        if not math.isfinite(xy_diff_derived_norm_clip) or xy_diff_derived_norm_clip < 0.0:
            raise ValueError(
                f"{name}: xy_diff_derived_norm_clip must be finite and non-negative, "
                f"got {xy_diff_derived_norm_clip}"
            )

        if model_name == TWO_STAGE_UNET_MODE:
            static_channel_tuple = tuple(cfg.unet_static_channels)
            static_channel_set = set(static_channel_tuple)
            for group_name in ("tw_channels", "h_channels", "shared_channels"):
                group = tuple(model_cfg.get(group_name, ()))
                if len(group) == 0 and group_name != "shared_channels":
                    raise ValueError(f"{name}: two_stage_unet {group_name} must not be empty")
                missing_group = sorted(set(group) - static_channel_set)
                if missing_group:
                    raise ValueError(
                        f"{name}: two_stage_unet {group_name} contains unknown channels={missing_group}"
                    )
            encoder_emb_dim = int(model_cfg.get("encoder_emb_dim", 64))
            encoder_stages = int(model_cfg.get("encoder_stages", 4))
            corr_matrix_dim = int(model_cfg.get("corr_matrix_dim", 8))
            if encoder_emb_dim <= 0:
                raise ValueError(f"{name}: encoder_emb_dim must be positive, got {encoder_emb_dim}")
            if encoder_stages <= 0:
                raise ValueError(f"{name}: encoder_stages must be positive, got {encoder_stages}")
            if corr_matrix_dim <= 0:
                raise ValueError(f"{name}: corr_matrix_dim must be positive, got {corr_matrix_dim}")
            if encoder_emb_dim % corr_matrix_dim != 0:
                raise ValueError(
                    f"{name}: encoder_emb_dim must be divisible by corr_matrix_dim, "
                    f"got {encoder_emb_dim} and {corr_matrix_dim}"
                )

        sample_model = cfg.sim_cfg["z_shift"].get("sample_model")
        if sample_model not in valid_z_sample_models:
            raise ValueError(f"{name}: unknown z_shift sample_model={sample_model!r}")

        z_noise_mode = cfg.sim_cfg["z_shift"].get("noise_mode")
        if isinstance(z_noise_mode, str):
            z_noise_weights = {z_noise_mode: 1.0}
        elif isinstance(z_noise_mode, dict):
            z_noise_weights = {str(mode): float(weight) for mode, weight in z_noise_mode.items()}
        else:
            raise TypeError(
                f"{name}: z_shift noise_mode must be a string or a dict of mode probabilities, "
                f"got {type(z_noise_mode).__name__}"
            )
        unknown_z_noise = sorted(set(z_noise_weights) - valid_z_noise_modes)
        if unknown_z_noise:
            raise ValueError(f"{name}: unknown z_shift noise_mode={unknown_z_noise!r}")
        nonfinite_z_noise = {mode: weight for mode, weight in z_noise_weights.items() if not math.isfinite(weight)}
        if nonfinite_z_noise:
            raise ValueError(f"{name}: z_shift noise_mode probabilities must be finite: {nonfinite_z_noise}")
        active_z_noise = {mode: weight for mode, weight in z_noise_weights.items() if weight > 0.0}
        negative_z_noise = {mode: weight for mode, weight in z_noise_weights.items() if weight < 0.0}
        if negative_z_noise:
            raise ValueError(f"{name}: z_shift noise_mode probabilities must be non-negative: {negative_z_noise}")
        if not active_z_noise:
            raise ValueError(f"{name}: z_shift noise_mode must give positive probability to at least one mode")

        z_shift_cfg = cfg.sim_cfg["z_shift"]
        diff_block_lengths = tuple(int(v) for v in z_shift_cfg.get("diff_block_lengths", ()))
        if len(diff_block_lengths) == 0 or any(v <= 0 for v in diff_block_lengths):
            raise ValueError(f"{name}: diff_block_lengths must contain positive integers, got {diff_block_lengths!r}")
        diff_block_weights = tuple(float(v) for v in z_shift_cfg.get("diff_block_weights", ()))
        if len(diff_block_weights) != len(diff_block_lengths):
            raise ValueError(
                f"{name}: diff_block_weights length must match diff_block_lengths, "
                f"got {len(diff_block_weights)} vs {len(diff_block_lengths)}"
            )
        if (
            any(not math.isfinite(v) for v in diff_block_weights)
            or any(v < 0.0 for v in diff_block_weights)
            or sum(diff_block_weights) <= 0.0
        ):
            raise ValueError(f"{name}: diff_block_weights must be finite non-negative values with positive sum")
        diff_block_join_width = int(z_shift_cfg.get("diff_block_join_width", 0))
        if diff_block_join_width < 0:
            raise ValueError(f"{name}: diff_block_join_width must be non-negative, got {diff_block_join_width}")
        diff_block_reverse_prob = float(z_shift_cfg.get("diff_block_reverse_prob", 0.0))
        if not 0.0 <= diff_block_reverse_prob <= 1.0:
            raise ValueError(f"{name}: diff_block_reverse_prob must be in [0, 1], got {diff_block_reverse_prob}")
        diff_block_candidate_count = int(z_shift_cfg.get("diff_block_candidate_count", 1))
        if diff_block_candidate_count <= 0:
            raise ValueError(f"{name}: diff_block_candidate_count must be positive, got {diff_block_candidate_count}")
        v2_windows = tuple(z_shift_cfg.get("TVT_bias_v2_windows", ()))
        if len(v2_windows) != 3 or any(int(v) <= 0 for v in v2_windows):
            raise ValueError(f"{name}: TVT_bias_v2_windows must contain three positive integers, got {v2_windows!r}")
        v2_weights = tuple(float(v) for v in z_shift_cfg.get("TVT_bias_v2_weights", ()))
        if len(v2_weights) != 3:
            raise ValueError(f"{name}: TVT_bias_v2_weights must contain three values, got {v2_weights!r}")
        if any(not math.isfinite(v) for v in v2_weights) or any(v < 0.0 for v in v2_weights) or sum(v2_weights) <= 0.0:
            raise ValueError(f"{name}: TVT_bias_v2_weights must be finite non-negative values with positive sum")
        v2_std = float(z_shift_cfg.get("TVT_bias_v2_tvt_std", 0.0))
        if not math.isfinite(v2_std) or v2_std < 0.0:
            raise ValueError(f"{name}: TVT_bias_v2_tvt_std must be finite and non-negative, got {v2_std}")
        v2_ranges = tuple(z_shift_cfg.get("TVT_bias_v2_gr_std_ranges", ()))
        v2_range_weights = tuple(float(v) for v in z_shift_cfg.get("TVT_bias_v2_gr_std_weights", ()))
        if len(v2_ranges) == 0 or len(v2_ranges) != len(v2_range_weights):
            raise ValueError(
                f"{name}: TVT_bias_v2_gr_std_ranges and TVT_bias_v2_gr_std_weights must be non-empty and equal length"
            )
        for value_range in v2_ranges:
            if len(value_range) != 2:
                raise ValueError(f"{name}: TVT_bias_v2_gr_std_ranges entries must have length 2, got {value_range!r}")
            low, high = float(value_range[0]), float(value_range[1])
            if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or low > high:
                raise ValueError(f"{name}: invalid TVT_bias_v2_gr_std range {value_range!r}")
        if (
            any(not math.isfinite(v) for v in v2_range_weights)
            or any(v < 0.0 for v in v2_range_weights)
            or sum(v2_range_weights) <= 0.0
        ):
            raise ValueError(f"{name}: TVT_bias_v2_gr_std_weights must be finite non-negative values with positive sum")
        if z_shift_cfg.get("TVT_bias_v2_scale_mode", "std") not in {"none", "std", "rmse"}:
            raise ValueError(f"{name}: unknown TVT_bias_v2_scale_mode={z_shift_cfg.get('TVT_bias_v2_scale_mode')!r}")
        v2_offset_std = float(z_shift_cfg.get("TVT_bias_v2_gr_offset_std", 0.0))
        if not math.isfinite(v2_offset_std) or v2_offset_std < 0.0:
            raise ValueError(f"{name}: TVT_bias_v2_gr_offset_std must be finite and non-negative, got {v2_offset_std}")
        v2_clip_abs = float(z_shift_cfg.get("TVT_bias_v2_clip_abs", 0.0))
        if not math.isfinite(v2_clip_abs) or v2_clip_abs < 0.0:
            raise ValueError(f"{name}: TVT_bias_v2_clip_abs must be finite and non-negative, got {v2_clip_abs}")
        cond_bins = int(z_shift_cfg.get("conditional_GR_bins", 0))
        if cond_bins < 2:
            raise ValueError(f"{name}: conditional_GR_bins must be at least 2, got {cond_bins}")
        cond_min_count = int(z_shift_cfg.get("conditional_GR_min_count", 0))
        if cond_min_count <= 0:
            raise ValueError(f"{name}: conditional_GR_min_count must be positive, got {cond_min_count}")
        cond_mean_shrink = float(z_shift_cfg.get("conditional_GR_mean_shrink", 0.0))
        if not math.isfinite(cond_mean_shrink) or cond_mean_shrink < 0.0:
            raise ValueError(
                f"{name}: conditional_GR_mean_shrink must be finite and non-negative, got {cond_mean_shrink}"
            )
        cond_scale_blend = float(z_shift_cfg.get("conditional_GR_scale_blend", 0.0))
        if not 0.0 <= cond_scale_blend <= 1.0:
            raise ValueError(f"{name}: conditional_GR_scale_blend must be in [0, 1], got {cond_scale_blend}")
        cond_windows = tuple(int(v) for v in z_shift_cfg.get("conditional_GR_windows", ()))
        if len(cond_windows) == 0 or any(v <= 0 for v in cond_windows):
            raise ValueError(f"{name}: conditional_GR_windows must contain positive integers, got {cond_windows!r}")
        cond_weights = tuple(float(v) for v in z_shift_cfg.get("conditional_GR_weights", ()))
        if len(cond_weights) != len(cond_windows) + 1:
            raise ValueError(
                f"{name}: conditional_GR_weights must contain one value per window plus one tail value, "
                f"got {len(cond_weights)} weights for {len(cond_windows)} windows"
            )
        if (
            any(not math.isfinite(v) for v in cond_weights)
            or any(v < 0.0 for v in cond_weights)
            or sum(cond_weights) <= 0.0
        ):
            raise ValueError(f"{name}: conditional_GR_weights must be finite non-negative values with positive sum")
        cond_tail_df = float(z_shift_cfg.get("conditional_GR_tail_df", 0.0))
        if not math.isfinite(cond_tail_df) or cond_tail_df <= 0.0:
            raise ValueError(f"{name}: conditional_GR_tail_df must be finite and positive, got {cond_tail_df}")
        cond_texture_weight = float(z_shift_cfg.get("conditional_GR_texture_weight", 0.0))
        if not 0.0 <= cond_texture_weight <= 1.0:
            raise ValueError(
                f"{name}: conditional_GR_texture_weight must be in [0, 1], got {cond_texture_weight}"
            )
        cond_tvt_std = float(z_shift_cfg.get("conditional_GR_tvt_std", 0.0))
        if not math.isfinite(cond_tvt_std) or cond_tvt_std < 0.0:
            raise ValueError(f"{name}: conditional_GR_tvt_std must be finite and non-negative, got {cond_tvt_std}")
        cond_offset_std = float(z_shift_cfg.get("conditional_GR_offset_std", 0.0))
        if not math.isfinite(cond_offset_std) or cond_offset_std < 0.0:
            raise ValueError(
                f"{name}: conditional_GR_offset_std must be finite and non-negative, got {cond_offset_std}"
            )
        cond_ranges = tuple(z_shift_cfg.get("conditional_GR_std_ranges", ()))
        cond_range_weights = tuple(float(v) for v in z_shift_cfg.get("conditional_GR_std_weights", ()))
        if len(cond_ranges) != len(cond_range_weights):
            raise ValueError(
                f"{name}: conditional_GR_std_ranges and conditional_GR_std_weights must have equal length"
            )
        if len(cond_ranges) > 0:
            for value_range in cond_ranges:
                if len(value_range) != 2:
                    raise ValueError(f"{name}: conditional_GR_std_ranges entries must have length 2, got {value_range!r}")
                low, high = float(value_range[0]), float(value_range[1])
                if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or low > high:
                    raise ValueError(f"{name}: invalid conditional_GR_std range {value_range!r}")
            if (
                any(not math.isfinite(v) for v in cond_range_weights)
                or any(v < 0.0 for v in cond_range_weights)
                or sum(cond_range_weights) <= 0.0
            ):
                raise ValueError(
                    f"{name}: conditional_GR_std_weights must be finite non-negative values with positive sum"
                )
        cond_clip_abs = float(z_shift_cfg.get("conditional_GR_clip_abs", 0.0))
        if not math.isfinite(cond_clip_abs) or cond_clip_abs < 0.0:
            raise ValueError(f"{name}: conditional_GR_clip_abs must be finite and non-negative, got {cond_clip_abs}")

        decomp_mean_mode = str(z_shift_cfg.get("decomposed_GR_mean_mode", "power"))
        if decomp_mean_mode not in {"power", "empirical"}:
            raise ValueError(f"{name}: unknown decomposed_GR_mean_mode={decomp_mean_mode!r}")
        decomp_power = float(z_shift_cfg.get("decomposed_GR_power", 0.9308))
        if not math.isfinite(decomp_power) or decomp_power <= 0.0:
            raise ValueError(f"{name}: decomposed_GR_power must be finite and positive, got {decomp_power}")
        for key in (
            "decomposed_GR_mean_shrink",
            "decomposed_GR_tvt_weight",
            "decomposed_GR_tvt_event_prob",
        ):
            value = float(z_shift_cfg.get(key, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}: {key} must be in [0, 1], got {value}")
        decomp_scale_blend = float(z_shift_cfg.get("decomposed_GR_scale_blend", 0.0))
        if not 0.0 <= decomp_scale_blend <= 1.0:
            raise ValueError(f"{name}: decomposed_GR_scale_blend must be in [0, 1], got {decomp_scale_blend}")
        decomp_random_windows = tuple(int(v) for v in z_shift_cfg.get("decomposed_GR_random_windows", ()))
        if len(decomp_random_windows) == 0 or any(v <= 0 for v in decomp_random_windows):
            raise ValueError(
                f"{name}: decomposed_GR_random_windows must contain positive integers, got {decomp_random_windows!r}"
            )
        decomp_random_weights = tuple(float(v) for v in z_shift_cfg.get("decomposed_GR_random_weights", ()))
        if len(decomp_random_weights) != len(decomp_random_windows) + 1:
            raise ValueError(
                f"{name}: decomposed_GR_random_weights must contain one value per window plus one tail value"
            )
        if (
            any(not math.isfinite(v) for v in decomp_random_weights)
            or any(v < 0.0 for v in decomp_random_weights)
            or sum(decomp_random_weights) <= 0.0
        ):
            raise ValueError(f"{name}: decomposed_GR_random_weights must be finite non-negative values with positive sum")
        decomp_tail_df = float(z_shift_cfg.get("decomposed_GR_tail_df", 0.0))
        if not math.isfinite(decomp_tail_df) or decomp_tail_df <= 0.0:
            raise ValueError(f"{name}: decomposed_GR_tail_df must be finite and positive, got {decomp_tail_df}")
        decomp_base_std = float(z_shift_cfg.get("decomposed_GR_tvt_base_std", 0.0))
        if not math.isfinite(decomp_base_std) or decomp_base_std < 0.0:
            raise ValueError(f"{name}: decomposed_GR_tvt_base_std must be finite and non-negative, got {decomp_base_std}")
        decomp_tvt_windows = tuple(int(v) for v in z_shift_cfg.get("decomposed_GR_tvt_base_windows", ()))
        if len(decomp_tvt_windows) != 3 or any(v <= 0 for v in decomp_tvt_windows):
            raise ValueError(
                f"{name}: decomposed_GR_tvt_base_windows must contain three positive integers, got {decomp_tvt_windows!r}"
            )
        decomp_tvt_weights = tuple(float(v) for v in z_shift_cfg.get("decomposed_GR_tvt_base_weights", ()))
        if len(decomp_tvt_weights) != 3:
            raise ValueError(f"{name}: decomposed_GR_tvt_base_weights must contain three values")
        if (
            any(not math.isfinite(v) for v in decomp_tvt_weights)
            or any(v < 0.0 for v in decomp_tvt_weights)
            or sum(decomp_tvt_weights) <= 0.0
        ):
            raise ValueError(f"{name}: decomposed_GR_tvt_base_weights must be finite non-negative values with positive sum")
        decomp_event_max = int(z_shift_cfg.get("decomposed_GR_tvt_event_max", 0))
        if decomp_event_max < 0:
            raise ValueError(f"{name}: decomposed_GR_tvt_event_max must be non-negative, got {decomp_event_max}")
        decomp_event_min_len = int(z_shift_cfg.get("decomposed_GR_tvt_event_min_len", 0))
        decomp_event_max_len = int(z_shift_cfg.get("decomposed_GR_tvt_event_max_len", 0))
        if decomp_event_min_len <= 0 or decomp_event_max_len < decomp_event_min_len:
            raise ValueError(
                f"{name}: decomposed_GR_tvt_event length bounds are invalid: "
                f"{decomp_event_min_len}, {decomp_event_max_len}"
            )
        for key in (
            "decomposed_GR_tvt_event_std",
            "decomposed_GR_target_rmse",
            "decomposed_GR_offset_std",
            "decomposed_GR_clip_abs",
        ):
            value = float(z_shift_cfg.get(key, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}: {key} must be finite and non-negative, got {value}")
        decomp_event_smooth = int(z_shift_cfg.get("decomposed_GR_tvt_event_smooth", 1))
        if decomp_event_smooth <= 0:
            raise ValueError(f"{name}: decomposed_GR_tvt_event_smooth must be positive, got {decomp_event_smooth}")
        decomp_ranges = tuple(z_shift_cfg.get("decomposed_GR_std_ranges", ()))
        decomp_range_weights = tuple(float(v) for v in z_shift_cfg.get("decomposed_GR_std_weights", ()))
        if len(decomp_ranges) != len(decomp_range_weights):
            raise ValueError(
                f"{name}: decomposed_GR_std_ranges and decomposed_GR_std_weights must have equal length"
            )
        if len(decomp_ranges) > 0:
            for value_range in decomp_ranges:
                if len(value_range) != 2:
                    raise ValueError(f"{name}: decomposed_GR_std_ranges entries must have length 2, got {value_range!r}")
                low, high = float(value_range[0]), float(value_range[1])
                if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or low > high:
                    raise ValueError(f"{name}: invalid decomposed_GR_std range {value_range!r}")
            if (
                any(not math.isfinite(v) for v in decomp_range_weights)
                or any(v < 0.0 for v in decomp_range_weights)
                or sum(decomp_range_weights) <= 0.0
            ):
                raise ValueError(
                    f"{name}: decomposed_GR_std_weights must be finite non-negative values with positive sum"
                )

        z_nan_mode = cfg.sim_cfg["z_shift"].get("nan_mode")
        if isinstance(z_nan_mode, str):
            z_nan_weights = {z_nan_mode: 1.0}
        elif isinstance(z_nan_mode, dict):
            z_nan_weights = {str(mode): float(weight) for mode, weight in z_nan_mode.items()}
        else:
            raise TypeError(
                f"{name}: z_shift nan_mode must be a string or a dict of mode probabilities, "
                f"got {type(z_nan_mode).__name__}"
            )
        unknown_z_nan = sorted(set(z_nan_weights) - valid_z_nan_modes)
        if unknown_z_nan:
            raise ValueError(f"{name}: unknown z_shift nan_mode={unknown_z_nan!r}")
        nonfinite_z_nan = {
            mode: weight for mode, weight in z_nan_weights.items() if not math.isfinite(weight)
        }
        if nonfinite_z_nan:
            raise ValueError(f"{name}: z_shift nan_mode probabilities must be finite: {nonfinite_z_nan}")
        negative_z_nan = {mode: weight for mode, weight in z_nan_weights.items() if weight < 0.0}
        if negative_z_nan:
            raise ValueError(f"{name}: z_shift nan_mode probabilities must be non-negative: {negative_z_nan}")
        if not any(weight > 0.0 for weight in z_nan_weights.values()):
            raise ValueError(f"{name}: z_shift nan_mode must give positive probability to at least one mode")
        z_apply_prob = float(cfg.sim_cfg["z_shift"].get("apply_prob", 0.0))
        if not 0.0 <= z_apply_prob <= 1.0:
            raise ValueError(f"{name}: z_shift apply_prob must be in [0, 1], got {z_apply_prob}")
        z_shift_range = float(cfg.sim_cfg["z_shift"].get("z_shift_range", 0.0))
        if not math.isfinite(z_shift_range) or z_shift_range < 0.0:
            raise ValueError(f"{name}: z_shift_range must be finite and non-negative, got {z_shift_range}")
        drift_ratio = float(cfg.sim_cfg["z_shift"].get("drift_ratio", 0.0))
        if not math.isfinite(drift_ratio) or not 0.0 <= drift_ratio <= 1.0:
            raise ValueError(f"{name}: z_shift drift_ratio must be finite and in [0, 1], got {drift_ratio}")
        fault_ratio = float(cfg.sim_cfg["z_shift"].get("fault_ratio", 0.0))
        if not 0.0 <= fault_ratio <= 1.0:
            raise ValueError(f"{name}: z_shift fault_ratio must be in [0, 1], got {fault_ratio}")
        fault_max_num = int(cfg.sim_cfg["z_shift"].get("fault_max_num", 0))
        if fault_max_num < 0:
            raise ValueError(f"{name}: z_shift fault_max_num must be non-negative, got {fault_max_num}")
        fault_std = float(cfg.sim_cfg["z_shift"].get("fault_std", 0.0))
        if fault_std < 0.0:
            raise ValueError(f"{name}: z_shift fault_std must be non-negative, got {fault_std}")

        typewell_mirror_cfg = cfg.sim_cfg.get("typewell_mirror", {})
        typewell_mirror_apply_prob = float(typewell_mirror_cfg.get("apply_prob", 0.0))
        if not 0.0 <= typewell_mirror_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: typewell_mirror apply_prob must be in [0, 1], "
                f"got {typewell_mirror_apply_prob}"
            )
        if not isinstance(typewell_mirror_cfg.get("skip_out_of_typewell", True), bool):
            raise ValueError(f"{name}: typewell_mirror skip_out_of_typewell must be boolean")

        switch_typewell_cfg = cfg.sim_cfg.get("switch_typewell", {})
        switch_typewell_apply_prob = float(switch_typewell_cfg.get("apply_prob", 0.0))
        if not 0.0 <= switch_typewell_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: switch_typewell apply_prob must be in [0, 1], "
                f"got {switch_typewell_apply_prob}"
            )
        switch_typewell_noise_mode = switch_typewell_cfg.get("noise_mode", "copy")
        if switch_typewell_noise_mode not in {"copy", "TVT_bias"}:
            raise ValueError(f"{name}: unknown switch_typewell noise_mode={switch_typewell_noise_mode!r}")
        if not isinstance(switch_typewell_cfg.get("same_nan", True), bool):
            raise ValueError(f"{name}: switch_typewell same_nan must be boolean")

        unet_arch = model_cfg.get("unet_arch")
        if unet_arch not in valid_unet_arches:
            raise ValueError(f"{name}: unknown unet_arch={unet_arch!r}")

        unet_mults = tuple(model_cfg.get("unet_mults", ()))
        if len(unet_mults) == 0 or any(int(v) <= 0 for v in unet_mults):
            raise ValueError(f"{name}: invalid unet_mults={unet_mults!r}")

        unet_stacks = int(model_cfg.get("unet_stacks", 1))
        if unet_stacks <= 0:
            raise ValueError(f"{name}: unet_stacks must be positive")
        if unet_arch == "convnext_small" and unet_stacks != 1:
            raise ValueError(f"{name}: convnext_small U-Net wrapper requires unet_stacks=1")

        unet_stem_stride_raw = model_cfg.get("unet_stem_stride", (1, 1))
        if isinstance(unet_stem_stride_raw, int):
            unet_stem_stride = (unet_stem_stride_raw, unet_stem_stride_raw)
        else:
            unet_stem_stride = tuple(unet_stem_stride_raw)
        if len(unet_stem_stride) != 2 or any(int(v) <= 0 for v in unet_stem_stride):
            raise ValueError(f"{name}: unet_stem_stride must contain two positive integers, got {unet_stem_stride!r}")

        unet_stem_down_mode = str(model_cfg.get("unet_stem_down_mode", "pool"))
        if unet_stem_down_mode not in {"pool", "conv"}:
            raise ValueError(
                f"{name}: unknown unet_stem_down_mode={unet_stem_down_mode!r}; "
                "expected 'pool' or 'conv'"
            )

        resblock_act = str(model_cfg.get("resblock_act", "silu"))
        if resblock_act not in valid_resblock_acts:
            raise ValueError(
                f"{name}: resblock_act must be one of {sorted(valid_resblock_acts)}, "
                f"got {resblock_act!r}"
            )
        resblock_norm = str(model_cfg.get("resblock_norm", "BN"))
        if resblock_norm not in valid_resblock_norms:
            raise ValueError(
                f"{name}: resblock_norm must be one of {sorted(valid_resblock_norms)}, "
                f"got {resblock_norm!r}"
            )
        convnext_norm_replace = model_cfg.get("unet_convnext_norm_replace", "BN")
        if (
            convnext_norm_replace is not None
            and convnext_norm_replace not in valid_convnext_norm_replacements
        ):
            raise ValueError(
                f"{name}: unet_convnext_norm_replace must be one of "
                f"{sorted(valid_convnext_norm_replacements)} or None, "
                f"got {convnext_norm_replace!r}"
            )

        unet_dropout = float(model_cfg.get("unet_dropout", 0.0))
        if not 0.0 <= unet_dropout <= 1.0:
            raise ValueError(f"{name}: unet_dropout must be in [0, 1], got {unet_dropout}")
        unet_channel_mask_prob = float(model_cfg.get("unet_channel_mask_prob", 0.0))
        if not 0.0 <= unet_channel_mask_prob <= 1.0:
            raise ValueError(
                f"{name}: unet_channel_mask_prob must be in [0, 1], got {unet_channel_mask_prob}"
            )
        unet_channel_mask_all_prob = float(model_cfg.get("unet_channel_mask_all_prob", 0.0))
        if not 0.0 <= unet_channel_mask_all_prob <= 1.0:
            raise ValueError(
                f"{name}: unet_channel_mask_all_prob must be in [0, 1], got {unet_channel_mask_all_prob}"
            )

        gr_noise_cfg = cfg.aug_cfg.get("GR_noise_shift", {})
        gr_noise_mode = gr_noise_cfg.get("noise_mode")
        if gr_noise_mode not in {"copy", "TVT_bias", "global_pool"}:
            raise ValueError(f"{name}: unknown GR_noise_shift noise_mode={gr_noise_mode!r}")
        gr_noise_apply_prob = float(gr_noise_cfg.get("apply_prob", 0.0))
        if not 0.0 <= gr_noise_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: GR_noise_shift apply_prob must be in [0, 1], got {gr_noise_apply_prob}"
            )

        horizontal_gr_blur_cfg = cfg.aug_cfg.get("horizontal_gr_blur", {})
        horizontal_gr_blur_apply_prob = float(horizontal_gr_blur_cfg.get("apply_prob", 0.0))
        if not 0.0 <= horizontal_gr_blur_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: horizontal_gr_blur apply_prob must be in [0, 1], "
                f"got {horizontal_gr_blur_apply_prob}"
            )
        flat_region_max_ratio = float(horizontal_gr_blur_cfg.get("flat_region_max_ratio", 0.0))
        if not math.isfinite(flat_region_max_ratio) or not 0.0 <= flat_region_max_ratio <= 1.0:
            raise ValueError(
                f"{name}: horizontal_gr_blur flat_region_max_ratio must be finite and in [0, 1], "
                f"got {flat_region_max_ratio}"
            )

        reverse_path_cfg = cfg.aug_cfg.get("reverse_path", {})
        reverse_path_apply_prob = float(reverse_path_cfg.get("apply_prob", 0.0))
        if not 0.0 <= reverse_path_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: reverse_path apply_prob must be in [0, 1], got {reverse_path_apply_prob}"
            )
        reverse_head_bound = float(reverse_path_cfg.get("head_TVT_bound", 0.0))
        if not math.isfinite(reverse_head_bound) or reverse_head_bound < 0.0:
            raise ValueError(f"{name}: reverse_path head_TVT_bound must be finite and non-negative")
        reverse_sim_head_range = int(reverse_path_cfg.get("sim_head_range", 0))
        if reverse_sim_head_range <= 0:
            raise ValueError(f"{name}: reverse_path sim_head_range must be positive")

        md_streching_cfg = cfg.aug_cfg.get("MD_streching", {})
        md_streching_apply_prob = float(md_streching_cfg.get("apply_prob", 0.0))
        if not 0.0 <= md_streching_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: MD_streching apply_prob must be in [0, 1], "
                f"got {md_streching_apply_prob}"
            )

        tail_cut_cfg = cfg.aug_cfg.get("tail_cut", {})
        tail_cut_apply_prob = float(tail_cut_cfg.get("apply_prob", 0.0))
        if not 0.0 <= tail_cut_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: tail_cut apply_prob must be in [0, 1], got {tail_cut_apply_prob}"
            )
        tail_cut_max_ratio = float(tail_cut_cfg.get("max_cut_ratio", 0.5))
        if not math.isfinite(tail_cut_max_ratio) or not 0.0 <= tail_cut_max_ratio <= 1.0:
            raise ValueError(
                f"{name}: tail_cut max_cut_ratio must be finite and in [0, 1], "
                f"got {tail_cut_max_ratio}"
            )

        seq_mask_cfg = cfg.aug_cfg.get("seq_mask", {})
        seq_mask_apply_prob = float(seq_mask_cfg.get("apply_prob", 0.0))
        if not 0.0 <= seq_mask_apply_prob <= 1.0:
            raise ValueError(f"{name}: seq_mask apply_prob must be in [0, 1], got {seq_mask_apply_prob}")
        seq_mask_ratio = float(seq_mask_cfg.get("mask_ratio", 0.0))
        if not 0.0 <= seq_mask_ratio <= 1.0:
            raise ValueError(f"{name}: seq_mask mask_ratio must be in [0, 1], got {seq_mask_ratio}")

        channel_mask_cfg = cfg.aug_cfg.get("channel_mask_2d", {})
        channel_mask_apply_prob = float(channel_mask_cfg.get("apply_prob", 0.0))
        if not 0.0 <= channel_mask_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: channel_mask_2d apply_prob must be in [0, 1], got {channel_mask_apply_prob}"
            )
        channel_mask_prob = float(channel_mask_cfg.get("mask_prob", 0.0))
        if not 0.0 <= channel_mask_prob <= 1.0:
            raise ValueError(
                f"{name}: channel_mask_2d mask_prob must be in [0, 1], got {channel_mask_prob}"
            )
        channel_pf_mask_prob = float(channel_mask_cfg.get("PF_mask_prob", 0.0))
        if not 0.0 <= channel_pf_mask_prob <= 1.0:
            raise ValueError(
                f"{name}: channel_mask_2d PF_mask_prob must be in [0, 1], got {channel_pf_mask_prob}"
            )
        pf_rotate_cfg = cfg.aug_cfg.get("PF_rotate_shift", {})
        pf_rotate_apply_prob = float(pf_rotate_cfg.get("apply_prob", 0.0))
        if not 0.0 <= pf_rotate_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: PF_rotate_shift apply_prob must be in [0, 1], got {pf_rotate_apply_prob}"
            )
        pf_rotate_ratio_jitter = float(pf_rotate_cfg.get("ratio_jitter", 0.25))
        if pf_rotate_ratio_jitter < 0.0:
            raise ValueError(
                f"{name}: PF_rotate_shift ratio_jitter must be non-negative, "
                f"got {pf_rotate_ratio_jitter}"
            )
        pf_rotate_max_abs_shift = float(pf_rotate_cfg.get("max_abs_shift", 20.0))
        if pf_rotate_max_abs_shift < 0.0:
            raise ValueError(
                f"{name}: PF_rotate_shift max_abs_shift must be non-negative, "
                f"got {pf_rotate_max_abs_shift}"
            )
        pf_rotate_k_clip = float(pf_rotate_cfg.get("k_clip", 0.0))
        if pf_rotate_k_clip < 0.0:
            raise ValueError(f"{name}: PF_rotate_shift k_clip must be non-negative, got {pf_rotate_k_clip}")
        pf_rotate_min_fit_denom = float(pf_rotate_cfg.get("min_fit_denom", 1e-6))
        if pf_rotate_min_fit_denom < 0.0:
            raise ValueError(
                f"{name}: PF_rotate_shift min_fit_denom must be non-negative, "
                f"got {pf_rotate_min_fit_denom}"
            )
        pf_rotate_shift_mode = str(pf_rotate_cfg.get("shift_mode", "absolute"))
        if pf_rotate_shift_mode not in {"absolute", "delta"}:
            raise ValueError(
                f"{name}: unknown PF_rotate_shift shift_mode={pf_rotate_shift_mode!r}; "
                "expected 'absolute' or 'delta'"
            )
        pf_rotate_center_mode = str(pf_rotate_cfg.get("center_mode", "first_valid"))
        if pf_rotate_center_mode not in {"first_valid", "zero"}:
            raise ValueError(
                f"{name}: unknown PF_rotate_shift center_mode={pf_rotate_center_mode!r}; "
                "expected 'first_valid' or 'zero'"
            )

        prior_corrupt_cfg = cfg.aug_cfg.get("prior_corrupt_2d", {})
        for prob_key in (
            "apply_prob",
            "geo_apply_prob",
            "pf_apply_prob",
            "geo_jump_prob",
            "pf_row_dropout_prob",
            "pf_wrong_mode_prob",
            "geo_pf_disagreement_prob",
        ):
            prob_value = float(prior_corrupt_cfg.get(prob_key, 0.0))
            if not 0.0 <= prob_value <= 1.0:
                raise ValueError(f"{name}: prior_corrupt_2d {prob_key} must be in [0, 1], got {prob_value}")
        for nonneg_key in (
            "geo_bias_std",
            "geo_ramp_std",
            "geo_smooth_noise_std",
            "geo_jump_std",
            "geo_max_abs_shift",
            "pf_path_noise_std",
            "pf_path_max_abs_shift",
        ):
            value = float(prior_corrupt_cfg.get(nonneg_key, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}: prior_corrupt_2d {nonneg_key} must be finite and non-negative, got {value}")
        for choice_key in ("geo_smooth_window_choices", "pf_path_smooth_window_choices"):
            choices = tuple(int(v) for v in prior_corrupt_cfg.get(choice_key, ()))
            if len(choices) == 0 or any(v <= 0 for v in choices):
                raise ValueError(f"{name}: prior_corrupt_2d {choice_key} must contain positive integers")
        for range_key in ("pf_temperature_range", "pf_wrong_mode_shift_range"):
            value_range = prior_corrupt_cfg.get(range_key, (1.0, 1.0))
            if len(value_range) != 2:
                raise ValueError(f"{name}: prior_corrupt_2d {range_key} must have length 2, got {value_range!r}")
            low, high = float(value_range[0]), float(value_range[1])
            if not math.isfinite(low) or not math.isfinite(high) or low < 0.0 or low > high:
                raise ValueError(f"{name}: invalid prior_corrupt_2d {range_key}={value_range!r}")
            if range_key == "pf_temperature_range" and low <= 0.0:
                raise ValueError(f"{name}: prior_corrupt_2d pf_temperature_range must be positive")
        corrupt_metric_channels = prior_corrupt_cfg.get("corrupt_metric_channels", False)
        if not isinstance(corrupt_metric_channels, bool):
            raise ValueError(f"{name}: prior_corrupt_2d corrupt_metric_channels must be boolean")
        if corrupt_metric_channels:
            raise ValueError(f"{name}: prior_corrupt_2d corrupt_metric_channels=True is not implemented")

        typewell_jitter_cfg = cfg.aug_cfg.get("typewell_gr_jitter", {})
        typewell_jitter_apply_prob = float(typewell_jitter_cfg.get("apply_prob", 0.0))
        if not 0.0 <= typewell_jitter_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: typewell_gr_jitter apply_prob must be in [0, 1], got {typewell_jitter_apply_prob}"
            )
        mode_weights = typewell_jitter_cfg.get("mode_weights", {"coherent_residual_copy": 1.0})
        if isinstance(mode_weights, str):
            mode_weights = {mode_weights: 1.0}
        if not isinstance(mode_weights, dict):
            raise TypeError(f"{name}: typewell_gr_jitter mode_weights must be a dict or string")
        mode_weights = {str(mode): float(weight) for mode, weight in mode_weights.items()}
        unknown_modes = sorted(set(mode_weights) - valid_typewell_gr_jitter_modes)
        if unknown_modes:
            raise ValueError(f"{name}: unknown typewell_gr_jitter modes={unknown_modes}")
        if any(not math.isfinite(weight) for weight in mode_weights.values()):
            raise ValueError(f"{name}: typewell_gr_jitter mode_weights must be finite")
        if any(weight < 0.0 for weight in mode_weights.values()):
            raise ValueError(f"{name}: typewell_gr_jitter mode_weights must be non-negative")
        if sum(weight for weight in mode_weights.values() if weight > 0.0) <= 0.0:
            raise ValueError(f"{name}: typewell_gr_jitter mode_weights must give positive probability")
        for nonneg_key in ("offset_std", "gain_std", "drift_std"):
            value = float(typewell_jitter_cfg.get(nonneg_key, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name}: typewell_gr_jitter {nonneg_key} must be finite and non-negative")
        gain_clip = typewell_jitter_cfg.get("gain_clip", (1.0, 1.0))
        if len(gain_clip) != 2:
            raise ValueError(f"{name}: typewell_gr_jitter gain_clip must have length 2")
        gain_low, gain_high = float(gain_clip[0]), float(gain_clip[1])
        if not math.isfinite(gain_low) or not math.isfinite(gain_high) or gain_low <= 0.0 or gain_low > gain_high:
            raise ValueError(f"{name}: invalid typewell_gr_jitter gain_clip={gain_clip!r}")
        drift_choices = tuple(float(v) for v in typewell_jitter_cfg.get("drift_window_ft_choices", ()))
        if len(drift_choices) == 0 or any((not math.isfinite(v)) or v <= 0.0 for v in drift_choices):
            raise ValueError(f"{name}: typewell_gr_jitter drift_window_ft_choices must contain positive values")
        smooth_prob = float(typewell_jitter_cfg.get("smooth_prob", 0.0))
        if not 0.0 <= smooth_prob <= 1.0:
            raise ValueError(f"{name}: typewell_gr_jitter smooth_prob must be in [0, 1], got {smooth_prob}")
        smooth_windows = tuple(int(v) for v in typewell_jitter_cfg.get("smooth_window_choices", ()))
        if len(smooth_windows) == 0 or any(v <= 0 for v in smooth_windows):
            raise ValueError(f"{name}: typewell_gr_jitter smooth_window_choices must contain positive integers")
        smooth_blend = typewell_jitter_cfg.get("smooth_blend_range", (0.0, 0.0))
        if len(smooth_blend) != 2:
            raise ValueError(f"{name}: typewell_gr_jitter smooth_blend_range must have length 2")
        blend_low, blend_high = float(smooth_blend[0]), float(smooth_blend[1])
        if not math.isfinite(blend_low) or not math.isfinite(blend_high) or blend_low < 0.0 or blend_high > 1.0 or blend_low > blend_high:
            raise ValueError(f"{name}: invalid typewell_gr_jitter smooth_blend_range={smooth_blend!r}")

        gr_trf_cfg = cfg.aug_cfg.get("GR_trf", {})
        gr_trf_apply_prob = float(gr_trf_cfg.get("apply_prob", 0.0))
        if not 0.0 <= gr_trf_apply_prob <= 1.0:
            raise ValueError(f"{name}: GR_trf apply_prob must be in [0, 1], got {gr_trf_apply_prob}")
        gr_trf_p_std = float(gr_trf_cfg.get("p_std", 0.0))
        if not math.isfinite(gr_trf_p_std) or gr_trf_p_std < 0.0:
            raise ValueError(f"{name}: GR_trf p_std must be finite and non-negative, got {gr_trf_p_std}")
        gr_trf_a_std = float(gr_trf_cfg.get("a_std", 0.0))
        if not math.isfinite(gr_trf_a_std) or gr_trf_a_std < 0.0:
            raise ValueError(f"{name}: GR_trf a_std must be finite and non-negative, got {gr_trf_a_std}")
        gr_trf_a_keep_mean = gr_trf_cfg.get("a_keep_mean", False)
        if not isinstance(gr_trf_a_keep_mean, bool):
            raise ValueError(f"{name}: GR_trf a_keep_mean must be boolean, got {gr_trf_a_keep_mean!r}")
        gr_trf_b_std = float(gr_trf_cfg.get("b_std", 0.0))
        if not math.isfinite(gr_trf_b_std) or gr_trf_b_std < 0.0:
            raise ValueError(f"{name}: GR_trf b_std must be finite and non-negative, got {gr_trf_b_std}")
        gr_trf_ab_window = gr_trf_cfg.get("ab_window")
        if gr_trf_ab_window is not None:
            gr_trf_ab_window_value = float(gr_trf_ab_window)
            if (
                not math.isfinite(gr_trf_ab_window_value)
                or not gr_trf_ab_window_value.is_integer()
                or gr_trf_ab_window_value <= 0
            ):
                raise ValueError(
                    f"{name}: GR_trf ab_window must be None or a positive integer, "
                    f"got {gr_trf_ab_window!r}"
                )
        gr_trf_center_mirror_prob = float(gr_trf_cfg.get("center_mirror_prob", 0.0))
        if not 0.0 <= gr_trf_center_mirror_prob <= 1.0:
            raise ValueError(
                f"{name}: GR_trf center_mirror_prob must be in [0, 1], got {gr_trf_center_mirror_prob}"
            )
        gr_trf_smooth_apply_prob = float(gr_trf_cfg.get("smooth_apply_prob", 0.0))
        gr_trf_sharpen_apply_prob = float(gr_trf_cfg.get("sharpen_apply_prob", 0.0))
        for branch_name, branch_prob in (
            ("smooth", gr_trf_smooth_apply_prob),
            ("sharpen", gr_trf_sharpen_apply_prob),
        ):
            if not 0.0 <= branch_prob <= 1.0:
                raise ValueError(
                    f"{name}: GR_trf {branch_name}_apply_prob must be in [0, 1], got {branch_prob}"
                )
        if gr_trf_smooth_apply_prob + gr_trf_sharpen_apply_prob > 1.0:
            raise ValueError(
                f"{name}: GR_trf smooth_apply_prob + sharpen_apply_prob must be <= 1, got "
                f"{gr_trf_smooth_apply_prob + gr_trf_sharpen_apply_prob}"
            )
        gr_trf_smooth_window_value = float(gr_trf_cfg.get("smooth_window", 5))
        if (
            not math.isfinite(gr_trf_smooth_window_value)
            or not gr_trf_smooth_window_value.is_integer()
            or gr_trf_smooth_window_value <= 0
            or int(gr_trf_smooth_window_value) % 2 == 0
        ):
            raise ValueError(
                f"{name}: GR_trf smooth_window must be a positive odd integer, "
                f"got {gr_trf_smooth_window_value}"
            )
        gr_trf_sharpen_alpha_range = gr_trf_cfg.get("sharpen_alpha_range", (0.0, 0.3))
        if len(gr_trf_sharpen_alpha_range) != 2:
            raise ValueError(f"{name}: GR_trf sharpen_alpha_range must have length 2")
        sharpen_alpha_low, sharpen_alpha_high = map(float, gr_trf_sharpen_alpha_range)
        if (
            not math.isfinite(sharpen_alpha_low)
            or not math.isfinite(sharpen_alpha_high)
            or sharpen_alpha_low < 0.0
            or sharpen_alpha_low > sharpen_alpha_high
        ):
            raise ValueError(
                f"{name}: invalid GR_trf sharpen_alpha_range={gr_trf_sharpen_alpha_range!r}"
            )

        typewell_drift_cfg = cfg.aug_cfg.get("typewell_drift", {})
        typewell_drift_apply_prob = float(typewell_drift_cfg.get("apply_prob", 0.0))
        if not 0.0 <= typewell_drift_apply_prob <= 1.0:
            raise ValueError(
                f"{name}: typewell_drift apply_prob must be in [0, 1], got {typewell_drift_apply_prob}"
            )
        typewell_drift_shift_std = float(typewell_drift_cfg.get("shift_std", 1.0))
        if not math.isfinite(typewell_drift_shift_std) or typewell_drift_shift_std < 0.0:
            raise ValueError(
                f"{name}: typewell_drift shift_std must be finite and non-negative, "
                f"got {typewell_drift_shift_std}"
            )

        hflip_cfg = cfg.aug_cfg.get("hflip", {})
        hflip_apply_prob = float(hflip_cfg.get("apply_prob", 0.0))
        if not 0.0 <= hflip_apply_prob <= 1.0:
            raise ValueError(f"{name}: hflip apply_prob must be in [0, 1], got {hflip_apply_prob}")
        mixup_prob = float(getattr(cfg, "mixup_prob", 0.0))
        if not 0.0 <= mixup_prob <= 1.0:
            raise ValueError(f"{name}: mixup_prob must be in [0, 1], got {mixup_prob}")
        mixup_alpha = float(getattr(cfg, "mixup_alpha", 1.0))
        if not math.isfinite(mixup_alpha) or mixup_alpha <= 0.0:
            raise ValueError(f"{name}: mixup_alpha must be positive and finite, got {mixup_alpha}")
        cutmix_prob = float(getattr(cfg, "cutmix_prob", 0.0))
        if not 0.0 <= cutmix_prob <= 1.0:
            raise ValueError(f"{name}: cutmix_prob must be in [0, 1], got {cutmix_prob}")
        cutmix_alpha = float(getattr(cfg, "cutmix_alpha", 1.0))
        if not math.isfinite(cutmix_alpha) or cutmix_alpha <= 0.0:
            raise ValueError(f"{name}: cutmix_alpha must be positive and finite, got {cutmix_alpha}")

        geo_prior_cfg = getattr(cfg, "geo_prior_cfg", None)
        if isinstance(geo_prior_cfg, dict):
            geo_prior_cfg = GeoPriorConfig(**geo_prior_cfg)
        if not isinstance(geo_prior_cfg, GeoPriorConfig):
            raise ValueError(f"{name}: geo_prior_cfg must be GeoPriorConfig, got {type(geo_prior_cfg)!r}")
        if geo_prior_cfg.method not in {"idw_dS_xy", "idw_dS_xy_wls"}:
            raise ValueError(f"{name}: unknown geo_prior_cfg.method={geo_prior_cfg.method!r}")
        for field_name in (
            "neighbor_wells",
            "point_neighbors",
            "train_bin_size",
            "query_bin_size",
            "prefix_bin_size",
            "max_train_points",
        ):
            value = int(getattr(geo_prior_cfg, field_name))
            if value <= 0:
                raise ValueError(f"{name}: geo_prior_cfg.{field_name} must be positive, got {value}")
        if int(geo_prior_cfg.prefix_support_rows) < 0:
            raise ValueError(
                f"{name}: geo_prior_cfg.prefix_support_rows must be non-negative, "
                f"got {geo_prior_cfg.prefix_support_rows}"
            )
        if geo_prior_cfg.support_region != "suffix":
            raise ValueError(
                f"{name}: geo_prior_cfg.support_region must be 'suffix', "
                f"got {geo_prior_cfg.support_region!r}"
            )
        if float(geo_prior_cfg.idw_power) <= 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.idw_power must be positive")
        if geo_prior_cfg.point_neighbor_metric not in {"euclidean", "anisotropic"}:
            raise ValueError(
                f"{name}: unknown geo_prior_cfg.point_neighbor_metric="
                f"{geo_prior_cfg.point_neighbor_metric!r}"
            )
        if geo_prior_cfg.query_upsample_mode not in {"repeat", "linear"}:
            raise ValueError(
                f"{name}: unknown geo_prior_cfg.query_upsample_mode="
                f"{geo_prior_cfg.query_upsample_mode!r}"
            )
        if not math.isfinite(float(geo_prior_cfg.anisotropy_angle_deg)):
            raise ValueError(f"{name}: geo_prior_cfg.anisotropy_angle_deg must be finite")
        if not math.isfinite(float(geo_prior_cfg.anisotropy_ratio)) or float(
            geo_prior_cfg.anisotropy_ratio
        ) <= 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.anisotropy_ratio must be positive and finite")
        if float(geo_prior_cfg.clip_ds_xy) < 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.clip_ds_xy must be non-negative")
        if float(geo_prior_cfg.filter_ds_thr) < 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.filter_ds_thr must be non-negative")
        if not isinstance(geo_prior_cfg.wls_normalize_dxy, bool):
            raise ValueError(f"{name}: geo_prior_cfg.wls_normalize_dxy must be boolean")
        if float(geo_prior_cfg.wls_ridge_frac) < 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.wls_ridge_frac must be non-negative")
        if not 0.0 <= float(geo_prior_cfg.wls_min_eig_ratio) <= 1.0:
            raise ValueError(
                f"{name}: geo_prior_cfg.wls_min_eig_ratio must be in [0, 1], "
                f"got {geo_prior_cfg.wls_min_eig_ratio}"
            )
        if float(geo_prior_cfg.wls_max_grad) < 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.wls_max_grad must be non-negative")
        if geo_prior_cfg.wls_robust_mode not in {"none", "huber", "tukey"}:
            raise ValueError(
                f"{name}: unknown geo_prior_cfg.wls_robust_mode="
                f"{geo_prior_cfg.wls_robust_mode!r}"
            )
        if not math.isfinite(float(geo_prior_cfg.wls_robust_scale)) or float(
            geo_prior_cfg.wls_robust_scale
        ) < 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.wls_robust_scale must be finite and non-negative")
        if int(geo_prior_cfg.wls_robust_iters) < 0:
            raise ValueError(f"{name}: geo_prior_cfg.wls_robust_iters must be non-negative")
        if float(geo_prior_cfg.distance_eps) <= 0.0:
            raise ValueError(f"{name}: geo_prior_cfg.distance_eps must be positive")

        optimizer_cfg = getattr(cfg, "optimizer_cfg", {})
        required_optimizer_keys = {"optimizer", "lr", "min_lr", "weight_decay", "scheduler"}
        missing_optimizer_keys = sorted(required_optimizer_keys - set(optimizer_cfg))
        if missing_optimizer_keys:
            raise ValueError(f"{name}: optimizer_cfg missing keys: {missing_optimizer_keys}")
        optimizer_name = str(optimizer_cfg["optimizer"]).lower()
        if optimizer_name not in valid_optimizers:
            raise ValueError(
                f"{name}: unknown optimizer_cfg optimizer={optimizer_name!r}; "
                f"expected one of {sorted(valid_optimizers)}"
            )
        scheduler = str(optimizer_cfg["scheduler"]).lower()
        if scheduler not in valid_schedulers:
            raise ValueError(
                f"{name}: unknown optimizer_cfg scheduler={scheduler!r}; "
                f"expected one of {sorted(valid_schedulers)}"
            )
        lr = float(optimizer_cfg["lr"])
        min_lr = float(optimizer_cfg["min_lr"])
        weight_decay = float(optimizer_cfg["weight_decay"])
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(f"{name}: optimizer_cfg lr must be finite and positive, got {lr}")
        if not math.isfinite(min_lr) or min_lr < 0.0:
            raise ValueError(f"{name}: optimizer_cfg min_lr must be finite and non-negative, got {min_lr}")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError(
                f"{name}: optimizer_cfg weight_decay must be finite and non-negative, got {weight_decay}"
            )
        constant_ratio = float(optimizer_cfg.get("constant_ratio", 0.0))
        if not 0.0 <= constant_ratio < 1.0:
            raise ValueError(f"{name}: optimizer constant_ratio must be in [0, 1), got {constant_ratio}")
        if constant_ratio > 0.0 and scheduler not in {"cos", "cosine"}:
            raise ValueError(
                f"{name}: optimizer constant_ratio is only supported for cos scheduler, got {scheduler!r}"
            )

        loss_weights = getattr(cfg, "unet_loss_weights", {})
        unknown_loss_weights = sorted(set(loss_weights) - set(CFG.unet_loss_weights))
        if unknown_loss_weights:
            raise ValueError(f"{name}: unknown unet_loss_weights keys={unknown_loss_weights}")
        for loss_name, loss_weight in loss_weights.items():
            if float(loss_weight) < 0.0:
                raise ValueError(f"{name}: loss weight {loss_name!r} must be non-negative")

        regression_head_mode = str(model_cfg.get("regression_head_mode", "logits"))
        if regression_head_mode not in {"logits", "prob+offset", "pool_mlp"}:
            raise ValueError(
                f"{name}: unknown regression_head_mode={regression_head_mode!r}; "
                "expected 'logits', 'prob+offset', or 'pool_mlp'"
            )
        share_logits_head = bool(model_cfg.get("share_logits_head", True))
        if share_logits_head and regression_head_mode not in {"logits", "prob+offset"}:
            raise ValueError(
                f"{name}: share_logits_head=True requires regression_head_mode='logits' or 'prob+offset'"
            )
        offset_detach = model_cfg.get("offset_detach_logits_head_in_loss", True)
        if not isinstance(offset_detach, bool):
            raise ValueError(f"{name}: offset_detach_logits_head_in_loss must be boolean, got {offset_detach!r}")
        if float(loss_weights.get("offset", 0.0)) > 0.0 and regression_head_mode != "prob+offset":
            raise ValueError(
                f"{name}: unet_loss_weights['offset'] requires regression_head_mode='prob+offset'"
            )

        min_seq_weight = float(getattr(cfg, "min_seq_weight", 1.0))
        if not 0.0 <= min_seq_weight <= 1.0:
            raise ValueError(f"{name}: min_seq_weight must be in [0, 1], got {min_seq_weight}")
        down_weight = float(getattr(cfg, "model_down_weight", 0.0))
        if not 0.0 <= down_weight <= 1.0:
            raise ValueError(f"{name}: model_down_weight must be in [0, 1], got {down_weight}")
        down_weight_wells = getattr(cfg, "model_down_weight_wells", [])
        if isinstance(down_weight_wells, (str, bytes)):
            raise ValueError(f"{name}: model_down_weight_wells must be a sequence of well ids, not a string")
        for well_id in down_weight_wells:
            if not isinstance(well_id, str):
                raise ValueError(
                    f"{name}: model_down_weight_wells must contain well_id strings, got {type(well_id)!r}"
                )
        seq_noise_smooth = float(getattr(cfg, "seq_noise_smooth", 1.0))
        if seq_noise_smooth <= 0.0:
            raise ValueError(f"{name}: seq_noise_smooth must be positive, got {seq_noise_smooth}")
        seq_noise_power = float(getattr(cfg, "seq_noise_power", 1.0))
        if seq_noise_power < 0.0:
            raise ValueError(f"{name}: seq_noise_power must be non-negative, got {seq_noise_power}")
        epochs = int(getattr(cfg, "epochs", 0))
        if epochs <= 0:
            raise ValueError(f"{name}: epochs must be positive, got {epochs}")
        min_epochs = int(getattr(cfg, "min_epochs", 0))
        if min_epochs < 0:
            raise ValueError(f"{name}: min_epochs must be non-negative, got {min_epochs}")
        if min_epochs > epochs:
            raise ValueError(f"{name}: min_epochs must be <= epochs, got {min_epochs} > {epochs}")
        early_stopping_rounds = int(getattr(cfg, "early_stopping_rounds", 0))
        if early_stopping_rounds < 0:
            raise ValueError(
                f"{name}: early_stopping_rounds must be non-negative, got {early_stopping_rounds}"
            )
        for batch_attr in ("batch_size", "val_batch_size", "grad_accum_steps"):
            batch_value = int(getattr(cfg, batch_attr, 0))
            if batch_value <= 0:
                raise ValueError(f"{name}: {batch_attr} must be positive, got {batch_value}")

        finetune_epochs = getattr(cfg, "finetune_epochs", None)
        if finetune_epochs is not None and int(finetune_epochs) <= 0:
            raise ValueError(f"{name}: finetune_epochs must be positive when set")
        if float(getattr(cfg, "finetune_lr_reduce", 0.3)) <= 0.0:
            raise ValueError(f"{name}: finetune_lr_reduce must be positive")

    return cfgs


def make_seq_focused_ablation_cfgs():
    cfgs = []
    for apply_prob in (0.1, 0.2, 0.4):
        cfgs.append(
            (
                f"switch_typewell_apply_prob_{_fmt_float_name(apply_prob)}",
                make_sim_cfg(switch_typewell={"apply_prob": apply_prob}),
            )
        )
    for huber_delta in (1.5, 3.0):
        cfgs.append(
            (
                f"huber_delta_{_fmt_float_name(huber_delta)}",
                make_unet_cfg(huber_delta=huber_delta),
            )
        )
    for min_seq_weight in (0.8, 0.6, 0.4):
        cfgs.append(
            (
                f"min_seq_weight_{_fmt_float_name(min_seq_weight)}",
                make_unet_cfg(min_seq_weight=min_seq_weight),
            )
        )
    return cfgs


def make_seq_unet_ablation_cfgs():
    cfgs = []
    cfgs.extend(
        [
            (
                "unet_emb_dim_64",
                make_unet_cfg(
                    model_overrides={
                        "unet_arch": "unet",
                        "unet_emb_dim": 64,
                    },
                ),
            ),
            (
                "unet_mults_1_2_4_8_16",
                make_unet_cfg(
                    model_overrides={
                        "unet_arch": "unet",
                        "unet_mults": (1, 2, 4, 8, 16),
                    },
                ),
            ),
        ]
    )
    for name, alignment_weight in (
        ("unet_loss_weights_alignment_0p05x4over3", 0.05 * 4.0 / 3.0),
        ("unet_loss_weights_alignment_0p05x1over3", 0.05 * 1.0 / 3.0),
    ):
        cfg = make_unet_cfg()
        cfg.unet_loss_weights["alignment"] = float(alignment_weight)
        cfgs.append((name, cfg.refresh()))
    return cfgs


def make_seq_requested_ablation_cfgs():
    offset_cfg = make_unet_cfg(model_overrides={"regression_head_mode": "prob+offset"})
    offset_cfg.unet_loss_weights["offset"] = 0.05 / 3
    offset_cfg = offset_cfg.refresh()

    cfgs = [
        (
            "exp_1p1_downsample_16_unet_emb_dim_16_stem_stride_4x4",
            make_unet_cfg(
                downsample=16,
                model_overrides={
                    "unet_emb_dim": 16,
                    "unet_stem_stride": (4, 4),
                },
            ),
        ),
        (
            "exp_2p1_epochs_150",
            make_unet_cfg(epochs=150, min_epochs=150),
        ),
        (
            "exp_3p1_typewell_calibration_blend_weight_0p85",
            make_unet_cfg(typewell_calibration_blend_weight=0.85),
        ),
        (
            "exp_4p1_prob_offset_offset_loss_0p05over3",
            offset_cfg,
        ),
        (
            "exp_5p1_weight_decay_1em1",
            make_optimizer_cfg(shared={"weight_decay": 1e-1}),
        ),
        (
            "exp_6p1_ln_onecycle_lr_5em4_epochs_250",
            make_optimizer_cfg(
                epochs=250,
                grad_accum_steps=2,
                amp_dtype="float16",
                model_overrides={"unet_convnext_norm_replace": "LN"},
                shared={
                    "lr": 5e-4,
                    "min_lr": 1e-6,
                    "scheduler": "onecycle",
                },
            ),
        ),
        (
            "exp_8p1_timm_convnext_small_fb_in22k_ft_in1k_384",
            make_unet_cfg(
                model_overrides={
                    "unet_timm_model_name": "timm/convnext_small.fb_in22k_ft_in1k_384",
                },
            ),
        ),
    ]
    return _validate_seq_train_cfgs(cfgs)


# Sequential training/submission experiment registry consumed by:
#   python src/seq_NN_main.py --run-mode seq_train --id <run_id>
#   python src/seq_NN_main.py --run-mode seq_full_train --id <run_id>
# In seq_train, each config is trained independently and writes only its own
# train log in the output dir. In seq_full_train, each config writes a full
# common-training artifact set under output_dir/<cfg_name>. Names are
# intentionally filename-safe because they become log filenames or directories.
SEQ_TRAIN_CFGS = make_seq_requested_ablation_cfgs()
