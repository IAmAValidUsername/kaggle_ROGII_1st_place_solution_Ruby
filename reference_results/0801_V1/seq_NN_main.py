import argparse
import pickle
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from seq_NN_cfg import (
    CFG,
    MODEL_MODES,
    SEQ_TARGET_MODES,
    SEQ_TRAIN_CFGS,
    TRF_UNET_MODE,
    TWO_STAGE_UNET_MODE,
    TYPEWELL_GR_GRID_TYPES,
    UNET_MODE,
)
from seq_NN_dataset import discover_well_ids
from seq_NN_data_prep import uses_pf_heatmap_channels
from seq_NN_trf_backbones import TRF_UNET_BACKBONES
import seq_NN_train


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def log_section(section_name):
    print(f"\n######## {section_name} #######", flush=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the standalone sequence NN pipeline.")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output-dir", type=Path, help="Exact output directory to create.")
    output_group.add_argument("--id", dest="run_id", help="Create outputs/<id>_seq_nn under the default output root.")
    parser.add_argument(
        "--run-mode",
        choices=["common", "seq_train", "seq_full_train"],
        default="common",
        help=(
            "common trains, saves artifacts, and predicts test; "
            "seq_train runs train-only cfg sweeps with logs only; "
            "seq_full_train runs cfg sweeps with one full common-training output dir per cfg."
        ),
    )
    parser.add_argument(
        "--submit-mode",
        action="store_true",
        help="Run inference-only submission mode from a previous training output directory.",
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        help="Previous sequence-NN training output directory containing saved models and optional cfg.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Dataset root containing train/ and test/ directories.",
    )
    parser.add_argument("--f", type=int, choices=[0, 1], help="Set to 1 to allow writing into an existing output dir.")
    parser.add_argument("--device", help="Torch device, default cfg.device.")
    parser.add_argument("--well-limit", type=int, help="Use only the first N wells for quick smoke runs.")
    parser.add_argument("--epochs", type=int, help="Override sequence NN epochs.")
    parser.add_argument("--log-freq", type=int, help="Print tqdm and epoch metrics every N epochs.")
    parser.add_argument("--fold-count", type=int, help="Override K-fold count.")
    parser.add_argument(
        "--cv-split-mode",
        choices=["kfold", "geo_skfold"],
        help="Override well-level CV split mode.",
    )
    parser.add_argument(
        "--cv-geo-map-path",
        type=Path,
        help="CSV with well_id,horizontal_avg_X,horizontal_avg_Y for geo_skfold.",
    )
    parser.add_argument("--cv-kmeans-clusters", type=int, help="Override geo_skfold KMeans cluster count.")
    parser.add_argument("--cv-kmeans-seed", type=int, help="Override geo_skfold KMeans random_state.")
    parser.add_argument("--batch-size", type=int, help="Override training batch size.")
    parser.add_argument("--val-batch-size", type=int, help="Override validation and inference batch size.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader worker count.")
    parser.add_argument(
        "--persistent-workers",
        type=int,
        choices=[0, 1],
        help="Keep dataloader workers alive between loader iterations.",
    )
    parser.add_argument("--grad-accum-steps", type=int, help="Accumulate gradients over N training batches.")
    parser.add_argument("--train-drop-last", type=int, choices=[0, 1], help="Drop the final incomplete training batch.")
    parser.add_argument("--optimizer", choices=["adamw", "radam", "muon"], help="Override model optimizer.")
    parser.add_argument("--lr", type=float, help="Override model optimizer learning rate.")
    parser.add_argument("--min-lr", type=float, help="Override model minimum scheduler learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override model optimizer weight decay.")
    parser.add_argument("--scheduler", choices=["cos", "onecycle"], help="Override model scheduler.")
    parser.add_argument(
        "--constant-ratio",
        type=float,
        help="Keep cosine scheduler at max LR for this fraction of updates before decay.",
    )
    parser.add_argument("--finetune-epochs", type=int, help="Train a second finetune phase for N epochs.")
    parser.add_argument("--finetune-log-freq", type=int, help="Print finetune tqdm and epoch metrics every N epochs.")
    parser.add_argument("--finetune-lr-reduce", type=float, help="Scale optimizer lr/min_lr for finetune.")
    parser.add_argument(
        "--finetune-disabled-augs",
        nargs="*",
        choices=tuple(CFG.sim_cfg) + tuple(CFG.aug_cfg),
        metavar="AUG",
        help="Simulation/augmentation names whose apply_prob is zeroed only during finetune.",
    )
    parser.add_argument("--typewell-len", type=int, help="Override local Typewell grid length.")
    parser.add_argument(
        "--tw-gr-grid-type",
        choices=TYPEWELL_GR_GRID_TYPES,
        help="Typewell GR grid sampler: exact linear interpolation or piecewise-linear cell average.",
    )
    parser.add_argument("--seq-target-mode", choices=SEQ_TARGET_MODES, help="Sequence NN target formula.")
    parser.add_argument("--huber-delta", type=float, help="Huber transition point on normalized target residuals.")
    parser.add_argument("--min-seq-weight", type=float, help="Minimum suffix-progress weight for model losses.")
    parser.add_argument(
        "--seq-noise-weight",
        type=int,
        choices=[0, 1],
        help="Weight model losses by inverse detached sequence RMSE in original TVT feet.",
    )
    parser.add_argument(
        "--seq-noise-smooth",
        type=float,
        help="Positive smoothing term for inverse sequence-noise model loss weighting.",
    )
    parser.add_argument(
        "--seq-noise-power",
        type=float,
        help="Non-negative power for inverse sequence-noise model loss weighting.",
    )
    parser.add_argument(
        "--gr-corr-weight",
        type=int,
        choices=[0, 1],
        help="Weight model-loss bins by smoothed within-sequence horizontal-GR leverage.",
    )
    parser.add_argument(
        "--gr-corr-weight-p",
        type=float,
        help="Positive power applied to absolute centered horizontal GR for loss weighting.",
    )
    parser.add_argument(
        "--gr-corr-weight-smooth-window",
        type=int,
        help="Moving-average window in sequence bins for horizontal-GR loss weighting.",
    )
    parser.add_argument(
        "--gr-corr-weight-smooth-factor",
        type=float,
        help="Non-negative sequence-mean floor factor for horizontal-GR loss weighting.",
    )
    parser.add_argument(
        "--gr-interpolate",
        type=int,
        choices=[0, 1],
        help="Linearly fill horizontal GR after simulation before feature construction.",
    )
    parser.add_argument(
        "--out-of-range-GR-mask",
        type=int,
        choices=[0, 1],
        help="Mask horizontal GR outside the local Typewell-row GR range before downsampling.",
    )
    parser.add_argument(
        "--model-name",
        choices=MODEL_MODES,
        help="Sequence NN architecture.",
    )
    parser.add_argument(
        "--trf-backbone",
        choices=tuple(TRF_UNET_BACKBONES),
        help="Override the timm feature-pyramid backbone for model_name='trf_unet'.",
    )
    parser.add_argument(
        "--trf-pretrained",
        type=int,
        choices=[0, 1],
        help="Load pretrained weights for the trf_unet backbone.",
    )
    parser.add_argument("--unet-updown-mode", choices=["param", "non-param"], help="2D U-Net down/up sampling mode.")
    parser.add_argument(
        "--unet-arch",
        choices=["unet", "convnext_small"],
        help="2D U-Net architecture.",
    )
    parser.add_argument("--unet-emb-dim", type=int, help="Override 2D U-Net embedding dimension.")
    parser.add_argument(
        "--unet-mults",
        type=int,
        nargs="+",
        metavar="M",
        help="Override 2D U-Net channel multipliers, e.g. --unet-mults 1 2 4 8 16.",
    )
    parser.add_argument(
        "--unet-pretrained",
        type=int,
        choices=[0, 1],
        help="Override pretrained timm backbone loading for pretrained U-Net arches.",
    )
    parser.add_argument(
        "--unet-stem-stride",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Override pretrained U-Net stem stride, e.g. --unet-stem-stride 2 4.",
    )
    parser.add_argument(
        "--unet-stem-down-mode",
        choices=["pool", "conv"],
        help="Pretrained U-Net stem downsampling: pool keeps AvgPool+1x1 projection, conv uses stride convolution.",
    )
    parser.add_argument(
        "--unet-convnext-norm-replace",
        choices=["BN", "IN", "LN"],
        help="Use BatchNorm or InstanceNorm in timm ConvNeXt, or LN to keep its native LayerNorm unchanged.",
    )
    parser.add_argument(
        "--unet-convnext-norm-to-bn",
        type=int,
        choices=[0, 1],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--unet-convnext-bn-eps",
        type=float,
        help="Epsilon for converted ConvNeXt BatchNorm or InstanceNorm layers.",
    )
    parser.add_argument(
        "--unet-convnext-bn-momentum",
        type=float,
        help="Momentum for converted ConvNeXt BatchNorm layers when running stats are tracked.",
    )
    parser.add_argument(
        "--unet-convnext-bn-track-running-stats",
        type=int,
        choices=[0, 1],
        help="Track running stats in converted ConvNeXt BatchNorm layers.",
    )
    parser.add_argument("--unet-dropout", type=float, help="Override 2D U-Net dropout.")
    parser.add_argument(
        "--model-down-weight-wells",
        nargs="+",
        help="Well ids whose model losses are downweighted.",
    )
    parser.add_argument(
        "--model-down-weight",
        type=float,
        help="Multiplicative weight applied to model losses for model-down-weight wells.",
    )
    parser.add_argument("--alignment-loss-weight", type=float, help="Override unet_loss_weights['alignment'].")
    parser.add_argument(
        "--stage-alignment-loss-weight",
        dest="alignment_loss_weight",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ema", type=int, choices=[0, 1], help="Enable model EMA.")
    parser.add_argument("--ema-decay", type=float, help="Override EMA decay.")
    parser.add_argument(
        "--deterministic",
        type=int,
        choices=[0, 1],
        help="Enable strict deterministic CUDA kernels. Seeds are fixed either way.",
    )
    parser.add_argument("--channels-last-2d", type=int, choices=[0, 1], help="Use channels-last tensors for 2D models.")
    parser.add_argument(
        "--pred-sg-smooth",
        type=int,
        choices=[0, 1],
        help="Apply row-level Savitzky-Golay smoothing to S=TVT_pred+Z before scoring/inference output.",
    )
    parser.add_argument("--pred-sg-smooth-window", type=int, help="Savitzky-Golay smoothing window in suffix rows.")
    parser.add_argument("--pred-sg-smooth-polyorder", type=int, help="Savitzky-Golay smoothing polynomial order.")
    parser.add_argument("--pred-sg-smooth-blend", type=float, help="Blend weight toward the SG-smoothed TVT surface.")
    parser.add_argument("--prefetch-factor", type=int, help="DataLoader worker prefetch factor.")
    args = parser.parse_args(argv)
    if args.submit_mode:
        if args.run_mode != "common":
            parser.error("--submit-mode cannot be combined with non-common --run-mode values")
        if args.train_output_dir is None:
            parser.error("--submit-mode requires --train-output-dir")
    elif args.train_output_dir is not None:
        parser.error("--train-output-dir is only used with --submit-mode")
    return args


TRAINING_ARG_OVERRIDES = (
    ("device", "device"),
    ("well_limit", "well_limit"),
    ("epochs", "epochs"),
    ("log_freq", "log_freq"),
    ("fold_count", "fold_count"),
    ("cv_split_mode", "cv_split_mode"),
    ("cv_geo_map_path", "cv_geo_map_path"),
    ("cv_kmeans_clusters", "cv_kmeans_clusters"),
    ("cv_kmeans_seed", "cv_kmeans_seed"),
    ("batch_size", "batch_size"),
    ("val_batch_size", "val_batch_size"),
    ("num_workers", "num_workers"),
    ("persistent_workers", "persistent_workers"),
    ("grad_accum_steps", "grad_accum_steps"),
    ("train_drop_last", "train_drop_last"),
    ("finetune_epochs", "finetune_epochs"),
    ("finetune_log_freq", "finetune_log_freq"),
    ("finetune_lr_reduce", "finetune_lr_reduce"),
    ("finetune_disabled_augs", "finetune_disabled_augs"),
    ("typewell_len", "typewell_len"),
    ("tw_gr_grid_type", "tw_gr_grid_type"),
    ("huber_delta", "huber_delta"),
    ("min_seq_weight", "min_seq_weight"),
    ("seq_noise_weight", "seq_noise_weight"),
    ("seq_noise_smooth", "seq_noise_smooth"),
    ("seq_noise_power", "seq_noise_power"),
    ("gr_corr_weight", "gr_corr_weight"),
    ("gr_corr_weight_p", "gr_corr_weight_p"),
    ("gr_corr_weight_smooth_window", "gr_corr_weight_smooth_window"),
    ("gr_corr_weight_smooth_factor", "gr_corr_weight_smooth_factor"),
    ("model_down_weight", "model_down_weight"),
    ("ema_decay", "ema_decay"),
    ("pred_sg_smooth_window", "pred_sg_smooth_window"),
    ("pred_sg_smooth_polyorder", "pred_sg_smooth_polyorder"),
    ("pred_sg_smooth_blend", "pred_sg_smooth_blend"),
    ("prefetch_factor", "prefetch_factor"),
)


def resolve_output_dir(cfg, args):
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if args.run_id:
        return (Path(cfg.output_dir) / f"{args.run_id}_seq_nn").resolve()
    return (Path(cfg.output_dir) / "seq_nn").resolve()


def apply_data_dir_override(cfg, data_dir):
    if data_dir is None:
        return
    data_dir = data_dir.expanduser().resolve()
    cfg.data_dir = data_dir
    cfg.train_path = data_dir / "train"
    cfg.test_path = data_dir / "test"


def apply_training_arg_overrides(cfg, args):
    apply_data_dir_override(cfg, args.data_dir)
    for arg_name, cfg_name in TRAINING_ARG_OVERRIDES:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    cfg.train_drop_last = bool(cfg.train_drop_last)
    cfg.seq_noise_weight = bool(cfg.seq_noise_weight)
    cfg.gr_corr_weight = bool(cfg.gr_corr_weight)
    if args.pred_sg_smooth is not None:
        cfg.pred_sg_smooth = bool(args.pred_sg_smooth)
    if args.model_name is not None:
        cfg.set_model_name(args.model_name)
    if args.seq_target_mode is not None:
        cfg.set_seq_target_mode(args.seq_target_mode)
    cfg.submit_mode = False
    model_cfg = cfg.model_cfg
    unet_arg_values = {
        "--unet-updown-mode": args.unet_updown_mode,
        "--unet-arch": args.unet_arch,
        "--unet-emb-dim": args.unet_emb_dim,
        "--unet-mults": args.unet_mults,
        "--unet-pretrained": args.unet_pretrained,
        "--unet-stem-stride": args.unet_stem_stride,
        "--unet-stem-down-mode": args.unet_stem_down_mode,
        "--unet-convnext-norm-replace": args.unet_convnext_norm_replace,
        "--unet-convnext-norm-to-bn": args.unet_convnext_norm_to_bn,
        "--unet-convnext-bn-eps": args.unet_convnext_bn_eps,
        "--unet-convnext-bn-momentum": args.unet_convnext_bn_momentum,
        "--unet-convnext-bn-track-running-stats": (
            args.unet_convnext_bn_track_running_stats
        ),
        "--unet-dropout": args.unet_dropout,
    }
    active_unet_args = [name for name, value in unet_arg_values.items() if value is not None]
    if active_unet_args and cfg.model_name not in {UNET_MODE, TWO_STAGE_UNET_MODE}:
        raise ValueError(
            f"{', '.join(active_unet_args)} apply only to model_name='unet' or "
            "'two_stage_unet'"
        )
    if cfg.model_name in {UNET_MODE, TWO_STAGE_UNET_MODE}:
        if args.unet_updown_mode is not None:
            model_cfg["updown_mode_2d"] = args.unet_updown_mode
        if args.unet_arch is not None:
            model_cfg["unet_arch"] = args.unet_arch
        if args.unet_emb_dim is not None:
            model_cfg["unet_emb_dim"] = int(args.unet_emb_dim)
        if args.unet_mults is not None:
            model_cfg["unet_mults"] = tuple(int(v) for v in args.unet_mults)
        if args.unet_pretrained is not None:
            model_cfg["unet_pretrained"] = bool(args.unet_pretrained)
        if args.unet_stem_stride is not None:
            model_cfg["unet_stem_stride"] = tuple(int(v) for v in args.unet_stem_stride)
        if args.unet_stem_down_mode is not None:
            model_cfg["unet_stem_down_mode"] = args.unet_stem_down_mode
        if args.unet_convnext_norm_replace is not None:
            model_cfg["unet_convnext_norm_replace"] = args.unet_convnext_norm_replace
        elif args.unet_convnext_norm_to_bn is not None:
            model_cfg["unet_convnext_norm_replace"] = (
                "BN" if bool(args.unet_convnext_norm_to_bn) else None
            )
        if args.unet_convnext_bn_eps is not None:
            model_cfg["unet_convnext_bn_eps"] = float(args.unet_convnext_bn_eps)
        if args.unet_convnext_bn_momentum is not None:
            model_cfg["unet_convnext_bn_momentum"] = float(args.unet_convnext_bn_momentum)
        if args.unet_convnext_bn_track_running_stats is not None:
            model_cfg["unet_convnext_bn_track_running_stats"] = bool(
                args.unet_convnext_bn_track_running_stats
            )
        if args.unet_dropout is not None:
            model_cfg["unet_dropout"] = args.unet_dropout
    active_trf_args = args.trf_backbone is not None or args.trf_pretrained is not None
    if active_trf_args and cfg.model_name != TRF_UNET_MODE:
        raise ValueError(
            "--trf-backbone and --trf-pretrained apply only to model_name='trf_unet'"
        )
    if cfg.model_name == TRF_UNET_MODE:
        if args.trf_backbone is not None:
            model_cfg["backbone_name"] = args.trf_backbone
        if args.trf_pretrained is not None:
            model_cfg["pretrained"] = bool(args.trf_pretrained)
    if args.model_down_weight_wells is not None:
        cfg.model_down_weight_wells = [str(well_id) for well_id in args.model_down_weight_wells]
    if args.alignment_loss_weight is not None:
        cfg.unet_loss_weights["alignment"] = float(args.alignment_loss_weight)
    if args.gr_interpolate is not None:
        cfg.gr_interpolate = bool(args.gr_interpolate)
    if args.out_of_range_GR_mask is not None:
        cfg.out_of_range_GR_mask = bool(args.out_of_range_GR_mask)
    for arg_name, key in (
        ("optimizer", "optimizer"),
        ("lr", "lr"),
        ("min_lr", "min_lr"),
        ("weight_decay", "weight_decay"),
        ("scheduler", "scheduler"),
        ("constant_ratio", "constant_ratio"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            cfg.optimizer_cfg[key] = value
    if args.ema is not None:
        cfg.ema_enabled = bool(args.ema)
    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay
    if args.deterministic is not None:
        cfg.deterministic = bool(args.deterministic)
    if args.channels_last_2d is not None:
        cfg.channels_last_2d = bool(args.channels_last_2d)
    cfg.refresh()


def apply_arg_overrides(cfg, args, include_output=True):
    if include_output:
        cfg.output_dir = resolve_output_dir(cfg, args)
        if args.f is not None:
            cfg.f = args.f
    apply_training_arg_overrides(cfg, args)


def apply_submit_arg_overrides(cfg, args):
    cfg.output_dir = resolve_output_dir(CFG(), args)
    cfg.f = args.f if args.f is not None else 0
    cfg.submit_mode = True
    apply_data_dir_override(cfg, args.data_dir)
    for arg_name in (
        "device",
        "well_limit",
        "val_batch_size",
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, arg_name, value)
    for arg_name in (
        "pred_sg_smooth_window",
        "pred_sg_smooth_polyorder",
        "pred_sg_smooth_blend",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, arg_name, value)
    if args.pred_sg_smooth is not None:
        cfg.pred_sg_smooth = bool(args.pred_sg_smooth)
    if args.gr_interpolate is not None:
        cfg.gr_interpolate = bool(args.gr_interpolate)
    if args.out_of_range_GR_mask is not None:
        cfg.out_of_range_GR_mask = bool(args.out_of_range_GR_mask)
    if args.deterministic is not None:
        cfg.deterministic = bool(args.deterministic)
    if args.channels_last_2d is not None:
        cfg.channels_last_2d = bool(args.channels_last_2d)
    cfg.refresh()


def save_cfg(cfg):
    cfg_path = cfg.output_dir / cfg.cfg_filename
    with open(cfg_path, "wb") as f:
        pickle.dump(cfg, f)
    log(f"saved cfg pickle: {cfg_path}")


def load_submit_cfg(train_output_dir):
    default_cfg = CFG()
    cfg_path = train_output_dir / default_cfg.cfg_filename
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
        cfg.refresh()
        return cfg, cfg_path
    return default_cfg, None


def save_models(models, cfg):
    model_path = cfg.output_dir / cfg.model_filename
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    log(f"saved models pickle: {model_path}")


def load_models(train_output_dir, cfg):
    model_path = train_output_dir / cfg.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"model pickle not found: {model_path}")
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    if not isinstance(models, dict) or len(models) == 0:
        raise ValueError(f"expected non-empty fold model dict in {model_path}")
    log(f"loaded models pickle: {model_path} folds={len(models):,}")
    return models


def pad_submission_tail(pred_df, test_path, cfg):
    pred_parts = []
    for well_id in discover_well_ids(test_path):
        if cfg.well_limit is not None and len(pred_parts) >= cfg.well_limit:
            break
        h_df = pd.read_csv(Path(test_path) / f"{well_id}__horizontal_well.csv", usecols=["TVT_input"])
        submit_indices = np.flatnonzero(h_df["TVT_input"].isna().to_numpy())
        well_pred = pred_df[pred_df["well_id"] == well_id].sort_values("submit_index").copy()
        if len(well_pred) == 0:
            raise ValueError(f"{well_id}: no model predictions to build submission")
        if len(well_pred) < len(submit_indices):
            extra_indices = submit_indices[len(well_pred) :]
            last_pred = float(well_pred["TVT_pred"].iloc[-1])
            pad_df = pd.DataFrame(
                {
                    "well_id": well_id,
                    "submit_index": extra_indices,
                    "TVT_pred": last_pred,
                }
            )
            well_pred = pd.concat([well_pred, pad_df], ignore_index=True)
        pred_parts.append(well_pred[["well_id", "submit_index", "TVT_pred"]])
    out = pd.concat(pred_parts, ignore_index=True)
    out["id"] = out["well_id"] + "_" + out["submit_index"].astype(str)
    out["tvt"] = out["TVT_pred"].astype(float)
    return out[["id", "tvt"]]


def pad_submission_details_tail(pred_df, test_path, cfg):
    pred_parts = []
    for well_id in discover_well_ids(test_path):
        if cfg.well_limit is not None and len(pred_parts) >= cfg.well_limit:
            break
        input_cols = {"MD", "X", "Y", "Z", "GR", "TVT_input"}
        h_df = pd.read_csv(
            Path(test_path) / f"{well_id}__horizontal_well.csv",
            usecols=lambda col: col in input_cols,
        )
        submit_indices = np.flatnonzero(h_df["TVT_input"].isna().to_numpy())
        well_pred = pred_df[pred_df["well_id"] == well_id].sort_values("submit_index").copy()
        if len(well_pred) == 0:
            raise ValueError(f"{well_id}: no model predictions to build submission details")
        if len(well_pred) < len(submit_indices):
            extra_indices = submit_indices[len(well_pred) :]
            pad_row = well_pred.iloc[[-1]].copy()
            pad_df = pd.concat([pad_row] * len(extra_indices), ignore_index=True)
            pad_df["submit_index"] = extra_indices
            pred_parts.append(pd.concat([well_pred, pad_df], ignore_index=True))
        else:
            pred_parts.append(well_pred)
        input_df = h_df.iloc[submit_indices].copy()
        input_df.insert(0, "submit_index", submit_indices)
        input_df.insert(0, "well_id", well_id)
        merge_pred = pred_parts[-1].drop(
            columns=[
                col
                for col in input_df.columns
                if col in pred_parts[-1].columns and col not in ("well_id", "submit_index")
            ]
        )
        pred_parts[-1] = input_df.merge(merge_pred, on=["well_id", "submit_index"], how="left", validate="one_to_one")
    out = pd.concat(pred_parts, ignore_index=True)
    out["id"] = out["well_id"] + "_" + out["submit_index"].astype(str)
    out["tvt"] = out["TVT_pred"].astype(float)
    return out


def prepare_output_dir(output_dir, force, log_func=log):
    if output_dir.exists():
        if force == 1:
            log_func(f"WARNING: output directory already exists and f=1 was set: {output_dir}")
            log_func("existing artifact files in this directory may be overwritten")
        else:
            log_func(f"output directory already exists: {output_dir}")
            log_func("exiting before any work to avoid overwriting results")
            return False
    else:
        output_dir.mkdir(parents=True)
    return True


def discover_run_wells(cfg, include_test=True):
    train_wells = discover_well_ids(cfg.train_path)
    test_wells = discover_well_ids(cfg.test_path) if include_test else []
    if cfg.well_limit is not None:
        train_wells = train_wells[: cfg.well_limit]
        if include_test:
            test_wells = test_wells[: cfg.well_limit]
    return train_wells, test_wells


def log_cfg_summary(cfg, train_wells, test_wells, include_test=True):
    log(f"output directory: {cfg.output_dir}")
    log(f"train path: {cfg.train_path}")
    if include_test:
        log(f"test path: {cfg.test_path}")
    log(f"device: {cfg.device}")
    log(f"well limit: {cfg.well_limit}")
    if include_test:
        log(f"train wells: {len(train_wells):,}, test wells: {len(test_wells):,}")
    else:
        log(f"train wells: {len(train_wells):,}")
    log(
        f"raw/downsample/bins: {cfg.raw_len}/{cfg.downsample}/{cfg.num_bins}, "
        f"typewell_len={cfg.typewell_len}, tw_gr_grid_type={getattr(cfg, 'tw_gr_grid_type', 'interpolate')}"
    )
    log(
        f"batch size: train={cfg.batch_size}, val={cfg.val_batch_size}, "
        f"grad_accum_steps={cfg.grad_accum_steps}, "
        f"train_drop_last={cfg.train_drop_last}, "
        f"epochs={cfg.epochs}, folds={cfg.fold_count}, log_freq={cfg.log_freq}, "
        f"workers={cfg.num_workers}, persistent_workers={bool(getattr(cfg, 'persistent_workers', True))}, "
        f"prefetch_factor={cfg.prefetch_factor}, pin_memory={cfg.pin_memory}"
    )
    cv_geo_map_path = seq_NN_train.resolve_cv_geo_map_path(cfg)
    log(
        f"CV split: mode={cfg.cv_split_mode}, "
        f"geo_map_path={cv_geo_map_path}, "
        f"kmeans_clusters={cfg.cv_kmeans_clusters}, "
        f"kmeans_seed={cfg.cv_kmeans_seed}, seed={cfg.seed}"
    )
    log(
        f"optimizer: {cfg.optimizer_cfg}, "
        f"betas={cfg.optimizer_betas}, eps={cfg.optimizer_eps:g}, "
        f"finetune_epochs={cfg.finetune_epochs}, "
        f"finetune_log_freq={cfg.finetune_log_freq}, "
        f"finetune_lr_reduce={cfg.finetune_lr_reduce:g}, "
        f"finetune_disabled_augs={list(cfg.finetune_disabled_augs)}"
    )
    if cfg.optimizer_cfg["optimizer"] == "muon":
        log(
            f"muon: momentum={cfg.muon_momentum:g}, ns_steps={cfg.muon_ns_steps}, "
            f"nesterov={cfg.muon_nesterov}"
        )
    alignment_sigma = (
        cfg.alignment_laplace_sigma
        if cfg.alignment_target_mode == "laplace"
        else cfg.alignment_exp_smooth_sigma
    )
    log(
        f"loss: unet weighted_huber_ce weights={cfg.unet_loss_weights}, "
        f"alignment_target={cfg.alignment_target_mode}, "
        f"sigma={alignment_sigma:g}, "
        f"tvt_diff_clip={cfg.tvt_diff_clip:g}, "
        f"model_down_weight={cfg.model_down_weight:g} wells={len(cfg.model_down_weight_wells)}, "
        f"min_seq_weight={cfg.min_seq_weight:g}, "
        f"seq_noise_weight={cfg.seq_noise_weight}, "
        f"seq_noise_smooth={cfg.seq_noise_smooth:g}, "
        f"seq_noise_power={cfg.seq_noise_power:g}, "
        f"gr_corr_weight={cfg.gr_corr_weight}, "
        f"gr_corr_weight_p={cfg.gr_corr_weight_p:g}, "
        f"gr_corr_weight_smooth_window={cfg.gr_corr_weight_smooth_window}, "
        f"gr_corr_weight_smooth_factor={cfg.gr_corr_weight_smooth_factor:g}"
    )
    log(f"simulation: sim_cfg={cfg.sim_cfg}")
    log(f"augmentation: aug_cfg={cfg.aug_cfg}")
    log(f"target mode: {cfg.seq_target_mode}, stats={cfg.target_stats[cfg.seq_target_mode]}")
    log(
        "pred_sg_smooth: "
        f"enabled={getattr(cfg, 'pred_sg_smooth', False)}, "
        f"window={getattr(cfg, 'pred_sg_smooth_window', 577)}, "
        f"polyorder={getattr(cfg, 'pred_sg_smooth_polyorder', 1)}, "
        f"blend={getattr(cfg, 'pred_sg_smooth_blend', 0.7261960515627183):g}"
    )
    geo_prior_cfg = cfg.geo_prior_cfg.to_dict() if hasattr(cfg.geo_prior_cfg, "to_dict") else dict(cfg.geo_prior_cfg)
    log(f"geo prior: {geo_prior_cfg}")
    log(f"model: {cfg.model_name}, cfg={cfg.model_cfg}")
    log(f"U-Net static channels: {cfg.unet_static_channels}")
    if uses_pf_heatmap_channels(cfg):
        log(
            "PF heatmap: "
            f"cache_dir={cfg.PF_heatmap_cache_dir}, "
            f"N={cfg.PF_heatmap_n_particles}, seeds={cfg.PF_heatmap_n_seeds}, "
            f"base_seed={cfg.PF_heatmap_base_seed}, "
            f"lik_scale={cfg.PF_heatmap_lik_scale}, "
            f"axis_sigma={cfg.PF_heatmap_axis_sigma}, "
            f"workers={cfg.PF_heatmap_num_workers}, "
            f"version={cfg.PF_heatmap_version}"
        )
    log(f"EMA: enabled={cfg.ema_enabled}, decay={cfg.ema_decay:.4f}")


def run_common(cfg):
    train_wells, test_wells = discover_run_wells(cfg, include_test=True)

    log_section("setup")
    log_cfg_summary(cfg, train_wells, test_wells, include_test=True)
    save_cfg(cfg)

    log_section("model training")
    models, oof_df = seq_NN_train.kfold_training(well_ids=train_wells, cfg=cfg, log=log)
    oof_path = cfg.output_dir / cfg.oof_filename
    oof_df.to_parquet(oof_path)
    log(f"saved OOF predictions: {oof_path}")

    log_section("save models")
    save_models(models=models, cfg=cfg)

    log_section("test inference")
    pred_df = seq_NN_train.predict_models(
        models=models,
        well_ids=test_wells,
        cfg=cfg,
        train_wells=train_wells,
        log=log,
    )
    submission = pad_submission_tail(pred_df, cfg.test_path, cfg)
    submission_details = pad_submission_details_tail(pred_df, cfg.test_path, cfg)
    submission_details_path = cfg.output_dir / getattr(cfg, "submission_details_filename", "submission_details.pqt")
    submission_details.to_parquet(submission_details_path, index=False)
    log(f"saved submission details: {submission_details_path}")
    submission_path = cfg.output_dir / cfg.submission_filename
    submission.to_csv(submission_path, index=False)
    log(f"saved submission: {submission_path}")
    log(f"submission shape: {submission.shape}")
    log("\n" + submission.head().to_string(index=False))


def run_submit(cfg, train_output_dir, cfg_source):
    if cfg_source is None:
        log(f"training cfg not found under {train_output_dir}; using default CFG")
    else:
        log(f"loaded training cfg: {cfg_source}")
    log(f"train output directory: {train_output_dir}")

    train_wells = discover_well_ids(cfg.train_path)
    test_wells = discover_well_ids(cfg.test_path)
    if cfg.well_limit is not None:
        train_wells = train_wells[: cfg.well_limit]
        test_wells = test_wells[: cfg.well_limit]

    log_section("setup")
    log_cfg_summary(cfg, train_wells, test_wells, include_test=True)

    log_section("load models")
    models = load_models(train_output_dir, cfg)

    log_section("test inference")
    pred_df = seq_NN_train.predict_models(
        models=models,
        well_ids=test_wells,
        cfg=cfg,
        train_wells=train_wells,
        log=log,
    )
    submission = pad_submission_tail(pred_df, cfg.test_path, cfg)
    submission_details = pad_submission_details_tail(pred_df, cfg.test_path, cfg)
    submission_details_path = cfg.output_dir / getattr(cfg, "submission_details_filename", "submission_details.pqt")
    submission_details.to_parquet(submission_details_path, index=False)
    log(f"saved submission details: {submission_details_path}")
    submission_path = cfg.output_dir / cfg.submission_filename
    submission.to_csv(submission_path, index=False)
    log(f"saved submission: {submission_path}")
    log(f"submission shape: {submission.shape}")
    log("\n" + submission.head().to_string(index=False))


def run_seq_train(base_cfg, args):
    for cfg_name, cfg_template in SEQ_TRAIN_CFGS:
        cfg = deepcopy(cfg_template)
        cfg.output_dir = base_cfg.output_dir
        cfg.f = base_cfg.f
        apply_arg_overrides(cfg, args, include_output=False)
        log_path = cfg.output_dir / f"seq_train_{cfg_name}.log"
        with log_path.open("w") as log_file, redirect_stdout(Tee(sys.stdout, log_file)):
            log_section(f"seq_train {cfg_name}")
            train_wells, test_wells = discover_run_wells(cfg, include_test=False)
            log(f"cfg name: {cfg_name}")
            log(f"train log: {log_path}")
            log_cfg_summary(cfg, train_wells, test_wells, include_test=False)
            seq_NN_train.kfold_training(well_ids=train_wells, cfg=cfg, log=log)
            log(f"finished train-only cfg: {cfg_name}")


def run_seq_full_train(base_cfg, args):
    for cfg_name, cfg_template in SEQ_TRAIN_CFGS:
        cfg = deepcopy(cfg_template)
        cfg.output_dir = base_cfg.output_dir / cfg_name
        cfg.f = base_cfg.f
        apply_arg_overrides(cfg, args, include_output=False)
        output_dir_existed = cfg.output_dir.exists()
        if not prepare_output_dir(cfg.output_dir, cfg.f):
            continue
        log_path = cfg.output_dir / cfg.log_filename
        with log_path.open("w") as log_file, redirect_stdout(Tee(sys.stdout, log_file)):
            if output_dir_existed and cfg.f == 1:
                log(f"WARNING: output directory already exists and f=1 was set: {cfg.output_dir}")
                log("existing artifact files in this directory may be overwritten")
            log_section(f"seq_full_train {cfg_name}")
            log(f"cfg name: {cfg_name}")
            log(f"full output directory: {cfg.output_dir}")
            run_common(cfg)
            log(f"finished full cfg: {cfg_name}")
            log(f"saved print log: {log_path}")


def main(argv=None):
    args = parse_args(argv)

    if args.submit_mode:
        train_output_dir = args.train_output_dir.expanduser().resolve()
        cfg, cfg_source = load_submit_cfg(train_output_dir)
        apply_submit_arg_overrides(cfg, args)
    else:
        cfg = CFG()
        apply_arg_overrides(cfg, args)

    output_dir_existed = cfg.output_dir.exists()
    if not prepare_output_dir(cfg.output_dir, cfg.f):
        return

    if args.submit_mode:
        if output_dir_existed and cfg.f == 1:
            log(f"WARNING: output directory already exists and f=1 was set: {cfg.output_dir}")
            log("existing artifact files in this directory may be overwritten")
        run_submit(cfg, train_output_dir, cfg_source)
        return

    if args.run_mode == "seq_train":
        run_seq_train(cfg, args)
        return

    if args.run_mode == "seq_full_train":
        run_seq_full_train(cfg, args)
        return

    log_path = cfg.output_dir / cfg.log_filename
    with log_path.open("w") as log_file, redirect_stdout(Tee(sys.stdout, log_file)):
        if output_dir_existed and cfg.f == 1:
            log(f"WARNING: output directory already exists and f=1 was set: {cfg.output_dir}")
            log("existing artifact files in this directory may be overwritten")
        run_common(cfg)
        log(f"saved print log: {log_path}")


if __name__ == "__main__":
    main()
