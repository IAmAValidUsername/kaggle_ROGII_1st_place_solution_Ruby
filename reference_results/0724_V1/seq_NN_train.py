import copy
import gc
import inspect
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

from seq_NN_data_prep import (
    ensure_pf_heatmap_cache,
    ensure_pf_sample_cache,
    uses_pf_heatmap_channels,
    uses_pf_sample_trend,
)
from seq_NN_dataset import (
    SeqUNetDataset,
    collate_seq_batch,
    discover_well_ids,
    remove_train_rm_wells,
    train_rm_wells_set,
)
from seq_NN_geo_prior import (
    POST_TRAIN_GEO_DIAGNOSTIC_NAMES,
    make_geo_prior_for_wells,
    resolve_geo_prior_cfg,
)
from seq_NN_models import SeqTwoStageUNetModel, SeqUNet2DModel
from seq_NN_cfg import (
    MODEL_MODES,
    TWO_STAGE_UNET_MODE,
    TYPEWELL_GR_GRID_TYPES,
    UNET_MODE,
)


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        *,
        lr,
        weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        muon_momentum=0.95,
        muon_ns_steps=5,
        muon_nesterov=True,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative: {weight_decay}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive: {eps}")
        if len(betas) != 2 or not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be two values in [0, 1): {betas}")
        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "betas": tuple(float(x) for x in betas),
            "eps": float(eps),
            "muon_momentum": float(muon_momentum),
            "muon_ns_steps": int(muon_ns_steps),
            "muon_nesterov": bool(muon_nesterov),
            "use_muon": False,
        }
        super().__init__(param_groups, defaults)

    @staticmethod
    def _zeropower_via_newton_schulz5(update, steps, eps):
        original_shape = update.shape
        matrix = update.reshape(update.shape[0], -1).float()
        if matrix.numel() == 0:
            return update
        transposed = matrix.shape[0] > matrix.shape[1]
        if transposed:
            matrix = matrix.T
        matrix = matrix / matrix.norm().clamp_min(eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(int(steps)):
            gram = matrix @ matrix.T
            matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix
        if transposed:
            matrix = matrix.T
        return matrix.reshape(original_shape).to(dtype=update.dtype)

    @staticmethod
    def _matrix_lr_scale(param):
        matrix = param.reshape(param.shape[0], -1)
        return math.sqrt(max(1.0, matrix.shape[0] / matrix.shape[1]))

    @torch.no_grad()
    def _adamw_step_param(self, param, grad, state, group):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)

        state["step"] += 1
        if weight_decay != 0.0:
            param.mul_(1.0 - lr * weight_decay)

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

    @torch.no_grad()
    def _muon_step_param(self, param, grad, state, group):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["muon_momentum"]
        ns_steps = group["muon_ns_steps"]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(param)
        if weight_decay != 0.0:
            param.mul_(1.0 - lr * weight_decay)

        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.mul_(momentum).add_(grad)
        if group["muon_nesterov"]:
            update = grad.add(momentum_buffer, alpha=momentum)
        else:
            update = momentum_buffer
        update = self._zeropower_via_newton_schulz5(update, ns_steps, group["eps"])
        param.add_(update, alpha=-lr * self._matrix_lr_scale(param))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            use_muon = bool(group.get("use_muon", False))
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonWithAdamW does not support sparse gradients")
                state = self.state[param]
                if use_muon:
                    self._muon_step_param(param, grad, state, group)
                else:
                    self._adamw_step_param(param, grad, state, group)
        return loss


class ModelEMA(torch.nn.Module):
    def __init__(self, model, *, decay):
        super().__init__()
        self.module = copy.deepcopy(source_model_for_train_ops(model))
        self.module.eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source_model = source_model_for_train_ops(model)
        model_state = source_model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name]
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value.detach(), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


@dataclass
class TrainOps:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def unscale(self, scaler):
        scaler.unscale_(self.optimizer)

    def step_scaled(self, scaler):
        old_scale = scaler.get_scale()
        scaler.step(self.optimizer)
        scaler.update()
        optimizer_stepped = scaler.get_scale() >= old_scale
        if optimizer_stepped:
            self.scheduler.step()
        return optimizer_stepped

    def step_plain(self):
        self.optimizer.step()
        self.scheduler.step()
        return True

    def lr_summary(self):
        return f"{self.scheduler.get_last_lr()[0]:.2e}"


def source_model_for_train_ops(model):
    source_model = getattr(model, "_orig_mod", model)
    if isinstance(source_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return source_model.module
    return source_model


def ema_enabled(cfg):
    return bool(getattr(cfg, "ema_enabled", False))


def resolve_ema_decay(cfg, steps_per_epoch):
    del steps_per_epoch
    decay = float(getattr(cfg, "ema_decay", 0.99))
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1): {decay}")
    return decay


def state_dict_for_best(model):
    return {k: v.detach().cpu().clone() for k, v in source_model_for_train_ops(model).state_dict().items()}


def set_training_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_group_folds(groups, fold_count, seed):
    subjects = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    val_subject_count = math.ceil(len(subjects) / fold_count)
    splits = []
    for fold in range(fold_count):
        val_subjects = subjects[val_subject_count * fold : val_subject_count * (fold + 1)]
        val_cond = np.isin(groups, val_subjects)
        train_index = np.flatnonzero(~val_cond)
        val_index = np.flatnonzero(val_cond)
        splits.append((train_index, val_index))
    return splits


def resolve_cv_split_seed(cfg):
    split_seed = getattr(cfg, "cv_split_seed", None)
    if split_seed is None:
        split_seed = getattr(cfg, "seed")
    return int(split_seed)


def resolve_cv_repeats(cfg):
    repeats = int(getattr(cfg, "cv_repeats", 1))
    if repeats < 1:
        raise ValueError(f"cv_repeats must be at least 1, got {repeats}")
    return repeats


def resolve_cv_geo_map_path(cfg):
    geo_map_path = getattr(cfg, "cv_geo_map_path", None)
    if geo_map_path is None:
        return Path(cfg.data_dir) / "train_png_typewell_map.csv"
    geo_map_path = Path(geo_map_path).expanduser()
    if not geo_map_path.is_absolute():
        geo_map_path = Path(cfg.project_root) / geo_map_path
    return geo_map_path


def make_geo_stratified_folds(well_ids, cfg, log, *, split_seed=None, repeat=None):
    geo_map_path = resolve_cv_geo_map_path(cfg)
    if not geo_map_path.exists():
        raise FileNotFoundError(f"geo CV map not found: {geo_map_path}")

    required_cols = {"well_id", "horizontal_avg_X", "horizontal_avg_Y"}
    geo_df = pd.read_csv(geo_map_path, dtype={"well_id": str})
    missing_cols = sorted(required_cols - set(geo_df.columns))
    if missing_cols:
        raise ValueError(f"geo CV map {geo_map_path} missing columns: {missing_cols}")
    if geo_df["well_id"].duplicated().any():
        duplicated = sorted(geo_df.loc[geo_df["well_id"].duplicated(), "well_id"].unique().tolist())
        raise ValueError(f"geo CV map {geo_map_path} has duplicate well_id rows: {duplicated[:10]}")

    well_ids = np.asarray(well_ids, dtype=str)
    split_df = pd.DataFrame({"well_id": well_ids})
    split_df = split_df.merge(
        geo_df[["well_id", "horizontal_avg_X", "horizontal_avg_Y"]],
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    missing_wells = split_df.loc[split_df[["horizontal_avg_X", "horizontal_avg_Y"]].isna().any(axis=1), "well_id"]
    if len(missing_wells) > 0:
        raise ValueError(
            f"geo CV map {geo_map_path} missing centroid rows for wells: "
            f"{missing_wells.head(10).tolist()}"
        )

    xy = split_df[["horizontal_avg_X", "horizontal_avg_Y"]].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError(f"geo CV map {geo_map_path} contains non-finite horizontal centroids")

    fold_count = min(int(cfg.fold_count), len(well_ids))
    n_clusters = min(int(getattr(cfg, "cv_kmeans_clusters", 20)), len(well_ids))
    if fold_count < 2:
        return [(np.arange(len(well_ids)), np.array([], dtype=int))]
    if n_clusters < 2:
        raise ValueError(f"cv_kmeans_clusters must be at least 2 for geo_skfold, got {n_clusters}")

    kmeans_seed = int(getattr(cfg, "cv_kmeans_seed", 0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_seed, n_init="auto").fit(xy)
    cluster = kmeans.labels_.astype(int)
    counts = np.bincount(cluster, minlength=n_clusters)
    too_small = np.flatnonzero(counts < fold_count)
    if len(too_small) > 0:
        raise ValueError(
            "geo_skfold requires every KMeans cluster to have at least one member per fold; "
            f"fold_count={fold_count}, undersized clusters="
            f"{[(int(c), int(counts[c])) for c in too_small[:10]]}"
        )

    if split_seed is None:
        split_seed = resolve_cv_split_seed(cfg)
    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=split_seed)
    splits = list(splitter.split(np.arange(len(well_ids)), cluster))
    repeat_text = "" if repeat is None else f"repeat={repeat}, "
    log(
        "geo_skfold: "
        f"{repeat_text}map={geo_map_path}, clusters={n_clusters}, "
        f"kmeans_seed={kmeans_seed}, split_seed={split_seed}, "
        f"cluster_counts={counts.tolist()}"
    )
    return splits


def make_cv_splits(well_ids, cfg, log):
    well_ids = np.asarray(well_ids, dtype=str)
    unique_groups = np.unique(well_ids)
    if len(unique_groups) < 2:
        return [(np.arange(len(well_ids)), np.array([], dtype=int))]

    split_mode = str(getattr(cfg, "cv_split_mode", "kfold"))
    base_split_seed = resolve_cv_split_seed(cfg)
    cv_repeats = resolve_cv_repeats(cfg)
    splits = []
    for repeat in range(cv_repeats):
        split_seed = base_split_seed + repeat
        repeat_arg = None if cv_repeats == 1 else repeat
        if split_mode == "kfold":
            if cv_repeats == 1:
                log(f"kfold: split_seed={split_seed}")
            else:
                log(f"kfold: repeat={repeat}, split_seed={split_seed}")
            repeat_splits = make_group_folds(
                groups=well_ids,
                fold_count=min(cfg.fold_count, len(unique_groups)),
                seed=split_seed,
            )
        elif split_mode == "geo_skfold":
            repeat_splits = make_geo_stratified_folds(
                well_ids=well_ids,
                cfg=cfg,
                log=log,
                split_seed=split_seed,
                repeat=repeat_arg,
            )
        else:
            raise ValueError(f"unknown cv_split_mode={split_mode!r}; expected 'kfold' or 'geo_skfold'")
        splits.extend(repeat_splits)
    if cv_repeats > 1:
        log(
            f"cv_repeats={cv_repeats}: generated {len(splits)} validation folds "
            f"from base_split_seed={base_split_seed}"
        )
    return splits


def optimizer_betas(cfg):
    betas = tuple(float(x) for x in getattr(cfg, "optimizer_betas", (0.9, 0.999)))
    if len(betas) != 2:
        raise ValueError(f"optimizer_betas must have two values: {betas}")
    return betas


def optimizer_eps(cfg):
    return float(getattr(cfg, "optimizer_eps", 1e-8))


def _optimizer_cfg(cfg):
    group_cfg = getattr(cfg, "optimizer_cfg", None)
    if group_cfg is None:
        raise ValueError("missing optimizer_cfg")
    required = {"optimizer", "lr", "min_lr", "weight_decay", "scheduler"}
    missing = sorted(required - set(group_cfg))
    if missing:
        raise ValueError(f"optimizer_cfg missing keys: {missing}")
    return group_cfg


def _named_model_params(model):
    unet_model = source_model_for_train_ops(model)
    if not isinstance(unet_model, SeqUNet2DModel):
        raise TypeError(f"expected SeqUNet2DModel, got {type(unet_model).__name__}")
    if hasattr(unet_model, "model_named_parameters"):
        return list(unet_model.model_named_parameters())
    return list(unet_model.named_parameters())


def _muon_param_groups_from_named_params(named_params):
    muon_params = []
    adamw_params = []
    for _, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    if len(muon_params) == 0:
        raise ValueError("optimizer='muon' found no ndim>=2 parameters for Muon updates")
    groups = [{"params": muon_params, "use_muon": True}]
    if len(adamw_params) > 0:
        groups.append({"params": adamw_params, "use_muon": False})
    return groups


def _make_optimizer(model, cfg, lr_scale=1.0):
    optimizer_cfg = _optimizer_cfg(cfg)
    name = str(optimizer_cfg["optimizer"]).lower()
    named_params = _named_model_params(model)
    params = [param for _, param in named_params if param.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    common_kwargs = {
        "lr": float(optimizer_cfg["lr"]) * float(lr_scale),
        "weight_decay": float(optimizer_cfg["weight_decay"]),
        "betas": optimizer_betas(cfg),
        "eps": optimizer_eps(cfg),
    }
    if name == "adamw":
        return torch.optim.AdamW(params, **common_kwargs)
    if name == "radam":
        radam_kwargs = dict(common_kwargs)
        if "decoupled_weight_decay" in inspect.signature(torch.optim.RAdam).parameters:
            radam_kwargs["decoupled_weight_decay"] = True
        return torch.optim.RAdam(params, **radam_kwargs)
    if name == "muon":
        return MuonWithAdamW(
            _muon_param_groups_from_named_params(named_params),
            **common_kwargs,
            muon_momentum=float(getattr(cfg, "muon_momentum", 0.95)),
            muon_ns_steps=int(getattr(cfg, "muon_ns_steps", 5)),
            muon_nesterov=bool(getattr(cfg, "muon_nesterov", True)),
        )
    raise ValueError(f"unknown optimizer: {name!r}; expected adamw, radam, or muon")


def _make_scheduler(
    optimizer,
    cfg,
    steps_per_epoch,
    *,
    epochs=None,
    lr_scale=1.0,
):
    optimizer_cfg = _optimizer_cfg(cfg)
    scheduler_name = str(optimizer_cfg["scheduler"]).lower()
    total_epochs = int(cfg.epochs) if epochs is None else int(epochs)
    active_epochs = max(1, total_epochs)
    max_lr = float(optimizer_cfg["lr"]) * float(lr_scale)
    min_lr = float(optimizer_cfg["min_lr"]) * float(lr_scale)
    constant_ratio = float(optimizer_cfg.get("constant_ratio", 0.0))
    if not 0.0 <= constant_ratio < 1.0:
        raise ValueError(f"optimizer constant_ratio must be in [0, 1), got {constant_ratio}")
    if constant_ratio > 0.0 and scheduler_name not in {"cos", "cosine"}:
        raise ValueError(f"optimizer constant_ratio is only supported for cos scheduler, got {scheduler_name!r}")
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=active_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.onecycle_pct_start,
            div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.onecycle_final_div_factor,
        )
    if scheduler_name in {"cos", "cosine"}:
        if constant_ratio > 0.0:
            total_steps = max(1, active_epochs * steps_per_epoch)
            constant_steps = min(total_steps - 1, int(round(total_steps * constant_ratio)))
            cosine_steps = max(1, total_steps - constant_steps)
            min_factor = min_lr / max_lr

            def lr_lambda(step):
                if step <= constant_steps:
                    return 1.0
                progress = min(1.0, (step - constant_steps) / cosine_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=active_epochs * steps_per_epoch,
            eta_min=min_lr,
        )
    raise ValueError(f"unknown scheduler: {scheduler_name!r}; expected cos or onecycle")


def make_train_ops(model, cfg, steps_per_epoch, *, epochs=None, lr_scale=1.0):
    steps_per_epoch = max(1, int(steps_per_epoch))
    optimizer = _make_optimizer(model, cfg, lr_scale=lr_scale)
    scheduler = _make_scheduler(
        optimizer,
        cfg,
        steps_per_epoch,
        epochs=epochs,
        lr_scale=lr_scale,
    )
    return TrainOps(optimizer=optimizer, scheduler=scheduler)


def unet_loss_weights(cfg):
    expected_keys = (
        "regression",
        "offset",
        "alignment",
        "GR_penalty",
        "dS_penalty",
        "unet_GR_RMSE",
    )
    raw_weights = getattr(cfg, "unet_loss_weights", {})
    unknown_keys = set(raw_weights) - set(expected_keys)
    if unknown_keys:
        raise ValueError(f"unknown unet loss weights: {sorted(unknown_keys)}")
    weights = {key: float(raw_weights.get(key, 0.0)) for key in expected_keys}
    negative = {key: weight for key, weight in weights.items() if weight < 0.0}
    if negative:
        raise ValueError(f"unet loss weights must be non-negative: {negative}")
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one unet loss weight must be positive")
    return weights


def active_multitask_loss_weights(cfg):
    return {key: weight for key, weight in unet_loss_weights(cfg).items() if weight > 0.0}


def model_down_weight_wells(cfg):
    return {str(well_id) for well_id in getattr(cfg, "model_down_weight_wells", [])}


def model_down_weight(cfg):
    weight = float(getattr(cfg, "model_down_weight", 0.0))
    if weight < 0.0 or weight > 1.0:
        raise ValueError(f"model_down_weight must be in [0, 1], got {weight}")
    return weight


def batch_model_loss_weight(batch_meta, target_mask, cfg):
    down_wells = model_down_weight_wells(cfg)
    sample_weights = None
    if down_wells:
        down_weight = model_down_weight(cfg)
        if down_weight != 1.0:
            sample_weights = np.asarray(
                [
                    down_weight if str(meta.get("well_id")) in down_wells else 1.0
                    for meta in batch_meta
                ],
                dtype=np.float32,
            )

    if bool(getattr(cfg, "strech_to_full_size", False)) and bool(
        getattr(cfg, "strech_to_full_size_match_seq_weight", False)
    ):
        target_len = float(cfg.target_len)
        original_lengths = np.asarray(
            [float(meta["suffix_len"]) for meta in batch_meta],
            dtype=np.float32,
        )
        sequence_weights = original_lengths / np.float32(target_len)
        sample_weights = (
            sequence_weights
            if sample_weights is None
            else sample_weights * sequence_weights
        )

    if sample_weights is None:
        return None
    weight = torch.as_tensor(
        sample_weights,
        device=target_mask.device,
        dtype=torch.float32,
    )[:, None]
    return weight.expand_as(target_mask.to(dtype=torch.float32))


def model_min_seq_weight(cfg):
    min_seq_weight = float(getattr(cfg, "min_seq_weight", 1.0))
    if min_seq_weight < 0.0 or min_seq_weight > 1.0:
        raise ValueError(f"min_seq_weight must be in [0, 1], got {min_seq_weight}")
    return min_seq_weight


def model_seq_noise_smooth(cfg):
    smooth = float(getattr(cfg, "seq_noise_smooth", 1.0))
    if smooth <= 0.0:
        raise ValueError(f"seq_noise_smooth must be positive, got {smooth}")
    return smooth


def model_seq_noise_power(cfg):
    power = float(getattr(cfg, "seq_noise_power", 1.0))
    if power < 0.0:
        raise ValueError(f"seq_noise_power must be non-negative, got {power}")
    return power


def model_seq_noise_weight_enabled(cfg):
    return bool(getattr(cfg, "seq_noise_weight", False))


def model_sequence_loss_weight(pred, target, target_mask, cfg):
    loss_weight = None
    min_seq_weight = model_min_seq_weight(cfg)
    target_mask = target_mask.to(device=pred.device, dtype=torch.bool)

    if min_seq_weight != 1.0:
        positions = target_mask.cumsum(dim=1).to(device=pred.device, dtype=torch.float32) - 1.0
        suffix_len = target_mask.sum(dim=1, keepdim=True).to(device=pred.device, dtype=torch.float32).clamp_min(1.0)
        ratio = positions / suffix_len
        suffix_progress_weight = (1.0 - ratio) + ratio * min_seq_weight
        loss_weight = torch.where(target_mask, suffix_progress_weight, torch.zeros_like(suffix_progress_weight))

    if model_seq_noise_weight_enabled(cfg):
        target_std = float(cfg.target_stats[UNET_MODE][1])
        if target_std <= 0.0:
            raise ValueError(f"unet target std must be positive, got {target_std}")
        mask_float = target_mask.to(device=pred.device, dtype=torch.float32)
        seq_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        seq_mse = ((pred - target).float().square() * mask_float).sum(dim=1, keepdim=True) / seq_count
        seq_rmse = seq_mse.sqrt().mul(target_std).detach()
        seq_noise_weight = (model_seq_noise_smooth(cfg) + seq_rmse).pow(-model_seq_noise_power(cfg))
        seq_noise_weight = torch.where(target_mask, seq_noise_weight, torch.zeros_like(mask_float))
        loss_weight = seq_noise_weight if loss_weight is None else loss_weight * seq_noise_weight

    return loss_weight


def _weighted_masked_mean(values, target_mask, loss_weight, loss_scale_weight=None):
    values = values[target_mask]
    if values.numel() == 0:
        return values.sum() * 0.0
    scale_weight = None
    if loss_scale_weight is not None:
        scale_weight = loss_scale_weight[target_mask].to(dtype=values.dtype)
    if loss_weight is None:
        if scale_weight is None:
            return values.mean()
        return (values * scale_weight).sum() / values.new_tensor(values.numel())
    weight = loss_weight[target_mask].to(dtype=values.dtype)
    weight_sum = weight.sum()
    if weight_sum <= 0.0:
        return values.sum() * 0.0
    weighted_values = values * weight
    if scale_weight is not None:
        weighted_values = weighted_values * scale_weight
    return weighted_values.sum() / weight_sum


def masked_regression_loss_values(
    pred,
    target,
    target_mask,
    loss_name,
    cfg,
    loss_weight=None,
    loss_scale_weight=None,
):
    if loss_weight is None and loss_scale_weight is None:
        diff = (pred[target_mask] - target[target_mask]).float()
        if diff.numel() == 0:
            return pred.float().sum() * 0.0
    else:
        diff = (pred - target).float()
    if loss_name == "mse":
        values = diff.square()
        return (
            values.mean()
            if loss_weight is None and loss_scale_weight is None
            else _weighted_masked_mean(values, target_mask, loss_weight, loss_scale_weight)
        )
    if loss_name == "mae":
        values = diff.abs()
        return (
            values.mean()
            if loss_weight is None and loss_scale_weight is None
            else _weighted_masked_mean(values, target_mask, loss_weight, loss_scale_weight)
        )
    if loss_name == "huber":
        delta = float(cfg.huber_delta)
        if delta <= 0.0:
            raise ValueError(f"huber_delta must be positive: {delta}")
        abs_diff = diff.abs()
        quadratic = torch.minimum(abs_diff, torch.full_like(abs_diff, delta))
        values = 0.5 * quadratic.square() + delta * (abs_diff - quadratic)
        return (
            values.mean()
            if loss_weight is None and loss_scale_weight is None
            else _weighted_masked_mean(values, target_mask, loss_weight, loss_scale_weight)
        )
    raise ValueError(f"unknown regression loss: {loss_name}")


def dS_loss_windows_weights(cfg):
    windows = tuple(int(window) for window in getattr(cfg, "dS_windows", (1,)))
    weights = tuple(float(weight) for weight in getattr(cfg, "dS_weights", (1.0,)))
    if len(windows) == 0:
        raise ValueError("dS_windows must contain at least one window")
    if len(windows) != len(weights):
        raise ValueError(f"dS_windows and dS_weights length mismatch: {windows} vs {weights}")
    if any(window <= 0 for window in windows):
        raise ValueError(f"dS_windows must be positive: {windows}")
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"dS_weights must be non-negative: {weights}")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError(f"at least one dS_weight must be positive: {weights}")
    weights = tuple(weight / weight_sum for weight in weights)
    return windows, weights


def _centered_window_sum(values, window):
    if window == 1:
        return values
    left = (window - 1) // 2
    right = window - 1 - left
    values = F.pad(values[:, None, :], (left, right))
    kernel = torch.ones(
        (1, 1, window),
        device=values.device,
        dtype=values.dtype,
    )
    return F.conv1d(values, kernel).squeeze(1)


def _masked_centered_window_avg(values, target_mask, window, bin_count):
    values = values.float()
    target_mask_float = target_mask.to(device=values.device, dtype=values.dtype)
    if bin_count is None:
        sample_weight = target_mask_float
    else:
        sample_weight = target_mask_float * bin_count.to(device=values.device, dtype=values.dtype).clamp_min(0.0)

    numerator = _centered_window_sum(values * sample_weight, window)
    denominator = _centered_window_sum(sample_weight, window)
    window_mask = target_mask & (denominator > 0.0)
    return numerator / denominator.clamp_min(1.0), window_mask


def masked_multiscale_dS_huber_loss(
    pred,
    target,
    target_mask,
    bin_count,
    cfg,
    loss_weight=None,
    loss_scale_weight=None,
):
    windows, weights = dS_loss_windows_weights(cfg)
    if windows == (1,):
        return masked_regression_loss_values(
            pred,
            target,
            target_mask,
            "huber",
            cfg,
            loss_weight=loss_weight,
            loss_scale_weight=loss_scale_weight,
        )

    loss = None
    for window, weight in zip(windows, weights):
        if weight == 0.0:
            continue
        pred_avg, window_mask = _masked_centered_window_avg(pred, target_mask, window, bin_count)
        target_avg, _ = _masked_centered_window_avg(target, target_mask, window, bin_count)
        window_loss = masked_regression_loss_values(
            pred_avg,
            target_avg,
            window_mask,
            "huber",
            cfg,
            loss_weight=loss_weight,
            loss_scale_weight=loss_scale_weight,
        )
        weighted = weight * window_loss
        loss = weighted if loss is None else loss + weighted
    if loss is None:
        return pred.float().sum() * 0.0
    return loss


def masked_sequence_gradient_mae(pred, target, target_mask, loss_weight=None, loss_scale_weight=None):
    left_valid = torch.zeros_like(target_mask)
    right_valid = torch.zeros_like(target_mask)
    left_valid[:, 1:] = target_mask[:, :-1]
    right_valid[:, :-1] = target_mask[:, 1:]
    grad_mask = target_mask & left_valid & right_valid
    if not grad_mask.any():
        return pred.float().sum() * 0.0
    pred_grad = torch.gradient(pred.float(), dim=1)[0]
    target_grad = torch.gradient(target.float(), dim=1)[0]
    values = (pred_grad - target_grad).abs()
    return _weighted_masked_mean(values, grad_mask, loss_weight, loss_scale_weight)


def masked_typewell_soft_cross_entropy(logits, target_probs, target_mask, loss_weight=None, loss_scale_weight=None):
    log_prob = F.log_softmax(logits.float(), dim=-1)
    per_bin_loss = -(target_probs.float() * log_prob).sum(dim=-1)
    return _weighted_masked_mean(per_bin_loss, target_mask, loss_weight, loss_scale_weight)


def masked_gr_penalty_loss(logits, gr_error, target_mask, loss_weight=None, loss_scale_weight=None):
    prob = F.softmax(logits.float(), dim=-1)
    per_bin_loss = (prob * gr_error.float()).sum(dim=-1)
    return _weighted_masked_mean(per_bin_loss, target_mask, loss_weight, loss_scale_weight)


def masked_surface_curvature_l1_loss(pred, z_rel, target_mask, cfg, loss_weight=None, loss_scale_weight=None):
    if z_rel is None:
        raise ValueError("dS_penalty requires z_rel")
    target_mean, target_std = cfg.target_stats[UNET_MODE]
    surface_mean, surface_std = cfg.surface_stats["S_rel"]
    target_std = float(target_std)
    surface_std = float(surface_std)
    if target_std <= 0.0 or surface_std <= 0.0:
        raise ValueError(f"target/surface std must be positive, got {target_std=} {surface_std=}")

    pred = pred.float()
    z_rel = z_rel.to(device=pred.device, dtype=pred.dtype)
    target_mask = target_mask.to(device=pred.device, dtype=torch.bool)
    tvt_rel = pred * target_std + float(target_mean)
    surface = (tvt_rel + z_rel - float(surface_mean)) / surface_std
    finite_mask = target_mask & torch.isfinite(surface)
    curvature_mask = finite_mask[:, :-2] & finite_mask[:, 1:-1] & finite_mask[:, 2:]
    if not curvature_mask.any():
        return pred.sum() * 0.0

    curvature = surface[:, 2:] - 2.0 * surface[:, 1:-1] + surface[:, :-2]
    curvature_weight = None if loss_weight is None else loss_weight[:, 1:-1]
    curvature_scale_weight = None if loss_scale_weight is None else loss_scale_weight[:, 1:-1]
    return _weighted_masked_mean(
        curvature.abs(),
        curvature_mask,
        curvature_weight,
        curvature_scale_weight,
    )


def _huber_values(diff, delta):
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.full_like(abs_diff, delta))
    return 0.5 * quadratic.square() + delta * (abs_diff - quadratic)


def masked_offset_huber_loss(
    logits,
    offset_pred,
    target,
    target_mask,
    cfg,
    loss_weight=None,
    loss_scale_weight=None,
    detach_logits=True,
):
    logits_for_prob = logits.detach() if detach_logits else logits
    prob = F.softmax(logits_for_prob.float(), dim=-1)
    axis_raw = torch.linspace(
        -float(cfg.typewell_window) + float(cfg.typewell_window) / float(cfg.typewell_len),
        float(cfg.typewell_window) - float(cfg.typewell_window) / float(cfg.typewell_len),
        int(cfg.typewell_len),
        device=prob.device,
        dtype=prob.dtype,
    )
    target_mean, target_std = cfg.target_stats[UNET_MODE]
    tvt_axis = ((axis_raw - float(target_mean)) / float(target_std)).reshape(1, 1, -1)
    target = target.to(device=prob.device, dtype=prob.dtype).unsqueeze(-1)
    diff = (tvt_axis + offset_pred.float()) - target
    values = _huber_values(diff, float(cfg.huber_delta))
    per_bin_loss = (prob * values).sum(dim=-1)
    return _weighted_masked_mean(per_bin_loss, target_mask, loss_weight, loss_scale_weight)


def unet_loss(
    pred,
    target,
    target_mask,
    typewell_target_probs,
    aux_targets,
    extra,
    cfg,
    return_details=False,
    bin_count=None,
    z_rel=None,
    loss_scale_weight=None,
):
    del bin_count
    if extra is None:
        raise ValueError("unet requires model extra_return=True")
    weights = unet_loss_weights(cfg)
    if (weights["alignment"] != 0.0 or weights["GR_penalty"] != 0.0) and "alignment_logits" not in extra:
        raise ValueError("unet requires extra['alignment_logits']")
    if weights["offset"] != 0.0 and ("offset_pred" not in extra or "regression_logits" not in extra):
        raise ValueError("unet requires extra['offset_pred'] and extra['regression_logits']")
    if weights["unet_GR_RMSE"] != 0.0 and "unet_GR_RMSE_pred" not in extra:
        raise ValueError("unet requires extra['unet_GR_RMSE_pred']")
    loss = None
    details = {}
    sequence_weight = None
    if (
        weights["regression"] != 0.0
        or weights["offset"] != 0.0
        or weights["alignment"] != 0.0
        or weights["GR_penalty"] != 0.0
        or weights["dS_penalty"] != 0.0
        or weights["unet_GR_RMSE"] != 0.0
    ):
        sequence_weight = model_sequence_loss_weight(pred, target, target_mask, cfg)

    if weights["regression"] != 0.0:
        regression_loss = masked_regression_loss_values(
            pred,
            target,
            target_mask,
            "huber",
            cfg,
            loss_weight=sequence_weight,
            loss_scale_weight=loss_scale_weight,
        )
        weighted = weights["regression"] * regression_loss
        loss = weighted if loss is None else loss + weighted
        if return_details:
            details["regression"] = regression_loss

    if weights["offset"] != 0.0:
        offset_loss = masked_offset_huber_loss(
            extra["regression_logits"],
            extra["offset_pred"],
            target,
            target_mask,
            cfg,
            loss_weight=sequence_weight,
            loss_scale_weight=loss_scale_weight,
            detach_logits=bool(getattr(cfg, "model_cfg", cfg.unet_cfg).get("offset_detach_logits_head_in_loss", True)),
        )
        weighted = weights["offset"] * offset_loss
        loss = weighted if loss is None else loss + weighted
        if return_details:
            details["offset"] = offset_loss

    if weights["alignment"] != 0.0:
        alignment_loss = masked_typewell_soft_cross_entropy(
            extra["alignment_logits"],
            typewell_target_probs,
            target_mask,
            loss_weight=sequence_weight,
            loss_scale_weight=loss_scale_weight,
        )
        weighted = weights["alignment"] * alignment_loss
        loss = weighted if loss is None else loss + weighted
        if return_details:
            details["alignment"] = alignment_loss

    if weights["GR_penalty"] != 0.0:
        if "GR_penalty_error" not in aux_targets:
            raise ValueError("unet requires aux target 'GR_penalty_error'")
        gr_penalty_loss = masked_gr_penalty_loss(
            extra["alignment_logits"],
            aux_targets["GR_penalty_error"],
            target_mask,
            loss_weight=sequence_weight,
            loss_scale_weight=loss_scale_weight,
        )
        weighted = weights["GR_penalty"] * gr_penalty_loss
        loss = weighted if loss is None else loss + weighted
        if return_details:
            details["GR_penalty"] = gr_penalty_loss

    if weights["dS_penalty"] != 0.0:
        ds_penalty_loss = masked_surface_curvature_l1_loss(
            pred,
            z_rel,
            target_mask,
            cfg,
            loss_weight=sequence_weight,
            loss_scale_weight=loss_scale_weight,
        )
        weighted = weights["dS_penalty"] * ds_penalty_loss
        loss = weighted if loss is None else loss + weighted
        if return_details:
            details["dS_penalty"] = ds_penalty_loss

    if weights["unet_GR_RMSE"] != 0.0:
        if "unet_GR_RMSE" not in aux_targets or "unet_GR_RMSE_mask" not in aux_targets:
            raise ValueError("unet requires aux targets 'unet_GR_RMSE' and 'unet_GR_RMSE_mask'")
        gr_rmse_mask = target_mask & aux_targets["unet_GR_RMSE_mask"].to(device=target_mask.device, dtype=torch.bool)
        gr_rmse_loss = masked_regression_loss_values(
            extra["unet_GR_RMSE_pred"],
            aux_targets["unet_GR_RMSE"],
            gr_rmse_mask,
            "huber",
            cfg,
            loss_weight=sequence_weight,
            loss_scale_weight=loss_scale_weight,
        )
        weighted = weights["unet_GR_RMSE"] * gr_rmse_loss
        loss = weighted if loss is None else loss + weighted
        if return_details:
            details["unet_GR_RMSE"] = gr_rmse_loss

    if loss is None:
        raise ValueError("unet has no active loss components")
    if return_details:
        return loss, details
    return loss


def validate_seq_target_support(cfg):
    model_name = getattr(cfg, "model_name", None)
    if model_name not in MODEL_MODES:
        raise ValueError(f"sequence NN model_name must be one of {MODEL_MODES}, got {model_name!r}")
    if getattr(cfg, "seq_target_mode", None) != UNET_MODE:
        raise ValueError("sequence NN now supports only seq_target_mode='unet'")
    tw_gr_grid_type = str(getattr(cfg, "tw_gr_grid_type", "interpolate")).lower()
    if tw_gr_grid_type not in TYPEWELL_GR_GRID_TYPES:
        raise ValueError(
            f"tw_gr_grid_type must be one of {TYPEWELL_GR_GRID_TYPES}, got {tw_gr_grid_type!r}"
        )
    unet_loss_weights(cfg)
    model_cfg = getattr(cfg, "model_cfg", cfg.unet_cfg)
    regression_head_mode = str(model_cfg.get("regression_head_mode", "logits"))
    if regression_head_mode not in {"logits", "prob+offset", "pool_mlp"}:
        raise ValueError(
            f"unknown regression_head_mode={regression_head_mode!r}; expected 'logits', 'prob+offset', or 'pool_mlp'"
        )
    share_logits_head = bool(model_cfg.get("share_logits_head", True))
    if share_logits_head and regression_head_mode not in {"logits", "prob+offset"}:
        raise ValueError(
            "share_logits_head=True requires regression_head_mode='logits' or 'prob+offset'"
        )
    resblock_act = str(model_cfg.get("resblock_act", "silu"))
    if resblock_act not in {"silu", "relu"}:
        raise ValueError(
            f"resblock_act must be 'silu' or 'relu', got {resblock_act!r}"
        )
    resblock_norm = str(model_cfg.get("resblock_norm", "BN"))
    if resblock_norm not in {"BN", "IN", "LN"}:
        raise ValueError(
            f"resblock_norm must be 'BN', 'IN', or 'LN', got {resblock_norm!r}"
        )
    convnext_norm_replace = model_cfg.get("unet_convnext_norm_replace", "BN")
    if (
        convnext_norm_replace is not None
        and convnext_norm_replace not in {"BN", "IN", "LN"}
    ):
        raise ValueError(
            "unet_convnext_norm_replace must be 'BN', 'IN', 'LN', or None, "
            f"got {convnext_norm_replace!r}"
        )
    if float(getattr(cfg, "unet_loss_weights", {}).get("offset", 0.0)) > 0.0 and regression_head_mode != "prob+offset":
        raise ValueError("unet_loss_weights['offset'] requires regression_head_mode='prob+offset'")
    if model_name == TWO_STAGE_UNET_MODE:
        channel_names = tuple(cfg.unet_static_channels)
        channel_set = set(channel_names)
        for group_name in ("tw_channels", "h_channels", "shared_channels"):
            group = tuple(model_cfg.get(group_name, ()))
            if len(group) == 0 and group_name != "shared_channels":
                raise ValueError(f"two_stage_unet {group_name} must not be empty")
            missing = sorted(set(group) - channel_set)
            if missing:
                raise ValueError(f"two_stage_unet {group_name} contains unknown channels={missing}")
        encoder_emb_dim = int(model_cfg.get("encoder_emb_dim", 64))
        corr_matrix_dim = int(model_cfg.get("corr_matrix_dim", 8))
        if encoder_emb_dim <= 0:
            raise ValueError(f"two_stage_unet encoder_emb_dim must be positive, got {encoder_emb_dim}")
        if int(model_cfg.get("encoder_stages", 4)) <= 0:
            raise ValueError(f"two_stage_unet encoder_stages must be positive")
        if corr_matrix_dim <= 0:
            raise ValueError(f"two_stage_unet corr_matrix_dim must be positive, got {corr_matrix_dim}")
        if encoder_emb_dim % corr_matrix_dim != 0:
            raise ValueError(
                "two_stage_unet encoder_emb_dim must be divisible by corr_matrix_dim: "
                f"{encoder_emb_dim} % {corr_matrix_dim} != 0"
            )
    geo_cfg = resolve_geo_prior_cfg(cfg)
    if geo_cfg.method not in {"idw_dS_xy", "idw_dS_xy_wls"}:
        raise ValueError(f"unsupported geo_prior_cfg.method={geo_cfg.method!r}")
    model_min_seq_weight(cfg)
    model_seq_noise_smooth(cfg)
    model_seq_noise_power(cfg)
    model_down_weight(cfg)
    raw_wells = getattr(cfg, "model_down_weight_wells", [])
    if isinstance(raw_wells, (str, bytes)):
        raise ValueError("model_down_weight_wells must be a sequence of well ids, not a string")
    optimizer_cfg = _optimizer_cfg(cfg)
    optimizer_name = str(optimizer_cfg["optimizer"]).lower()
    if optimizer_name not in {"adamw", "radam", "muon"}:
        raise ValueError(f"unknown optimizer={optimizer_name!r}; expected adamw, radam, or muon")
    scheduler_name = str(optimizer_cfg["scheduler"]).lower()
    if scheduler_name not in {"cos", "cosine", "onecycle"}:
        raise ValueError(f"unknown scheduler={scheduler_name!r}; expected cos or onecycle")
    if float(optimizer_cfg["lr"]) <= 0.0:
        raise ValueError(f"optimizer lr must be positive: {optimizer_cfg['lr']}")
    if float(optimizer_cfg["min_lr"]) < 0.0:
        raise ValueError(f"optimizer min_lr must be non-negative: {optimizer_cfg['min_lr']}")
    if float(optimizer_cfg["weight_decay"]) < 0.0:
        raise ValueError(f"optimizer weight_decay must be non-negative: {optimizer_cfg['weight_decay']}")
    grad_accum_steps = int(getattr(cfg, "grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive: {grad_accum_steps}")
    min_epochs = int(getattr(cfg, "min_epochs", 0))
    if min_epochs < 0:
        raise ValueError(f"min_epochs must be non-negative: {min_epochs}")
    finetune_epochs = getattr(cfg, "finetune_epochs", None)
    if finetune_epochs is not None:
        finetune_epochs = int(finetune_epochs)
        if finetune_epochs < 0:
            raise ValueError(f"finetune_epochs must be non-negative or None: {finetune_epochs}")
        finetune_log_freq = int(getattr(cfg, "finetune_log_freq", 1))
        if finetune_log_freq <= 0:
            raise ValueError(f"finetune_log_freq must be positive: {finetune_log_freq}")
        finetune_lr_reduce = float(getattr(cfg, "finetune_lr_reduce", 0.3))
        if finetune_lr_reduce <= 0.0:
            raise ValueError(f"finetune_lr_reduce must be positive: {finetune_lr_reduce}")


def needs_extra_return(cfg):
    del cfg
    return True


def prepare_pf_cache_if_needed(split, path, well_ids, cfg, log):
    if not uses_pf_heatmap_channels(cfg):
        return None
    if float(cfg.aug_cfg.get("start_point_shift", {}).get("apply_prob", 0.0)) > 0.0:
        log(
            "WARNING: PF heatmap channels assume start_point_shift stays disabled; "
            "z_shift/xy_shift/reverse_path/MD_streching/tail_cut are remapped from original cached path density"
        )
    return ensure_pf_heatmap_cache(split=split, data_path=path, well_ids=well_ids, cfg=cfg, log=log)


def prepare_pf_sample_cache_if_needed(split, path, well_ids, cfg, log):
    if not uses_pf_sample_trend(cfg):
        return None
    return ensure_pf_sample_cache(split=split, data_path=path, well_ids=well_ids, cfg=cfg, log=log)


def make_loader(
    path,
    well_ids,
    cfg,
    training,
    simulation,
    shuffle,
    batch_size,
    seed,
    pf_cache_dir=None,
    pf_sample_cache_dir=None,
    geo_prior=None,
    drop_last=False,
):
    validate_seq_target_support(cfg)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = {}
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 1))
        kwargs["persistent_workers"] = bool(getattr(cfg, "persistent_workers", True))
    return DataLoader(
        SeqUNetDataset(
            path=path,
            well_ids=well_ids,
            cfg=cfg,
            training=training,
            simulation=simulation,
            pf_cache_dir=pf_cache_dir,
            pf_sample_cache_dir=pf_sample_cache_dir,
            geo_prior=geo_prior,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.device(cfg.device).type == "cuda",
        collate_fn=collate_seq_batch,
        drop_last=bool(drop_last),
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


def make_model(cfg):
    validate_seq_target_support(cfg)
    model_cfg = copy.deepcopy(cfg.model_cfg)
    model_name = getattr(cfg, "model_name", UNET_MODE)
    target_mean, target_std = cfg.target_stats[UNET_MODE]
    tvt_rel_norm_mean, tvt_rel_norm_std = cfg.typewell_stats["tvt_rel"]
    if bool(getattr(cfg, "submit_mode", False)):
        unet_pretrained = False
    elif model_cfg.get("unet_pretrained", None) is None:
        unet_pretrained = True
    else:
        unet_pretrained = bool(model_cfg["unet_pretrained"])
    model_cfg.pop("enable_geo_tvt_rel_channel", None)
    model_cfg.pop("enable_geo_diff_channel", None)
    model_cfg.update(
        {
            "unet_pretrained": unet_pretrained,
            "typewell_window": float(cfg.typewell_window),
            "typewell_len": int(cfg.typewell_len),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "tvt_diff_clip": float(getattr(cfg, "tvt_diff_clip", 100.0)),
            "tvt_rel_norm_mean": float(tvt_rel_norm_mean),
            "tvt_rel_norm_std": float(tvt_rel_norm_std),
            "typewell_gr_mean": float(cfg.typewell_stats["gr"][0]),
            "typewell_gr_std": float(cfg.typewell_stats["gr"][1]),
        }
    )
    if model_name == UNET_MODE:
        return SeqUNet2DModel(
            unet_static_in_dim=len(cfg.unet_static_channels),
            **model_cfg,
        )
    if model_name == TWO_STAGE_UNET_MODE:
        model_cfg.setdefault("tw_channels", tuple(getattr(cfg, "two_stage_tw_channels", ())))
        model_cfg.setdefault("h_channels", tuple(getattr(cfg, "two_stage_h_channels", ())))
        model_cfg.setdefault("shared_channels", tuple(getattr(cfg, "two_stage_shared_channels", ())))
        model_cfg["static_channel_names"] = tuple(cfg.unet_static_channels)
        return SeqTwoStageUNetModel(
            unet_static_in_dim=len(cfg.unet_static_channels),
            **model_cfg,
        )
    raise ValueError(f"unknown model_name={model_name!r}")


def uses_channels_last_unet(cfg):
    return (
        bool(getattr(cfg, "channels_last_2d", False))
        and torch.device(cfg.device).type == "cuda"
        and not bool(getattr(cfg, "deterministic", False))
    )


def move_model_to_device(model, cfg):
    model = model.to(torch.device(cfg.device))
    if uses_channels_last_unet(cfg):
        source_model_for_train_ops(model).unet.to(memory_format=torch.channels_last)
    return model


def release_fold_cuda_cache(cfg, log, fold_name):
    gc.collect()
    device = torch.device(cfg.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    gib = float(1024**3)
    log(
        f"{fold_name}: CUDA cleanup "
        f"allocated={allocated_before / gib:.2f}->{allocated_after / gib:.2f} GiB, "
        f"reserved={reserved_before / gib:.2f}->{reserved_after / gib:.2f} GiB"
    )


def model_forward(
    model,
    unet_static,
    typewell_aux,
    z_rel=None,
    bin_count=None,
    z_diff=None,
    extra_return=False,
):
    return model(
        unet_static,
        typewell_aux,
        z_rel=z_rel,
        bin_count=bin_count,
        z_diff=z_diff,
        extra_return=extra_return,
    )


def predict_one_model(model, loader, cfg, move_cpu=True, return_meta=False):
    device = torch.device(cfg.device)
    amp_dtype = getattr(torch, cfg.amp_dtype)
    amp_enabled = cfg.amp_dtype != "float32" and device.type == "cuda"
    model = move_model_to_device(model, cfg)
    model.eval()
    preds = []
    metas = []
    with torch.no_grad():
        for batch in loader:
            unet_static = batch["unet_static"].to(device, non_blocking=True)
            typewell_aux = batch["typewell_aux"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model_forward(
                    model,
                    unet_static,
                    typewell_aux,
                    extra_return=False,
                )
            preds.append(pred.float().cpu().numpy())
            if return_meta:
                metas.extend(batch["meta"])
    if move_cpu:
        model.cpu()
    pred_arr = np.concatenate(preds, axis=0)
    if return_meta:
        return pred_arr, metas
    return pred_arr


def _expand_full_size_bin_values(bin_values, meta, cfg):
    """Map canonical suffix-bin values back to every original suffix row."""
    suffix_len = int(meta["suffix_len"])
    if suffix_len <= 0:
        return np.empty(0, dtype="float32")
    dest_pos = np.asarray(meta.get("stretch_suffix_dest_pos"), dtype=np.float32)
    if dest_pos.shape != (suffix_len,):
        raise ValueError(
            f"{meta['well_id']}: full-stretch inverse map length mismatch, "
            f"expected={suffix_len} got={dest_pos.shape}"
        )
    values = np.asarray(bin_values, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"{meta['well_id']}: full-stretch bin values must be a non-empty 1D array"
        )
    query = (np.float32(cfg.prefix_len) + dest_pos) / np.float32(cfg.downsample)
    source = np.arange(values.size, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.full(suffix_len, np.nan, dtype=np.float32)
    if not finite.any():
        return out
    finite_source = source[finite]
    finite_values = values[finite]
    valid_query = np.isfinite(query)
    if finite_values.size == 1:
        nearest = np.flatnonzero(valid_query)
        if nearest.size:
            nearest = nearest[np.argmin(np.abs(query[nearest] - finite_source[0]))]
            out[nearest] = finite_values[0]
        return out
    out[valid_query] = np.interp(
        query[valid_query],
        finite_source,
        finite_values,
    ).astype(np.float32)
    return out


def expand_target_prediction(bin_pred, meta, cfg):
    suffix_len = int(meta["suffix_len"])
    full_stretch = bool(getattr(cfg, "strech_to_full_size", False)) or meta.get(
        "stretch_to_full_size",
        False,
    )
    if full_stretch:
        mean, std = cfg.target_stats[UNET_MODE]
        return (
            _expand_full_size_bin_values(bin_pred, meta, cfg) * np.float32(std)
            + np.float32(mean)
        ).astype(np.float32)
    kept_len = min(suffix_len, cfg.target_len)
    if kept_len <= 0:
        return np.empty(0, dtype="float32")

    target_pred = np.empty(suffix_len, dtype="float32")
    mean, std = cfg.target_stats[UNET_MODE]
    for raw_idx in range(kept_len):
        bin_idx = (cfg.prefix_len + raw_idx) // cfg.downsample
        target_pred[raw_idx] = bin_pred[bin_idx] * std + mean
    if suffix_len > cfg.target_len:
        target_pred[kept_len:] = target_pred[kept_len - 1]
    return target_pred


def expand_bin_values(bin_values, meta, cfg):
    suffix_len = int(meta["suffix_len"])
    full_stretch = bool(getattr(cfg, "strech_to_full_size", False)) or meta.get(
        "stretch_to_full_size",
        False,
    )
    if full_stretch:
        return _expand_full_size_bin_values(bin_values, meta, cfg)
    kept_len = min(suffix_len, cfg.target_len)
    if kept_len <= 0:
        return np.empty(0, dtype="float32")

    out = np.empty(suffix_len, dtype="float32")
    bin_values = np.asarray(bin_values, dtype=np.float32)
    for raw_idx in range(kept_len):
        bin_idx = (cfg.prefix_len + raw_idx) // cfg.downsample
        out[raw_idx] = bin_values[bin_idx]
    if suffix_len > cfg.target_len:
        out[kept_len:] = out[kept_len - 1]
    return out


def target_value_to_tvt(target_value, meta, cfg):
    del cfg
    tvt0 = np.float32(meta["tvt0"])
    suffix_z = meta["suffix_z"].astype("float32", copy=False)
    if len(suffix_z) != len(target_value):
        raise ValueError(
            f"{meta['well_id']}: suffix Z length mismatch, z={len(suffix_z)} target={len(target_value)}"
        )
    return tvt0 + target_value


def _savgol_1d(values, window, polyorder):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or int(window) <= 1:
        return values.copy()
    use_window = int(window)
    if use_window % 2 == 0:
        use_window += 1
    if use_window > values.size:
        use_window = values.size if values.size % 2 == 1 else values.size - 1
    if use_window < 3 or use_window <= int(polyorder):
        return values.copy()
    return savgol_filter(
        values,
        window_length=use_window,
        polyorder=int(polyorder),
        mode="interp",
    )


def apply_pred_sg_smooth(df, cfg):
    if not getattr(cfg, "pred_sg_smooth", False):
        return df
    for col in ("well_id", "TVT_pred", "Z"):
        if col not in df.columns:
            raise KeyError(f"prediction dataframe missing {col!r} for pred_sg_smooth")

    out = df.copy()
    raw = out["TVT_pred"].to_numpy(dtype=np.float64, copy=True)
    z = out["Z"].to_numpy(dtype=np.float64, copy=False)
    candidate = raw.copy()
    window = int(getattr(cfg, "pred_sg_smooth_window", 1025))
    polyorder = int(getattr(cfg, "pred_sg_smooth_polyorder", 1))
    blend = float(getattr(cfg, "pred_sg_smooth_blend", 0.6828376753426914))
    for pos in out.groupby("well_id", sort=False).indices.values():
        pos = np.asarray(pos, dtype=np.int64)
        surface = raw[pos] + z[pos]
        candidate[pos] = _savgol_1d(surface, window=window, polyorder=polyorder) - z[pos]
    final = raw + blend * (candidate - raw)
    out["TVT_pred_raw"] = raw.astype("float32")
    out["TVT_sg_surface"] = candidate.astype("float32")
    out["pred_sg_smooth"] = final.astype("float32")
    out["TVT_pred"] = out["pred_sg_smooth"]
    return out


def make_prediction_df(pred_arr, metas, cfg, include_target, apply_smooth=False):
    rows = []
    mean, std = cfg.target_stats[UNET_MODE]
    for sample_idx, meta in enumerate(metas):
        pred = pred_arr[sample_idx]
        target_pred = expand_target_prediction(pred, meta, cfg)
        normalized_pred = (target_pred - mean) / std
        tvt_pred = target_value_to_tvt(target_pred, meta, cfg)
        submit_indices = meta["submit_indices"]
        if len(submit_indices) != len(tvt_pred):
            raise ValueError(
                f"{meta['well_id']}: prediction length mismatch, "
                f"indices={len(submit_indices)} pred={len(tvt_pred)}"
            )
        suffix_z = meta["suffix_z"].astype("float32", copy=False)
        if len(suffix_z) != len(tvt_pred):
            raise ValueError(
                f"{meta['well_id']}: suffix Z length mismatch, "
                f"indices={len(submit_indices)} z={len(suffix_z)} pred={len(tvt_pred)}"
            )
        sidecars = {}
        if "pf_tvt_rel_pred" in meta:
            pf_rel = expand_bin_values(meta["pf_tvt_rel_pred"], meta, cfg)
            pf_tvt = target_value_to_tvt(pf_rel, meta, cfg)
            if len(pf_tvt) != len(tvt_pred):
                raise ValueError(
                    f"{meta['well_id']}: PF sidecar length mismatch, "
                    f"indices={len(submit_indices)} pf={len(pf_tvt)}"
                )
            sidecars["PF_TVT"] = pf_tvt.astype("float32")
        if "geo_prior_TVT" in meta:
            geo_prior_tvt = np.asarray(meta["geo_prior_TVT"], dtype=np.float32)
            if len(geo_prior_tvt) != len(tvt_pred):
                raise ValueError(
                    f"{meta['well_id']}: geo prior sidecar length mismatch, "
                    f"indices={len(submit_indices)} geo={len(geo_prior_tvt)}"
                )
            sidecars["geo_prior_TVT"] = geo_prior_tvt.astype("float32", copy=False)
        item = pd.DataFrame(
            {
                "well_id": meta["well_id"],
                "submit_index": submit_indices,
                "Z": suffix_z,
                "TVT_pred": tvt_pred.astype("float32"),
                **sidecars,
            }
        )
        item["pred"] = normalized_pred.astype("float32")
        item["target_pred"] = target_pred.astype("float32")
        item["target_mode"] = UNET_MODE
        item["a"] = std
        item["b"] = meta["tvt0"] + mean
        item["target_a"] = std
        item["target_b"] = mean
        if include_target:
            item["TVT"] = meta["TVT"].astype("float32")
        rows.append(item)
    out = pd.concat(rows, ignore_index=True)
    if apply_smooth:
        out = apply_pred_sg_smooth(out, cfg)
    return out


def add_geo_prior_diagnostic_columns(df, geo_prior):
    """Attach inference-safe original-row geo diagnostics for post-training analysis."""

    out = df.copy()
    for name in POST_TRAIN_GEO_DIAGNOSTIC_NAMES:
        out[name] = np.full(len(out), np.nan, dtype=np.float32)
    for well_id, index in out.groupby("well_id", sort=False).groups.items():
        prior_item = geo_prior[str(well_id)]
        submit_index = out.loc[index, "submit_index"].to_numpy(dtype=np.int64)
        for name in POST_TRAIN_GEO_DIAGNOSTIC_NAMES:
            if isinstance(prior_item, dict):
                values = np.asarray(prior_item[name], dtype=np.float32)
            else:
                values = np.asarray(getattr(prior_item, name), dtype=np.float32)
            out.loc[index, name] = values[submit_index]
    for name in POST_TRAIN_GEO_DIAGNOSTIC_NAMES:
        out[name] = out[name].astype(np.float32)
    return out


def add_geo_prior_well_summary_columns(df):
    """Repeat suffix-level geo support summaries on each well's output rows."""

    out = df.copy()
    out["geo_nbr_distance_q10"] = (
        out.groupby("well_id", sort=False, observed=True)["geo_nbr_distance"]
        .transform("quantile", q=0.10)
        .astype(np.float32)
    )
    return out


def lookup_typewell_gr(data_path, well_id, tvt_values):
    typewell_df = pd.read_csv(Path(data_path) / f"{well_id}__typewell.csv", usecols=["TVT", "GR"])
    order = np.argsort(typewell_df["TVT"].to_numpy(dtype=np.float64))
    typewell_tvt = typewell_df["TVT"].to_numpy(dtype=np.float64)[order]
    typewell_gr = typewell_df["GR"].to_numpy(dtype=np.float64)[order]
    tvt_values = np.asarray(tvt_values, dtype=np.float64)
    matched_gr = np.full(tvt_values.shape, np.nan, dtype=np.float32)
    valid = (
        np.isfinite(tvt_values)
        & (tvt_values >= typewell_tvt[0])
        & (tvt_values <= typewell_tvt[-1])
    )
    if valid.any():
        matched_gr[valid] = np.interp(tvt_values[valid], typewell_tvt, typewell_gr).astype(np.float32)
    return matched_gr


def add_typewell_matched_gr_columns(df, data_path, tvt_col="TVT_pred", gr_col="pred_lookup_GR"):
    out = df.copy()
    if out.empty:
        out[gr_col] = np.full(len(out), np.nan, dtype=np.float32)
        return out
    if tvt_col not in out.columns:
        raise KeyError(f"prediction dataframe missing {tvt_col!r}")
    column_pairs = [(tvt_col, gr_col)]
    if tvt_col == "TVT_pred" and gr_col == "pred_lookup_GR":
        for sidecar_tvt_col, sidecar_gr_col in (
            ("PF_TVT", "PF_lookup_GR"),
            ("geo_prior_TVT", "geo_prior_lookup_GR"),
        ):
            if sidecar_tvt_col in out.columns:
                column_pairs.append((sidecar_tvt_col, sidecar_gr_col))
    for _, output_col in column_pairs:
        out[output_col] = np.full(len(out), np.nan, dtype=np.float32)
    for well_id, index in out.groupby("well_id", sort=False).groups.items():
        for input_col, output_col in column_pairs:
            matched_gr = lookup_typewell_gr(
                data_path,
                well_id,
                out.loc[index, input_col].to_numpy(dtype=np.float64),
            )
            out.loc[index, output_col] = matched_gr
    for _, output_col in column_pairs:
        out[output_col] = out[output_col].astype("float32")
    return out


def repeated_oof_average_columns(df):
    base_columns = (
        "TVT_pred",
        "pred",
        "target_pred",
        "TVT_pred_raw",
        "TVT_sg_surface",
        "pred_sg_smooth",
        "PF_TVT",
        "geo_prior_TVT",
        *POST_TRAIN_GEO_DIAGNOSTIC_NAMES,
    )
    columns = [col for col in base_columns if col in df.columns]
    for col in df.columns:
        if col.endswith("_TVT") and col != "TVT" and col not in columns:
            columns.append(col)
    return columns


def average_repeated_oof_predictions(df, cfg, log=print):
    cv_repeats = resolve_cv_repeats(cfg)
    if cv_repeats == 1 or df.empty:
        return df

    key_cols = ["well_id", "submit_index"]
    missing_key_cols = [col for col in key_cols if col not in df.columns]
    if missing_key_cols:
        raise KeyError(f"OOF dataframe missing repeat-average key columns: {missing_key_cols}")

    repeat_counts = df.groupby(key_cols, sort=False).size()
    bad_counts = repeat_counts[repeat_counts != cv_repeats]
    if len(bad_counts) > 0:
        examples = bad_counts.head(10).to_dict()
        raise ValueError(
            f"expected every OOF row to have exactly cv_repeats={cv_repeats} predictions; "
            f"bad group count examples={examples}"
        )

    avg_cols = repeated_oof_average_columns(df)
    lookup_cols = [
        col
        for col in ("pred_lookup_GR", "PF_lookup_GR", "geo_prior_lookup_GR")
        if col in df.columns
    ]
    drop_for_first = set(avg_cols) | set(lookup_cols)
    first_cols = [col for col in df.columns if col not in drop_for_first]
    first_df = df[first_cols].groupby(key_cols, sort=False, as_index=False).first()
    avg_df = df[key_cols + avg_cols].groupby(key_cols, sort=False, as_index=False).mean()
    out = first_df.merge(avg_df, on=key_cols, how="left", validate="one_to_one")
    for col in avg_cols:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32")

    pre_lookup_order = [col for col in df.columns if col in out.columns and col not in lookup_cols]
    out = out[pre_lookup_order + [col for col in out.columns if col not in pre_lookup_order]]
    out = add_typewell_matched_gr_columns(out, cfg.train_path)

    final_order = [col for col in df.columns if col in out.columns]
    out = out[final_order + [col for col in out.columns if col not in final_order]]
    if log is not None:
        log(
            f"OOF repeat average: cv_repeats={cv_repeats}, "
            f"input_rows={len(df):,}, output_rows={len(out):,}, "
            f"averaged_columns={avg_cols}"
        )
    return out


def score_prediction_df(df, pred_col="TVT_pred"):
    if pred_col not in df.columns:
        raise KeyError(f"prediction dataframe missing {pred_col!r}")
    return np.sqrt(np.mean(np.square(df["TVT"].to_numpy(dtype=np.float64) - df[pred_col].to_numpy(dtype=np.float64))))


def well_rmse_summary(df, pred_col="TVT_pred"):
    if df.empty or "TVT" not in df.columns:
        return None
    if pred_col not in df.columns:
        raise KeyError(f"prediction dataframe missing {pred_col!r}")
    err2 = df.assign(_err2=(df["TVT"] - df[pred_col]) ** 2)
    values = np.sqrt(err2.groupby("well_id")["_err2"].mean().to_numpy(dtype=float))
    if values.size == 0:
        return None
    q95 = float(np.quantile(values, 0.95))
    below_q95 = values[values < q95]
    return {
        "mean": float(np.mean(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": q95,
        "q99": float(np.quantile(values, 0.99)),
        "mean_lt_q95": float(np.mean(below_q95)) if below_q95.size > 0 else float("nan"),
    }


def format_well_rmse_summary(summary):
    if summary is None:
        return "NA"
    return (
        f"mean={summary['mean']:.4f}, q50={summary['q50']:.4f}, "
        f"q75={summary['q75']:.4f}, q95={summary['q95']:.4f}, "
        f"q99={summary['q99']:.4f}, mean_lt_q95={summary['mean_lt_q95']:.4f}"
    )


def oof_sample_rmse_summary(df, sample_wells=150, sample_count=10000, seed=20260622, pred_col="TVT_pred"):
    if df.empty or "TVT" not in df.columns:
        return None
    if pred_col not in df.columns:
        raise KeyError(f"prediction dataframe missing {pred_col!r}")
    sample_wells = int(sample_wells)
    sample_count = int(sample_count)
    if sample_wells <= 0 or sample_count <= 0:
        raise ValueError(
            f"sample_wells and sample_count must be positive, got {sample_wells}, {sample_count}"
        )

    err2 = df.assign(_err2=(df["TVT"] - df[pred_col]) ** 2)
    well_stats = err2.groupby("well_id", sort=True).agg(
        sse=("_err2", "sum"),
        count=("_err2", "size"),
    )
    if well_stats.empty:
        return None
    n_wells = len(well_stats)
    if sample_wells > n_wells:
        return {
            "sample_wells": sample_wells,
            "sample_count": sample_count,
            "seed": int(seed),
            "well_count": n_wells,
            "skipped_reason": f"only {n_wells} OOF wells",
        }

    sse = well_stats["sse"].to_numpy(dtype=np.float64)
    count = well_stats["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    sample_rmse = np.empty(sample_count, dtype=np.float64)
    for i in range(sample_count):
        idx = rng.choice(n_wells, size=sample_wells, replace=False)
        sample_rmse[i] = math.sqrt(float(sse[idx].sum() / count[idx].sum()))

    return {
        "sample_wells": sample_wells,
        "sample_count": sample_count,
        "seed": int(seed),
        "well_count": n_wells,
        "mean": float(np.mean(sample_rmse)),
        "q10": float(np.quantile(sample_rmse, 0.10)),
        "q25": float(np.quantile(sample_rmse, 0.25)),
        "q50": float(np.quantile(sample_rmse, 0.50)),
        "q75": float(np.quantile(sample_rmse, 0.75)),
        "q90": float(np.quantile(sample_rmse, 0.90)),
    }


def format_oof_sample_rmse_summary(summary):
    if summary is None:
        return "NA"
    if "skipped_reason" in summary:
        return (
            f"sample_wells={summary['sample_wells']}, "
            f"sample_count={summary['sample_count']}, "
            f"seed={summary['seed']}, "
            f"skipped={summary['skipped_reason']}"
        )
    return (
        f"sample_wells={summary['sample_wells']}, "
        f"sample_count={summary['sample_count']}, "
        f"seed={summary['seed']}, "
        f"mean={summary['mean']:.4f}, "
        f"q10={summary['q10']:.4f}, q25={summary['q25']:.4f}, "
        f"q50={summary['q50']:.4f}, q75={summary['q75']:.4f}, "
        f"q90={summary['q90']:.4f}"
    )


def summarize_kfold_epoch_rmse(fold_val_histories, phase):
    fold_maps = {}
    for fold_name, history in fold_val_histories.items():
        epoch_map = {
            int(item["epoch"]): float(item["rmse"])
            for item in history
            if item.get("phase", item.get("stage", "train")) == phase and np.isfinite(item["rmse"])
        }
        if epoch_map:
            fold_maps[fold_name] = epoch_map
    if not fold_maps:
        return None

    min_fold_epoch = min(max(epoch_map) for epoch_map in fold_maps.values())
    common_epochs = sorted(
        set.intersection(
            *[
                {epoch for epoch in epoch_map if epoch <= min_fold_epoch}
                for epoch_map in fold_maps.values()
            ]
        )
    )
    if not common_epochs:
        return None

    avg_by_epoch = []
    for epoch in common_epochs:
        scores = [epoch_map[epoch] for epoch_map in fold_maps.values()]
        avg_by_epoch.append((epoch, float(np.mean(scores))))
    best_epoch, best_avg_rmse = min(avg_by_epoch, key=lambda item: item[1])
    return {
        "avg_by_epoch": avg_by_epoch,
        "best_epoch": int(best_epoch),
        "best_avg_rmse": float(best_avg_rmse),
        "min_fold_epoch": int(min_fold_epoch),
        "fold_count": len(fold_maps),
        "phase": phase,
    }


def should_log_epoch(epoch, cfg, *, epochs=None, log_freq=None):
    total_epochs = int(cfg.epochs) if epochs is None else int(epochs)
    freq = int(cfg.log_freq) if log_freq is None else int(log_freq)
    return epoch == 1 or epoch == total_epochs or epoch % freq == 0


def format_train_loss(losses, loss_detail_values):
    loss_text = f"{np.mean(losses):.6f}"
    detail_parts = [
        f"{name}:{np.mean(values):.4f}"
        for name, values in loss_detail_values.items()
        if len(values) > 0
    ]
    if detail_parts:
        loss_text += f" ({','.join(detail_parts)})"
    return loss_text


def _loader_batch_sample_counts(loader):
    batch_size = getattr(loader, "batch_size", None)
    dataset = getattr(loader, "dataset", None)
    if batch_size is None or dataset is None:
        return [1] * len(loader)
    batch_size = int(batch_size)
    dataset_len = len(dataset)
    counts = [batch_size] * len(loader)
    if counts and not bool(getattr(loader, "drop_last", False)):
        last_count = dataset_len - batch_size * (len(loader) - 1)
        if last_count > 0:
            counts[-1] = last_count
    return counts


def _raise_if_nonfinite_loss(loss, loss_details, batch, fold_name, epoch, batch_idx):
    return
    if torch.isfinite(loss.detach()).all():
        return
    detail_text = []
    for name, value in loss_details.items():
        value = value.detach().float()
        value_text = f"{float(value.cpu()):.6g}" if value.numel() == 1 else "<tensor>"
        detail_text.append(f"{name}={value_text}")
    well_ids = ",".join(str(meta.get("well_id", "?")) for meta in batch.get("meta", []))
    raise FloatingPointError(
        f"{fold_name} epoch {epoch} batch {batch_idx}: non-finite loss={float(loss.detach().float().cpu()):.6g}; "
        f"wells=[{well_ids}]; details={';'.join(detail_text)}"
    )


def run_training(model, train_loader, val_loader, cfg, log, fold_name, finetune_loader=None):
    device = torch.device(cfg.device)
    amp_dtype = getattr(torch, cfg.amp_dtype)
    amp_enabled = cfg.amp_dtype != "float32" and device.type == "cuda"
    scaler_enabled = cfg.amp_dtype == "float16" and device.type == "cuda"
    model = move_model_to_device(model, cfg)
    grad_accum_steps = int(getattr(cfg, "grad_accum_steps", 1))
    best_score = math.inf
    best_epoch = "NA"
    best_state = None
    val_history = []

    def run_phase(
        *,
        phase_name,
        phase_loader,
        epochs,
        log_freq,
        lr_scale,
        early_stop,
    ):
        nonlocal best_score, best_epoch, best_state, val_history

        phase_label = fold_name if phase_name == "train" else f"{fold_name} {phase_name}"
        optimizer_steps_per_epoch = max(1, math.ceil(len(phase_loader) / grad_accum_steps))
        train_ops = make_train_ops(
            model,
            cfg,
            optimizer_steps_per_epoch,
            epochs=epochs,
            lr_scale=lr_scale,
        )
        scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
        ema_model = None
        if ema_enabled(cfg):
            ema_decay = resolve_ema_decay(cfg, len(phase_loader))
            ema_model = ModelEMA(model, decay=ema_decay)
            log(f"{phase_label}: EMA enabled decay={ema_decay:.4f}")

        phase_best_epoch = 0
        last_epoch = 0
        for epoch in range(1, int(epochs) + 1):
            last_epoch = epoch
            model.train()
            losses = []
            loss_detail_values = {name: [] for name in active_multitask_loss_weights(cfg)}
            do_log = should_log_epoch(epoch, cfg, epochs=epochs, log_freq=log_freq)
            if do_log:
                train_iter = tqdm(
                    phase_loader,
                    desc=f"{phase_label} epoch {epoch}/{epochs}",
                    dynamic_ncols=True,
                )
            else:
                train_iter = phase_loader
            batch_sample_counts = _loader_batch_sample_counts(phase_loader)
            accum_sample_counts = [
                sum(batch_sample_counts[start : start + grad_accum_steps])
                for start in range(0, len(phase_loader), grad_accum_steps)
            ]
            train_ops.zero_grad()
            for batch_idx, batch in enumerate(train_iter, start=1):
                unet_static = batch["unet_static"].to(device, non_blocking=True)
                typewell_aux = batch["typewell_aux"].to(device, non_blocking=True)
                z_rel = batch["z_rel"].to(device, non_blocking=True)
                bin_count = batch["bin_count"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                typewell_target_probs = batch["typewell_target_probs"].to(device, non_blocking=True)
                aux_targets = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch["aux_targets"].items()
                }
                target_mask = batch["target_mask"].to(device, non_blocking=True)
                loss_scale_weight = batch_model_loss_weight(batch["meta"], target_mask, cfg)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    model_output = model_forward(
                        model,
                        unet_static,
                        typewell_aux,
                        z_rel=z_rel,
                        bin_count=bin_count,
                        extra_return=needs_extra_return(cfg),
                    )
                    pred, extra = model_output
                    loss, loss_details = unet_loss(
                        pred,
                        target,
                        target_mask,
                        typewell_target_probs,
                        aux_targets,
                        extra,
                        cfg,
                        return_details=True,
                        bin_count=bin_count,
                        z_rel=z_rel,
                        loss_scale_weight=loss_scale_weight,
                    )
                    _raise_if_nonfinite_loss(
                        loss,
                        loss_details,
                        batch,
                        phase_label,
                        epoch,
                        batch_idx,
                    )
                    accum_index = (batch_idx - 1) // grad_accum_steps
                    scaled_loss = loss * (batch_sample_counts[batch_idx - 1] / accum_sample_counts[accum_index])
                scaler.scale(scaled_loss).backward()
                should_step = batch_idx % grad_accum_steps == 0 or batch_idx == len(phase_loader)
                if should_step:
                    if scaler_enabled:
                        train_ops.unscale(scaler)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    if scaler_enabled:
                        optimizer_stepped = train_ops.step_scaled(scaler)
                    else:
                        optimizer_stepped = train_ops.step_plain()
                    train_ops.zero_grad()
                    if optimizer_stepped and ema_model is not None:
                        ema_model.update(model)
                losses.append(loss.item())
                for name, value in loss_details.items():
                    if name in loss_detail_values:
                        loss_detail_values[name].append(float(value.detach().float().cpu()))
                if do_log:
                    train_iter.set_postfix(loss=f"{np.mean(losses[-50:]):.5f}", lr=train_ops.lr_summary())

            did_validate = False
            train_loss_text = format_train_loss(losses, loss_detail_values)
            if val_loader is None or not do_log:
                val_rmse = math.inf
                if do_log:
                    log(f"{phase_label} epoch {epoch}: train_loss={train_loss_text}, lr={train_ops.lr_summary()}")
            else:
                eval_model = ema_model.module if ema_model is not None else model
                pred, metas = predict_one_model(
                    eval_model,
                    val_loader,
                    cfg,
                    move_cpu=False,
                    return_meta=True,
                )
                val_df = make_prediction_df(pred, metas, cfg, include_target=True, apply_smooth=False)
                val_rmse = score_prediction_df(val_df)
                val_history.append(
                    {
                        "phase": phase_name,
                        "epoch": int(epoch),
                        "rmse": float(val_rmse),
                        "rmse_raw": float(val_rmse),
                    }
                )
                did_validate = True
                log(
                    f"{phase_label} epoch {epoch}: train_loss={train_loss_text}, "
                    f"val_rmse_raw={val_rmse:.4f}, "
                    f"lr={train_ops.lr_summary()}"
                )
            if did_validate and val_rmse < best_score:
                best_score = val_rmse
                best_epoch = int(epoch) if phase_name == "train" else f"{phase_name}:{epoch}"
                state_source = ema_model.module if ema_model is not None else model
                best_state = state_dict_for_best(state_source)
                phase_best_epoch = epoch
            patience_start_epoch = phase_best_epoch
            min_epochs = int(getattr(cfg, "min_epochs", 0))
            if (
                early_stop
                and did_validate
                and epoch >= min_epochs
                and epoch - patience_start_epoch >= cfg.early_stopping_rounds
            ):
                log(
                    f"{phase_label}: early stop at epoch {epoch}, best_epoch={best_epoch}, "
                    f"patience_start_epoch={patience_start_epoch}, "
                    f"min_epochs={min_epochs}, best_rmse={best_score:.4f}"
                )
                break

        if val_loader is None and last_epoch > 0:
            state_source = ema_model.module if ema_model is not None else model
            best_state = state_dict_for_best(state_source)
            best_epoch = int(last_epoch) if phase_name == "train" else f"{phase_name}:{last_epoch}"

    run_phase(
        phase_name="train",
        phase_loader=train_loader,
        epochs=int(cfg.epochs),
        log_freq=int(cfg.log_freq),
        lr_scale=1.0,
        early_stop=True,
    )

    if best_state is None:
        best_state = state_dict_for_best(model)
    model.load_state_dict(best_state)

    finetune_epochs = getattr(cfg, "finetune_epochs", None)
    finetune_epochs = 0 if finetune_epochs is None else int(finetune_epochs)
    if finetune_epochs > 0:
        if finetune_loader is None:
            raise ValueError("finetune_epochs > 0 requires a finetune_loader")
        finetune_lr_reduce = float(getattr(cfg, "finetune_lr_reduce", 0.3))
        log(
            f"{fold_name}: starting finetune from best checkpoint "
            f"best_epoch={best_epoch}, best_rmse={best_score:.4f}, "
            f"epochs={finetune_epochs}, lr_reduce={finetune_lr_reduce:g}, "
            f"simulation=False"
        )
        run_phase(
            phase_name="finetune",
            phase_loader=finetune_loader,
            epochs=finetune_epochs,
            log_freq=int(getattr(cfg, "finetune_log_freq", 1)),
            lr_scale=finetune_lr_reduce,
            early_stop=False,
        )

    model.load_state_dict(best_state)
    model.cpu()
    return model, best_score, best_epoch, val_history


def kfold_training(well_ids, cfg, log):
    well_ids = np.asarray(well_ids, dtype=str)
    configured_train_rm_wells = train_rm_wells_set(cfg)
    missing_train_rm_wells = sorted(configured_train_rm_wells - set(well_ids.tolist()))
    if missing_train_rm_wells:
        raise ValueError(f"train_rm_wells contains wells absent from the current train split: {missing_train_rm_wells}")
    all_wells = set(well_ids.tolist())
    missing_down_weight_wells = sorted(model_down_weight_wells(cfg) - all_wells)
    if missing_down_weight_wells:
        raise ValueError(
            "model_down_weight_wells contains wells absent from the current train split: "
            f"{missing_down_weight_wells}"
        )
    splits = make_cv_splits(well_ids=well_ids, cfg=cfg, log=log)
    models = {}
    oof_df = []
    fold_scores = []
    raw_fold_scores = []
    fold_val_histories = {}
    set_training_seed(cfg.seed, deterministic=bool(getattr(cfg, "deterministic", True)))
    train_pf_cache_dir = prepare_pf_cache_if_needed("train", cfg.train_path, well_ids.tolist(), cfg, log)
    train_pf_sample_cache_dir = prepare_pf_sample_cache_if_needed("train", cfg.train_path, well_ids.tolist(), cfg, log)
    cv_repeats = resolve_cv_repeats(cfg)
    splits_per_repeat = len(splits) // cv_repeats if cv_repeats > 0 and len(splits) % cv_repeats == 0 else len(splits)

    for fold, (train_index, val_index) in enumerate(splits):
        set_training_seed(cfg.seed + fold, deterministic=bool(getattr(cfg, "deterministic", True)))
        if cv_repeats > 1:
            fold_name = f"repeat{fold // splits_per_repeat}_fold{fold % splits_per_repeat}"
        else:
            fold_name = f"fold{fold}"
        train_wells = well_ids[train_index].tolist()
        val_wells = well_ids[val_index].tolist()
        raw_train_count = len(train_wells)
        train_wells, removed_train_wells = remove_train_rm_wells(train_wells, cfg)
        if raw_train_count > 0 and len(train_wells) == 0:
            raise ValueError(f"{fold_name}: train_rm_wells removed every training well")
        fold_text = f"{fold_name}: training wells={len(train_wells):,}, validation wells={len(val_wells):,}"
        if removed_train_wells:
            fold_text += f", train_rm_wells removed={len(removed_train_wells):,}"
        down_count = len(set(train_wells) & model_down_weight_wells(cfg))
        if down_count > 0:
            fold_text += f", model_down_weight wells={down_count:,} weight={model_down_weight(cfg):g}"
        log(fold_text)
        geo_cfg = resolve_geo_prior_cfg(cfg)
        train_geo_prior, train_prior_summary = make_geo_prior_for_wells(
            cfg.train_path,
            cfg.train_path,
            train_wells,
            train_wells,
            geo_cfg,
            exclude_query_from_support=True,
        )
        val_geo_prior = None
        val_prior_summary = None
        if len(val_wells) > 0:
            val_geo_prior, val_prior_summary = make_geo_prior_for_wells(
                cfg.train_path,
                cfg.train_path,
                train_wells,
                val_wells,
                geo_cfg,
                exclude_query_from_support=False,
            )
        train_prior_rmse = train_prior_summary.get("rmse")
        val_prior_rmse = None if val_prior_summary is None else val_prior_summary.get("rmse")
        train_prior_text = "NA" if train_prior_rmse is None else f"{train_prior_rmse:.4f}"
        val_prior_text = "NA" if val_prior_rmse is None else f"{val_prior_rmse:.4f}"
        log(
            f"{fold_name}: geo_prior {geo_cfg.method} "
            f"train_rmse={train_prior_text} "
            f"val_rmse={val_prior_text} "
            f"elapsed={train_prior_summary['elapsed_sec'] + (0.0 if val_prior_summary is None else val_prior_summary['elapsed_sec']):.2f}s"
        )
        train_loader = make_loader(
            path=cfg.train_path,
            well_ids=train_wells,
            cfg=cfg,
            training=True,
            simulation=True,
            shuffle=True,
            batch_size=cfg.batch_size,
            seed=cfg.seed + fold * 1000 + 1,
            pf_cache_dir=train_pf_cache_dir,
            pf_sample_cache_dir=train_pf_sample_cache_dir,
            geo_prior=train_geo_prior,
            drop_last=bool(getattr(cfg, "train_drop_last", True)),
        )
        finetune_loader = None
        finetune_epochs = getattr(cfg, "finetune_epochs", None)
        finetune_epochs = 0 if finetune_epochs is None else int(finetune_epochs)
        if finetune_epochs > 0:
            finetune_loader = make_loader(
                path=cfg.train_path,
                well_ids=train_wells,
                cfg=cfg,
                training=True,
                simulation=False,
                shuffle=True,
                batch_size=cfg.batch_size,
                seed=cfg.seed + fold * 1000 + 3,
                pf_cache_dir=train_pf_cache_dir,
                pf_sample_cache_dir=None,
                geo_prior=train_geo_prior,
                drop_last=bool(getattr(cfg, "train_drop_last", True)),
            )
        val_loader = None
        if len(val_wells) > 0:
            val_loader = make_loader(
                path=cfg.train_path,
                well_ids=val_wells,
                cfg=cfg,
                training=True,
                simulation=False,
                shuffle=False,
                batch_size=cfg.val_batch_size,
                seed=cfg.seed + fold * 1000 + 2,
                pf_cache_dir=train_pf_cache_dir,
                pf_sample_cache_dir=None,
                geo_prior=val_geo_prior,
                drop_last=False,
            )
        model = make_model(cfg)
        model, best_score, best_epoch, val_history = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            log=log,
            fold_name=fold_name,
            finetune_loader=finetune_loader,
        )
        fold_val_histories[fold_name] = val_history
        fold_score = None
        fold_well_summary = None
        if val_loader is not None:
            pred, metas = predict_one_model(model, val_loader, cfg, return_meta=True)
            tmp_raw = make_prediction_df(pred, metas, cfg, include_target=True, apply_smooth=False)
            raw_fold_score = score_prediction_df(tmp_raw)
            raw_fold_well_summary = well_rmse_summary(tmp_raw)
            if getattr(cfg, "pred_sg_smooth", False):
                tmp = apply_pred_sg_smooth(tmp_raw, cfg)
                fold_score = score_prediction_df(tmp)
                fold_well_summary = well_rmse_summary(tmp)
            else:
                tmp = tmp_raw
                fold_score = raw_fold_score
                fold_well_summary = raw_fold_well_summary
            tmp = add_geo_prior_diagnostic_columns(tmp, val_geo_prior)
            tmp = add_typewell_matched_gr_columns(tmp, cfg.train_path)
            fold_scores.append(fold_score)
            raw_fold_scores.append(raw_fold_score)
            oof_df.append(tmp)
        models[fold_name] = model
        fold_summary_text = f"{fold_name}: best_epoch={best_epoch}, best_rmse={best_score:.4f}"
        if fold_score is not None:
            fold_summary_text += (
                f", final_rmse={fold_score:.4f}, "
                f"raw_rmse={raw_fold_score:.4f}, "
                f"well_rmse={format_well_rmse_summary(fold_well_summary)}"
            )
            if getattr(cfg, "pred_sg_smooth", False):
                fold_summary_text += f", raw_well_rmse={format_well_rmse_summary(raw_fold_well_summary)}"
        log(fold_summary_text)
        del train_loader, finetune_loader, val_loader
        del train_geo_prior, val_geo_prior
        release_fold_cuda_cache(cfg, log, fold_name)

    oof_df = pd.concat(oof_df, ignore_index=True) if len(oof_df) > 0 else pd.DataFrame()
    oof_df = average_repeated_oof_predictions(oof_df, cfg, log=log)
    if len(oof_df) > 0:
        oof_df = add_geo_prior_well_summary_columns(oof_df)
        oof_score = score_prediction_df(oof_df)
        raw_oof_pred_col = "TVT_pred_raw" if "TVT_pred_raw" in oof_df.columns else "TVT_pred"
        raw_oof_score = score_prediction_df(oof_df, pred_col=raw_oof_pred_col)
        fold_scores = np.asarray(fold_scores, dtype=float)
        raw_fold_scores = np.asarray(raw_fold_scores, dtype=float)
        fold_score_text = "/".join(f"{score:.4f}" for score in fold_scores)
        raw_fold_score_text = "/".join(f"{score:.4f}" for score in raw_fold_scores)
        log(
            f"OOF RMSE: {oof_score:.4f} | "
            f"FOLD RMSE: {fold_score_text} "
            f"(mean:{fold_scores.mean():.4f} +- std:{fold_scores.std():.4f})"
        )
        if raw_oof_pred_col != "TVT_pred" or not np.allclose(raw_fold_scores, fold_scores):
            log(
                f"OOF raw RMSE: {raw_oof_score:.4f} | "
                f"RAW FOLD RMSE: {raw_fold_score_text} "
                f"(mean:{raw_fold_scores.mean():.4f} +- std:{raw_fold_scores.std():.4f})"
            )
        log(f"OOF well_rmse: {format_well_rmse_summary(well_rmse_summary(oof_df))}")
        if raw_oof_pred_col != "TVT_pred":
            log(
                "OOF raw well_rmse: "
                f"{format_well_rmse_summary(well_rmse_summary(oof_df, pred_col=raw_oof_pred_col))}"
            )
        train_epoch_rmse_summary = summarize_kfold_epoch_rmse(fold_val_histories, phase="train")
        if train_epoch_rmse_summary is not None:
            log(
                "KFold train-phase avg RMSE best_epoch="
                f"{train_epoch_rmse_summary['best_epoch']}, "
                f"best_avg_rmse={train_epoch_rmse_summary['best_avg_rmse']:.4f}, "
                f"folds={train_epoch_rmse_summary['fold_count']}"
            )
            log(f"OOF sample150 RMSE: {format_oof_sample_rmse_summary(oof_sample_rmse_summary(oof_df))}")
            if raw_oof_pred_col != "TVT_pred":
                log(
                    "OOF raw sample150 RMSE: "
                    f"{format_oof_sample_rmse_summary(oof_sample_rmse_summary(oof_df, pred_col=raw_oof_pred_col))}"
                )
        finetune_epoch_rmse_summary = summarize_kfold_epoch_rmse(fold_val_histories, phase="finetune")
        if finetune_epoch_rmse_summary is not None:
            log(
                "KFold finetune-phase avg RMSE best_epoch="
                f"{finetune_epoch_rmse_summary['best_epoch']}, "
                f"best_avg_rmse={finetune_epoch_rmse_summary['best_avg_rmse']:.4f}, "
                f"folds={finetune_epoch_rmse_summary['fold_count']}"
            )
    return models, oof_df


def predict_models(models, well_ids, cfg, train_wells=None, log=print):
    model_count = len(models)
    if model_count <= 0:
        raise ValueError("predict_models requires at least one trained model")
    log(f"test inference: ensembling {model_count:,} models")
    if train_wells is None:
        train_wells = discover_well_ids(cfg.train_path)
    if getattr(cfg, "well_limit", None) is not None:
        train_wells = train_wells[: cfg.well_limit]
    geo_cfg = resolve_geo_prior_cfg(cfg)
    test_geo_prior, test_prior_summary = make_geo_prior_for_wells(
        cfg.train_path,
        cfg.test_path,
        train_wells,
        well_ids,
        geo_cfg,
        exclude_query_from_support=False,
    )
    rmse_text = "NA" if test_prior_summary.get("rmse") is None else f"{test_prior_summary['rmse']:.4f}"
    log(
        f"test geo_prior {geo_cfg.method}: "
        f"support_wells={len(train_wells):,}, query_wells={len(well_ids):,}, "
        f"rmse={rmse_text}, elapsed={test_prior_summary['elapsed_sec']:.2f}s"
    )
    test_pf_cache_dir = prepare_pf_cache_if_needed("test", cfg.test_path, list(well_ids), cfg, log)
    loader = make_loader(
        path=cfg.test_path,
        well_ids=well_ids,
        cfg=cfg,
        training=False,
        simulation=False,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        seed=cfg.seed + 100000,
        pf_cache_dir=test_pf_cache_dir,
        pf_sample_cache_dir=None,
        geo_prior=test_geo_prior,
        drop_last=False,
    )
    pred_sum = None
    metas = None
    for model in models.values():
        pred, metas = predict_one_model(model, loader, cfg, return_meta=True)
        if pred_sum is None:
            pred_sum = np.zeros_like(pred)
        pred_sum += pred / model_count
    pred_df = make_prediction_df(
        pred_sum,
        metas,
        cfg,
        include_target=False,
        apply_smooth=bool(getattr(cfg, "pred_sg_smooth", False)),
    )
    pred_df = add_geo_prior_diagnostic_columns(pred_df, test_geo_prior)
    pred_df = add_geo_prior_well_summary_columns(pred_df)
    return add_typewell_matched_gr_columns(pred_df, cfg.test_path)
