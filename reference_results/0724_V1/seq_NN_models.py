import torch
from torch import nn
import torch.nn.functional as F

from seq_NN_pretrained_unet import ChannelLayerNorm, PretrainedUNet2d


def _make_resblock_act(act):
    if act == "silu":
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    raise ValueError(f"resblock_act must be 'silu' or 'relu', got {act!r}")


def _make_resblock_norm(norm, num_features, ndim):
    if norm == "BN":
        return nn.BatchNorm1d(num_features) if ndim == 1 else nn.BatchNorm2d(num_features)
    if norm == "IN":
        instance_norm = nn.InstanceNorm1d if ndim == 1 else nn.InstanceNorm2d
        return instance_norm(num_features, affine=True)
    if norm == "LN":
        return ChannelLayerNorm(num_features)
    raise ValueError(f"resblock_norm must be 'BN', 'IN', or 'LN', got {norm!r}")


class Conv2dBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0, act="silu", norm="BN"):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = _make_resblock_norm(norm, out_ch, ndim=2)
        self.act = _make_resblock_act(act)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class ResBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0, act="silu", norm="BN"):
        super().__init__()
        self.conv1 = Conv2dBNAct(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dropout=dropout,
            act=act,
            norm=norm,
        )
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = _make_resblock_norm(norm, out_ch, ndim=2)
        self.drop = nn.Dropout2d(dropout)
        self.act = _make_resblock_act(act)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.drop(self.bn2(self.conv2(x)))
        return self.act(x + residual)


class Conv1dBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0, act="silu", norm="BN"):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = _make_resblock_norm(norm, out_ch, ndim=1)
        self.act = _make_resblock_act(act)
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0, act="silu", norm="BN"):
        super().__init__()
        self.conv1 = Conv1dBNAct(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dropout=dropout,
            act=act,
            norm=norm,
        )
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = _make_resblock_norm(norm, out_ch, ndim=1)
        self.drop = nn.Dropout1d(dropout)
        self.act = _make_resblock_act(act)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.drop(self.bn2(self.conv2(x)))
        return self.act(x + residual)


class UNet1d(nn.Module):
    def __init__(
        self,
        in_ch,
        emb_dim=64,
        stages=4,
        kernel_size=3,
        dropout=0.0,
        resblock_act="silu",
        resblock_norm="BN",
    ):
        super().__init__()
        emb_dim = int(emb_dim)
        stages = int(stages)
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        if stages <= 0:
            raise ValueError(f"stages must be positive, got {stages}")
        self.stem = ResBlock1d(
            in_ch,
            emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )
        self.down_blocks = nn.ModuleList()
        for _ in range(stages - 1):
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "down": nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True),
                        "block": ResBlock1d(
                            emb_dim,
                            emb_dim,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            act=resblock_act,
                            norm=resblock_norm,
                        ),
                    }
                )
            )
        self.up_blocks = nn.ModuleList()
        for _ in range(stages - 1):
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "proj": nn.Conv1d(emb_dim * 2, emb_dim, kernel_size=1, bias=False),
                        "block": ResBlock1d(
                            emb_dim,
                            emb_dim,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            act=resblock_act,
                            norm=resblock_norm,
                        ),
                    }
                )
            )
        self.out = ResBlock1d(
            emb_dim,
            emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )

    def forward(self, x):
        skips = [self.stem(x)]
        x = skips[-1]
        for layer in self.down_blocks:
            x = layer["down"](x)
            x = layer["block"](x)
            skips.append(x)
        for layer, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = layer["proj"](x)
            x = layer["block"](x)
        return self.out(x)


class RandomChannelMask(nn.Module):
    def __init__(self, mask_prob=0.1, mask_all_prob=0.0, mask_value=0.0):
        super().__init__()
        self.mask_prob = float(mask_prob)
        self.mask_all_prob = float(mask_all_prob)
        self.mask_value = float(mask_value)
        for name, value in (
            ("mask_prob", self.mask_prob),
            ("mask_all_prob", self.mask_all_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def forward(self, x):
        if not self.training or (self.mask_prob == 0.0 and self.mask_all_prob == 0.0):
            return x
        if x.ndim < 2:
            raise ValueError(f"RandomChannelMask expects at least 2 dimensions, got shape={tuple(x.shape)}")

        mask_shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        mask = torch.rand(mask_shape, device=x.device) > self.mask_prob
        if self.mask_all_prob > 0.0:
            batch_mask_all = torch.rand(x.shape[0], device=x.device) < self.mask_all_prob
            if batch_mask_all.any():
                mask[batch_mask_all] = False
        return torch.where(mask, x, x.new_tensor(self.mask_value))


class UNet2d(nn.Module):
    def __init__(
        self,
        in_ch,
        emb_dim,
        mults=(1, 2, 4),
        kernel_size=3,
        dropout=0.0,
        updown_mode="param",
        resblock_act="silu",
        resblock_norm="BN",
    ):
        super().__init__()
        if updown_mode not in {"param", "non-param"}:
            raise ValueError(f"unknown updown_mode: {updown_mode}")
        channels = [emb_dim * m for m in mults]
        self.stem = ResBlock2d(
            in_ch,
            channels[0],
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )
        self.down_blocks = nn.ModuleList()
        in_channels = channels[0]
        for out_channels in channels[1:]:
            if updown_mode == "param":
                down = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(),
                )
                block_in_channels = out_channels
            else:
                down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
                block_in_channels = in_channels
            self.down_blocks.append(
                nn.ModuleDict(
                    {
                        "down": down,
                        "block": ResBlock2d(
                            block_in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            act=resblock_act,
                            norm=resblock_norm,
                        ),
                    }
                )
            )
            in_channels = out_channels
        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.up_blocks.append(
                nn.ModuleDict(
                    {
                        "proj": nn.Conv2d(in_channels + skip_channels, skip_channels, kernel_size=1, bias=False),
                        "block": ResBlock2d(
                            skip_channels,
                            skip_channels,
                            kernel_size=kernel_size,
                            dropout=dropout,
                            act=resblock_act,
                            norm=resblock_norm,
                        ),
                    }
                )
            )
            in_channels = skip_channels
        self.out = ResBlock2d(
            in_channels,
            emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )

    def forward(self, x):
        skips = [self.stem(x)]
        x = skips[-1]
        for layer in self.down_blocks:
            x = layer["down"](x)
            x = layer["block"](x)
            skips.append(x)
        for layer, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = layer["proj"](x)
            x = layer["block"](x)
        return self.out(x)


class SeqUNet2DModel(nn.Module):
    def __init__(
        self,
        unet_static_in_dim,
        unet_emb_dim=32,
        unet_mults=(1, 2, 4, 4),
        unet_stacks=1,
        unet_arch="unet",
        unet_pretrained=False,
        unet_timm_model_name=None,
        unet_stem_stride=(2, 4),
        unet_stem_down_mode="pool",
        unet_convnext_norm_replace="BN",
        unet_convnext_bn_eps=1e-3,
        unet_convnext_bn_momentum=0.01,
        unet_convnext_bn_track_running_stats=False,
        res_kernel_size=3,
        resblock_act="silu",
        resblock_norm="BN",
        unet_dropout=0.0,
        head_hidden_mult=2,
        updown_mode_2d="non-param",
        regression_head_mode="logits",
        share_logits_head=True,
        offset_detach_logits_head_in_loss=True,
        unet_channel_mask_prob=0.0,
        unet_channel_mask_all_prob=0.0,
        unet_channel_mask_value=0.0,
        typewell_window=100.0,
        typewell_len=400,
        target_mean=0.0,
        target_std=1.0,
        tvt_diff_clip=100.0,
        tvt_rel_norm_mean=0.0,
        tvt_rel_norm_std=57.73,
        typewell_gr_mean=0.0,
        typewell_gr_std=1.0,
    ):
        super().__init__()
        regression_head_mode = str(regression_head_mode)
        if regression_head_mode not in {"logits", "prob+offset", "pool_mlp"}:
            raise ValueError(
                f"unknown regression_head_mode={regression_head_mode!r}; expected 'logits', 'prob+offset', or 'pool_mlp'"
            )
        if float(target_std) <= 0.0:
            raise ValueError(f"target_std must be positive: {target_std}")
        if float(tvt_rel_norm_std) <= 0.0:
            raise ValueError(f"tvt_rel_norm_std must be positive: {tvt_rel_norm_std}")
        unet_arch = str(unet_arch)
        if unet_arch not in {"unet", "convnext_small"}:
            raise ValueError(
                f"unknown unet_arch={unet_arch!r}; expected 'unet' or 'convnext_small'"
            )
        self.unet_arch = unet_arch
        self.regression_head_mode = regression_head_mode
        self.share_logits_head = bool(share_logits_head)
        self.offset_detach_logits_head_in_loss = bool(offset_detach_logits_head_in_loss)
        if self.share_logits_head and self.regression_head_mode not in {"logits", "prob+offset"}:
            raise ValueError("share_logits_head=True requires regression_head_mode='logits' or 'prob+offset'")
        self.unet_channel_mask = RandomChannelMask(
            mask_prob=unet_channel_mask_prob,
            mask_all_prob=unet_channel_mask_all_prob,
            mask_value=unet_channel_mask_value,
        )
        self.tvt_diff_clip = float(tvt_diff_clip)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.tvt_rel_norm_mean = float(tvt_rel_norm_mean)
        self.tvt_rel_norm_std = float(tvt_rel_norm_std)
        if float(typewell_gr_std) <= 0.0:
            raise ValueError(f"typewell_gr_std must be positive: {typewell_gr_std}")
        self.typewell_gr_mean = float(typewell_gr_mean)
        self.typewell_gr_std = float(typewell_gr_std)
        unet_dropout = float(unet_dropout)
        if not 0.0 <= unet_dropout <= 1.0:
            raise ValueError(f"unet_dropout must be in [0, 1]: {unet_dropout}")

        unet_in_dim = int(unet_static_in_dim)
        unet_stacks = int(unet_stacks)
        if unet_stacks <= 0:
            raise ValueError(f"unet_stacks must be positive: {unet_stacks}")
        if unet_arch == "unet":
            self.unet = [
                UNet2d(
                    unet_in_dim,
                    unet_emb_dim,
                    mults=unet_mults,
                    kernel_size=res_kernel_size,
                    dropout=unet_dropout,
                    updown_mode=updown_mode_2d,
                    resblock_act=resblock_act,
                    resblock_norm=resblock_norm,
                )
            ]
            for _ in range(unet_stacks - 1):
                self.unet.append(
                    UNet2d(
                        unet_emb_dim,
                        unet_emb_dim,
                        mults=unet_mults,
                        kernel_size=res_kernel_size,
                        dropout=unet_dropout,
                        updown_mode=updown_mode_2d,
                        resblock_act=resblock_act,
                        resblock_norm=resblock_norm,
                    )
                )
            self.unet = nn.Sequential(*self.unet)
        else:
            if unet_stacks != 1:
                raise ValueError("unet_arch='convnext_small' does not support unet_stacks != 1")
            self.unet = PretrainedUNet2d(
                unet_in_dim,
                unet_emb_dim,
                backbone_name=unet_arch,
                pretrained=bool(unet_pretrained),
                timm_model_name=unet_timm_model_name,
                stem_stride=unet_stem_stride,
                stem_down_mode=unet_stem_down_mode,
                convnext_norm_replace=unet_convnext_norm_replace,
                convnext_bn_eps=float(unet_convnext_bn_eps),
                convnext_bn_momentum=float(unet_convnext_bn_momentum),
                convnext_bn_track_running_stats=bool(unet_convnext_bn_track_running_stats),
                kernel_size=res_kernel_size,
                resblock_act=resblock_act,
                resblock_norm=resblock_norm,
                dropout=unet_dropout,
            )
        self.alignment_head = nn.Conv2d(unet_emb_dim, 1, kernel_size=1)
        typewell_tvt_axis_raw = torch.linspace(
            -typewell_window + typewell_window / typewell_len,
            typewell_window - typewell_window / typewell_len,
            typewell_len,
            dtype=torch.float32,
        )
        self.register_buffer("typewell_tvt_axis_raw", typewell_tvt_axis_raw, persistent=False)
        typewell_tvt_axis = (typewell_tvt_axis_raw - self.target_mean) / self.target_std
        self.register_buffer("typewell_tvt_axis", typewell_tvt_axis, persistent=False)

        self.offset_head = None
        hidden_dim = int(unet_emb_dim * head_hidden_mult)
        if self.regression_head_mode in {"logits", "prob+offset"}:
            self.regression_logits_head = None if self.share_logits_head else nn.Conv2d(unet_emb_dim, 1, kernel_size=1)
            self.regression_mlp_head = None
            if self.regression_head_mode == "prob+offset":
                self.offset_head = nn.Conv2d(unet_emb_dim, 1, kernel_size=1)
                nn.init.zeros_(self.offset_head.weight)
                if self.offset_head.bias is not None:
                    nn.init.zeros_(self.offset_head.bias)
        else:
            self.regression_logits_head = None
            self.regression_mlp_head = nn.Sequential(
                nn.Linear(unet_emb_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Dropout(unet_dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.unet_gr_rmse_head = nn.Sequential(
            nn.Linear(unet_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(unet_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def model_modules(self):
        modules = []
        unet_channel_mask = getattr(self, "unet_channel_mask", None)
        if unet_channel_mask is not None:
            modules.append(unet_channel_mask)
        modules.extend([self.unet, self.alignment_head])
        if self.regression_logits_head is not None:
            modules.append(self.regression_logits_head)
        if self.regression_mlp_head is not None:
            modules.append(self.regression_mlp_head)
        offset_head = getattr(self, "offset_head", None)
        if offset_head is not None:
            modules.append(offset_head)
        unet_gr_rmse_head = getattr(self, "unet_gr_rmse_head", None)
        if unet_gr_rmse_head is not None:
            modules.append(unet_gr_rmse_head)
        return tuple(modules)

    def model_named_parameters(self):
        module_names = [("unet", self.unet), ("alignment_head", self.alignment_head)]
        if self.regression_logits_head is not None:
            module_names.append(("regression_logits_head", self.regression_logits_head))
        if self.regression_mlp_head is not None:
            module_names.append(("regression_mlp_head", self.regression_mlp_head))
        offset_head = getattr(self, "offset_head", None)
        if offset_head is not None:
            module_names.append(("offset_head", offset_head))
        unet_gr_rmse_head = getattr(self, "unet_gr_rmse_head", None)
        if unet_gr_rmse_head is not None:
            module_names.append(("unet_gr_rmse_head", unet_gr_rmse_head))
        seen = set()
        for module_name, module in module_names:
            for name, param in module.named_parameters():
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                yield f"{module_name}.{name}", param

    def _unet_input(self, unet_static):
        unet_input = unet_static
        unet_channel_mask = getattr(self, "unet_channel_mask", None)
        if unet_channel_mask is not None:
            unet_input = unet_channel_mask(unet_input)
        return unet_input

    def _regression_from_logits(self, logits):
        prob = F.softmax(logits, dim=-1)
        tvt_axis = self.typewell_tvt_axis.to(device=prob.device, dtype=prob.dtype)
        return (prob * tvt_axis.reshape(1, 1, -1)).sum(dim=-1)

    def _regression_from_prob_and_offset(self, logits, offset):
        prob = F.softmax(logits, dim=-1)
        tvt_axis = self.typewell_tvt_axis.to(device=prob.device, dtype=prob.dtype)
        offset = offset.to(device=prob.device, dtype=prob.dtype)
        return (prob * (tvt_axis.reshape(1, 1, -1) + offset)).sum(dim=-1)

    def forward(
        self,
        unet_static,
        typewell_aux,
        z_rel=None,
        bin_count=None,
        z_diff=None,
        extra_return=False,
    ):
        del typewell_aux, z_rel, bin_count, z_diff
        unet_input = self._unet_input(unet_static)
        x = self.unet(unet_input)
        x_pooled = None
        alignment_logits = None
        offset_pred = None
        if self.regression_head_mode in {"logits", "prob+offset"}:
            if self.share_logits_head:
                regression_logits = self.alignment_head(x).squeeze(1)
                alignment_logits = regression_logits
            else:
                regression_logits = self.regression_logits_head(x).squeeze(1)
            if self.regression_head_mode == "logits":
                pred = self._regression_from_logits(regression_logits)
            else:
                offset_pred = self.offset_head(x).squeeze(1)
                pred = self._regression_from_prob_and_offset(regression_logits, offset_pred)
        else:
            x_pooled = x.mean(dim=-1).transpose(1, 2)
            batch_size, seq_len, feature_dim = x_pooled.shape
            flat_x = x_pooled.reshape(batch_size * seq_len, feature_dim)
            pred = self.regression_mlp_head(flat_x).reshape(batch_size, seq_len)
            regression_logits = None
        unet_gr_rmse_pred = None
        unet_gr_rmse_head = getattr(self, "unet_gr_rmse_head", None)
        if unet_gr_rmse_head is not None:
            if x_pooled is None:
                x_pooled = x.mean(dim=-1).transpose(1, 2)
            batch_size, seq_len, feature_dim = x_pooled.shape
            flat_x = x_pooled.reshape(batch_size * seq_len, feature_dim)
            unet_gr_rmse_pred = F.softplus(unet_gr_rmse_head(flat_x)).reshape(batch_size, seq_len)
        if not extra_return:
            return pred
        if alignment_logits is None:
            alignment_logits = self.alignment_head(x).squeeze(1)
        extra = {
            "alignment_logits": alignment_logits,
        }
        if regression_logits is not None:
            extra["regression_logits"] = regression_logits
        if offset_pred is not None:
            extra["offset_pred"] = offset_pred
        if unet_gr_rmse_pred is not None:
            extra["unet_GR_RMSE_pred"] = unet_gr_rmse_pred
        return pred, extra


class SeqTwoStageUNetModel(SeqUNet2DModel):
    def __init__(
        self,
        unet_static_in_dim,
        static_channel_names,
        tw_channels=("tw_gr", "tw_gr_is_nan"),
        h_channels=("gr", "gr_isnan_rate", "gr_std", "gr_first_last_delta", "gr_slope"),
        shared_channels=None,
        encoder_emb_dim=64,
        encoder_stages=4,
        corr_matrix_dim=8,
        encoder_kernel_size=3,
        encoder_dropout=None,
        **kwargs,
    ):
        static_channel_names = tuple(static_channel_names)
        if int(unet_static_in_dim) != len(static_channel_names):
            raise ValueError(
                "unet_static_in_dim must match static_channel_names length: "
                f"{unet_static_in_dim} != {len(static_channel_names)}"
            )
        channel_to_idx = {name: idx for idx, name in enumerate(static_channel_names)}
        if len(channel_to_idx) != len(static_channel_names):
            raise ValueError(f"static_channel_names must be unique: {static_channel_names}")

        tw_channels = tuple(tw_channels)
        h_channels = tuple(h_channels)
        if len(tw_channels) == 0:
            raise ValueError("tw_channels must contain at least one channel")
        if len(h_channels) == 0:
            raise ValueError("h_channels must contain at least one channel")
        if shared_channels is None:
            used_1d = set(tw_channels) | set(h_channels)
            shared_channels = tuple(name for name in static_channel_names if name not in used_1d)
        else:
            shared_channels = tuple(shared_channels)

        grouped_channels = tw_channels + h_channels + shared_channels
        missing = sorted({name for name in grouped_channels if name not in channel_to_idx})
        if missing:
            raise ValueError(f"two-stage channel groups reference unknown channels: {missing}")

        encoder_emb_dim = int(encoder_emb_dim)
        corr_matrix_dim = int(corr_matrix_dim)
        if encoder_emb_dim <= 0:
            raise ValueError(f"encoder_emb_dim must be positive, got {encoder_emb_dim}")
        if corr_matrix_dim <= 0:
            raise ValueError(f"corr_matrix_dim must be positive, got {corr_matrix_dim}")
        if encoder_emb_dim % corr_matrix_dim != 0:
            raise ValueError(
                "encoder_emb_dim must be divisible by corr_matrix_dim: "
                f"{encoder_emb_dim} % {corr_matrix_dim} != 0"
            )

        decoder_in_dim = len(shared_channels) + corr_matrix_dim
        super().__init__(
            unet_static_in_dim=decoder_in_dim,
            **kwargs,
        )
        dropout = float(kwargs.get("unet_dropout", 0.0)) if encoder_dropout is None else float(encoder_dropout)
        self.static_channel_names = static_channel_names
        self.tw_channels = tw_channels
        self.h_channels = h_channels
        self.shared_channels = shared_channels
        self.tw_channel_indices = tuple(channel_to_idx[name] for name in tw_channels)
        self.h_channel_indices = tuple(channel_to_idx[name] for name in h_channels)
        self.shared_channel_indices = tuple(channel_to_idx[name] for name in shared_channels)
        self.encoder_emb_dim = encoder_emb_dim
        self.corr_matrix_dim = corr_matrix_dim
        self.corr_group_dim = encoder_emb_dim // corr_matrix_dim
        self.tw_encoder = UNet1d(
            len(self.tw_channel_indices),
            emb_dim=encoder_emb_dim,
            stages=int(encoder_stages),
            kernel_size=int(encoder_kernel_size),
            dropout=dropout,
            resblock_act=kwargs.get("resblock_act", "silu"),
            resblock_norm=kwargs.get("resblock_norm", "BN"),
        )
        self.h_encoder = UNet1d(
            len(self.h_channel_indices),
            emb_dim=encoder_emb_dim,
            stages=int(encoder_stages),
            kernel_size=int(encoder_kernel_size),
            dropout=dropout,
            resblock_act=kwargs.get("resblock_act", "silu"),
            resblock_norm=kwargs.get("resblock_norm", "BN"),
        )

    def model_modules(self):
        return (self.tw_encoder, self.h_encoder) + super().model_modules()

    def model_named_parameters(self):
        for module_name, module in (("tw_encoder", self.tw_encoder), ("h_encoder", self.h_encoder)):
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param
        yield from super().model_named_parameters()

    def _groupwise_correlation(self, h_emb, tw_emb):
        batch_size, emb_dim, horizontal_len = h_emb.shape
        tw_batch_size, tw_emb_dim, typewell_len = tw_emb.shape
        if tw_batch_size != batch_size or tw_emb_dim != emb_dim:
            raise ValueError(
                "horizontal/typewell embeddings must share batch and channel dimensions: "
                f"h={tuple(h_emb.shape)}, tw={tuple(tw_emb.shape)}"
            )
        group_count = self.corr_matrix_dim
        group_dim = self.corr_group_dim
        h_grouped = h_emb.reshape(batch_size, group_count, group_dim, horizontal_len)
        tw_grouped = tw_emb.reshape(batch_size, group_count, group_dim, typewell_len)
        corr = torch.einsum("bgdh,bgdt->bght", h_grouped, tw_grouped)
        return corr * (group_dim ** -0.5)

    def _unet_input(self, unet_static):
        tw_seq = unet_static[:, list(self.tw_channel_indices), 0, :]
        h_seq = unet_static[:, list(self.h_channel_indices), :, 0]
        tw_emb = self.tw_encoder(tw_seq)
        h_emb = self.h_encoder(h_seq)
        corr = self._groupwise_correlation(h_emb, tw_emb)
        if self.shared_channel_indices:
            shared = unet_static[:, list(self.shared_channel_indices), :, :]
            unet_input = torch.cat([shared, corr], dim=1)
        else:
            unet_input = corr
        unet_channel_mask = getattr(self, "unet_channel_mask", None)
        if unet_channel_mask is not None:
            unet_input = unet_channel_mask(unet_input)
        return unet_input
