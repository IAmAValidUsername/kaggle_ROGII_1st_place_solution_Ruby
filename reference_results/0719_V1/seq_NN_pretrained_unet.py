import time

import torch
from torch import nn
import torch.nn.functional as F


PRETRAINED_UNET_BACKBONES = {
    "convnext_small": {
        "timm_model_name": "hf_hub:timm/convnext_small.in12k_ft_in1k_384",
        "stage_channels": (96, 192, 384, 768),
    },
}

_TIMM_PRETRAINED_MODEL_DOWNLOAD_RETRIES = 3
_TIMM_PRETRAINED_MODEL_DOWNLOAD_RETRY_BASE_SLEEP_SEC = 30.0


def _make_resblock_act(act):
    if act == "silu":
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    raise ValueError(f"resblock_act must be 'silu' or 'relu', got {act!r}")


class ChannelLayerNorm(nn.LayerNorm):
    def __init__(self, num_features, eps=1e-5):
        super().__init__(int(num_features), eps=float(eps))

    def forward(self, x):
        x = x.movedim(1, -1)
        x = super().forward(x)
        return x.movedim(-1, 1)


def _make_resblock_norm(norm, num_features):
    if norm == "BN":
        return nn.BatchNorm2d(num_features)
    if norm == "IN":
        return nn.InstanceNorm2d(num_features, affine=True)
    if norm == "LN":
        return ChannelLayerNorm(num_features)
    raise ValueError(f"resblock_norm must be 'BN', 'IN', or 'LN', got {norm!r}")


class Conv2dBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.0, act="silu", norm="BN"):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False)
        self.bn = _make_resblock_norm(norm, out_ch)
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
        self.bn2 = _make_resblock_norm(norm, out_ch)
        self.drop = nn.Dropout2d(dropout)
        self.act = _make_resblock_act(act)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.drop(self.bn2(self.conv2(x)))
        return self.act(x + residual)


class BatchNorm2dFp32(nn.Module):
    is_convnext_converted_bn = True

    def __init__(
        self,
        num_features,
        eps=1e-3,
        momentum=0.01,
        affine=True,
        track_running_stats=False,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            int(num_features),
            eps=float(eps),
            momentum=float(momentum),
            affine=bool(affine),
            track_running_stats=bool(track_running_stats),
        )

    def forward(self, x):
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            return self.bn(x.float()).to(dtype=dtype)
        return self.bn(x)


class ChannelLastBatchNorm2d(nn.Module):
    is_convnext_converted_bn = True

    def __init__(
        self,
        num_features,
        eps=1e-3,
        momentum=0.01,
        affine=True,
        track_running_stats=False,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(
            int(num_features),
            eps=float(eps),
            momentum=float(momentum),
            affine=bool(affine),
            track_running_stats=bool(track_running_stats),
        )

    def forward(self, x):
        dtype = x.dtype
        x = x.permute(0, 3, 1, 2)
        if dtype in (torch.float16, torch.bfloat16):
            x = self.bn(x.float()).to(dtype=dtype)
        else:
            x = self.bn(x)
        return x.permute(0, 2, 3, 1)


class InstanceNorm2dFp32(nn.Module):
    def __init__(self, num_features, eps=1e-3, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(
            int(num_features),
            eps=float(eps),
            affine=bool(affine),
            track_running_stats=False,
        )

    def forward(self, x):
        dtype = x.dtype
        if dtype in (torch.float16, torch.bfloat16):
            return self.norm(x.float()).to(dtype=dtype)
        return self.norm(x)


class ChannelLastInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-3, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm2d(
            int(num_features),
            eps=float(eps),
            affine=bool(affine),
            track_running_stats=False,
        )

    def forward(self, x):
        dtype = x.dtype
        x = x.permute(0, 3, 1, 2)
        if dtype in (torch.float16, torch.bfloat16):
            x = self.norm(x.float()).to(dtype=dtype)
        else:
            x = self.norm(x)
        return x.permute(0, 2, 3, 1)


class BilinearResUpBlock2d(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        kernel_size=3,
        dropout=0.0,
        resblock_act="silu",
        resblock_norm="BN",
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1, bias=False)
        self.block = ResBlock2d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.block(x)


class BilinearAddResUpBlock2d(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        kernel_size=3,
        dropout=0.0,
        resblock_act="silu",
        resblock_norm="BN",
    ):
        super().__init__()
        self.proj_x = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.proj_skip = nn.Identity() if skip_ch == out_ch else nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.block = ResBlock2d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj_x(x) + self.proj_skip(skip)
        return self.block(x)


def _pair(value, name):
    if isinstance(value, int):
        out = (value, value)
    else:
        out = tuple(value)
    if len(out) != 2:
        raise ValueError(f"{name} must be an int or length-2 tuple, got {value!r}")
    out = (int(out[0]), int(out[1]))
    if out[0] <= 0 or out[1] <= 0:
        raise ValueError(f"{name} entries must be positive, got {value!r}")
    return out


def _stride_conv_kernel_and_padding(stride):
    return tuple(2 * value - 1 for value in stride), tuple(value - 1 for value in stride)


def _stage_channels_from_timm_model(model, backbone_name):
    feature_info = getattr(model, "feature_info", None)
    if feature_info is not None:
        try:
            channels = [int(info["num_chs"]) for info in feature_info]
        except TypeError:
            channels = [int(info["num_chs"]) for info in list(feature_info)]
        if len(channels) > 0:
            return tuple(channels)
    return PRETRAINED_UNET_BACKBONES[backbone_name]["stage_channels"]


def _norm_num_features(module):
    if hasattr(module, "num_features"):
        return int(module.num_features)
    shape = getattr(module, "normalized_shape", None)
    if shape is None:
        return None
    if isinstance(shape, int):
        return int(shape)
    if len(shape) != 1:
        return None
    return int(shape[0])


def _copy_norm_affine(source, target):
    if not hasattr(target, "weight") or not hasattr(target, "bias"):
        return
    source_weight = getattr(source, "weight", None)
    source_bias = getattr(source, "bias", None)
    with torch.no_grad():
        if source_weight is not None and target.weight is not None:
            target.weight.copy_(source_weight.reshape_as(target.weight))
        if source_bias is not None and target.bias is not None:
            target.bias.copy_(source_bias.reshape_as(target.bias))


def _converted_norm_eps(source_norm, norm_eps):
    return max(float(getattr(source_norm, "eps", 1e-5)), float(norm_eps))


def _convert_convnext_norms(
    module,
    *,
    norm_replace,
    bn_eps=1e-3,
    bn_momentum=0.01,
    bn_track_running_stats=False,
):
    if norm_replace not in {"BN", "IN"}:
        raise ValueError(
            f"convnext_norm_replace must be 'BN' or 'IN', got {norm_replace!r}"
        )
    for child_name, child in list(module.named_children()):
        child_type = type(child).__name__
        if child_type == "LayerNorm2d":
            num_features = _norm_num_features(child)
            if num_features is None:
                raise TypeError(f"cannot infer channel count for {child_type}")
            if norm_replace == "BN":
                replacement = BatchNorm2dFp32(
                    num_features,
                    eps=_converted_norm_eps(child, bn_eps),
                    momentum=bn_momentum,
                    affine=bool(getattr(child, "elementwise_affine", True)),
                    track_running_stats=bn_track_running_stats,
                )
                replacement_norm = replacement.bn
            else:
                replacement = InstanceNorm2dFp32(
                    num_features,
                    eps=_converted_norm_eps(child, bn_eps),
                    affine=bool(getattr(child, "elementwise_affine", True)),
                )
                replacement_norm = replacement.norm
            _copy_norm_affine(child, replacement_norm)
            setattr(module, child_name, replacement)
        elif child_type == "LayerNorm":
            num_features = _norm_num_features(child)
            if num_features is None:
                raise TypeError(f"cannot infer channel count for {child_type}")
            if norm_replace == "BN":
                replacement = ChannelLastBatchNorm2d(
                    num_features,
                    eps=_converted_norm_eps(child, bn_eps),
                    momentum=bn_momentum,
                    affine=bool(getattr(child, "elementwise_affine", True)),
                    track_running_stats=bn_track_running_stats,
                )
                replacement_norm = replacement.bn
            else:
                replacement = ChannelLastInstanceNorm2d(
                    num_features,
                    eps=_converted_norm_eps(child, bn_eps),
                    affine=bool(getattr(child, "elementwise_affine", True)),
                )
                replacement_norm = replacement.norm
            _copy_norm_affine(child, replacement_norm)
            setattr(module, child_name, replacement)
        else:
            _convert_convnext_norms(
                child,
                norm_replace=norm_replace,
                bn_eps=bn_eps,
                bn_momentum=bn_momentum,
                bn_track_running_stats=bn_track_running_stats,
            )


def _convert_convnext_norms_to_bn(
    module,
    *,
    bn_eps=1e-3,
    bn_momentum=0.01,
    bn_track_running_stats=False,
):
    _convert_convnext_norms(
        module,
        norm_replace="BN",
        bn_eps=bn_eps,
        bn_momentum=bn_momentum,
        bn_track_running_stats=bn_track_running_stats,
    )


def _create_timm_model_with_retries(timm, model_name, *, pretrained, num_classes):
    model_name = str(model_name)
    if model_name.startswith("timm/"):
        model_name = f"hf_hub:{model_name}"

    max_attempts = 1
    if pretrained:
        max_attempts += _TIMM_PRETRAINED_MODEL_DOWNLOAD_RETRIES

    for attempt_idx in range(max_attempts):
        try:
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:
            if attempt_idx + 1 >= max_attempts:
                raise
            sleep_sec = _TIMM_PRETRAINED_MODEL_DOWNLOAD_RETRY_BASE_SLEEP_SEC * (2**attempt_idx)
            print(
                f"timm.create_model failed while loading pretrained model {model_name!r} "
                f"(attempt {attempt_idx + 1}/{max_attempts}); retrying in {sleep_sec:.1f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(sleep_sec)

    raise RuntimeError("unreachable timm model creation retry state")


class PretrainedUNet2d(nn.Module):
    def __init__(
        self,
        in_ch,
        emb_dim,
        backbone_name="convnext_small",
        pretrained=True,
        timm_model_name=None,
        stem_stride=(2, 4),
        stem_down_mode="pool",
        convnext_norm_replace="BN",
        convnext_bn_eps=1e-3,
        convnext_bn_momentum=0.01,
        convnext_bn_track_running_stats=False,
        kernel_size=3,
        resblock_act="silu",
        resblock_norm="BN",
        dropout=0.0,
    ):
        super().__init__()
        backbone_name = str(backbone_name)
        if backbone_name not in PRETRAINED_UNET_BACKBONES:
            available = ", ".join(sorted(PRETRAINED_UNET_BACKBONES))
            raise ValueError(f"unknown pretrained U-Net backbone {backbone_name!r}; available: {available}")
        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        if (
            convnext_norm_replace is not None
            and convnext_norm_replace not in {"BN", "IN", "LN"}
        ):
            raise ValueError(
                "convnext_norm_replace must be 'BN', 'IN', 'LN', or None; "
                f"got {convnext_norm_replace!r}"
            )
        self.convnext_norm_replace = convnext_norm_replace
        self.convnext_bn_eps = float(convnext_bn_eps)
        self.convnext_bn_momentum = float(convnext_bn_momentum)
        self.convnext_bn_track_running_stats = bool(convnext_bn_track_running_stats)
        self.stem_stride = _pair(stem_stride, "stem_stride")
        self.stem_down_mode = str(stem_down_mode)
        if self.stem_down_mode not in {"pool", "conv"}:
            raise ValueError(f"stem_down_mode must be 'pool' or 'conv', got {stem_down_mode!r}")
        self.emb_dim = int(emb_dim)
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        if self.convnext_bn_eps <= 0.0:
            raise ValueError(f"convnext_bn_eps must be positive, got {convnext_bn_eps}")
        if self.convnext_bn_momentum < 0.0:
            raise ValueError(f"convnext_bn_momentum must be non-negative, got {convnext_bn_momentum}")

        timm_model_name = timm_model_name or PRETRAINED_UNET_BACKBONES[backbone_name]["timm_model_name"]
        self.timm_model_name = str(timm_model_name)
        try:
            import timm
        except ImportError as exc:
            raise ImportError("unet_arch='convnext_small' requires the timm package") from exc

        backbone = _create_timm_model_with_retries(
            timm,
            self.timm_model_name,
            pretrained=self.pretrained,
            num_classes=0,
        )
        if not hasattr(backbone, "stages"):
            raise TypeError(f"timm model {self.timm_model_name!r} does not expose ConvNeXt-style stages")
        stages = list(backbone.stages)
        if len(stages) != 4:
            raise ValueError(f"{self.timm_model_name!r} expected 4 down stages, got {len(stages)}")
        stage_channels = _stage_channels_from_timm_model(backbone, backbone_name)
        if len(stage_channels) != len(stages):
            raise ValueError(
                f"{self.timm_model_name!r} stage/channel mismatch: "
                f"{len(stages)} stages vs {len(stage_channels)} channel entries"
            )
        if self.convnext_norm_replace in {"BN", "IN"}:
            for stage in stages:
                _convert_convnext_norms(
                    stage,
                    norm_replace=self.convnext_norm_replace,
                    bn_eps=self.convnext_bn_eps,
                    bn_momentum=self.convnext_bn_momentum,
                    bn_track_running_stats=self.convnext_bn_track_running_stats,
                )

        self.stem = ResBlock2d(
            in_ch,
            self.emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )
        if self.stem_stride == (1, 1):
            self.stem_down = nn.Identity()
            self.to_backbone = nn.Conv2d(self.emb_dim, stage_channels[0], kernel_size=1, bias=True)
        elif self.stem_down_mode == "pool":
            self.stem_down = nn.AvgPool2d(
                kernel_size=self.stem_stride,
                stride=self.stem_stride,
                ceil_mode=True,
            )
            self.to_backbone = nn.Conv2d(self.emb_dim, stage_channels[0], kernel_size=1, bias=True)
        else:
            self.stem_down = nn.Identity()
            down_kernel_size, down_padding = _stride_conv_kernel_and_padding(self.stem_stride)
            self.to_backbone = nn.Conv2d(
                self.emb_dim,
                stage_channels[0],
                kernel_size=down_kernel_size,
                stride=self.stem_stride,
                padding=down_padding,
                bias=True,
            )
        self.backbone_stages = nn.ModuleList(stages)
        self.up_blocks = nn.ModuleList(
            BilinearResUpBlock2d(
                in_channels,
                skip_channels,
                skip_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                resblock_act=resblock_act,
                resblock_norm=resblock_norm,
            )
            for in_channels, skip_channels in zip(
                reversed(stage_channels[1:]),
                reversed(stage_channels[:-1]),
            )
        )
        self.to_full_resolution = nn.Conv2d(stage_channels[0], self.emb_dim, kernel_size=1, bias=False)
        self.full_up_block = BilinearAddResUpBlock2d(
            self.emb_dim,
            self.emb_dim,
            self.emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            resblock_act=resblock_act,
            resblock_norm=resblock_norm,
        )
        self.out = ResBlock2d(
            self.emb_dim,
            self.emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )

    def forward(self, x):
        stem_skip = self.stem(x)
        x = self.stem_down(stem_skip)
        x = self.to_backbone(x)

        stage_skips = []
        for stage_idx, stage in enumerate(self.backbone_stages):
            x = stage(x)
            if stage_idx < len(self.backbone_stages) - 1:
                stage_skips.append(x)

        for up_block, skip in zip(self.up_blocks, reversed(stage_skips)):
            x = up_block(x, skip)
        x = self.to_full_resolution(x)
        x = self.full_up_block(x, stem_skip)
        return self.out(x)
