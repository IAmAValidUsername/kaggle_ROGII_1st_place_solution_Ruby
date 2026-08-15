import math

import torch
from torch import nn
import torch.nn.functional as F

from seq_NN_pretrained_unet import (
    BilinearAddResUpBlock2d,
    BilinearResUpBlock2d,
    ResBlock2d,
    _create_timm_model_with_retries,
)
from seq_NN_trf_backbones import TRF_UNET_BACKBONES


def _pair(value, name):
    if isinstance(value, int):
        out = (value, value)
    else:
        out = tuple(value)
    if len(out) != 2:
        raise ValueError(f"{name} must be an int or length-2 tuple, got {value!r}")
    out = (int(out[0]), int(out[1]))
    if any(entry <= 0 for entry in out):
        raise ValueError(f"{name} entries must be positive, got {value!r}")
    return out


def _padded_size(input_size, multiple):
    return tuple(
        int(math.ceil(size / stride) * stride)
        for size, stride in zip(input_size, multiple)
    )


def _feature_info_values(feature_info, method_name):
    method = getattr(feature_info, method_name, None)
    if callable(method):
        return tuple(int(value) for value in method())
    values = []
    for info in list(feature_info):
        key = "num_chs" if method_name == "channels" else "reduction"
        values.append(int(info[key]))
    return tuple(values)


class PretrainedTransformerUNet2d(nn.Module):
    """Timm feature-pyramid backbone with a local ResBlock U-Net decoder."""

    def __init__(
        self,
        in_ch,
        emb_dim,
        input_size,
        backbone_name="swin_tiny",
        pretrained=True,
        pad_value=0.0,
        kernel_size=3,
        resblock_act="silu",
        resblock_norm="BN",
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        backbone_name = str(backbone_name)
        if backbone_name not in TRF_UNET_BACKBONES:
            available = ", ".join(sorted(TRF_UNET_BACKBONES))
            raise ValueError(
                f"unknown Transformer U-Net backbone {backbone_name!r}; "
                f"available: {available}"
            )
        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        self.input_size = _pair(input_size, "input_size")
        self.pad_value = float(pad_value)
        if not math.isfinite(self.pad_value):
            raise ValueError(f"pad_value must be finite, got {pad_value}")
        self.emb_dim = int(emb_dim)
        if self.emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        dropout = float(dropout)
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be finite and in [0, 1], got {dropout}")
        self.drop_path_rate = float(drop_path_rate)
        if not math.isfinite(self.drop_path_rate) or not 0.0 <= self.drop_path_rate <= 1.0:
            raise ValueError(
                f"drop_path_rate must be finite and in [0, 1], got {drop_path_rate}"
            )

        backbone_cfg = TRF_UNET_BACKBONES[backbone_name]
        self.timm_model_name = backbone_cfg["timm_model_name"]
        self.out_indices = tuple(backbone_cfg["out_indices"])
        self.feature_layout = str(backbone_cfg["feature_layout"])
        self.padded_size = _padded_size(self.input_size, (32, 32))
        try:
            import timm
        except ImportError as exc:
            raise ImportError("model_name='trf_unet' requires the timm package") from exc

        model_kwargs = {
            "features_only": True,
            "out_indices": self.out_indices,
            "in_chans": int(in_ch),
            "drop_path_rate": self.drop_path_rate,
        }
        if backbone_cfg["accepts_img_size"]:
            model_kwargs["img_size"] = self.padded_size
        self.backbone = _create_timm_model_with_retries(
            timm,
            self.timm_model_name,
            pretrained=self.pretrained,
            **model_kwargs,
        )

        feature_info = getattr(self.backbone, "feature_info", None)
        if feature_info is None:
            raise TypeError(
                f"timm model {self.timm_model_name!r} does not expose feature_info"
            )
        stage_channels = _feature_info_values(feature_info, "channels")
        stage_reductions = _feature_info_values(feature_info, "reduction")
        if len(stage_channels) != 4 or stage_reductions != (4, 8, 16, 32):
            raise ValueError(
                f"{self.timm_model_name!r} must expose four reductions (4, 8, 16, 32), "
                f"got channels={stage_channels}, reductions={stage_reductions}"
            )
        self.stage_channels = stage_channels
        self.stage_reductions = stage_reductions

        self.stem = ResBlock2d(
            int(in_ch),
            self.emb_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            act=resblock_act,
            norm=resblock_norm,
        )
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
        self.deep_sup_stage_channels = tuple(stage_channels[2::-1]) + (self.emb_dim,)
        self.to_full_resolution = nn.Conv2d(
            stage_channels[0],
            self.emb_dim,
            kernel_size=1,
            bias=False,
        )
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

    @staticmethod
    def _to_nchw(feature, channels, layout, backbone_name):
        if feature.ndim != 4:
            raise ValueError(
                f"{backbone_name}: expected a 4D feature map, got {tuple(feature.shape)}"
            )
        if layout == "NCHW":
            if feature.shape[1] != channels:
                raise ValueError(
                    f"{backbone_name}: expected {channels} NCHW feature channels, "
                    f"got shape={tuple(feature.shape)}"
                )
            return feature
        if layout == "NHWC":
            if feature.shape[-1] != channels:
                raise ValueError(
                    f"{backbone_name}: expected {channels} NHWC feature channels, "
                    f"got shape={tuple(feature.shape)}"
                )
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"{backbone_name}: unsupported feature layout {layout!r}"
        )

    def forward(self, x, return_stages=False):
        if tuple(x.shape[-2:]) != self.input_size:
            raise ValueError(
                f"{self.backbone_name} expects input spatial size {self.input_size}, "
                f"got {tuple(x.shape[-2:])}"
            )
        pad_h = self.padded_size[0] - self.input_size[0]
        pad_w = self.padded_size[1] - self.input_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h), value=self.pad_value)
        stem_skip = self.stem(x)
        features = self.backbone(x)
        if len(features) != len(self.stage_channels):
            raise ValueError(
                f"{self.backbone_name}: expected {len(self.stage_channels)} feature maps, "
                f"got {len(features)}"
            )
        features = tuple(
            self._to_nchw(
                feature,
                channels,
                self.feature_layout,
                self.backbone_name,
            )
            for feature, channels in zip(features, self.stage_channels)
        )

        x = features[-1]
        stage_features = [] if return_stages else None
        for up_block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = up_block(x, skip)
            if return_stages:
                stage_features.append(x)
        x = self.to_full_resolution(x)
        x = self.full_up_block(x, stem_skip)
        if return_stages:
            stage_features.append(x[..., : self.input_size[0], : self.input_size[1]])
        output = self.out(x)
        output = output[..., : self.input_size[0], : self.input_size[1]]
        if return_stages:
            return output, tuple(stage_features)
        return output
