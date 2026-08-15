TRF_UNET_BACKBONES = {
    "caformer_s18": {
        "timm_model_name": "caformer_s18.sail_in22k_ft_in1k_384",
        "out_indices": (0, 1, 2, 3),
        "accepts_img_size": True,
        "feature_layout": "NCHW",
    },
    "swin_tiny": {
        "timm_model_name": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "out_indices": (0, 1, 2, 3),
        "accepts_img_size": True,
        "feature_layout": "NHWC",
    },
    "swin_small": {
        "timm_model_name": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        "out_indices": (0, 1, 2, 3),
        "accepts_img_size": True,
        "feature_layout": "NHWC",
    },
    "swin_base": {
        "timm_model_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "out_indices": (0, 1, 2, 3),
        "accepts_img_size": True,
        "feature_layout": "NHWC",
    },
    "pvt_v2_b2": {
        "timm_model_name": "pvt_v2_b2.in1k",
        "out_indices": (0, 1, 2, 3),
        "accepts_img_size": False,
        "feature_layout": "NCHW",
    },
    "twins_svt_small": {
        "timm_model_name": "twins_svt_small.in1k",
        "out_indices": (0, 1, 2, 3),
        "accepts_img_size": True,
        "feature_layout": "NCHW",
    },
    "maxvit_rmlp_small": {
        "timm_model_name": "maxvit_rmlp_small_rw_224.sw_in1k",
        "out_indices": (1, 2, 3, 4),
        "accepts_img_size": True,
        "feature_layout": "NCHW",
    },
    "coatnet_1": {
        "timm_model_name": "coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k",
        "out_indices": (1, 2, 3, 4),
        "accepts_img_size": True,
        "feature_layout": "NCHW",
    },
    "coatnet_2": {
        "timm_model_name": "coatnet_2_rw_224.sw_in12k_ft_in1k",
        "out_indices": (1, 2, 3, 4),
        "accepts_img_size": True,
        "feature_layout": "NCHW",
    },
}
