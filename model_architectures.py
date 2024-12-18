from model import HybridBlock

FRAMES = {
    'frames_stanford_v1': [ # 440k params
        HybridBlock(k_size=3, out_filters=16, strides=(1, 1), se_ratio=0.25, expand_ratio=1),
        HybridBlock(k_size=3, out_filters=24, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=3, out_filters=24, strides=(1, 1), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=48, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=48, strides=(1, 1), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=96, strides=(3, 3), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=192, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=256, strides=(2, 2), se_ratio=0.25, expand_ratio=1),
    ],
    'frames_stanford_v2': [ # 230k params
        HybridBlock(k_size=3, out_filters=16, strides=(1, 1), se_ratio=0.25, expand_ratio=1),
        HybridBlock(k_size=3, out_filters=24, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=3, out_filters=24, strides=(1, 1), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=48, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=48, strides=(1, 1), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=96, strides=(3, 3), se_ratio=0.25, expand_ratio=3),
        HybridBlock(k_size=5, out_filters=192, strides=(2, 2), se_ratio=0.25, expand_ratio=2),
        # HybridBlock(k_size=5, out_filters=256, strides=(2, 2), se_ratio=0.25, expand_ratio=1),
    ],
    'frames_stanford_half_v2': [  # 230k params
        HybridBlock(k_size=3, out_filters=16, strides=(1, 1), se_ratio=0.25, expand_ratio=1),
        HybridBlock(k_size=3, out_filters=24, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=3, out_filters=24, strides=(1, 1), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=48, strides=(2, 2), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=48, strides=(1, 1), se_ratio=0.25, expand_ratio=4),
        HybridBlock(k_size=5, out_filters=96, strides=(3, 3), se_ratio=0.25, expand_ratio=3),
        # HybridBlock(k_size=5, out_filters=192, strides=(2, 2), se_ratio=0.25, expand_ratio=2),
        # HybridBlock(k_size=5, out_filters=256, strides=(2, 2), se_ratio=0.25, expand_ratio=1),
    ]
}

FUSION_MODEL = {
    'fusion_layerless': [

    ],
    'fusion_b67': [
        HybridBlock(k_size=5, out_filters=192, strides=(2, 2), se_ratio=0.25, expand_ratio=2),
        HybridBlock(k_size=5, out_filters=256, strides=(2, 2), se_ratio=0.25, expand_ratio=1),
    ]
}