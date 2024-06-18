import warp as wp
import torch

# half precision
WP_FLOAT16 = wp.float16
# originally using bf16 but warp only supports float16 for now
WP_FLOAT32 = wp.float32

WP_INT = wp.int32
WP_VEC2 = wp.vec2
WP_VEC2H = wp.vec2h

DEVICE = "cuda:0"  # "cpu"
TORCH_FLOAT = torch.float32