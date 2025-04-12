import numpy as np
import warp as wp
from config import *
# Initialize Warp
wp.init()

syncthreads_snippet = """
__syncthreads();
"""

@wp.func_native(syncthreads_snippet)
def syncthreads():
    return

TILE_M = wp.constant(32)
TILE_N = wp.constant(32)
TILE_THREADS = 128

@wp.kernel
def wp_embedding(embedding: wp.array2d(dtype=WP_FLOAT32), tokens: wp.array(dtype=WP_INT), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    idx = tokens[i]
    output[i, j] = embedding[idx, j]

@wp.kernel
def wp_pow(tensor: wp.array2d(dtype=WP_FLOAT32), power: WP_FLOAT32, output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.pow(tensor[i, j], power)

# actually not used
@wp.kernel
def wp_mean(tensor: wp.array2d(dtype=WP_FLOAT32), axis: WP_INT, output: wp.array(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    if axis == 0:
        wp.atomic_add(output, i, tensor[i, j])
    elif axis == 1:
        wp.atomic_add(output, j, tensor[i, j])

@wp.kernel
def wp_sqrt(tensor: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.sqrt(tensor[i, j])

@wp.kernel
def wp_rms_norm(tensor: wp.array2d(dtype=WP_FLOAT32), norm_eps:WP_FLOAT32, norm_weights: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i = wp.tid()
    tensor_2_mean_tid = WP_FLOAT32(0.0)
    for j in range(tensor.shape[1]):
        tensor_2_mean_tid = tensor_2_mean_tid + tensor[i, j] * tensor[i, j]
    tensor_2_mean_tid = tensor_2_mean_tid / WP_FLOAT32(tensor.shape[1]) + norm_eps
    rms_tid = wp.sqrt(tensor_2_mean_tid)
    for j in range(tensor.shape[1]):
        output[i, j] = tensor[i, j] * norm_weights[j] / rms_tid

@wp.kernel
def wp_compute_freqs(rope_theta: WP_FLOAT32, zero_to_one_split_into_64_parts: wp.array(dtype=WP_FLOAT32), freqs: wp.array(dtype=WP_FLOAT32)):
    tid = wp.tid()
    freqs[tid] = WP_FLOAT32(1.0) / wp.pow(rope_theta, zero_to_one_split_into_64_parts[tid])

@wp.kernel
def wp_outer_1d_1d_2d(a: wp.array(dtype=WP_FLOAT32), b: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = a[i] * b[j]

@wp.kernel
def wp_polar(inputs: wp.array2d(dtype=WP_FLOAT32), outputs: wp.array3d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    outputs[i, j][0] = wp.cos(inputs[i, j])
    outputs[i, j][1] = wp.sin(inputs[i, j])

@wp.kernel
def wp_complex_multiply(complex1: wp.array3d(dtype=WP_FLOAT32), complex2: wp.array3d(dtype=WP_FLOAT32), output: wp.array3d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    # Extract real and imaginary parts
    real1 = complex1[i, j][0]
    imag1 = complex1[i, j][1]
    real2 = complex2[i, j][0]
    imag2 = complex2[i, j][1]

    # Perform complex multiplication
    real_output = real1 * real2 - imag1 * imag2
    imag_output = real1 * imag2 + imag1 * real2

    # Store the result in the output array
    output[i, j][0] = real_output
    output[i, j][1] = imag_output

@wp.kernel
def wp_triu(mask: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()  # Get the thread indices
    if j < i + 1:
        mask[i, j] = 0.0  # Keep the values above the diagonal (including diagonal) as 0

@wp.kernel
def wp_softmax_exp_sum(qk: wp.array2d(dtype=WP_FLOAT32), mask: wp.array2d(dtype=WP_FLOAT32), exp: wp.array(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    wp.atomic_add(exp, i, wp.exp(qk[i, j] + mask[i, j]))

@wp.kernel
def wp_softmax_exp_normalize(qk: wp.array2d(dtype=WP_FLOAT32), mask: wp.array2d(dtype=WP_FLOAT32), exp: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.exp(qk[i, j] + mask[i, j]) / exp[i]

@wp.kernel
def wp_add(a: wp.array2d(dtype=WP_FLOAT32), b: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = a[i, j] + b[i, j]

@wp.kernel
def wp_mul(a: wp.array2d(dtype=WP_FLOAT32), b: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = a[i, j] * b[i, j]

@wp.kernel
def wp_stack(head: WP_INT, stack_tensor: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j + head * stack_tensor.shape[1]] = stack_tensor[i, j]

@wp.kernel
def wp_silu(x: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()  # Get the row index
    sig = 1.0 / (1.0 + wp.exp(-x[i, j]))  # Sigmoid function
    output[i, j] = x[i, j] * sig  # SiLU activation

@wp.kernel
def wp_access_d1(inputs: wp.array2d(dtype=WP_FLOAT32), outputs: wp.array2d(dtype=WP_FLOAT32), index: WP_INT):
    j = wp.tid()
    outputs[0, j] = inputs[index, j]

@wp.kernel
def wp_argmax(logits: wp.array2d(dtype=WP_FLOAT32), max_logit: wp.array(dtype=WP_FLOAT32)):
    i = wp.tid()
    wp.atomic_max(max_logit, 0, logits[0, i])

@wp.kernel
def wp_index(logits: wp.array2d(dtype=WP_FLOAT32), logit_value: wp.array(dtype=WP_FLOAT32), logit_index: wp.array(dtype=WP_INT)):
    i = wp.tid()
    if logits[0, i] == logit_value[0]:
        logit_index[0] = i

# @wp.kernel
# def wp_softmax(qk: wp.array2d(dtype=WP_FLOAT32), mask: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
#     i = wp.tid()
#     exp_sum = WP_FLOAT32(0.0)
#     for j in range(qk.shape[1]):
#         exp_sum = exp_sum + wp.exp(qk[i, j] + mask[i, j])

#     for j in range(qk.shape[1]):
#         output[i, j] = wp.exp(qk[i, j] + mask[i, j]) / exp_sum

# I think we have to use for loops sometimes
# Maybe it's better to use for loops than launch a kernel more than once
def wp_softmax(qk_per_token, mask):
    qk_per_token_after_masking_after_softmax = wp.zeros(shape=qk_per_token.shape, dtype=WP_FLOAT32)
    exp = wp.zeros(shape=qk_per_token.shape[0], dtype=WP_FLOAT32)
    wp.launch(kernel=wp_softmax_exp_sum, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp])
    wp.launch(kernel=wp_softmax_exp_normalize, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp], outputs=[qk_per_token_after_masking_after_softmax])
    return qk_per_token_after_masking_after_softmax

@wp.kernel
def wp_exp(qk: wp.array2d(dtype=WP_FLOAT32), mask: wp.array2d(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    output[i, j] = wp.exp(qk[i, j] + mask[i, j])

def tiled_softmax(qk_per_token, mask):
    qk_per_token_after_masking_after_softmax = wp.zeros(shape=qk_per_token.shape, dtype=WP_FLOAT32)
    exp = wp.zeros(shape=qk_per_token.shape, dtype=WP_FLOAT32)
    wp.launch(wp_exp, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp])
    exp_sum = wp.zeros(shape=qk_per_token.shape[0], dtype=WP_FLOAT32)
    wp.launch_tiled(wp_tile_sum_d0, dim=(qk_per_token.shape[0], qk_per_token.shape[1] // TILE_N + 1), inputs=[exp, exp_sum], block_dim=TILE_THREADS)
    wp.launch(wp_softmax_exp_normalize, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp_sum], outputs=[qk_per_token_after_masking_after_softmax])
    return qk_per_token_after_masking_after_softmax

@wp.kernel
def wp_tile_sum_d0(tensor: wp.array2d(dtype=WP_FLOAT32), output: wp.array(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    tile = wp.tile_load(tensor, shape=(1, TILE_N), offset=(i, j*TILE_N))
    tile_sum = wp.tile_sum(tile)
    wp.tile_atomic_add(output, tile_sum, offset=(i,))

@wp.kernel
def wp_simt_rms_norm(tensor: wp.array2d(dtype=WP_FLOAT32), sum_tensor: wp.array(dtype=WP_FLOAT32), norm_eps:WP_FLOAT32, norm_weights: wp.array(dtype=WP_FLOAT32), output: wp.array2d(dtype=WP_FLOAT32)):
    i, j = wp.tid()
    norm = sum_tensor[i] / WP_FLOAT32(tensor.shape[1]) + norm_eps
    rms = wp.sqrt(norm)
    output[i, j] = tensor[i, j] * norm_weights[j] / rms

def tiled_rms_norm(tensor, norm_eps, norm_weights):
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    tensor_square = wp.zeros(shape=(rows, cols), dtype=WP_FLOAT32)
    wp.launch(
        wp_pow,
        dim=(rows, cols),
        inputs=[tensor, 2, tensor_square],
    )
    tensor_square_sum_d0 = wp.zeros(shape=(rows,), dtype=WP_FLOAT32)
    wp.launch_tiled(
        wp_tile_sum_d0,
        dim=(rows, cols // TILE_N),
        inputs=[tensor_square, tensor_square_sum_d0],
        block_dim=TILE_THREADS
    )
    output = wp.zeros(shape=(rows, cols), dtype=WP_FLOAT32)
    wp.launch(
        wp_simt_rms_norm,
        dim=(rows, cols),
        inputs=[tensor, tensor_square_sum_d0, norm_eps, norm_weights, output]
    )
    return output