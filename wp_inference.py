from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import wp_module as wpm
from config import *
import time
import argparse
import contextlib
from utils import WarpProfilingManager

parser = argparse.ArgumentParser(description="Optional profiling.")
parser.add_argument('--profile', action='store_true', help="Enable profiling.")
args = parser.parse_args()

pm = WarpProfilingManager(args.profile) 
t1 = time.time()
with pm.wp_profile_function("Warp Inference"):
    tokenizer_path = "Meta-Llama-3-8B/tokenizer.model"
    special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",  # end of turn
            ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    tokenizer = tiktoken.Encoding(
        name=Path(tokenizer_path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
    )

    model = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
    model = {k: wp.array(v.to(torch.float32), dtype=WP_FLOAT32) for k, v in model.items()}

    with open("Meta-Llama-3-8B/params.json", "r") as f:
        config = json.load(f)

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    multiple_of = config["multiple_of"]
    ffn_dim_multiplier = config["ffn_dim_multiplier"]
    norm_eps = config["norm_eps"]
    rope_theta = WP_FLOAT32(config["rope_theta"])

    prompt = "the answer to the ultimate question of life, the universe, and everything is "
    tokens_ = [128000] + tokenizer.encode(prompt)
    tokens_ = torch.tensor(tokens_, dtype=torch.int32)
    tokens = wp.from_torch(tokens_.to(DEVICE), dtype=WP_INT)
    
    with pm.wp_profile_function("Embedding Layer"):
        embedding_layer_weight = model["tok_embeddings.weight"]
        token_embeddings_unnormalized = wp.zeros((tokens.shape[0], dim), dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_embedding, dim=(tokens.shape[0], embedding_layer_weight.shape[1]), inputs=[embedding_layer_weight, tokens, token_embeddings_unnormalized])

    with pm.wp_profile_function("Q Layer Prep"):
        q_layer0 = model["layers.0.attention.wq.weight"]
        head_dim = q_layer0.shape[0] // n_heads
        q_layer0 = q_layer0.reshape((n_heads, head_dim, dim))

    with pm.wp_profile_function("Freqs Prep"):
        zero_to_one_split_into_64_parts = wp.array(data=np.arange(64)/64, dtype=WP_FLOAT32)
        freqs = wp.zeros(shape=zero_to_one_split_into_64_parts.shape, dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_compute_freqs, dim=zero_to_one_split_into_64_parts.shape[0], inputs=[rope_theta, zero_to_one_split_into_64_parts, freqs])

        freqs_for_each_token = wp.zeros(shape=(17, freqs.shape[0]), dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_outer_1d_1d_2d, dim=freqs_for_each_token.shape, inputs=[wp.array(data=np.arange(17), dtype=WP_FLOAT32), freqs, freqs_for_each_token])

        freqs_cis = wp.array(shape=(freqs_for_each_token.shape[0], freqs_for_each_token.shape[1], 2), dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_polar, dim=freqs_for_each_token.shape, inputs=[freqs_for_each_token, freqs_cis])

    with pm.wp_profile_function("K and V Layer Prep"):
        k_layer0 = model["layers.0.attention.wk.weight"]
        k_layer0 = k_layer0.reshape((n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim))
        v_layer0 = model["layers.0.attention.wv.weight"]
        v_layer0 = v_layer0.reshape((n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim))

    final_embedding = token_embeddings_unnormalized

    for layer in range(n_layers):
        with pm.wp_profile_function(f"Layer {layer}"):
            qkv_attention_store_ = []
            stacked_qkv_attention = wp.zeros(shape=(tokens.shape[0], v_layer0[0].shape[0] * n_heads), dtype=WP_FLOAT32)

            layer_embedding_norm = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
            wp.launch(kernel=wpm.wp_rms_norm, dim=final_embedding.shape[0], inputs=[final_embedding, WP_FLOAT32(norm_eps), model[f"layers.{layer}.attention_norm.weight"], layer_embedding_norm])

            q_layer = model[f"layers.{layer}.attention.wq.weight"]
            head_dim = q_layer.shape[0] // n_heads
            q_layer = q_layer.reshape((n_heads, head_dim, dim))
            k_layer = model[f"layers.{layer}.attention.wk.weight"]
            k_layer = k_layer.reshape((n_kv_heads, k_layer.shape[0] // n_kv_heads, dim))
            v_layer = model[f"layers.{layer}.attention.wv.weight"]
            v_layer = v_layer.reshape((n_kv_heads, v_layer.shape[0] // n_kv_heads, dim))
            w_layer = model[f"layers.{layer}.attention.wo.weight"]

            for head in range(n_heads):
                with pm.wp_profile_function(f"Layer {layer} Attention Head {head}"):
                    q_layer_head = q_layer[head]
                    k_layer_head = k_layer[head//4]
                    v_layer_head = v_layer[head//4]
                    q_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], q_layer_head.shape[0]), dtype=WP_FLOAT32)
                    c = wp.zeros(shape=(layer_embedding_norm.shape[0], q_layer_head.shape[0]), dtype=WP_FLOAT32)
                    wp.matmul(layer_embedding_norm, q_layer_head.transpose(), c, q_per_token)

                    k_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], k_layer_head.shape[0]), dtype=WP_FLOAT32)
                    c = wp.zeros(shape=(layer_embedding_norm.shape[0], k_layer_head.shape[0]), dtype=WP_FLOAT32)
                    wp.matmul(layer_embedding_norm, k_layer_head.transpose(), c, k_per_token)
                    v_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], v_layer_head.shape[0]), dtype=WP_FLOAT32)
                    c = wp.zeros(shape=(layer_embedding_norm.shape[0], v_layer_head.shape[0]), dtype=WP_FLOAT32)
                    wp.matmul(layer_embedding_norm, v_layer_head.transpose(), c, v_per_token)

                    q_per_token_split_into_pair = q_per_token.reshape((q_per_token.shape[0], -1, 2))
                    q_per_token_split_into_pairs_rotated = wp.zeros(shape=q_per_token_split_into_pair.shape, dtype=WP_FLOAT32)
                    wp.launch(kernel=wpm.wp_complex_multiply, dim=(q_per_token_split_into_pair.shape[0], q_per_token_split_into_pair.shape[1]), inputs=[q_per_token_split_into_pair, freqs_cis, q_per_token_split_into_pairs_rotated])
                    q_per_token_rotated = q_per_token_split_into_pairs_rotated.reshape(q_per_token.shape)

                    k_per_token_split_into_pairs = k_per_token.reshape((k_per_token.shape[0], -1, 2))
                    k_per_token_split_into_pairs_rotated = wp.zeros(shape=k_per_token_split_into_pairs.shape, dtype=WP_FLOAT32)
                    wp.launch(kernel=wpm.wp_complex_multiply, dim=(k_per_token_split_into_pairs.shape[0], k_per_token_split_into_pairs.shape[1]), inputs=[k_per_token_split_into_pairs, freqs_cis, k_per_token_split_into_pairs_rotated])
                    k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)

                    qk_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]), dtype=WP_FLOAT32)
                    c = wp.zeros(shape=(layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]), dtype=WP_FLOAT32)
                    alpha = WP_FLOAT32(1.0 / wp.sqrt(WP_FLOAT32(128)))
                    wp.matmul(q_per_token_rotated, k_per_token_rotated.transpose(), c, qk_per_token, alpha=alpha)

                    mask = wp.full((layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]), WP_FLOAT32(float("-inf")), dtype=WP_FLOAT32, device=DEVICE)
                    wp.launch(kernel=wpm.wp_triu, dim=mask.shape, inputs=[mask])

                    qk_per_token_after_masking_after_softmax = wpm.wp_softmax(qk_per_token, mask)

                    qkv_attention = wp.zeros(shape=(layer_embedding_norm.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
                    c = wp.zeros(shape=(layer_embedding_norm.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
                    wp.matmul(qk_per_token_after_masking_after_softmax, v_per_token, c, qkv_attention)
                    wp.launch(kernel=wpm.wp_stack, dim=qkv_attention.shape, inputs=[WP_INT(head), qkv_attention, stacked_qkv_attention])

            w_layer = model[f"layers.{layer}.attention.wo.weight"]
            embedding_delta = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
            c = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
            wp.matmul(stacked_qkv_attention, w_layer.transpose(), c, embedding_delta)
            embedding_after_edit = wp.zeros(shape=final_embedding.shape, dtype=WP_FLOAT32)
            wp.launch(kernel=wpm.wp_add, dim=final_embedding.shape, inputs=[final_embedding, embedding_delta, embedding_after_edit])
            embedding_after_edit_normalized = wp.zeros(shape=embedding_after_edit.shape, dtype=WP_FLOAT32)
            wp.launch(kernel=wpm.wp_rms_norm, dim=embedding_after_edit.shape[0], inputs=[embedding_after_edit, WP_FLOAT32(norm_eps), model[f"layers.{layer}.ffn_norm.weight"], embedding_after_edit_normalized])

            w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
            tmp_left = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w1.shape[0]), dtype=WP_FLOAT32)
            c = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w1.shape[0]), dtype=WP_FLOAT32)
            wp.matmul(embedding_after_edit_normalized, w1.transpose(), c, tmp_left)
            tmp_left_silu = wp.zeros(shape=tmp_left.shape, dtype=WP_FLOAT32)
            wp.launch(kernel=wpm.wp_silu, dim=tmp_left.shape, inputs=[tmp_left, tmp_left_silu])

            tmp_right = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w3.shape[0]), dtype=WP_FLOAT32)
            c = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w3.shape[0]), dtype=WP_FLOAT32)
            wp.matmul(embedding_after_edit_normalized, w3.transpose(), c, tmp_right)

            tmp = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w2.shape[1]), dtype=WP_FLOAT32)
            wp.launch(kernel=wpm.wp_mul, dim=tmp.shape, inputs=[tmp_left_silu, tmp_right, tmp])

            output_after_feedforward = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w2.shape[0]), dtype=WP_FLOAT32)
            c = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w2.shape[0]), dtype=WP_FLOAT32)
            wp.matmul(tmp, w2.transpose(), c, output_after_feedforward)
            final_embedding = wp.zeros(shape=embedding_after_edit.shape, dtype=WP_FLOAT32)
            wp.launch(kernel=wpm.wp_add, dim=embedding_after_edit.shape, inputs=[embedding_after_edit, output_after_feedforward, final_embedding])

    with pm.wp_profile_function(f"Decode Token"):
        final_embedding_rms = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_rms_norm, dim=final_embedding.shape[0], inputs=[final_embedding, WP_FLOAT32(norm_eps), model["norm.weight"], final_embedding_rms])

        last_final_embedding = wp.zeros(shape=(1, final_embedding_rms.shape[1]), dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_access_d1, dim=final_embedding_rms.shape[1], inputs=[final_embedding_rms, last_final_embedding, WP_INT(final_embedding_rms.shape[0]-1)])
        logits = wp.zeros(shape=(1, model["output.weight"].shape[0]), dtype=WP_FLOAT32)
        c = wp.zeros(shape=(1, model["output.weight"].shape[0]), dtype=WP_FLOAT32)
        wp.matmul(last_final_embedding, model["output.weight"].transpose(), c, logits)

        next_token = wp.zeros(shape=(1,), dtype=WP_INT)
        max_logit = wp.zeros(shape=(1,), dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_argmax, dim=logits.shape[1], inputs=[logits, max_logit])
        wp.launch(kernel=wpm.wp_index, dim=logits.shape[1], inputs=[logits, max_logit, next_token])
        print(tokenizer.decode([wp.to_torch(next_token).item()]))

t2 = time.time()
print("Time taken: ", t2 - t1)