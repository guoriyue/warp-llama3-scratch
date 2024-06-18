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

# DEVICE = 'cpu'
DEVICE = 'cuda:0'

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

tokenizer.decode(tokenizer.encode("hello world!"))

# for correctness check, we use float32 for both warp and torch
# because the result is not accurate enough with float16
# and we run the torch version on the CPU because of the memory limitation
model_ = torch.load("Meta-Llama-3-8B/consolidated.00.pth")
model_ = {k: v.to(torch.float32) for k, v in model_.items()}

# wp
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
rope_theta_ = torch.tensor(config["rope_theta"])
rope_theta = WP_FLOAT32(config["rope_theta"])

# ============= Embedding =============
prompt = "the answer to the ultimate question of life, the universe, and everything is "
tokens_ = [128000] + tokenizer.encode(prompt)
tokens_ = torch.tensor(tokens_, dtype=torch.int32)
prompt_split_as_tokens_ = [tokenizer.decode([token.item()]) for token in tokens_]

embedding_layer_ = torch.nn.Embedding(vocab_size, dim)
embedding_layer_.weight.data.copy_(model_["tok_embeddings.weight"])
# originally bfloat16 but we need float16 to be compatible with warp
token_embeddings_unnormalized_ = embedding_layer_(tokens_)

# Warp version
tokens = wp.from_torch(tokens_.to(DEVICE), dtype=WP_INT)
embedding_layer_weight = model["tok_embeddings.weight"]
token_embeddings_unnormalized = wp.zeros((tokens.shape[0], dim), dtype=WP_FLOAT32) 
wp.launch(kernel=wpm.wp_embedding, dim=(tokens.shape[0], embedding_layer_weight.shape[1]), inputs=[embedding_layer_weight, tokens, token_embeddings_unnormalized])

assert torch.allclose(token_embeddings_unnormalized_, wp.to_torch(token_embeddings_unnormalized).cpu(), rtol=1e-05, atol=1e-08), "token_embeddings_unnormalized incorrect!"



# ============= RMS Norm =============
# def rms_norm(tensor, norm_weights):
#     rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
#     return tensor * (norm_weights / rms)

def rms_norm(tensor, norm_weights):
    rms = (tensor.pow(2).mean(-1, keepdim=True) + norm_eps)**0.5
    return tensor * (norm_weights / rms)

token_embeddings_ = rms_norm(token_embeddings_unnormalized_, model_["layers.0.attention_norm.weight"])
print("layers.0.attention_norm.weight", model_["layers.0.attention_norm.weight"].shape)
print("layers.0.attention.wq.weight", model_["layers.0.attention.wq.weight"].shape)
print("layers.0.attention.wk.weight", model_["layers.0.attention.wk.weight"].shape)
print("layers.0.attention.wv.weight", model_["layers.0.attention.wv.weight"].shape)
print("layers.0.attention.wo.weight", model_["layers.0.attention.wo.weight"].shape)

# Warp version
token_embeddings = wp.zeros(token_embeddings_unnormalized.shape, dtype=WP_FLOAT32)
model_layer_0_attention_norm_weight = wp.array(data=wp.to_torch(model["layers.0.attention_norm.weight"]).to(torch.float32), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_rms_norm, dim=token_embeddings.shape[0], inputs=[token_embeddings_unnormalized, WP_FLOAT32(norm_eps), model_layer_0_attention_norm_weight, token_embeddings])

assert torch.allclose(token_embeddings_, wp.to_torch(token_embeddings).cpu(), rtol=1e-05, atol=1e-08), "token_embeddings incorrect!"

token_embeddings_ = token_embeddings_

# assert torch.allclose(token_embeddings_, wp.to_torch(token_embeddings), rtol=1e-05, atol=1e-05), "FLOAT16: token_embeddings incorrect!"

# ============= Q =============
q_layer0_ = model_["layers.0.attention.wq.weight"]
head_dim_ = q_layer0_.shape[0] // n_heads
q_layer0_ = q_layer0_.view(n_heads, head_dim_, dim)


q_layer0_head0_ = q_layer0_[0]

q_per_token_ = torch.matmul(token_embeddings_, q_layer0_head0_.T)

q_per_token_split_into_pairs_ = q_per_token_.float().view(q_per_token_.shape[0], -1, 2)

zero_to_one_split_into_64_parts_ = (torch.tensor(range(64))/64)

freqs_ = 1.0 / (rope_theta_ ** zero_to_one_split_into_64_parts_)

freqs_for_each_token_ = torch.outer(torch.arange(17), freqs_)
# polar not implemented for Half
freqs_cis_ = torch.polar(torch.ones_like(freqs_for_each_token_), freqs_for_each_token_)

# Warp version
q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.reshape((n_heads, head_dim, dim))
q_layer0_head0 = q_layer0[0]

assert torch.allclose(q_layer0_head0_.T, wp.to_torch(q_layer0_head0.transpose()).cpu(), rtol=1e-05, atol=1e-08), "q_layer0_head0.transpose() incorrect!"


# `d = alpha * (a @ b) + beta * c`
q_per_token = wp.zeros(shape=(token_embeddings.shape[0], q_layer0_head0.shape[0]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(token_embeddings.shape[0], q_layer0_head0.shape[0]), dtype=WP_FLOAT32)
wp.matmul(token_embeddings, q_layer0_head0.transpose(), c, q_per_token)

assert torch.allclose(q_per_token_, wp.to_torch(q_per_token).cpu(), rtol=1e-05, atol=1e-05), "q_per_token incorrect!"


q_per_token_split_into_pairs = q_per_token.reshape((q_per_token.shape[0], -1, 2))

zero_to_one_split_into_64_parts = wp.array(data=np.arange(64)/64, dtype=WP_FLOAT32)

freqs = wp.zeros(shape=zero_to_one_split_into_64_parts.shape, dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_compute_freqs, dim=zero_to_one_split_into_64_parts.shape[0], inputs=[rope_theta, zero_to_one_split_into_64_parts, freqs])

assert torch.allclose(freqs_, wp.to_torch(freqs).cpu(), rtol=1e-05, atol=1e-08), "freqs incorrect!"

freqs_for_each_token = wp.zeros(shape=(17, freqs.shape[0]), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_outer_1d_1d_2d, dim=freqs_for_each_token.shape, inputs=[wp.array(data=np.arange(17), dtype=WP_FLOAT32), freqs, freqs_for_each_token])

assert torch.allclose(freqs_for_each_token_, wp.to_torch(freqs_for_each_token).cpu(), rtol=1e-05, atol=1e-05), "freqs_for_each_token incorrect!"

# vec2, 0 for real part and 1 for imag part
freqs_cis = wp.array(shape=(freqs_for_each_token.shape[0], freqs_for_each_token.shape[1], 2), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_polar, dim=freqs_for_each_token.shape, inputs=[freqs_for_each_token, freqs_cis])

q_per_token_as_complex_numbers_ = torch.view_as_complex(q_per_token_split_into_pairs_)
q_per_token_as_complex_numbers_rotated_ = q_per_token_as_complex_numbers_ * freqs_cis_
q_per_token_split_into_pairs_rotated_ = torch.view_as_real(q_per_token_as_complex_numbers_rotated_)
q_per_token_rotated_ = q_per_token_split_into_pairs_rotated_.view(q_per_token_.shape)


# q_per_token_split_into_pairs_.shape torch.Size([17, 64, 2])
# freqs_cis_.shape torch.Size([17, 64])
# q_per_token_as_complex_numbers_rotated_.shape torch.Size([17, 64])

# don't use vec2, I don't want to do data conversion 
# result is complex number, we choose to view it as real number
q_per_token_as_complex_numbers_rotated = wp.array(shape=(q_per_token_split_into_pairs.shape[0], q_per_token_split_into_pairs.shape[1], 2), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_complex_multiply, dim=(q_per_token_split_into_pairs.shape[0], q_per_token_split_into_pairs.shape[1]), inputs=[q_per_token_split_into_pairs, freqs_cis, q_per_token_as_complex_numbers_rotated])

# correctness check
assert torch.allclose(torch.view_as_real(q_per_token_as_complex_numbers_rotated_), wp.to_torch(q_per_token_as_complex_numbers_rotated).cpu(), rtol=1e-05, atol=1e-05), "q_per_token_split_into_pairs_rotated incorrect!"
q_per_token_rotated = q_per_token_as_complex_numbers_rotated.reshape(q_per_token.shape)


# ============= K =============
k_layer0_ = model_["layers.0.attention.wk.weight"]
k_layer0_ = k_layer0_.view(n_kv_heads, k_layer0_.shape[0] // n_kv_heads, dim)
k_layer0_head0_ = k_layer0_[0]
k_per_token_ = torch.matmul(token_embeddings_, k_layer0_head0_.T)
k_per_token_split_into_pairs_ = k_per_token_.float().view(k_per_token_.shape[0], -1, 2)
k_per_token_as_complex_numbers_ = torch.view_as_complex(k_per_token_split_into_pairs_)
k_per_token_split_into_pairs_rotated_ = torch.view_as_real(k_per_token_as_complex_numbers_ * freqs_cis_)
k_per_token_rotated_ = k_per_token_split_into_pairs_rotated_.view(k_per_token_.shape)
qk_per_token_ = torch.matmul(q_per_token_rotated_, k_per_token_rotated_.T)/(head_dim_)**0.5

# k_layer0_.shape torch.Size([8, 128, 4096])
# k_layer0_head0_.shape torch.Size([128, 4096])
# k_per_token.shape torch.Size([17, 128])
# k_per_token_split_into_pairs.shape torch.Size([17, 64, 2])
# k_per_token_as_complex_numbers.shape torch.Size([17, 64])
# k_per_token_split_into_pairs_rotated.shape torch.Size([17, 64, 2])
# k_per_token_rotated.shape torch.Size([17, 128])
# qk_per_token.shape torch.Size([17, 17])

# Warp version
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.reshape((n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim))
k_layer0_head0 = k_layer0[0]
k_per_token = wp.zeros(shape=(token_embeddings.shape[0], k_layer0_head0.shape[0]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(token_embeddings.shape[0], k_layer0_head0.shape[0]), dtype=WP_FLOAT32)
wp.matmul(token_embeddings, k_layer0_head0.transpose(), c, k_per_token)

assert torch.allclose(k_per_token_, wp.to_torch(k_per_token).cpu(), rtol=1e-05, atol=1e-05), "k_per_token incorrect!"

k_per_token_split_into_pairs = k_per_token.reshape((k_per_token.shape[0], -1, 2))
# view as complex
k_per_token_split_into_pairs_rotated = wp.array(shape=(k_per_token_split_into_pairs.shape[0], k_per_token_split_into_pairs.shape[1], 2), dtype=WP_FLOAT32)

# If dim is not correct, the result will be wrong
# e.g. freqs_cis might be changed
wp.launch(kernel=wpm.wp_complex_multiply, dim=(k_per_token_split_into_pairs.shape[0], k_per_token_split_into_pairs.shape[1]), inputs=[k_per_token_split_into_pairs, freqs_cis, k_per_token_split_into_pairs_rotated])

assert torch.allclose(k_per_token_split_into_pairs_rotated_, wp.to_torch(k_per_token_split_into_pairs_rotated).cpu(), rtol=1e-05, atol=1e-05), "k_per_token_split_into_pairs_rotated incorrect!"

k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)
qk_per_token = wp.zeros(shape=(token_embeddings.shape[0], token_embeddings.shape[0]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(token_embeddings.shape[0], token_embeddings.shape[0]), dtype=WP_FLOAT32)
# d = alpha * (a @ b) + beta * c
alpha = WP_FLOAT32(1.0 / wp.sqrt(WP_FLOAT32(head_dim)))
wp.matmul(q_per_token_rotated, k_per_token_rotated.transpose(), c, qk_per_token, alpha=alpha)

assert torch.allclose(qk_per_token_, wp.to_torch(qk_per_token).cpu(), rtol=1e-05, atol=1e-05), "qk_per_token incorrect!"

mask_ = torch.full((len(tokens_), len(tokens_)), float("-inf"), device=tokens_.device)
mask_ = torch.triu(mask_, diagonal=1)

qk_per_token_after_masking_ = qk_per_token_ + mask_
qk_per_token_after_masking_after_softmax_ = torch.nn.functional.softmax(qk_per_token_after_masking_, dim=1)

mask = wp.full((token_embeddings.shape[0], token_embeddings.shape[0]), WP_FLOAT32(float("-inf")), dtype=WP_FLOAT32, device=DEVICE)
wp.launch(kernel=wpm.wp_triu, dim=mask.shape, inputs=[mask])

# add mask to qk_per_token and apply softmax
# qk_per_token_after_masking_after_softmax = wp.zeros(shape=qk_per_token.shape, dtype=WP_FLOAT32)
# exp = wp.zeros(shape=qk_per_token.shape[0], dtype=WP_FLOAT32)
# wp.launch(kernel=wpm.wp_softmax, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp, qk_per_token_after_masking_after_softmax])
qk_per_token_after_masking_after_softmax = wpm.wp_softmax(qk_per_token, mask)
print(qk_per_token_after_masking_after_softmax_ - wp.to_torch(qk_per_token_after_masking_after_softmax).cpu())
assert torch.allclose(qk_per_token_after_masking_after_softmax_, wp.to_torch(qk_per_token_after_masking_after_softmax).cpu(), rtol=1e-05, atol=1e-05), "qk_per_token_after_masking_after_softmax incorrect!"


# ============= V =============

v_layer0_ = model_["layers.0.attention.wv.weight"]
v_layer0_ = v_layer0_.view(n_kv_heads, v_layer0_.shape[0] // n_kv_heads, dim)
v_layer0_head0_ = v_layer0_[0]
v_per_token_ = torch.matmul(token_embeddings_, v_layer0_head0_.T)
qkv_attention_ = torch.matmul(qk_per_token_after_masking_after_softmax_, v_per_token_)

# v_layer0.shape torch.Size([8, 128, 4096])
# v_layer0_head0.shape torch.Size([128, 4096])
# v_per_token.shape torch.Size([17, 128])
# qkv_attention.shape torch.Size([17, 128])

v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.reshape((n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim))
v_layer0_head0 = v_layer0[0]
v_per_token = wp.zeros(shape=(token_embeddings.shape[0], v_layer0_head0.shape[0]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(token_embeddings.shape[0], v_layer0_head0.shape[0]), dtype=WP_FLOAT32)
wp.matmul(token_embeddings, v_layer0_head0.transpose(), c, v_per_token)

assert torch.allclose(v_per_token_, wp.to_torch(v_per_token).cpu(), rtol=1e-05, atol=1e-05), "v_per_token incorrect!"

qkv_attention = wp.zeros(shape=(token_embeddings.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(token_embeddings.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
wp.matmul(qk_per_token_after_masking_after_softmax, v_per_token, c, qkv_attention)

assert torch.allclose(qkv_attention_, wp.to_torch(qkv_attention).cpu(), rtol=1e-05, atol=1e-05), "qkv_attention incorrect!"

qkv_attention_store_ = []
stacked_qkv_attention = wp.zeros(shape=(token_embeddings.shape[0], v_per_token.shape[1] * n_heads), dtype=WP_FLOAT32)
print(stacked_qkv_attention.shape)
for head in range(n_heads):
    q_layer0_head_ = q_layer0_[head]
    k_layer0_head_ = k_layer0_[head//4] # key weights are shared across 4 heads
    v_layer0_head_ = v_layer0_[head//4] # value weights are shared across 4 heads
    q_per_token_ = torch.matmul(token_embeddings_, q_layer0_head_.T)
    k_per_token_ = torch.matmul(token_embeddings_, k_layer0_head_.T)
    v_per_token_ = torch.matmul(token_embeddings_, v_layer0_head_.T)

    q_per_token_split_into_pairs_ = q_per_token_.float().view(q_per_token_.shape[0], -1, 2)
    q_per_token_as_complex_numbers_ = torch.view_as_complex(q_per_token_split_into_pairs_)
    q_per_token_split_into_pairs_rotated_ = torch.view_as_real(q_per_token_as_complex_numbers_ * freqs_cis_[:len(tokens_)])
    q_per_token_rotated_ = q_per_token_split_into_pairs_rotated_.view(q_per_token_.shape)

    k_per_token_split_into_pairs_ = k_per_token_.float().view(k_per_token_.shape[0], -1, 2)
    k_per_token_as_complex_numbers_ = torch.view_as_complex(k_per_token_split_into_pairs_)
    k_per_token_split_into_pairs_rotated_ = torch.view_as_real(k_per_token_as_complex_numbers_ * freqs_cis_[:len(tokens_)])
    k_per_token_rotated_ = k_per_token_split_into_pairs_rotated_.view(k_per_token_.shape)

    qk_per_token_ = torch.matmul(q_per_token_rotated_, k_per_token_rotated_.T)/(128)**0.5
    mask_ = torch.full((len(tokens_), len(tokens_)), float("-inf"), device=tokens_.device)
    mask_ = torch.triu(mask_, diagonal=1)
    qk_per_token_after_masking_ = qk_per_token_ + mask_
    qk_per_token_after_masking_after_softmax_ = torch.nn.functional.softmax(qk_per_token_after_masking_, dim=1)
    qkv_attention_ = torch.matmul(qk_per_token_after_masking_after_softmax_, v_per_token_)
    qkv_attention_store_.append(qkv_attention_)
    
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4]
    v_layer0_head = v_layer0[head//4]
    q_per_token = wp.zeros(shape=(token_embeddings.shape[0], q_layer0_head.shape[0]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(token_embeddings.shape[0], q_layer0_head.shape[0]), dtype=WP_FLOAT32)
    wp.matmul(token_embeddings, q_layer0_head.transpose(), c, q_per_token)
    k_per_token = wp.zeros(shape=(token_embeddings.shape[0], k_layer0_head.shape[0]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(token_embeddings.shape[0], k_layer0_head.shape[0]), dtype=WP_FLOAT32)
    wp.matmul(token_embeddings, k_layer0_head.transpose(), c, k_per_token)
    v_per_token = wp.zeros(shape=(token_embeddings.shape[0], v_layer0_head.shape[0]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(token_embeddings.shape[0], v_layer0_head.shape[0]), dtype=WP_FLOAT32)
    wp.matmul(token_embeddings, v_layer0_head.transpose(), c, v_per_token)
    
    q_per_token_split_into_pairs = q_per_token.reshape((q_per_token.shape[0], -1, 2))
    q_per_token_split_into_pairs_rotated = wp.zeros(shape=q_per_token_split_into_pairs.shape, dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_complex_multiply, dim=(q_per_token_split_into_pairs.shape[0], q_per_token_split_into_pairs.shape[1]), inputs=[q_per_token_split_into_pairs, freqs_cis, q_per_token_split_into_pairs_rotated])
    assert torch.allclose(q_per_token_split_into_pairs_rotated_, wp.to_torch(q_per_token_split_into_pairs_rotated).cpu(), rtol=1e-05, atol=1e-05), "q_per_token_split_into_pairs_rotated incorrect!"
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.reshape(q_per_token.shape)
    
    k_per_token_split_into_pairs = k_per_token.reshape((k_per_token.shape[0], -1, 2))
    k_per_token_split_into_pairs_rotated = wp.zeros(shape=k_per_token_split_into_pairs.shape, dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_complex_multiply, dim=(k_per_token_split_into_pairs.shape[0], k_per_token_split_into_pairs.shape[1]), inputs=[k_per_token_split_into_pairs, freqs_cis, k_per_token_split_into_pairs_rotated])
    assert torch.allclose(k_per_token_split_into_pairs_rotated_, wp.to_torch(k_per_token_split_into_pairs_rotated).cpu(), rtol=1e-05, atol=1e-05), "k_per_token_split_into_pairs_rotated incorrect!"
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)
    
    qk_per_token = wp.zeros(shape=(token_embeddings.shape[0], token_embeddings.shape[0]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(token_embeddings.shape[0], token_embeddings.shape[0]), dtype=WP_FLOAT32)
    alpha = WP_FLOAT32(1.0 / wp.sqrt(WP_FLOAT32(128)))
    wp.matmul(q_per_token_rotated, k_per_token_rotated.transpose(), c, qk_per_token, alpha=alpha)
    assert torch.allclose(qk_per_token_, wp.to_torch(qk_per_token).cpu(), rtol=1e-05, atol=1e-05), "qk_per_token incorrect!"
    
    mask = wp.full((token_embeddings.shape[0], token_embeddings.shape[0]), WP_FLOAT32(float("-inf")), dtype=WP_FLOAT32, device=DEVICE)
    wp.launch(kernel=wpm.wp_triu, dim=mask.shape, inputs=[mask])
    # qk_per_token_after_masking_after_softmax = wp.zeros(shape=qk_per_token.shape, dtype=WP_FLOAT32)
    # exp = wp.zeros(shape=qk_per_token.shape[0], dtype=WP_FLOAT32)
    # wp.launch(kernel=wpm.wp_softmax, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp, qk_per_token_after_masking_after_softmax])
    qk_per_token_after_masking_after_softmax = wpm.wp_softmax(qk_per_token, mask)
    print(qk_per_token_after_masking_after_softmax_)
    print(wp.to_torch(qk_per_token_after_masking_after_softmax).cpu())
    print(qk_per_token_after_masking_after_softmax_ - wp.to_torch(qk_per_token_after_masking_after_softmax).cpu())
    assert torch.allclose(qk_per_token_after_masking_after_softmax_, wp.to_torch(qk_per_token_after_masking_after_softmax).cpu(), rtol=1e-05, atol=1e-05), "qk_per_token_after_masking_after_softmax incorrect!"
    
    qkv_attention = wp.zeros(shape=(token_embeddings.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(token_embeddings.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
    wp.matmul(qk_per_token_after_masking_after_softmax, v_per_token, c, qkv_attention)
    assert torch.allclose(qkv_attention_, wp.to_torch(qkv_attention).cpu(), rtol=1e-05, atol=1e-05), "qkv_attention incorrect!"
    # qkv_attention_store.append(qkv_attention)
    wp.launch(kernel=wpm.wp_stack, dim=qkv_attention.shape, inputs=[WP_INT(head), qkv_attention, stacked_qkv_attention])

# qkv_attention_store_ 32
# torch.Size([17, 128])
# stacked_qkv_attention_ torch.Size([17, 4096])
stacked_qkv_attention_ = torch.cat(qkv_attention_store_, dim=-1)
w_layer0_ = model_["layers.0.attention.wo.weight"]
embedding_delta_ = torch.matmul(stacked_qkv_attention_, w_layer0_.T)
embedding_after_edit_ = token_embeddings_unnormalized_ + embedding_delta_
embedding_after_edit_normalized_ = rms_norm(embedding_after_edit_, model_["layers.0.ffn_norm.weight"])


w1_ = model_["layers.0.feed_forward.w1.weight"]
w2_ = model_["layers.0.feed_forward.w2.weight"]
w3_ = model_["layers.0.feed_forward.w3.weight"]
output_after_feedforward_ = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized_, w1_.T)) * torch.matmul(embedding_after_edit_normalized_, w3_.T), w2_.T)
layer_0_embedding_ = embedding_after_edit_ + output_after_feedforward_
final_embedding_ = token_embeddings_unnormalized_

assert torch.allclose(stacked_qkv_attention_, wp.to_torch(stacked_qkv_attention).cpu(), rtol=1e-05, atol=1e-05), "stacked_qkv_attention incorrect!"

w_layer = model["layers.0.attention.wo.weight"]
embedding_delta = wp.zeros(shape=(token_embeddings.shape[0], token_embeddings.shape[1]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(token_embeddings.shape[0], token_embeddings.shape[1]), dtype=WP_FLOAT32)
wp.matmul(stacked_qkv_attention, w_layer.transpose(), c, embedding_delta)
embedding_after_edit = wp.zeros(shape=token_embeddings.shape, dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_add, dim=token_embeddings.shape, inputs=[token_embeddings_unnormalized, embedding_delta, embedding_after_edit])
embedding_after_edit_normalized = wp.zeros(shape=embedding_after_edit.shape, dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_rms_norm, dim=embedding_after_edit.shape[0], inputs=[embedding_after_edit, WP_FLOAT32(norm_eps), model["layers.0.ffn_norm.weight"], embedding_after_edit_normalized])

assert torch.allclose(embedding_after_edit_normalized_, wp.to_torch(embedding_after_edit_normalized).cpu(), rtol=1e-05, atol=1e-05), "embedding_after_edit_normalized incorrect!"

final_embedding = token_embeddings_unnormalized

assert torch.allclose(final_embedding_, wp.to_torch(final_embedding).cpu(), rtol=1e-05, atol=1e-05), "final_embedding incorrect!"

# need higher tolerance
for layer in range(n_layers):
    qkv_attention_store_ = []
    stacked_qkv_attention = wp.zeros(shape=(token_embeddings.shape[0], v_per_token.shape[1] * n_heads), dtype=WP_FLOAT32)
    
    layer_embedding_norm_= rms_norm(final_embedding_, model_[f"layers.{layer}.attention_norm.weight"])
    q_layer_ = model_[f"layers.{layer}.attention.wq.weight"]
    q_layer_ = q_layer_.view(n_heads, q_layer_.shape[0] // n_heads, dim)
    k_layer_ = model_[f"layers.{layer}.attention.wk.weight"]
    k_layer_ = k_layer_.view(n_kv_heads, k_layer_.shape[0] // n_kv_heads, dim)
    v_layer_ = model_[f"layers.{layer}.attention.wv.weight"]
    v_layer_ = v_layer_.view(n_kv_heads, v_layer_.shape[0] // n_kv_heads, dim)
    w_layer_ = model_[f"layers.{layer}.attention.wo.weight"]
    
    # assert torch.allclose(final_embedding_, wp.to_torch(final_embedding).cpu(), rtol=1e-05, atol=1e-05), "final_embedding incorrect!"
    # assert torch.allclose(model_[f"layers.{layer}.attention_norm.weight"], wp.to_torch(model[f"layers.{layer}.attention_norm.weight"]).cpu(), rtol=1e-05, atol=1e-05), "model[f'layers.{layer}.attention_norm.weight'] incorrect!"
    
    layer_embedding_norm = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_rms_norm, dim=final_embedding.shape[0], inputs=[final_embedding, WP_FLOAT32(norm_eps), model[f"layers.{layer}.attention_norm.weight"], layer_embedding_norm])

    # assert torch.allclose(layer_embedding_norm_, wp.to_torch(layer_embedding_norm).cpu(), rtol=1e-05, atol=1e-05), "layer_embedding_norm incorrect!"
    
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    head_dim = q_layer.shape[0] // n_heads
    q_layer = q_layer.reshape((n_heads, head_dim, dim))
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.reshape((n_kv_heads, k_layer.shape[0] // n_kv_heads, dim))
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.reshape((n_kv_heads, v_layer.shape[0] // n_kv_heads, dim))
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    
    for head in range(n_heads):
        q_layer_head_ = q_layer_[head]
        k_layer_head_ = k_layer_[head//4]
        v_layer_head_ = v_layer_[head//4]
        q_per_token_ = torch.matmul(layer_embedding_norm_, q_layer_head_.T)
        k_per_token_ = torch.matmul(layer_embedding_norm_, k_layer_head_.T)
        v_per_token_ = torch.matmul(layer_embedding_norm_, v_layer_head_.T)
        q_per_token_split_into_pairs_ = q_per_token_.float().view(q_per_token_.shape[0], -1, 2)
        q_per_token_as_complex_numbers_ = torch.view_as_complex(q_per_token_split_into_pairs_)
        q_per_token_split_into_pairs_rotated_ = torch.view_as_real(q_per_token_as_complex_numbers_ * freqs_cis_)
        q_per_token_rotated_ = q_per_token_split_into_pairs_rotated_.view(q_per_token_.shape)
        k_per_token_split_into_pairs_ = k_per_token_.float().view(k_per_token_.shape[0], -1, 2)
        k_per_token_as_complex_numbers_ = torch.view_as_complex(k_per_token_split_into_pairs_)
        k_per_token_split_into_pairs_rotated_ = torch.view_as_real(k_per_token_as_complex_numbers_ * freqs_cis_)
        k_per_token_rotated_ = k_per_token_split_into_pairs_rotated_.view(k_per_token_.shape)
        qk_per_token_ = torch.matmul(q_per_token_rotated_, k_per_token_rotated_.T)/(128)**0.5
        mask_ = torch.full((len(token_embeddings_unnormalized_), len(token_embeddings_unnormalized_)), float("-inf"))
        mask_ = torch.triu(mask_, diagonal=1)
        qk_per_token_after_masking_ = qk_per_token_ + mask_
        qk_per_token_after_masking_after_softmax_ = torch.nn.functional.softmax(qk_per_token_after_masking_, dim=1)
        qkv_attention_ = torch.matmul(qk_per_token_after_masking_after_softmax_, v_per_token_)
        qkv_attention_store_.append(qkv_attention_)
        
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], q_layer_head.shape[0]), dtype=WP_FLOAT32)
        c = wp.zeros(shape=(layer_embedding_norm.shape[0], q_layer_head.shape[0]), dtype=WP_FLOAT32)
        wp.matmul(layer_embedding_norm, q_layer_head.transpose(), c, q_per_token)
        # assert torch.allclose(layer_embedding_norm_, wp.to_torch(layer_embedding_norm).cpu(), rtol=1e-05, atol=1e-05), "layer_embedding_norm incorrect!"
        # assert torch.allclose(q_layer_head_.T, wp.to_torch(q_layer_head.transpose()).cpu(), rtol=1e-05, atol=1e-05), "q_layer_head.transpose() incorrect!"
        
        k_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], k_layer_head.shape[0]), dtype=WP_FLOAT32)
        c = wp.zeros(shape=(layer_embedding_norm.shape[0], k_layer_head.shape[0]), dtype=WP_FLOAT32)
        wp.matmul(layer_embedding_norm, k_layer_head.transpose(), c, k_per_token)
        v_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], v_layer_head.shape[0]), dtype=WP_FLOAT32)
        c = wp.zeros(shape=(layer_embedding_norm.shape[0], v_layer_head.shape[0]), dtype=WP_FLOAT32)
        wp.matmul(layer_embedding_norm, v_layer_head.transpose(), c, v_per_token)
        # assert torch.allclose(q_per_token_, wp.to_torch(q_per_token).cpu(), rtol=1e-05, atol=1e-05), "q_per_token incorrect!"
        q_per_token_split_into_pair = q_per_token.reshape((q_per_token.shape[0], -1, 2))
        # assert torch.allclose(q_per_token_split_into_pairs_, wp.to_torch(q_per_token_split_into_pair).cpu(), rtol=1e-05, atol=1e-05), "q_per_token_split_into_pair incorrect!"
        
        q_per_token_split_into_pairs_rotated = wp.zeros(shape=q_per_token_split_into_pair.shape, dtype=WP_FLOAT32)
        
        wp.launch(kernel=wpm.wp_complex_multiply, dim=(q_per_token_split_into_pair.shape[0], q_per_token_split_into_pair.shape[1]), inputs=[q_per_token_split_into_pair, freqs_cis, q_per_token_split_into_pairs_rotated])
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.reshape(q_per_token.shape)
        
        # assert torch.allclose(q_per_token_rotated_, wp.to_torch(q_per_token_rotated).cpu(), rtol=1e-05, atol=1e-05), "q_per_token_split_into_pairs_rotated incorrect!"
        k_per_token_split_into_pairs = k_per_token.reshape((k_per_token.shape[0], -1, 2))
        k_per_token_split_into_pairs_rotated = wp.zeros(shape=k_per_token_split_into_pairs.shape, dtype=WP_FLOAT32)
        wp.launch(kernel=wpm.wp_complex_multiply, dim=(k_per_token_split_into_pairs.shape[0], k_per_token_split_into_pairs.shape[1]), inputs=[k_per_token_split_into_pairs, freqs_cis, k_per_token_split_into_pairs_rotated])
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.reshape(k_per_token.shape)
        # assert torch.allclose(k_per_token_rotated_, wp.to_torch(k_per_token_rotated).cpu(), rtol=1e-05, atol=1e-05), "k_per_token_split_into_pairs_rotated incorrect!"
        qk_per_token = wp.zeros(shape=(layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]), dtype=WP_FLOAT32)
        c = wp.zeros(shape=(layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]), dtype=WP_FLOAT32)
        alpha = WP_FLOAT32(1.0 / wp.sqrt(WP_FLOAT32(128)))
        wp.matmul(q_per_token_rotated, k_per_token_rotated.transpose(), c, qk_per_token, alpha=alpha)
        # assert torch.allclose(qk_per_token_, wp.to_torch(qk_per_token).cpu(), rtol=1e-05, atol=1e-05), "qk_per_token incorrect!"
        mask = wp.full((layer_embedding_norm.shape[0], layer_embedding_norm.shape[0]), WP_FLOAT32(float("-inf")), dtype=WP_FLOAT32, device=DEVICE)
        wp.launch(kernel=wpm.wp_triu, dim=mask.shape, inputs=[mask])
        # qk_per_token_after_masking_after_softmax = wp.zeros(shape=qk_per_token.shape, dtype=WP_FLOAT32)
        # exp = wp.zeros(shape=qk_per_token.shape[0], dtype=WP_FLOAT32)
        # wp.launch(kernel=wpm.wp_softmax, dim=qk_per_token.shape, inputs=[qk_per_token, mask, exp, qk_per_token_after_masking_after_softmax])
        qk_per_token_after_masking_after_softmax = wpm.wp_softmax(qk_per_token, mask)

        # assert torch.allclose(qk_per_token_after_masking_after_softmax_, wp.to_torch(qk_per_token_after_masking_after_softmax).cpu(), rtol=1e-05, atol=1e-05), "qk_per_token_after_masking_after_softmax incorrect!"
        qkv_attention = wp.zeros(shape=(layer_embedding_norm.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
        c = wp.zeros(shape=(layer_embedding_norm.shape[0], v_per_token.shape[1]), dtype=WP_FLOAT32)
        wp.matmul(qk_per_token_after_masking_after_softmax, v_per_token, c, qkv_attention)
        # assert torch.allclose(qkv_attention_, wp.to_torch(qkv_attention).cpu(), rtol=1e-05, atol=1e-05), "qkv_attention incorrect!"
        wp.launch(kernel=wpm.wp_stack, dim=qkv_attention.shape, inputs=[WP_INT(head), qkv_attention, stacked_qkv_attention])
        
    stacked_qkv_attention_ = torch.cat(qkv_attention_store_, dim=-1)
    # assert torch.allclose(stacked_qkv_attention_, wp.to_torch(stacked_qkv_attention).cpu(), rtol=1e-05, atol=1e-05), "stacked_qkv_attention incorrect!"
    
    w_layer_ = model_[f"layers.{layer}.attention.wo.weight"]
    embedding_delta_ = torch.matmul(stacked_qkv_attention_, w_layer_.T)
    embedding_after_edit_ = final_embedding_ + embedding_delta_
    embedding_after_edit_normalized_ = rms_norm(embedding_after_edit_, model_[f"layers.{layer}.ffn_norm.weight"])
    w1_ = model_[f"layers.{layer}.feed_forward.w1.weight"]
    w2_ = model_[f"layers.{layer}.feed_forward.w2.weight"]
    w3_ = model_[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward_ = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized_, w1_.T)) * torch.matmul(embedding_after_edit_normalized_, w3_.T), w2_.T)
    final_embedding_ = embedding_after_edit_ + output_after_feedforward_
    
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
    wp.matmul(stacked_qkv_attention, w_layer.transpose(), c, embedding_delta)
    embedding_after_edit = wp.zeros(shape=final_embedding.shape, dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_add, dim=final_embedding.shape, inputs=[final_embedding, embedding_delta, embedding_after_edit])
    embedding_after_edit_normalized = wp.zeros(shape=embedding_after_edit.shape, dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_rms_norm, dim=embedding_after_edit.shape[0], inputs=[embedding_after_edit, WP_FLOAT32(norm_eps), model[f"layers.{layer}.ffn_norm.weight"], embedding_after_edit_normalized])
    
    # assert torch.allclose(embedding_after_edit_normalized_, wp.to_torch(embedding_after_edit_normalized).cpu(), rtol=1e-05, atol=1e-05), "embedding_after_edit_normalized incorrect!"
    
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    tmp_left = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w1.shape[0]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w1.shape[0]), dtype=WP_FLOAT32)
    wp.matmul(embedding_after_edit_normalized, w1.transpose(), c, tmp_left)
    tmp_left_silu = wp.zeros(shape=tmp_left.shape, dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_silu, dim=tmp_left.shape, inputs=[tmp_left, tmp_left_silu])
    
    tmp_right = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w3.shape[0]), dtype=WP_FLOAT32)
    # torch.matmul(embedding_after_edit_normalized_, w3_.T)
    c = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w3.shape[0]), dtype=WP_FLOAT32)
    wp.matmul(embedding_after_edit_normalized, w3.transpose(), c, tmp_right)
    
    tmp = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w2.shape[1]), dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_mul, dim=tmp.shape, inputs=[tmp_left_silu, tmp_right, tmp])
    
    output_after_feedforward = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w2.shape[0]), dtype=WP_FLOAT32)
    c = wp.zeros(shape=(embedding_after_edit_normalized.shape[0], w2.shape[0]), dtype=WP_FLOAT32)
    wp.matmul(tmp, w2.transpose(), c, output_after_feedforward)
    
    # assert torch.allclose(output_after_feedforward_, wp.to_torch(output_after_feedforward).cpu(), rtol=1e-05, atol=1e-05), "output_after_feedforward incorrect!"
    final_embedding = wp.zeros(shape=embedding_after_edit.shape, dtype=WP_FLOAT32)
    wp.launch(kernel=wpm.wp_add, dim=embedding_after_edit.shape, inputs=[embedding_after_edit, output_after_feedforward, final_embedding])
    
    # assert torch.allclose(final_embedding_, wp.to_torch(final_embedding).cpu(), rtol=1e-05, atol=1e-05), "final_embedding incorrect!"
    
final_embedding_ = rms_norm(final_embedding_, model_["norm.weight"])
# final_embedding_[-1] torch.Size([4096])
# torch.Size([4096, 128256])
# torch.Size([128256])
logits_ = torch.matmul(final_embedding_[-1], model_["output.weight"].T)
next_token_ = torch.argmax(logits_, dim=-1)
tokenizer.decode([next_token_.item()])

final_embedding_rms = wp.zeros(shape=(final_embedding.shape[0], final_embedding.shape[1]), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_rms_norm, dim=final_embedding.shape[0], inputs=[final_embedding, WP_FLOAT32(norm_eps), model["norm.weight"], final_embedding_rms])

assert torch.allclose(final_embedding_, wp.to_torch(final_embedding_rms).cpu(), rtol=1e-05, atol=1e-03), "final_embedding incorrect!"

# logits for all tokens
last_final_embedding = wp.zeros(shape=(1, final_embedding_rms.shape[1]), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_access_d1, dim=final_embedding_rms.shape[1], inputs=[final_embedding_rms, last_final_embedding, WP_INT(final_embedding_rms.shape[0]-1)])
logits = wp.zeros(shape=(1, model["output.weight"].shape[0]), dtype=WP_FLOAT32)
c = wp.zeros(shape=(1, model["output.weight"].shape[0]), dtype=WP_FLOAT32)
wp.matmul(last_final_embedding, model["output.weight"].transpose(), c, logits)

assert torch.allclose(logits_, wp.to_torch(logits).cpu(), rtol=1e-05, atol=1e-03), "logits incorrect!"

next_token = wp.zeros(shape=(1,), dtype=WP_INT)
max_logit = wp.zeros(shape=(1,), dtype=WP_FLOAT32)
wp.launch(kernel=wpm.wp_argmax, dim=logits.shape[1], inputs=[logits, max_logit])
wp.launch(kernel=wpm.wp_index, dim=logits.shape[1], inputs=[logits, max_logit, next_token])

print(tokenizer.decode([next_token_.item()]))
print(tokenizer.decode([wp.to_torch(next_token).item()]))

