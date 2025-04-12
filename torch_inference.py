from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import torch.autograd.profiler as profiler
import argparse
import time
import contextlib
from config import *
from utils import TorchProfilingManager

parser = argparse.ArgumentParser(description="Optional profiling.")
parser.add_argument('--profile', action='store_true', help="Enable profiling.")
parser.add_argument('--profile_iter', type=int, default=100, help="Number of iterations to profile.")
parser.add_argument('--compile', action='store_true', help="Use torch.compile for optimization.")
args = parser.parse_args()

pm = TorchProfilingManager(args.profile)

# Define the full model inference function
def run_llama_inference(tokens, model, config):
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])

    # Embedding Layer
    embedding_layer = torch.nn.Embedding(vocab_size, dim, device=DEVICE)
    embedding_layer.weight.data = model["tok_embeddings.weight"]
    token_embeddings = embedding_layer(tokens)
    token_embeddings = token_embeddings.to(TORCH_FLOAT)

    # Freqs Prep
    zero_to_one_split_into_64_parts = (torch.tensor(range(64)) / 64).to(DEVICE)
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(tokens.shape[0]).to(DEVICE), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token).to(DEVICE), freqs_for_each_token)

    # Define RMS Norm
    def rms_norm(tensor, norm_weights):
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

    # Process all layers
    final_embedding = token_embeddings
    for layer in range(n_layers):
        qkv_attention_store = []
        layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
        q_layer = model[f"layers.{layer}.attention.wq.weight"]
        q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
        k_layer = model[f"layers.{layer}.attention.wk.weight"]
        k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
        v_layer = model[f"layers.{layer}.attention.wv.weight"]
        v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
        t_head = time.time()
        for head in range(n_heads):
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head//4]
            v_layer_head = v_layer[head//4]
            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
            q_per_token_split_into_pairs = q_per_token.to(TORCH_FLOAT).view(q_per_token.shape[0], -1, 2)
            q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
            q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
            q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
            k_per_token_split_into_pairs = k_per_token.to(TORCH_FLOAT).view(k_per_token.shape[0], -1, 2)
            k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
            k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
            mask = torch.full((len(token_embeddings), len(token_embeddings)), float("-inf"))
            mask = torch.triu(mask, diagonal=1).to(DEVICE)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(TORCH_FLOAT)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)
        t_head = time.time() - t_head
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        final_embedding = embedding_after_edit + output_after_feedforward

    # Final norm and logits
    final_embedding = rms_norm(final_embedding, model["norm.weight"])
    logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
    next_token = torch.argmax(logits, dim=-1)

    return next_token

t1 = time.time()
with pm.torch_profile_section():
    with pm.torch_profile_function("Torch Inference"):
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
        model = {k: v.to(TORCH_FLOAT).to(DEVICE) for k, v in model.items()}

        with open("Meta-Llama-3-8B/params.json", "r") as f:
            config = json.load(f)

        prompt = "the answer to the ultimate question of life, the universe, and everything is "
        tokens = [128000] + tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).to(DEVICE)

        # Apply torch.compile to the entire inference function
        if args.compile:
            print("Using torch.compile for full model inference")
            compiled_inference = torch.compile(run_llama_inference, mode="max-autotune")
            t_inference = time.time()
            next_token = compiled_inference(tokens, model, config)
            t_inference = time.time() - t_inference
            print("Time taken for inference (torch.compile):", t_inference)

            # Initialize sequence for continuous generation
            generated_tokens = tokens
            avg_time = 0
            
            for i in range(args.profile_iter):
                # Use the growing sequence for next token prediction
                t_inference = time.time()
                next_token = compiled_inference(generated_tokens, model, config)
                t_inference = time.time() - t_inference
                avg_time += t_inference
                
                # Append the new token to our generated sequence
                next_token_tensor = torch.tensor([next_token], dtype=torch.int32).to(DEVICE)
                generated_tokens = torch.cat([generated_tokens, next_token_tensor], dim=0)

                # Display information
                token_text = tokenizer.decode([next_token.cpu().item()])
                
                # print(f"Iteration {i+1}: New token = '{token_text}', Time = {t_inference:.4f}s, Token ID = {next_token}")
            
                # Every 10 tokens, print the full text
                if (i+1) % 10 == 0:
                    full_text = tokenizer.decode(generated_tokens.cpu().tolist())
                    print(f"Full text so far: '{full_text}'")

            print("Average time taken for inference:", avg_time / args.profile_iter)
        else:
            t_inference = time.time()
            next_token = run_llama_inference(tokens, model, config)
            t_inference = time.time() - t_inference
            print("Time taken for inference:", t_inference)

            # Initialize sequence for continuous generation
            generated_tokens = tokens
            avg_time = 0
            
            for i in range(args.profile_iter):
                # Use the growing sequence for next token prediction
                t_inference = time.time()
                next_token = run_llama_inference(generated_tokens, model, config)
                t_inference = time.time() - t_inference
                avg_time += t_inference
                
                # Append the new token to our generated sequence
                next_token_tensor = torch.tensor([next_token], dtype=torch.int32).to(DEVICE)
                generated_tokens = torch.cat([generated_tokens, next_token_tensor], dim=0)

                # Decode and display results
                token_text = tokenizer.decode([next_token.cpu().item()])
                full_text = tokenizer.decode(generated_tokens.cpu().tolist())
                # print(f"Iteration {i+1}: New token = '{token_text}', Time = {t_inference:.4f}s, Token ID = {next_token.cpu().item()}")
                
                # Every 10 tokens, print the full text so far
                if (i+1) % 10 == 0:
                    print(f"Full text so far: '{full_text}'")

            print("Average time taken for inference:", avg_time / args.profile_iter)

        token_text = tokenizer.decode([next_token.item()])

t2 = time.time()
print("Time taken:", t2 - t1)
if args.profile:
    print(pm.prof.key_averages().table(row_limit=100))