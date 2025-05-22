# Llama3-Warp

Implement a warp version of Llama-3 from scratch!
Everything on warp except loading model parameter and tokenization.

Original repo: https://github.com/naklecha/llama3-from-scratch

## Introduction

wp_inference.py: for Warp library Llama3 inference.

torch_inference.py: for Torch library Llama3 inference.

config.py: Define some warp data types.

wp_kernels.py: Define some warp functions useful for Llama-3. But we probably need to refactor them.

equivalence_check.py: Check the correctness of the float32 version of the warp code. Already passed! A lot of duplicate code, just for easier debugging.

## Some Useful Commands

```
./inference.sh wp # Warp Llama3 inference
./inference.sh torch # Torch Llama3 inference
./inference.sh wp prof # Detailed profile information for Warp Llama3 inference
./inference.sh torch prof # Detailed profile information for Torch Llama3 inference
```

Remove cache dir (like tokenizer.model) if you get tokenization error:

```
rm -rf /tmp/data-gym-cache
```

## Profile

| Implementation | Configuration | Time (s) |
|----------------|---------------|----------|
| Torch | Standard | 0.3053 |
| Torch | Compile | 0.4146 |
| Warp | No Tile, No Compile | 1.1500 |
| Warp | No Tile, Compile | 1.1578 |
| Warp | Tile, No Compile | 1.0648 |
| Warp | Tile, Compile | 1.0689 |


The tiling optimization shows only small impact on performance, likely because our Warp kernels are already SIMT. The torch.compile results show a slight performance degradation, needs further investigation.

## TODO

1. Check how torch.compile works
2. Refactor wp_kernels.py for better support, maybe add other models in the future. llm / nerf / diffusion model
