# Llama3-Warp

Implement a warp version of Llama-3 from scratch!
Everything on warp except loading model parameter and tokenization.

Original repo: https://github.com/naklecha/llama3-from-scratch

## Introduction

wp_inference.py: for Warp library Llama3 inference.

torch_inference.py: for Torch library Llama3 inference.

config.py: Define some warp data types.

wp_module.py: Define some warp functions useful for Llama-3. But we probably need to refactor them.

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

Warp time taken : 10.25156021118164 

Torch time taken : 10.623606204986572

## TODO

1. The precision problem (just use float32 now, but I think for llm we should use bf16)
4. Profile the code (the result is really gooood, need to check again)
5. Refactor wp_module.py for better support, maybe add other models in the future. llm / nerf / diffusion model