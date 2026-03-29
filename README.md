# fused-turboquant

[![PyPI](https://img.shields.io/pypi/v/fused-turboquant?color=blue)](https://pypi.org/project/fused-turboquant/)
[![Python](https://img.shields.io/pypi/pyversions/fused-turboquant)](https://pypi.org/project/fused-turboquant/)
[![License](https://img.shields.io/github/license/Argonaut790/fused-turboquant)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b)](https://arxiv.org/abs/2504.19874)
[![GitHub stars](https://img.shields.io/github/stars/Argonaut790/fused-turboquant?style=social)](https://github.com/Argonaut790/fused-turboquant)

**Fused Triton encode/decode kernels for TurboQuant KV cache compression, powered by Randomized Hadamard Transform.**

- Compresses LLM KV cache to **2-4 bits** using [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026)
- Fuses the entire encode/decode pipeline into **single Triton kernels** (1 kernel vs 5+ in other implementations)
- Uses **RHT** instead of dense QR rotation -- O(d log d) compute, O(d) storage, fits in registers
- Drop-in **HuggingFace** integration and experimental **vLLM** plugin
- Auto-detects CUDA + Triton; falls back to unfused PyTorch on CPU

## Installation

```bash
pip install fused-turboquant[cuda]          # core + Triton fused kernels
pip install fused-turboquant[cuda,hf]       # + HuggingFace transformers
pip install fused-turboquant[vllm]          # + vLLM plugin
pip install fused-turboquant                # core only (torch + scipy + numpy)
```

> If `torch.cuda.is_available()` returns `False`, install CUDA-enabled PyTorch first:
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`

**From source** (development):

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
pip install -e ".[dev]"       # or: uv sync --extra dev
```

## Quick Start

```python
import torch
from fused_turboquant import TurboQuantMSE

tq = TurboQuantMSE(head_dim=256, bits=4, device="cuda")
keys = torch.randn(1, 4, 128, 256, device="cuda")

compressed = tq.encode(keys)   # 1 fused Triton kernel
decoded = tq.decode(compressed) # 1 fused Triton kernel

print(f"Compression: {compressed.compression_ratio:.1f}x")  # 3.9x
```

## Usage

### HuggingFace Integration

Requires `pip install fused-turboquant[cuda,hf]`.

**Simulation cache** -- measure quality impact (perplexity) without changing memory layout:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fused_turboquant.hf import make_simulation_cache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

cache = make_simulation_cache(bits=4)
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50, use_cache=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

**Fused cache** -- real compressed key storage with fused attention kernel:

```python
from fused_turboquant.hf import patch_model

cache = patch_model(model, bits=4)  # patches all full-attention layers
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50, use_cache=True)
```

### vLLM Integration

> **Status**: Registers a `TURBOQUANT_RHT` backend via entry point. Full PagedAttention integration is WIP.

```bash
pip install fused-turboquant[vllm]
```

The plugin auto-registers via Python entry points -- no code changes needed to vLLM.

## How It Works

TurboQuant compresses KV vectors by rotating them into a uniform distribution, then quantizing each coordinate independently. The pipeline:

```
Encode: input -> RHT rotate -> normalize -> Lloyd-Max quantize -> pack nibbles
Decode: unpack -> dequantize -> denormalize -> inverse RHT -> output
```

**Why fusion matters**: Other implementations use dense QR rotation (O(d^2) matrix, 256 KB at d=256), which forces a separate cuBLAS matmul and prevents kernel fusion. RHT needs only a d-element sign vector (1 KB) that fits in SRAM, so the entire pipeline runs in **one kernel launch** with zero HBM round-trips between stages:

```
Other implementations:  [rotation] -> HBM -> [norm] -> HBM -> [quantize] -> HBM -> [pack]    5+ kernels
This project:           [single Triton kernel: RHT -> norm -> quantize -> pack]                1 kernel
```

## Benchmarks

### Real Model Throughput (Qwen2.5-0.5B, 4-bit)

NVIDIA GB10, PyTorch 2.10+cu128, `max_new_tokens=200`, averaged over 5 prompts.

| Method | Avg TPS | Peak Memory | Relative Speed |
|--------|:-:|:-:|:-:|
| FP16 baseline | **105.0/s** | 962 MB | 1.00x |
| Ours (TQ4) | 98.2/s | 962 MB | 0.94x |
| Dejan.ai (TQ4) | 86.1/s | 963 MB | 0.82x |

Our TurboQuant adds only **7% overhead** vs FP16 baseline. Dejan.ai adds **18% overhead** due to Dense QR rotation. Memory is similar here because Qwen2.5-0.5B is a small model where KV cache is a tiny fraction of total VRAM -- the memory savings become significant on larger models with longer contexts.

> Reproduce: `uv run python benchmarks/bench_e2e.py --model Qwen/Qwen2.5-0.5B --bits 4`

### Kernel Throughput (batch=2048, head_dim=256)

NVIDIA RTX 5070 Ti, Triton 3.6.0, PyTorch 2.10+cu128, 4-bit.

| Metric | Ours (fused) | Ours (unfused) | Dejan.ai | FP16 baseline |
|--------|:-:|:-:|:-:|:-:|
| Encode TPS | **51.8M** | 9.8M | 15.5M | -- |
| Decode TPS | **84.4M** | 13.6M | 28.0M | -- |
| Cosine similarity | 0.986 | 0.995 | 0.995 | 1.000 |
| Compression | **3.9x** | **3.9x** | 2.0x | 1.0x |

### vs Dejan.ai

| Feature | Ours | Dejan.ai |
|---------|:-:|:-:|
| Pipeline | **Fused 1-kernel Triton** | Multi-kernel PyTorch |
| Rotation | **RHT O(d log d)** | Dense QR O(d^2) |
| Storage/layer | **1 KB** | 256 KB |
| Kernel launches (enc+dec) | **2** | 6+ |
| V compression | **Yes** | No (K only) |

Full benchmark sweep: `uv run python benchmarks/run_fused_benchmark.py`

## Architecture

```
src/fused_turboquant/
+-- core/
|   +-- hadamard.py         # RHT rotation (Triton primary, PyTorch fallback)
|   +-- lloyd_max.py        # Lloyd-Max quantizer for Beta distribution
|   +-- quantizer.py        # TurboQuantMSE: auto-selects fused/unfused
|   +-- packing.py          # Sub-byte packing (4-bit: 2/byte, 2-bit: 4/byte)
+-- kernels/
|   +-- triton_rht.py       # Standalone RHT kernel
|   +-- triton_encode.py    # Fused encode: RHT + norm + quantize + pack
|   +-- triton_decode.py    # Fused decode: unpack + dequant + denorm + inv RHT
|   +-- triton_attention.py # Fused Q.K^T from uint8 indices
+-- hf/
|   +-- simulation_cache.py # DynamicCache with compress->decompress roundtrip
|   +-- fused_cache.py      # Compressed key storage + fused attention forward
+-- vllm_plugin/            # vLLM backend registration + KV cache hooks
+-- cache/kv_cache.py       # Standalone KV cache wrapper
```

## Compatibility

| Dependency | Minimum | Tested up to | Install extra |
|:----------:|:-------:|:------------:|:-------------:|
| Python | 3.10 | 3.12 | -- |
| PyTorch | 2.4 | 2.11 | -- |
| Triton | 3.0 | 3.6 | `[cuda]` |
| transformers | 4.45 | latest | `[hf]` |
| vLLM | 0.8 | 0.18 | `[vllm]` |

## Development

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
pip install -e ".[dev]"

pytest                                           # unit tests
python benchmarks/run_fused_benchmark.py         # kernel microbenchmarks
python benchmarks/run_full_comparison.py         # quality + memory benchmarks
```

**Real model benchmarks** (requires `pip install -e ".[dev,hf]"`):

```bash
# 4-way comparison: FP16 vs ours vs Dejan.ai (throughput + memory)
python benchmarks/bench_e2e.py --model Qwen/Qwen2.5-0.5B --bits 4

# Include WikiText-2 perplexity
python benchmarks/bench_e2e.py --model Qwen/Qwen3.5-9B --bits 4 --quality

# Export results to JSON
python benchmarks/bench_e2e.py --model Qwen/Qwen2.5-0.5B --bits 4 --json results.json
```

### Target Model

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B): 60 layers = 15 blocks of (3 Gated DeltaNet + 1 Full Attention). Only the 15 full-attention layers use KV cache with `head_dim=256` (power of 2, ideal for Hadamard).

## Citation

This is an **independent, community-driven implementation** of TurboQuant. It is not affiliated with or endorsed by Google Research or the original paper authors. Our contribution is the fused Triton kernel design using RHT (replacing dense QR rotation) and the integrations with HuggingFace / vLLM.

If you use this implementation in your work, please cite both the original paper and this project:

```bibtex
@software{fused_turboquant,
  title   = {fused-turboquant: Fused Triton Kernels for TurboQuant KV Cache Compression},
  author  = {fused-turboquant Contributors},
  url     = {https://github.com/Argonaut790/fused-turboquant},
  year    = {2025},
  license = {Apache-2.0},
}

@inproceedings{lindgren2026turboquant,
  title     = {TurboQuant: Online and Offline KV Cache Quantization via Norm-Tweaked Lloyd-Max},
  author    = {Lindgren, Erik and Awasthi, Pranjal and Kumar, Sanjiv},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2504.19874},
}
```

## Acknowledgements

- [TurboQuant](https://arxiv.org/abs/2504.19874) -- Lindgren, Awasthi, Kumar. ICLR 2026. The algorithm this project implements.
- [Fast JL Transform (RHT)](https://doi.org/10.1137/060673096) -- Ailon & Chazelle. SICOMP 39(1), 2009. The rotation that enables kernel fusion.
- [Dejan.ai TurboQuant](https://dejan.ai/blog/turboquant/) -- Dense QR implementation, benchmarked against.
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## License

Apache 2.0
