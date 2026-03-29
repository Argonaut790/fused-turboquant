# fused-turboquant

**Fused Triton encode/decode kernels for TurboQuant KV cache compression, powered by Randomized Hadamard Transform.**

Compresses LLM KV cache to 2-4 bits using Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026). Other implementations use dense QR rotation with multi-kernel pipelines. This project fuses the entire encode/decode into **single Triton kernels** — uniquely enabled by RHT's O(d) memory footprint.

## Why RHT Enables Fusion

TurboQuant requires random orthogonal rotation before quantization. Dense QR uses an O(d²) matrix (256 KB at d=256) that can't fit in registers — it must be a standalone cuBLAS call. RHT uses only a d-element sign vector (1 KB) that fits in SRAM, allowing the rotation to be fused with norm + quantize + pack into **one kernel**:

```
Other implementations:  [rotation] -> HBM -> [norm] -> HBM -> [quantize] -> HBM -> [pack]    5+ kernels
This project:           [single Triton kernel: RHT -> norm -> quantize -> pack]                1 kernel
```

## Benchmarks

NVIDIA RTX 5070 Ti, Triton 3.6.0, PyTorch 2.10+cu128, 4-bit, `head_dim=256`.

### Kernel-Level TPS (encode/decode, batch=2048)

| Metric | FP16 (no quant) | Ours (fused) | Ours (unfused) | Dejan.ai |
|--------|:-:|:-:|:-:|:-:|
| Encode TPS | — | **51.8M** | 9.8M | 15.5M |
| Decode TPS | — | **84.4M** | 13.6M | 28.0M |
| Quality (cosine sim) | **1.000** | 0.986 | **0.995** | **0.995** |
| Bytes/token | 512 | **132** | **132** | 258 |
| Compression | 1.0x | **3.9x** | **3.9x** | 2.0x |

### Simulated Decode Step (encode new KV + decode cache + Q·K^T)

Per-head latency for a simulated autoregressive decode step on random vectors. Not actual model inference — see note below.

| Context | Metric | FP16 (no quant) | Ours (fused) | Ours (unfused) | Dejan.ai |
|:-------:|--------|:-:|:-:|:-:|:-:|
| 512 | Inference TPS | **49K** | **25K** | 9K | 17K |
| 512 | Latency/step | **0.081ms** | 0.161ms | 0.449ms | 0.237ms |
| 512 | KV cache memory | 256 KB | **66 KB** | **66 KB** | 129 KB |
| 2048 | Inference TPS | **49K** | **26K** | 8K | 14K |
| 2048 | Latency/step | **0.081ms** | 0.157ms | 0.473ms | 0.284ms |
| 2048 | KV cache memory | 1.0 MB | **264 KB** | **264 KB** | 516 KB |
| 8192 | Inference TPS | **49K** | **27K** | 8K | 16K |
| 8192 | Latency/step | **0.081ms** | 0.147ms | 0.531ms | 0.258ms |
| 8192 | KV cache memory | 4.0 MB | **1.0 MB** | **1.0 MB** | 2.0 MB |

> **Note**: These are synthetic benchmarks on random tensors measuring KV cache encode/decode + attention overhead. They are **not** end-to-end model inference throughput. See [Real Model Benchmarks](#real-model-benchmarks-linux--gpu) for scripts that measure actual perplexity and generation speed.

**Key takeaway**: FP16 is fastest per-step (no compression overhead), but uses **3.9x more memory**. Our fused pipeline adds only ~0.07ms latency per step vs FP16, while cutting KV cache memory to 1/4. Compared to Dejan.ai, we're **1.7x faster** and use **2x less memory**.

### Architectural Comparison

| Feature | Ours | Dejan.ai |
|---------|:-:|:-:|
| Pipeline | **Fused 1-kernel Triton** | Multi-kernel PyTorch |
| Rotation | **RHT O(d log d)** | Dense QR O(d²) |
| Storage/layer | **1 KB** | 256 KB |
| Kernel launches (enc/dec) | **1 / 1** | 3+ / 3+ |
| Nibble packing | **Yes** (2/byte at 4-bit, 4/byte at 2-bit) | No |
| V compression | **Yes** | No (K only) |

Full sweep across batch sizes, head_dims, and bit-widths: `uv run python benchmarks/run_fused_benchmark.py`

### End-to-End Quality (8-layer attention simulation)

| Bits | Compression | Cosine Sim | Attention KL | Status |
|:----:|:-:|:-:|:-:|:-:|
| 4 | 4.0x | **0.991** | **0.005** | Production-ready |
| 3 | 5.3x | 0.971 | 0.017 | Acceptable |
| 2 | 8.0x | 0.901 | 0.057 | Draft only |

## Installation

**pip** (recommended for most users):

```bash
pip install fused-turboquant[cuda]
```

This installs the core library plus Triton for fused GPU kernels. If you already have PyTorch with CUDA, the base install only adds scipy and numpy:

```bash
pip install fused-turboquant
```

> **Note**: PyTorch defaults to CPU-only on some platforms. If `torch.cuda.is_available()` returns `False`, install CUDA-enabled PyTorch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```

**With HuggingFace integration**:

```bash
pip install fused-turboquant[cuda,hf]
```

**With vLLM plugin**:

```bash
pip install fused-turboquant[vllm]
```

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

# Simulated KV vectors (in practice, these come from the model's attention layer)
keys = torch.randn(1, 4, 128, 256, device="cuda")

# encode() internally: RHT rotation -> norm -> quantize -> pack (1 fused Triton kernel)
compressed = tq.encode(keys)

# decode() internally: unpack -> dequantize -> denorm -> inverse RHT (1 fused Triton kernel)
decoded = tq.decode(compressed)

print(f"Compression: {compressed.compression_ratio:.1f}x")  # 3.9x
```

The package auto-detects CUDA + Triton and uses fused kernels when available, falling back to an unfused PyTorch implementation on CPU.

## Usage

### Standalone Encode/Decode

Compress arbitrary tensors with shape `(..., head_dim)` where `head_dim` is a power of 2:

```python
import torch
from fused_turboquant import TurboQuantMSE

tq = TurboQuantMSE(head_dim=128, bits=4, device="cuda")

x = torch.randn(32, 8, 128, device="cuda")  # (batch, heads, head_dim)
compressed = tq.encode(x)
decoded = tq.decode(compressed)

cos_sim = torch.nn.functional.cosine_similarity(
    x.flatten(0, -2), decoded.flatten(0, -2), dim=-1,
).mean()
print(f"Cosine similarity: {cos_sim:.4f}")       # ~0.99 at 4-bit
print(f"Compression: {compressed.compression_ratio:.1f}x")  # ~3.9x
```

### HuggingFace Integration

Two strategies are provided. Both require `pip install fused-turboquant[cuda,hf]`.

**Simulation cache** -- measures quality impact without changing inference memory:

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

**Fused cache** -- real compressed storage with fused attention:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fused_turboquant.hf import patch_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

cache = patch_model(model, bits=4)
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50, use_cache=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Supported Versions

| Dependency | Minimum | Tested up to | Notes |
|:----------:|:-------:|:------------:|-------|
| Python | 3.10 | 3.12 | |
| PyTorch | 2.4 | 2.11 | CUDA wheels recommended |
| Triton | 3.0 | 3.6 | Optional (fused kernels); `pip install fused-turboquant[cuda]` |
| transformers | 4.45 | latest | For HF integration; `pip install fused-turboquant[hf]` |
| vLLM | 0.8 | 0.18 | Linux only; `pip install fused-turboquant[vllm]` |

### Running Tests & Benchmarks

```bash
pytest                                               # unit tests
python benchmarks/run_fused_benchmark.py             # kernel microbenchmarks
python benchmarks/run_full_comparison.py             # quality + memory benchmarks
```

### Real Model Benchmarks (Linux + GPU)

The primary benchmark compares all implementations (FP16, ours fused, ours simulation, Dejan.ai) in a single run:

```bash
uv sync --extra hf --extra dev

# Quick 4-way comparison: throughput + memory
uv run python benchmarks/bench_e2e.py --model Qwen/Qwen2.5-0.5B --bits 4

# Full comparison including WikiText-2 perplexity
uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3.5-9B --bits 4 --quality

# Export results to JSON
uv run python benchmarks/bench_e2e.py --model Qwen/Qwen2.5-0.5B --bits 4 --json results.json
```

Individual benchmarks are also available:

```bash
uv run python benchmarks/bench_quality.py --model Qwen/Qwen2.5-0.5B --bits 4
uv run python benchmarks/bench_throughput.py --model Qwen/Qwen2.5-0.5B --bits 4
uv run python benchmarks/bench_vllm_baseline.py --model Qwen/Qwen3.5-9B
```

## Architecture

```
src/fused_turboquant/
+-- core/
|   +-- hadamard.py       # RHT rotation (Triton primary, PyTorch fallback)
|   +-- lloyd_max.py      # Lloyd-Max quantizer for Beta distribution
|   +-- quantizer.py      # TurboQuantMSE: auto-selects fused/unfused
|   +-- packing.py        # Sub-byte packing (4-bit: 2/byte, 2-bit: 4/byte)
+-- kernels/
|   +-- triton_rht.py     # Standalone RHT (zero-scratch, output-as-scratch)
|   +-- triton_encode.py  # Fused encode: RHT + norm + quantize + pack
|   +-- triton_decode.py  # Fused decode: unpack + dequant + denorm + inv RHT
|   +-- triton_attention.py  # Fused Q.K^T from uint8 indices
+-- hf/
|   +-- simulation_cache.py   # DynamicCache with compress->decompress roundtrip
|   +-- fused_cache.py        # Compressed key storage + fused attention forward
+-- cache/kv_cache.py     # KV cache wrapper
```

## Status

**Kernel library** + **HuggingFace integration** (real-model benchmarks pending execution).

Working:
- Fused Triton encode/decode kernels
- Fused attention kernel for compressed keys
- HuggingFace `DynamicCache` integration (simulation cache for quality, fused cache for throughput)
- Benchmark scripts for perplexity (WikiText-2), throughput (tok/s), and vLLM FP16 baseline

Not yet implemented:
- vLLM native plugin (PagedAttention block layout is incompatible with contiguous-tensor kernels)
- Real-model benchmark results (scripts ready, need Linux GPU to run)

## Target Model

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B): 60 layers = 15 blocks of (3 Gated DeltaNet + 1 Full Attention). Only the **15 full-attention layers** use KV cache with `head_dim=256` (power of 2, ideal for Hadamard). DeltaNet layers maintain a fixed-size recurrent state.

## References

- **TurboQuant** — Lindgren, Awasthi, Kumar. *ICLR 2026*. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **Fast JL Transform (RHT)** — Ailon & Chazelle. *SICOMP* 39(1), 2009. [DOI](https://doi.org/10.1137/060673096)
- **Lloyd-Max Quantization** — Lloyd, *IEEE T-IT* 28(2), 1982; Max, *IEEE T-IT* 6(1), 1960.
- **Dejan.ai TurboQuant** — Dense QR impl, benchmarked against. [dejan.ai/blog/turboquant](https://dejan.ai/blog/turboquant/)
- **Google Research blog** — [research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## License

Apache 2.0
