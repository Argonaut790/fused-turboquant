# ⚡ fused-turboquant

[![PyPI](https://img.shields.io/pypi/v/fused-turboquant?color=blue)](https://pypi.org/project/fused-turboquant/)
[![Python](https://img.shields.io/pypi/pyversions/fused-turboquant)](https://pypi.org/project/fused-turboquant/)
[![License](https://img.shields.io/github/license/Argonaut790/fused-turboquant)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19874-b31b1b)](https://arxiv.org/abs/2504.19874)
[![GitHub stars](https://img.shields.io/github/stars/Argonaut790/fused-turboquant?style=social)](https://github.com/Argonaut790/fused-turboquant)

**fused-turboquant** is a high-performance Python library for compressing LLM KV caches to 2-4 bits using the TurboQuant algorithm (Google Research, ICLR 2026). It fuses the entire quantization pipeline — Randomized Hadamard Transform, normalization, Lloyd-Max quantization, and bit packing — into single Triton GPU kernels, achieving up to 4.9x memory compression with near-lossless quality. Drop-in support for HuggingFace Transformers and vLLM. Works with Llama, Qwen, Mistral, Phi, and more.

- 🗜️ Compresses both **K and V** caches to **2-4 bits** using [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) — up to **~4.9x KV cache compression** (3-bit) or **~3.8x** (4-bit)
- 🔥 Fuses the entire encode/decode pipeline into **single Triton kernels** (1 kernel vs 5+ in other implementations)
- 🦋 Uses **RHT** instead of dense QR rotation — O(d log d) compute, O(d) storage, fits in registers
- 🤗 Drop-in **HuggingFace** integration and **vLLM** attention backend
- 🔄 Auto-detects CUDA + Triton; falls back to unfused PyTorch on CPU

## 📦 Installation

```bash
pip install fused-turboquant[cuda]          # core + Triton fused kernels
pip install fused-turboquant[cuda,hf]       # + HuggingFace transformers
pip install fused-turboquant[vllm]          # + vLLM plugin
pip install fused-turboquant                # core only (torch + scipy + numpy)
```

> ⚠️ If `torch.cuda.is_available()` returns `False`, install CUDA-enabled PyTorch first:
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`

**From source** (development):

```bash
git clone https://github.com/Argonaut790/fused-turboquant.git
cd fused-turboquant
pip install -e ".[dev]"       # or: uv sync --extra dev
```

## 🚀 Quick Start

```python
import torch
from fused_turboquant import TurboQuantMSE

tq = TurboQuantMSE(head_dim=256, bits=4, device="cuda")
keys = torch.randn(1, 4, 128, 256, device="cuda")

compressed = tq.encode(keys)   # 1 fused Triton kernel
decoded = tq.decode(compressed) # 1 fused Triton kernel

print(f"Compression: {compressed.compression_ratio:.1f}x")  # 3.9x
```

## 🛠️ Usage

### 🤗 HuggingFace Integration

Requires `pip install fused-turboquant[cuda,hf]`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from fused_turboquant.hf import patch_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

cache = patch_model(model, bits=4)  # compresses K+V cache in all attention layers
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=50, use_cache=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

`patch_model` compresses both keys and values via fused Triton encode. Keys are stored packed (nibble-packed for 4-bit) and Q·K^T is computed directly from packed indices with inline unpacking — no dequantization pass. Values are decompressed on the fly during the attention-weighted sum. Supports `compress_v=False` (K-only), `compress_v="boundary"` (first/last 2 layers at fp16), or a custom `callable(layer_idx, n_layers) -> bool`.

### 🔌 vLLM Integration

Serve any supported model with compressed KV cache through vLLM's standard serving stack — continuous batching, OpenAI-compatible API, and paged block management all work transparently with compressed blocks.

```bash
pip install fused-turboquant[vllm]
```

**Online serving** (OpenAI-compatible API):

```bash
vllm serve Qwen/Qwen3-8B --attention-backend FUSED_TURBOQUANT
```

**Offline batch inference** (Python API):

```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-8B", attention_backend="FUSED_TURBOQUANT")
outputs = llm.generate(
    ["Explain KV cache compression in one paragraph."],
    SamplingParams(max_tokens=200, temperature=0.0),
)
print(outputs[0].outputs[0].text)
```

**Configuration** via environment variables:

| Variable | Default | Description |
|---|---|---|
| `TURBOQUANT_BITS` | `4` | Quantization bit-width (2, 3, or 4) |
| `TURBOQUANT_COMPRESS_V` | `1` | Compress values too (`1`) or K-only (`0`) |

```bash
# 3-bit K+V compression for maximum memory savings
TURBOQUANT_BITS=3 vllm serve Qwen/Qwen3-8B --attention-backend FUSED_TURBOQUANT
```

The plugin auto-registers via Python entry points — no code changes needed to vLLM. KV cache blocks are stored as packed uint8 indices + fp32 norms within vLLM's paged block system. Prefill uses FlashAttention/SDPA on full-precision KV, then compresses for storage. Decode uses our fused Triton QK kernel directly on packed indices.

**Compatibility**: vLLM 0.8 – 0.18+ | Requires CUDA with Triton

## ✅ Supported Models

### Requirements

For compression to apply, a model's attention layers must satisfy:

- 🔢 **`head_dim` is a power of 2** (64, 128, 256) — required by the Randomized Hadamard Transform
- 🎛️ **`bits` is 2, 3, or 4** — Lloyd-Max codebooks are precomputed for these bit-widths
- 🖥️ **CUDA + Triton** for fused kernels (unfused PyTorch fallback on CPU for encode/decode, but not fused attention)

`patch_model` (fused cache) additionally requires:

- 🔑 **Separate QKV projections** — `q_proj`, `k_proj`, `v_proj` (not fused `qkv_proj` or `c_attn`)
- 🔄 **RoPE via `position_embeddings=(cos, sin)`** — the convention used by Llama, Qwen, and most modern models
- ➗ **`n_q_heads` divisible by `n_kv_heads`** — standard for GQA/MQA models
- 🚫 **No sliding window or logit softcapping** — these are validated and rejected upfront

### Model Compatibility

> 📋 `head_dim` values verified from official HuggingFace `config.json` files (March 2026).

#### Dense Decoder-Only

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🦙 **Llama 3.1/3.2/3.3** (1B–405B) | 128 | ✅ Tested | Standard GQA + RoPE |
| 🟣 **Qwen3** (0.6B–235B) | 128 | ✅ Tested | GQA + RoPE |
| 🟣 **Qwen2.5** (0.5B–72B) | 128 | ✅ Tested | |
| 🔷 **Mistral 7B / Small 3** | 128 | ✅ Expected | Standard GQA + RoPE |
| 🔵 **Phi-4** (14B) | 128 | ✅ Expected | `hidden=5120 / 40 heads = 128` |
| 🟢 **Command R / R+** | 128 | ✅ Expected | |
| 🟢 **Yi** (1.5, 6B–34B) | 128 | ✅ Expected | |
| 🟢 **InternLM 2 / 3** | 128 | ✅ Expected | |

#### MoE (Mixture of Experts)

MoE models share attention with their dense variants — only FFN layers use expert routing. fused-turboquant patches attention only, so MoE routing is completely unaffected.

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🟣 **Qwen3-MoE** (30B-A3B, 235B-A22B) | 128 | ✅ Expected | Same attention as Qwen3 dense |
| 🔷 **Mixtral** (8x7B, 8x22B) | 128 | ✅ Expected | Same attention as Mistral 7B |
| 🟢 **OLMoE** (6.9B, 1.3B active) | 128 | ✅ Expected | Allen AI; standard GQA |
| 🟢 **Zen4-Max** (30B, 3B active) | 128 | ✅ Expected | Apache 2.0 |

#### Multimodal (Vision-Language)

For multimodal models, fused-turboquant **patches only the text decoder** attention layers. Vision encoder attention is automatically skipped by module-path detection (`visual`, `vision_model`, `vision_tower`, etc.).

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🟣 **Qwen2-VL** | 128 | ✅ Expected | Vision encoder skipped; text decoder patched |
| 🟢 **InternVL 2/3** | 128 | ✅ Expected | Vision encoder skipped; text decoder patched |
| 🟢 **LLaVA** (Llama/Vicuna backbone) | 128 | ✅ Expected | CLIP ViT skipped (different naming) |

#### Hybrid Architectures

Models mixing full attention with linear attention (DeltaNet, Mamba, etc.). Only the full-attention layers are patched; linear layers are auto-skipped.

| Model | `head_dim` | `patch_model` | Notes |
|:------|:----------:|:-------------:|:------|
| 🟣 **Qwen3.5-MoE** (35B-A3B, 397B-A17B) | 256 | ✅ Expected | 10 of 40 layers patched (every 4th = full attn); DeltaNet + MoE layers skipped |

#### Not Compatible

| Model | Reason |
|:------|:-------|
| 🌊 **DeepSeek V3 / V3.2 / V4** | Multi-Latent Attention (MLA) — no standard QKV projections |
| 🦙 **Llama 4** Scout / Maverick | QK norm (`use_qk_norm=true`) + chunked attention — both unsupported |
| 💎 **Gemma 3** (1B–27B) | Alternating sliding window / full attention — sliding window unsupported |
| 🔷 **Mistral Small 4** (119B MoE) | Sliding window attention — unsupported |
| 🔵 **Phi-3 / 3.5** (mini, small) | Fused `qkv_proj` + `head_dim=96` (not power of 2) |
| 🟢 **GPT-2 / BLOOM / Falcon** | No RoPE; fused QKV projection (`c_attn` / `query_key_value`) |
| 🔴 **NVIDIA Nemotron 3 Super** | Hybrid Mamba-Transformer MoE — Mamba layers not patchable |

**Legend**: ✅ Tested = verified end-to-end, ✅ Expected = compatible architecture (not yet tested), ❌ = incompatible.

> 🔧 **Smart detection**: `patch_model` validates each layer upfront and raises clear errors for unsupported features (sliding window, fused QKV, logit softcapping, cross-attention). QK layer norm (Qwen3, Gemma3) is fully supported. It also runs a smoke test to catch any silent issues. Use `verify=False` to skip the smoke test.
>
> 🔍 **Vision-safe**: For multimodal models, attention layers inside vision encoders (`visual.*`, `vision_model.*`, `vision_tower.*`) are automatically skipped — only the text decoder is patched.

**🔍 Check before patching** — use `check_model_compatibility()` to diagnose any model:

```python
from fused_turboquant.hf import check_model_compatibility

result = check_model_compatibility(model)
print(result)
# {'compatible': True, 'head_dim': 128, 'eligible_layers': 24,
#  'rope_detected': True, 'unsupported_features': [],
#  'vision_layers_skipped': 0, 'known_compatible': True, ...}
```

## 🧠 How It Works

TurboQuant compresses KV vectors by rotating them into a uniform distribution, then quantizing each coordinate independently. Both keys and values go through the same pipeline:

```
Encode: input → RHT rotate → normalize → Lloyd-Max quantize → pack nibbles
Decode: unpack → dequantize → denormalize → inverse RHT → output
```

**Why fusion matters**: Other implementations use dense QR rotation (O(d²) matrix, 256 KB at d=256), which forces a separate cuBLAS matmul and prevents kernel fusion. RHT needs only a d-element sign vector (1 KB) that fits in SRAM, so the entire pipeline runs in **one kernel launch** with zero HBM round-trips between stages:

```
Other implementations:  [rotation] → HBM → [norm] → HBM → [quantize] → HBM → [pack]    5+ kernels
This project:           [single Triton kernel: RHT → norm → quantize → pack]              1 kernel
```

**Prefill vs Decode**: Compression applies to the **decode (generation) phase**. During prefill (processing the prompt), we use Flash Attention (SDPA) on full FP16 keys and values for maximum speed — the KV tensors are then compressed and stored in packed form. During autoregressive decode, each new token's KV is compressed on arrival, and the fused Triton kernel computes Q·K^T directly from packed indices. The memory savings come from the **stored KV cache** being compressed — this is what dominates memory at long contexts.

## 📊 Benchmarks

All benchmarks: NVIDIA GB10 (Blackwell, unified memory), Qwen3-8B, PyTorch 2.10+cu128.

### Memory Savings

**The primary benefit of TurboQuant KV cache compression is memory reduction** — fitting longer contexts and larger batches in the same GPU memory.

#### Compression Ratio

Both K and V caches are compressed by default (`compress_v=True`). All bit-widths use true packed storage — nibble-packed for 4-bit, bitstream-packed for 3-bit (8 values per 3 bytes), 2-bit packed for 2-bit. Keys are unpacked inline by the fused attention kernel (shift+mask, no separate dequant pass).

| Config | Nominal bits | Effective bits/elem | KV Compression | Per-position storage (head_dim=128) |
|--------|:---:|:---:|:---:|:---|
| FP16 baseline | 16 | 16.0 | 1.0x | K: 256B + V: 256B = 512B |
| **FusedTQ4** (K+V) | 4 | 4.25 | **~3.8x** | K: 68B + V: 68B = 136B |
| **FusedTQ3** (K+V) | 3 | 3.25 | **~4.9x** | K: 52B + V: 52B = 104B |

> **Effective bit-rate**: The nominal `--bits` is the index width; the per-vector fp32 norm adds overhead. For head_dim=128: `bits=3` → 3.25 bits/elem, `bits=4` → 4.25 bits/elem.
>
> **Layer-aware V compression**: Use `compress_v="boundary"` to keep the first 2 and last 2 layers at fp16 V precision while compressing the rest. This can recover quality on sensitive models with negligible memory cost. Custom per-layer strategies are also supported via `compress_v=callable`.

#### Measured End-to-End Peak Memory

Peak GPU memory (model weights + KV cache + buffers) during generation on NVIDIA GB10. Compression applies to the **decode phase** — prefill uses Flash Attention on full FP16, then KV is compressed for storage.

Qwen3-8B, K+V compressed, 50 decode tokens per context length:

**3-bit (FusedTQ3):**

| Context | FP16 Peak | FusedTQ3 K+V (ours) | Saved | TQ3 (Dejan.ai) |
|--------:|:---------:|:-------------------:|:-----:|:---------------:|
| 4,096 | 16,626 MB | **16,179 MB** | 447 MB | 17,630 MB |
| 8,192 | 17,620 MB | **16,726 MB** | 894 MB | 18,625 MB |
| 16,384 | 19,608 MB | **17,790 MB** | 1,818 MB | 20,613 MB |
| 32,768 | 23,585 MB | **19,949 MB** | 3,636 MB | 24,590 MB |

**4-bit (FusedTQ4):**

| Context | FP16 Peak | FusedTQ4 K+V (ours) | Saved | TQ4 (Dejan.ai) |
|--------:|:---------:|:-------------------:|:-----:|:---------------:|
| 4,096 | 16,626 MB | **16,208 MB** | 418 MB | 17,913 MB |
| 8,192 | 17,620 MB | **16,783 MB** | 837 MB | 18,907 MB |
| 16,384 | 19,608 MB | **17,934 MB** | 1,674 MB | 20,895 MB |
| 32,768 | 23,585 MB | **20,237 MB** | 3,348 MB | 24,872 MB |

> **Note**: Dejan TQ uses *more* memory than FP16 because it materializes decompressed keys for standard attention. FusedTQ avoids this entirely by computing Q·K^T directly from packed compressed indices.

#### Projected KV Cache at 1M Context

Based on the measured per-token compression ratios, the KV cache (excluding model weights) at extreme context lengths for Qwen3-8B (32 layers, 8 KV heads, head_dim=128):

| Context | FP16 KV Cache | FusedTQ4 K+V (~3.8x) | FusedTQ3 K+V (~4.9x) |
|--------:|:-------------:|:---------------------:|:---------------------:|
| 131,072 (128K) | 16 GB | 4.2 GB | **3.2 GB** |
| 524,288 (512K) | 64 GB | 17 GB | **13 GB** |
| **1,048,576 (1M)** | **128 GB** | **34 GB** | **26 GB** |

> *Calculated from measured per-token storage: FP16=128 KB/token, FusedTQ4=34 KB/token, FusedTQ3=26 KB/token.* At 1M context, FP16 KV cache alone exceeds most GPU VRAM (128 GB). FusedTQ3 reduces this to **26 GB** — fitting on a single 80 GB GPU (A100/H100).

### Quality

Measured on Qwen3-8B via teacher-forced decode: 20 tokens prefilled, then ground-truth tokens fed one at a time through the compressed KV attention path. Logit cosine similarity compared against FP16 reference at each decode position.

| Config | Avg Logit Cosine Sim | Min Logit Cosine Sim | Quality Impact |
|--------|:---:|:---:|:---|
| FusedTQ4, K+V | **0.9869** | 0.9048 | Near-lossless at the paper's quality-neutral point |
| FusedTQ3, K+V | **0.9538** | 0.5087 | Good average quality, occasional outlier positions |
| FusedTQ4, boundary V | **0.9825** | 0.8284 | First/last 2 layers at fp16 V |

> Run full WikiText-2 perplexity evaluation:
> `uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 4 --quality`

### Throughput

Measured decode throughput (tok/s) on Qwen3-8B, NVIDIA GB10:

| Context | FP16 | FusedTQ3 | FusedTQ4 |
|--------:|:----:|:--------:|:--------:|
| 4,096 | 3.9 | 3.7 | 2.8 |
| 8,192 | 2.5 | 2.3 | 1.7 |
| 16,384 | 1.4 | 1.3 | 0.9 |
| 32,768 | 0.7 | 0.6 | 0.6 |

> KV cache compression is primarily a **memory optimization** — the goal is fitting more context and larger batches, not faster single-sequence generation. FusedTQ3 runs within 5-15% of FP16 throughput. FusedTQ4 has higher overhead due to nibble packing/unpacking in the attention kernel.
>
> Reproduce: `uv run python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --long-context`

### 🆚 vs TQ (Dejan.ai) / TQ+ (TheTom)

| Feature | FusedTQ (ours) | TQ (Dejan.ai) | TQ+ (TheTom) |
|---------|:-:|:-:|:-:|
| Compression target | **K + V (packed)** | K only | K + V |
| 4-bit compression ratio | **~3.8x** | ~1.3x | ~3.8x |
| 3-bit compression ratio | **~4.9x** | N/A | N/A |
| Kernel pipeline | **Fused 1-kernel (Triton)** | Multi-kernel (PyTorch) | Multi-kernel (C / Metal / CUDA) |
| Fused Q·K^T from packed | **Yes** | No | No |
| Rotation | **RHT O(d log d)** | Dense QR O(d²) | Walsh-Hadamard O(d log d) |
| Rotation storage/layer | **~1 KB** | ~256 KB | ~1 KB |
| Layer-aware V compression | **Yes (configurable callable)** | No | Yes (fixed boundary only) |
| HuggingFace integration | **Yes** | No | No (llama.cpp only) |
| QK norm support | **Yes** | No | N/A |

Full benchmark sweep: `uv run python benchmarks/run_fused_benchmark.py`

## 🏗️ Architecture

```
src/fused_turboquant/
├── core/
│   ├── hadamard.py         # RHT rotation (Triton primary, PyTorch fallback)
│   ├── lloyd_max.py        # Lloyd-Max quantizer for Beta distribution
│   ├── quantizer.py        # TurboQuantMSE: auto-selects fused/unfused
│   └── packing.py          # Sub-byte packing (4-bit: nibble, 3-bit: bitstream, 2-bit: 4/byte)
├── kernels/
│   ├── triton_rht.py       # Standalone RHT kernel
│   ├── triton_encode.py    # Fused encode: RHT + norm + quantize + pack
│   ├── triton_decode.py    # Fused decode: unpack + dequant + denorm + inv RHT
│   └── triton_attention.py # Fused Q·K^T from packed indices (inline nibble unpack)
├── hf/
│   └── fused_cache.py      # Compressed K+V storage + fused attention forward
├── vllm_plugin/            # Full vLLM attention backend (paged compressed KV cache)
└── cache/kv_cache.py       # Standalone KV cache wrapper
```

## 📋 Compatibility

| Dependency | Minimum | Tested up to | Install extra |
|:----------:|:-------:|:------------:|:-------------:|
| Python | 3.10 | 3.12 | — |
| PyTorch | 2.4 | 2.11 | — |
| Triton | 3.0 | 3.6 | `[cuda]` |
| transformers | 4.45 | latest | `[hf]` |
| vLLM | 0.8 | 0.18 | `[vllm]` |

## 🧪 Development

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
# 3-way comparison: FP16 vs FusedTQ vs TQ (Dejan.ai)
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --max-new-tokens 512

# Long-context decode sweep (4K-32K tokens)
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --long-context

# Batch throughput search (find max batch per method)
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --batch-search

# Include WikiText-2 perplexity
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --quality

# Export results to JSON
python benchmarks/bench_e2e.py --model Qwen/Qwen3-8B --bits 3 --json results.json
```

## 📝 Citation

This is an **independent, community-driven implementation** based on the TurboQuant algorithm described in [arXiv:2504.19874](https://arxiv.org/abs/2504.19874). It is **not affiliated with, endorsed by, or derived from code by** Google Research, Google DeepMind, or the original paper authors. The paper is published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Our contribution is the fused Triton kernel design using RHT (replacing dense QR rotation) and the integrations with HuggingFace / vLLM — the implementation is entirely original.

If you use this implementation in your work, please cite both the original paper and this project:

```bibtex
@software{fused_turboquant,
  title   = {fused-turboquant: Fused Triton Kernels for TurboQuant KV Cache Compression},
  author  = {fused-turboquant Contributors},
  url     = {https://github.com/Argonaut790/fused-turboquant},
  year    = {2025},
  license = {Apache-2.0},
}

@inproceedings{zandieh2026turboquant,
  title     = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author    = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2504.19874},
}
```

## 🙏 Acknowledgements

- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni. ICLR 2026. The algorithm this project implements.
- [Fast JL Transform (RHT)](https://doi.org/10.1137/060673096) — Ailon & Chazelle. SICOMP 39(1), 2009. The rotation that enables kernel fusion.
- [Dejan.ai TurboQuant](https://dejan.ai/blog/turboquant/) — Dense QR implementation, benchmarked against.
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

## 📄 License

Apache 2.0
