# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-29

### Added

- Fused Triton encode kernel (RHT + norm + quantize + pack in one kernel)
- Fused Triton decode kernel (unpack + dequant + denorm + inverse RHT in one kernel)
- Fused attention kernel for compressed keys (Q.K^T from uint8 indices)
- Standalone Triton RHT kernel with zero-scratch design
- Lloyd-Max quantizer for Beta distribution
- Sub-byte packing (4-bit: 2/byte, 2-bit: 4/byte)
- PyTorch fallback paths for all operations
- HuggingFace `DynamicCache` integration (simulation + fused cache)
- Benchmark scripts for kernel throughput, quality, and memory
- Real-model benchmark scripts for perplexity (WikiText-2) and generation throughput
- Quick start example

[Unreleased]: https://github.com/Argonaut790/fused-turboquant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Argonaut790/fused-turboquant/releases/tag/v0.1.0
