# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | Yes                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.

Instead, please report it privately by emailing the maintainers or using
[GitHub's private vulnerability reporting](https://github.com/Argonaut790/fused-turboquant/security/advisories/new).

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days.

## Scope

This project runs Triton GPU kernels and processes numerical tensors. Security concerns most likely to apply:

- Malicious model weights or inputs causing unexpected kernel behavior
- Denial of service via crafted tensor shapes
- Dependency vulnerabilities in PyTorch, Triton, or other dependencies
