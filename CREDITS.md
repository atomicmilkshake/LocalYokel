# Credits

LocalYokel is **not** a from-scratch inference engine. It is a branded,
Windows-and-lab-oriented continuation of FreeToken.

## FreeToken (upstream)

- Project: [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
- Site: [flashml.ai](https://www.flashml.ai/)
- Paper: [arXiv:2608.16157](https://arxiv.org/abs/2608.16157)
- License: Apache 2.0

The MoE-offload runtime, OpenAI/Anthropic-compatible server, kernel cache,
and core architecture are FreeToken's. Use their names when you mean the
upstream project.

## Cruz (vcruz305)

- Engine fork: [vcruz305/FreeToken](https://github.com/vcruz305/FreeToken)
- Models: [huggingface.co/vcruz305](https://huggingface.co/vcruz305)

Cruz widened GGUF beyond Gemma-4: Qwen3 / Qwen3.5 / Qwen3.6 (`qwen3moe`,
`qwen35moe`, `qwen35`), Qwen3.8-Flash-Next (`qwen4exp`, lightning indexer),
DeepSeek-V4, ggml quant types (including mixed per-layer expert banks), and
multi-shard checkpoints. LocalYokel merged that work with official FreeToken
(QuantConfig, GLM-5.3-Flash, Qwen3.8-Flash-Next HF/FTW).

## LocalYokel (this repository)

- GitHub: [atomicmilkshake/LocalYokel](https://github.com/atomicmilkshake/LocalYokel)

Windows MSVC + CUDA 13.2 bring-up, CUDA TurboQuant tq4 KV (opt-in
`--kv-quant tq4`), graph-safe materialize, and this packaging. The CLI
`localyokel` is an alias of FreeToken's `ft` / `freetoken.cli`.
