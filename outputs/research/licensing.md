# License and provenance register

This is a planning register, not legal advice. Before distribution, counsel should review the exact pinned commits, weights, notices, model card and intended license.

| Dependency/source | License | Compatibility / obligations |
|---|---|---|
| [Meta Llama models](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) / Llama 3.2 weights | Llama 3.2 Community License | Custom terms/AUP govern weights and derivatives; preserve required notices and comply with redistribution/use conditions. It is not permissive OSS. |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | permissive with LICENSE/NOTICE obligations; preferred reference/dependency. |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/blob/master/LICENSE) | Apache-2.0 | use package/API rather than copying. `sharded_moe.py` states portions derive from FairScale and retains BSD notice. |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/LICENSE) | Apache-2.0 | permissive code; keep notices. |
| [FairScale](https://github.com/facebookresearch/fairscale) | BSD-style | archived; preserve required attribution if copying. |
| [vLLM](https://github.com/vllm-project/vllm/blob/main/LICENSE) | Apache-2.0 | server dependency; its support does not grant weight rights. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) | MIT | permissive code; model conversion still bound by Llama terms. |
| [DeepSeek-V3 code](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README.md) | MIT | model weights have separate model license; avoid copying model material. |

## Pinning rule

Record repository URL, immutable commit SHA, release/package version, files used, and retrieval date in the conversion manifest. This report references upstream `main` because research inspected current source; **do not ship against a floating branch**. No copied source is proposed: composition through package APIs/clean-room thin glue minimizes notice and copyleft risk.
