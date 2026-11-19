# LLM Engineering — From Transformer Internals to Instruction Fine-Tuning

**An end-to-end engineering exploration of GPT-style Large Language Models, covering Transformer implementation, autoregressive generation, pretrained weight integration, downstream adaptation, and supervised instruction fine-tuning.**

## Repo Overview

This repository documents the implementation and experimentation of GPT-style language models, with an emphasis on understanding the computational mechanisms behind modern LLMs.

Rather than treating an LLM as a black-box inference interface, the project decomposes its architecture into individual computational components and reconstructs the forward-pass and training pipeline using PyTorch.

The work progresses from token-level representations and attention mechanisms to complete decoder-only Transformer architecture, followed by pretrained GPT-2 adaptation and task-specific fine-tuning.

The primary objective was to establish a practical understanding of how language models represent context, learn token dependencies, generate sequences, and adapt to downstream objectives.

---

## 01. Transformer Architecture Engineering

Implemented the fundamental components of a decoder-only Transformer architecture.

**Core components:**

- BPE-based tokenization and vocabulary encoding.
- Token embeddings and positional embeddings.
- Query, Key, and Value projections.
- Scaled dot-product self-attention.
- Causal attention masking for autoregressive modeling.
- Multi-head attention with parallel attention heads.
- Layer normalization and residual pathways.
- GELU-activated position-wise feed-forward networks.
- Stacked Transformer blocks.
- Output projection and next-token probability distribution.

<img width="968" height="584" alt="image" src="https://github.com/user-attachments/assets/f3abee72-78aa-44a9-b550-f91bee8bc458" />

**Generation mechanism:** Autoregressive decoding with temperature scaling and Top-k sampling.

<img width="827" height="1280" alt="WhatsApp Image 2026-11-19 at 00 31 16" src="https://github.com/user-attachments/assets/7f412476-1f33-46ef-ba49-7b17c7d56c3d" />

## 02. Model Parameterization & Pretrained Weight Integration

Integrated pretrained GPT-2 parameters into the implemented architecture.
<img width="1679" height="937" alt="image" src="https://github.com/user-attachments/assets/ebf7cac3-4abc-49e4-b7b2-5895364e3f1f" />

- GPT-2 checkpoint retrieval and parameter extraction.
- Mapping checkpoint tensors to corresponding model components.
- Parameter compatibility and dimensional consistency.
- Weight transfer into the implemented Transformer.

This stage connects architectural implementation with practical transfer learning.

## 03. Downstream Task Adaptation — Text Classification

Adapted pretrained GPT-2 for binary SMS spam classification.

**Implementation pipeline:**

`Dataset → Tokenization → Sequence Batching → GPT-2 → Classification Head → Binary Prediction`

<img width="1062" height="820" alt="image" src="https://github.com/user-attachments/assets/3e5b74a3-deda-4eea-9f96-c044b2aef8c6" />

Engineering components:

- Dataset preprocessing and label encoding.
- Train/validation/test partitioning.
- Custom PyTorch Dataset and DataLoader.
- Sequence representation and padding.
- Classification head integration.
- Supervised optimization and validation.
- Held-out test evaluation.

**Experimental results**

| Metric | Score |
|---|---:|
| Training Accuracy | 97.21% |
| Validation Accuracy | 97.32% |
| Test Accuracy | 95.67% |

## 04. Supervised Instruction Fine-Tuning

Extended the pretrained GPT-2 workflow towards instruction-following behavior using supervised instruction-response data.

<img width="2536" height="1430" alt="image" src="https://github.com/user-attachments/assets/ef3a4de2-05cd-4abb-9c55-673ebfc29061" />

**Implementation pipeline:**

`Instruction Dataset → Formatting → Tokenization → Input/Target Construction → Fine-Tuning → Response Generation → Evaluation`

Engineering components:

- Instruction-response dataset preparation.
- Alpaca-style prompt formatting.
- Sequence construction and padding.
- Target masking and loss computation.
- Custom batch collation.
- Supervised fine-tuning.
- Response generation on held-out examples.
- Exploratory LLM-as-judge evaluation.

**Experimental configuration:**

| Parameter | Value |
|---|---|
| Fine-tuning epochs | 2 |
| Evaluation samples | 110 |
| Successfully scored responses | 109 |
| Average judge score | 46.15/100 |

<img width="1844" height="1432" alt="image" src="https://github.com/user-attachments/assets/5879eabe-862d-443a-a293-9bf2577b6d82" />

The evaluation also exposed an important distinction between fluent language generation and reliable instruction adherence.

## Repository Scope

This repository focuses on architectural implementation, pretrained model integration, downstream adaptation, and experimental evaluation.

It does not claim to pretrain a foundation-scale GPT model from random initialization.

## Acknowledgements

Developed as a hands-on LLM engineering journey through iterative implementation, experimentation, debugging, and architectural analysis.

**The focus: understanding the mechanisms behind LLMs by implementing and testing them, rather than only consuming them through inference APIs.**
