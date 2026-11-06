# Small Language Model from Scratch on Intel Arc XPU

An experimental Small Language Model (SLM) implementation developed using PyTorch, adapted to run on Intel Arc GPUs through the PyTorch XPU backend.

## Project Overview

This project explores the development and training of a Small Language Model from scratch, with a focus on adapting the training pipeline to Intel GPU hardware.

## Objectives

- Implement a Small Language Model architecture.
- Understand the underlying training pipeline of transformer-based language models.
- Adapt PyTorch operations for Intel Arc XPU acceleration.
- Experiment with model training and evaluation.
- Explore performance and compatibility considerations when using Intel GPU hardware.

## Project Structure

```text
SLM-Intel-Arc-XPU/
├── SLM_IntelArc_XPU.ipynb
├── Vizuara_AI_Labs_Small_Language_Model_Scratch_Final.ipynb
├── requirements.txt
├── loss_curves.png
├── README.md
└── .gitignore
```

## Hardware and Software

- GPU: Intel Arc
- Framework: PyTorch
- Acceleration: PyTorch XPU
- Language: Python
- Environment: Windows

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Ensure your PyTorch installation supports Intel XPU before running the notebooks.

## Model Architecture

This project implements a compact, GPT-style decoder-only Transformer language model trained from scratch using the TinyStories dataset.

The architecture is inspired by the GPT family of language models and consists of token embeddings, learned positional embeddings, stacked Transformer decoder blocks, and a language modelling head.

### Architecture Configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context window | 128 tokens |
| Transformer layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Feed-forward dimension | 1,536 |
| Dropout | 0.1 |
| Activation function | GELU |
| Normalization | LayerNorm |
| Tokenizer | GPT-2 |
| Parameters | Approximately 20 million |

### Transformer Components

**1. Token and Positional Embeddings**

Input tokens are converted into 384-dimensional vectors using a learned token embedding matrix. Learned positional embeddings are added to provide information about token order.

**2. Causal Multi-Head Self-Attention**

Each Transformer block contains a causal self-attention mechanism with six attention heads. The causal attention mask prevents the model from accessing future tokens when predicting the next token.

Where supported, the implementation uses PyTorch's `scaled_dot_product_attention` operation, with a manual masked-attention fallback.

**3. Feed-Forward Network**

Each block contains a two-layer feed-forward network that expands the hidden dimension from 384 to 1,536 before projecting it back to 384. GELU is used as the activation function.

**4. Residual Connections and Layer Normalization**

Residual connections help information and gradients flow through the network, while LayerNorm stabilizes intermediate representations.

**5. Language Modelling Head**

The final hidden representations are projected into the vocabulary space to produce next-token predictions. The token embedding matrix and language modelling head share weights through weight tying.

### Training Objective

The model is trained using next-token prediction with cross-entropy loss. Given a sequence of input tokens, it learns to predict the next token at each position.

### Hardware Compatibility

The implementation supports multiple PyTorch execution backends:

- Intel Arc GPUs through PyTorch XPU.
- NVIDIA GPUs through CUDA.
- CPU execution as a fallback.

Automatic device detection and device-aware mixed-precision configuration allow the same implementation to target different hardware environments.

## Datasets and Model Checkpoints

Large datasets and trained model checkpoints are maintained separately from the Git repository.

Instructions for obtaining these artifacts will be added here.

## Acknowledgements

This project builds upon the Small Language Model learning material by Vizuara AI Labs. The original notebook is retained for reference and attribution.

## Project Status

Experimental / In Development.
