# Transformer From Scratch in JAX

A decoder-only GPT-style Transformer implemented from scratch using
low-level JAX operations.

The goal of this project is to understand the mathematical foundations,
implementation, and computational performance of modern Transformer
architectures.

Rather than relying on high-level neural network libraries, the model
will be implemented primarily using JAX and jax.numpy.

## Objectives

1. Understand the mathematics behind Transformer architectures.
2. Implement neural network components from scratch.
3. Understand automatic differentiation and backpropagation.
4. Build and train a decoder-only language model.
5. Experiment with different Transformer architectures.
6. Study GPU execution and performance optimization.

## Technologies

- Python
- JAX
- XLA
- NVIDIA CUDA
- NVIDIA Blackwell GPU

## Architecture

The model will follow a decoder-only Transformer architecture.

```text
Input Tokens
    |
    v
Token Embeddings
    |
    v
Positional Information
    |
    v
Transformer Blocks (N layers)
    |
    |-- RMSNorm
    |-- Multi-Head Causal Self-Attention
    |-- Residual Connection
    |-- RMSNorm
    |-- Feed-Forward Network
    |-- Residual Connection
    |
    v
Final RMSNorm
    |
    v
Output Projection
    |
    v
Next-Token Logits
```

## Implementation Roadmap

### Phase 1: Neural Network Fundamentals

- [ ] Implement linear layers
- [ ] Implement activation functions
- [ ] Implement softmax
- [ ] Implement RMSNorm
- [ ] Understand JAX automatic differentiation
- [ ] Implement weight initialization

### Phase 2: Attention Mechanisms

- [ ] Implement query, key, and value projections
- [ ] Implement scaled dot-product attention
- [ ] Implement causal masking
- [ ] Implement multi-head attention
- [ ] Understand attention computational complexity

### Phase 3: Transformer Architecture

- [ ] Implement token embeddings
- [ ] Implement positional embeddings
- [ ] Implement feed-forward networks
- [ ] Implement residual connections
- [ ] Implement Transformer blocks
- [ ] Assemble a complete decoder-only Transformer

### Phase 4: Training

- [ ] Implement cross-entropy loss
- [ ] Implement AdamW optimizer
- [ ] Implement training loop
- [ ] Apply JAX JIT compilation
- [ ] Train on Tiny Shakespeare
- [ ] Implement autoregressive text generation

### Phase 5: Experiments

- [ ] Compare normalization methods
- [ ] Compare activation functions
- [ ] Experiment with positional encodings
- [ ] Benchmark training throughput
- [ ] Analyze GPU memory consumption
- [ ] Experiment with model scaling

### Phase 6: Advanced GPU Optimization (Optional)

- [ ] Profile JAX/XLA execution
- [ ] Investigate attention kernel performance
- [ ] Explore JAX Pallas
- [ ] Compare custom kernels against standard JAX operations

## Dataset

Initial experiments will use the Tiny Shakespeare dataset.

The dataset is small enough to train a compact language model
while studying Transformer behavior.

## Hardware

Development and training will be performed on Northwestern University's
DeepDish GPU infrastructure.

Primary target:

- NVIDIA RTX PRO 5000 Blackwell
- 48 GB GPU memory

## Project Philosophy

The objective is not to reproduce an existing implementation
line by line.

Instead, each component will be developed by:

1. Understanding the underlying mathematics.
2. Implementing the operations using low-level JAX.
3. Verifying correctness through unit tests.
4. Studying gradients and tensor dimensions.
5. Measuring computational performance.

## Status

Project setup complete. Implementation pending.