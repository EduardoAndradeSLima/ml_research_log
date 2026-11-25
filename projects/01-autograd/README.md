# 01 — Scalar Autograd Engine

From-scratch scalar reverse-mode automatic differentiation engine in pure Python, inspired by Karpathy's `micrograd`.

## Goal
Implement a computation graph with dynamic topological sorting and backpropagation, then train a small MLP on a toy dataset without PyTorch.

## Implemented
- `engine.py`: `Value` class with `+`, `*`, `**`, `tanh`, `relu`, and reverse-mode `backward()`.
- `nn.py`: `Neuron`, `Layer`, and `MLP` built on top of `Value`.
- `test_engine.py`: Gradient verification against PyTorch / numerical finite differences.

## Run
```bash
python test_engine.py