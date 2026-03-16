# Neural Networks: From Scratch

## Overview
This repository documents an experimental approach of building neural networks from first principles. Instead of relying on high-level machine learning frameworks, the project focuses on implementing the underlying mechanics manually in order to understand how neural networks function internally.

The repository acts as both:
* A technical learning log
* A progressive implementation archive

It shows how simple, intuitive ideas gradually evolve into more structured and computationally efficient systems.

---

## Motivation
Modern deep learning frameworks abstract away much of the underlying computation. While convenient, this abstraction can obscure how neural networks actually operate.

The objective of this project is to:
* Understand neural network mechanics at a fundamental level.
* Implement core concepts manually.
* Observe how architectural and computational decisions affect performance and scalability.

The approach intentionally begins with naive implementations and progressively improves them.

---

## Architectural Evolution

### Initial Architecture
The earliest implementations focused primarily on conceptual correctness and intuitive structure.

**Characteristics:**
* List-based computations using basic Python structures.
* Manual loops for neuron and layer calculations.
* Minimal abstraction.

Although logically valid, several issues emerged:
* **Backpropagation incompatibility:** The original structure was not designed with gradient propagation in mind.
* **Lack of network abstraction:** Without a dedicated network structure, scaling to deeper architectures became difficult.
* **Performance limitations:** Python lists and nested loops resulted in slow computations and poor memory efficiency.

These limitations motivated a structural redesign.

### Revised Architecture
The new architecture introduces improvements aimed at scalability, efficiency, and training compatibility.

**Key changes include:**
* **Matrix-Based Computation:** Linear algebra operations are implemented using NumPy, allowing faster vectorized computations and cleaner mathematical representation.
* **Modular Network Design:** A dedicated `Network` abstraction manages layers, parameters, and forward propagation, enabling easier experimentation with architectures.
* **Training Compatibility:** The structure now aligns with the requirements of backpropagation, preparing the system for gradient-based learning algorithms.

---

The repository will continue to evolve as both a learning tool and a practical implementation reference.

---
## Repository Structure

```text
Neural-Networks-From-First-Principles/
├── Old Architecture/
│   Early experimental implementations.
│   These versions demonstrate conceptual exploration
│   and highlight computational limitations.
├── New Architecture/
│   Refactored implementations using matrix operations
│   and modular design, intended to support training
│   and scalable experimentation.

