
# Neural Networks: From First Principles

## Overview
This repository documents the process of building neural networks from scratch, focusing on fundamental principles rather than relying on high-level libraries. It serves as a technical log of iterative development, highlighting both conceptual and computational challenges encountered.

## Motivation for Architectural Evolution

### Old Architecture
The initial implementations prioritized logical correctness and intuitive structure. However, several limitations became apparent:
- **Lack of Backpropagation Compatibility:** Early designs were not structured to support backpropagation, making training infeasible.
- **Absence of Network Abstraction:** Without a dedicated network class, managing and scaling architectures was cumbersome.
- **Performance Bottlenecks:** The use of basic Python lists and manual loops resulted in slow computations and inefficient memory usage.

### Need for New Architecture
To address these issues, the project transitioned to a new architecture:
- **Matrix-Based Computation:** Leveraging libraries like NumPy enabled faster and more efficient mathematical operations.
- **Improved Scalability:** Introducing a network class and modular design allowed for easier expansion and experimentation.
- **Preparation for Training:** The new structure is designed to be compatible with backpropagation and future training routines.

## Repository Structure
- **Old Architecture/**: Contains early prototypes and logically valid structures, illustrating the learning process and highlighting computational bottlenecks.
- **New Architecture/**: Features optimized, scalable, and training-ready implementations.

## Objective
The primary goal is to discover and understand neural network design through hands-on implementation. By starting with naive approaches and iteratively refining them, this project aims to reveal the rationale behind industry-standard practices.
