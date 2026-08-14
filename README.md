
# Generative Flow Models for Quantum States

A PyTorch implementation of **Generative Flow Models for quantum states**.

This project investigates whether flow-based generative models can learn the distribution of quantum states and efficiently generate new states conditioned on physical or structural properties of the state.

## Overview

Generative models provide a way to learn complicated probability distributions from data and subsequently sample new instances from the learned distribution.

In this project, we apply **Flow Matching** to the problem of quantum-state generation.

Given a quantum state represented by its state vector

$$
|\psi\rangle =
\begin{pmatrix}
\psi_0 \
\psi_1 \
\vdots \
\psi_{2^n-1}
\end{pmatrix},
$$

the model learns a time-dependent vector field

$$
v_\theta(x,t,y),
$$

which transports samples from a simple base distribution, such as a Gaussian distribution, toward the distribution of quantum states.

The model can additionally be conditioned on properties of the quantum state, allowing it to learn different regions of the state space.

---

## Method

### Flow Matching

The training procedure constructs an interpolation between a Gaussian noise sample $\epsilon$ and a target quantum state $x$:

$$
x_t = t x + (1-t)\epsilon,
$$

where

* $x$ is a target quantum state,
* $\epsilon$ is a sample from a Gaussian base distribution,
* $t \sim U(0,1)$ is the flow time.

The corresponding target vector field is

$$
u_t(x_t) = x-\epsilon.
$$

The neural network is trained to predict this vector field:

$$
v_\theta(x_t,t,y) \approx x-\epsilon.
$$

Training therefore minimizes the mean-squared error

$$
\mathcal{L}
===========

\mathbb{E}*{x,\epsilon,t}
\left[
\left|
v*\theta(x_t,t,y)-(x-\epsilon)
\right|^2
\right].
$$

This allows the trained model to define an ODE that transports samples from the base distribution toward the learned quantum-state distribution.

---

## Model Architecture

The repository currently contains two approaches to parameterizing the flow field.

### 1. MLP Flow Model

`Flow_Model_NN.py` implements a fully-connected neural network for learning the flow vector field.

The network takes as input:

* the current state $x_t$,
* a time embedding $t$,
* a conditioning label $y$.

Time and scalar labels are transformed using **Fourier embeddings** before being passed to the network. The resulting representation is processed by a multilayer SiLU network that outputs a vector field with the same dimensionality as the input state.

The basic architecture is:

```text
Quantum state x_t
       │
       ├──────────────┐
       │              │
       ▼              ▼
 State representation   Time Fourier embedding
       │              │
       │        Label Fourier embedding
       │              │
       └───────┬──────┘
               ▼
        Fully Connected
               │
             SiLU
               │
        Fully Connected
               │
             SiLU
               │
        Fully Connected
               │
             SiLU
               │
               ▼
        Predicted Vector Field
```

### 2. Transformer Flow Model

`Flow_Model_transformer.py` explores a transformer-based architecture for quantum-state representations.

Rather than treating the entire state vector as a single input, the state vector can be divided into smaller **tokens**. These tokens are then projected into a higher-dimensional representation and processed using transformer-style components.

The transformer implementation includes components for:

* zero-padding states with different dimensions,
* splitting state vectors into patches,
* patch embeddings,
* Fourier time embeddings,
* scalar label embeddings,
* one-hot class embeddings,
* adaptive normalization,
* feed-forward transformer layers.

This architecture is intended to provide a more scalable representation for quantum states as the number of qubits increases.

---

## Conditioning

The flow model supports conditional generation.

A scalar physical quantity can be encoded using a Fourier embedding:

$$
y
\rightarrow
[
\cos(2\pi\omega_1y),
\dots,
\cos(2\pi\omega_dy),
\sin(2\pi\omega_1y),
\dots,
\sin(2\pi\omega_dy)
].
$$

Discrete quantum-state classes can additionally be represented using one-hot embeddings.

The model therefore has the general form

$$
v_\theta(x_t,t,y),
$$

where $y$ specifies the desired condition.

The implementation also supports **classifier-free guidance-style conditioning dropout**, where the conditioning information is randomly removed during training.

---

## Repository Structure

```text
Generative-Flow-Model-for-Quantum-States/
│
├── Flow_Model_NN.py
│   └── MLP-based flow-field network
│
├── Flow_Model_Training.py
│   └── Flow-matching training procedure
│
├── Flow_Model_transformer.py
│   └── Transformer-based flow model
│
├── Flow_model_sampler.py
│   └── Sampling / ODE integration
│
├── plots_func.py
│   └── Visualization utilities
│
├── Benchmarking/
│   └── Benchmarking experiments
│
├── data/
│   └── Training datasets
│
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/flaa-ux/Generative-Flow-Model-for-Quantum-States.git
cd Generative-Flow-Model-for-Quantum-States
```

Install the required Python packages:

```bash
pip install torch torchvision numpy matplotlib
```

The implementation is written in **PyTorch** and automatically uses an available accelerator when supported, otherwise falling back to the CPU.

---

## Training

The flow model is trained by sampling:

1. A target quantum state $x$.
2. A Gaussian noise vector $\epsilon$.
3. A random time $t\in[0,1]$.
4. An interpolated state $x_t$.
5. The target vector field $x-\epsilon$.

The network then minimizes the MSE between the predicted and target vector fields.

A simplified version of the training procedure is:

```python
gaussian_noise = torch.randn(...)
time = torch.rand(...)

interpolated_state = (
    time * target_state
    + (1 - time) * gaussian_noise
)

target_vector_field = target_state - gaussian_noise

predicted_vector_field = model(
    interpolated_state,
    label,
    time
)

loss = MSELoss(
    predicted_vector_field,
    target_vector_field
)
```

The training implementation uses the Adam optimizer and saves the resulting model parameters as a PyTorch `.pth` checkpoint.

---

## Sampling

After training, new quantum states can be generated by integrating the learned vector field.

Starting from Gaussian noise,

$$
x_0 \sim \mathcal{N}(0,I),
$$

the model solves

$$
\frac{dx}{dt}
=============

v_\theta(x,t,y)
$$

from $t=0$ to $t=1$.

The resulting state $x_1$ is a sample from the distribution learned by the model.

The sampling implementation is contained in `Flow_model_sampler.py`.

---

## Experiments

The repository is intended to investigate questions such as:

* Can flow models learn distributions of quantum states?
* How well do generated states reproduce the statistics of the training distribution?
* How does conditional generation perform across different quantum-state properties?
* How does an MLP architecture compare with a transformer architecture?
* How does the model scale with the number of qubits?
* Can the learned flow efficiently generate physically meaningful quantum states?

---

## Current Status

This repository is an ongoing research project.

Current work includes:

* [x] Basic flow-matching formulation
* [x] MLP flow-field model
* [x] Fourier conditioning
* [x] Conditional generation
* [x] Conditioning dropout
* [x] Flow-model sampling
* [x] Initial transformer architecture
* [ ] Systematic benchmarking
* [ ] Scaling to larger numbers of qubits
* [ ] Improved quantum-state representations
* [ ] Comprehensive evaluation of generated states

---

## Quantum State Representation

For an $n$-qubit system, a pure quantum state is represented by a complex vector of dimension

$$
2^n.
$$

In numerical experiments, the real and imaginary components can be represented separately, giving a real-valued vector of dimension

$$
2^{n+1}.
$$

This representation allows standard neural-network architectures to operate directly on the state-vector representation.

An important direction for future work is incorporating the physical structure of quantum states directly into the generative model rather than treating the state vector simply as a high-dimensional Euclidean vector.

---

## References

This project builds on ideas from **Flow Matching** and modern generative modeling.

* Lipman et al., *Flow Matching for Generative Modeling*, 2022.
* Chen et al., *Flow Matching for Generative Modeling*, 2022.
* Related work on generative modeling of quantum states.

---

## License

This project is licensed under the MIT License.

---

## Author

**Fansen Funata**

This repository contains research and experimental code for investigating generative flow models applied to quantum-state distributions.
