# Generative Flow Models for Quantum States

A PyTorch implementation of **generative flow models for SRE-based quantum-state generation**.

This project investigates whether generative flow models can learn the distribution of quantum states **conditioned on their Stabilizer Rényi Entropy (SRE)**, enabling efficient generation of quantum states with a desired level of non-stabilizerness.

## Overview

The goal of this project is to develop a generative model capable of producing quantum states with a specified **Stabilizer Rényi Entropy (SRE)**.

The Stabilizer Rényi Entropy provides a measure of the **non-stabilizerness**, or *magic*, of a quantum state. Stabilizer states have zero SRE, while states with increasing SRE exhibit increasingly non-stabilizer character.

Rather than sampling quantum states uniformly or generating them without regard to their physical properties, this project aims to learn a conditional distribution

$$
p(\psi \mid \mathrm{SRE}),
$$

where $\psi$ is a quantum state and the conditioning variable specifies its desired Stabilizer Rényi Entropy.

The ultimate objective is therefore:

$$
\boxed{
\text{SRE} \longrightarrow \text{Generative Flow Model}
\longrightarrow \text{Quantum State}
}
$$

Given a target SRE value, the trained model can be used to generate quantum states whose SRE is concentrated around the desired value.

---

## Motivation

### Why generate quantum states based on SRE?

Non-stabilizerness is an important resource in quantum computation and quantum information. Stabilizer states can be efficiently simulated classically, while non-stabilizer states provide computational resources that enable quantum advantage.

The **Stabilizer Rényi Entropy** gives a quantitative way to characterize this resource.

However, obtaining large collections of quantum states with controlled values of SRE can be computationally expensive. A generative model provides an alternative approach:

1. Generate or collect a dataset of quantum states.
2. Calculate the SRE of each state.
3. Train a conditional generative model on the resulting $(\psi,\mathrm{SRE})$ pairs.
4. Specify a target SRE.
5. Generate new quantum states conditioned on that target.

This transforms quantum-state generation into a **conditional generative modeling problem**.

---

## Problem Formulation

For an $n$-qubit pure state,

$$
|\psi\rangle =
\sum_{i=0}^{2^n-1} c_i |i\rangle,
$$

we associate each state with its Stabilizer Rényi Entropy,

$$
S_R(\psi).
$$

The model is trained to learn the conditional distribution

$$
p(\psi \mid S_R).
$$

During generation, we provide a desired SRE value $s$ and sample from

$$
\psi \sim p_\theta(\psi \mid S_R=s).
$$

The generated state can then be evaluated independently to determine whether its actual SRE is close to the requested value.

This gives a natural evaluation loop:

```text
       Target SRE
           │
           ▼
   ┌─────────────────┐
   │ Conditional     │
   │ Flow Model      │
   └────────┬────────┘
            │
            ▼
    Generated State ψ
            │
            ▼
      Calculate SRE
            │
            ▼
     Compare with
      Target SRE
```

---

## Flow Matching

The generative model is based on **Flow Matching**.

Instead of directly learning the probability density of quantum states, the model learns a time-dependent vector field

$$
v_\theta(x,t,s),
$$

which transports samples from a simple base distribution toward the target quantum-state distribution.

Here:

* $x$ is the quantum-state representation,
* $t\in[0,1]$ is the flow time,
* $s$ is the target SRE,
* $v_\theta$ is the learned vector field.

A Gaussian noise sample $\epsilon$ is interpolated with a target state $x$:

$$
x_t = t x + (1-t)\epsilon.
$$

The corresponding target vector field is

$$
u_t = x-\epsilon.
$$

The model minimizes

$$
\mathcal{L}
===========

\mathbb{E}
\left[
\left|
v_\theta(x_t,t,s)-(x-\epsilon)
\right|^2
\right].
$$

The learned vector field can subsequently be integrated as an ODE to transform Gaussian noise into a generated quantum state.

---

## SRE Conditioning

The central feature of the model is **conditioning on Stabilizer Rényi Entropy**.

The SRE value is embedded into a high-dimensional representation using a Fourier feature embedding. This allows the network to learn a smooth dependence between the target SRE and the generated state distribution.

Conceptually:

$$
s
\rightarrow
\text{Fourier Embedding}
\rightarrow
\text{Conditional Flow Model}
\rightarrow
|\psi\rangle.
$$

This allows the same model to generate states across a range of SRE values instead of requiring a separate model for each target.

For example:

```text
SRE = 0.0  ──────►  Generate low/non-magic states

SRE = 0.5  ──────►  Generate moderately non-stabilizer states

SRE = 1.0  ──────►  Generate more strongly non-stabilizer states

SRE = target ────►  Generate states conditioned on target SRE
```

---

## Model Architecture

The repository currently explores two approaches for parameterizing the conditional flow field.

### MLP Flow Model

`Flow_Model_NN.py` implements a fully-connected neural network that takes the current state, flow time, and SRE conditioning as inputs.

The architecture uses Fourier embeddings for the conditioning variables and SiLU-activated fully connected layers to predict the flow vector field.

```text
                  Quantum State x_t
                         │
                         │
                         ▼
                   State Input
                         │
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
  Time Embedding                    SRE Embedding
        │                                 │
        └────────────────┬────────────────┘
                         ▼
                  Neural Network
                         │
                  SiLU / Linear
                         │
                         ▼
                Predicted Flow Field
                         │
                         ▼
                       dx/dt
```

### Transformer Flow Model

`Flow_Model_transformer.py` explores a transformer-based architecture for representing higher-dimensional quantum states.

The state vector can be divided into patches/tokens, embedded into a latent representation, and processed using transformer-style blocks.

The goal is to investigate whether transformer architectures can better capture structure in quantum-state distributions as the number of qubits increases.

---

## Generation

Once the model has been trained, generation begins from Gaussian noise:

$$
x_0\sim\mathcal{N}(0,I).
$$

The learned flow is then integrated according to

$$
\frac{dx}{dt}
=============

v_\theta(x,t,S_R)
$$

from $t=0$ to $t=1$.

The resulting $x_1$ represents the generated quantum state.

The desired SRE is supplied throughout the integration, meaning that the generated state is explicitly conditioned on the requested non-stabilizerness.

---

## Evaluation

A central question is whether the model actually generates states with the requested SRE.

For a target SRE $S_{\mathrm{target}}$, generated states are evaluated independently:

$$
S_{\mathrm{generated}}
======================

S_R(\psi_{\mathrm{generated}}).
$$

The generation error can then be quantified as

$$
\Delta S
========

\left|
S_{\mathrm{generated}}
----------------------

S_{\mathrm{target}}
\right|.
$$

A successful model should produce a distribution satisfying

$$
\mathbb{E}[S_{\mathrm{generated}}]
\approx
S_{\mathrm{target}},
$$

while also reproducing other relevant properties of the underlying quantum-state distribution.

---

## Research Questions

This project is primarily concerned with the following questions:

* Can flow matching learn the distribution of quantum states conditioned on SRE?
* Can a single model generate states across a continuous range of SRE values?
* How accurately can the generated SRE match the requested SRE?
* Does conditioning allow efficient sampling of rare or difficult-to-generate SRE regimes?
* How does the quality of generation scale with the number of qubits?
* Do transformer-based architectures improve over MLP-based flow models?
* Beyond SRE itself, do generated states reproduce other statistical properties of the training distribution?

---

## Repository Structure

```text
Generative-Flow-Model-for-Quantum-States/
│
├── Flow_Model_NN.py
│   └── MLP-based conditional flow model
│
├── Flow_Model_Training.py
│   └── Flow-matching training procedure
│
├── Flow_Model_transformer.py
│   └── Transformer-based flow model
│
├── Flow_model_sampler.py
│   └── ODE-based state generation
│
├── plots_func.py
│   └── Visualization utilities
│
├── Benchmarking/
│   └── Benchmarking experiments
│
├── data/
│   └── Training data
│
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/flaa-ux/Generative-Flow-Model-for-Quantum-States.git
cd Generative-Flow-Model-for-Quantum-States

pip install torch torchvision numpy matplotlib
```

---

## Current Status

This is an ongoing research project exploring **SRE-conditioned generative modeling of quantum states**.

Current directions include:

* [x] Flow-matching formulation
* [x] Conditional flow-field model
* [x] Fourier SRE embeddings
* [x] Conditional generation
* [x] ODE-based sampling
* [x] Initial transformer architecture
* [ ] Quantitative SRE-generation benchmarks
* [ ] Generation across a broad SRE range
* [ ] Scaling to larger numbers of qubits
* [ ] Comparison between MLP and transformer architectures
* [ ] Evaluation of generated-state statistics
* [ ] Investigation of generation accuracy in high-SRE regimes

---

## References

This project builds upon work in:

* **Flow Matching for Generative Modeling**
* **Stabilizer Rényi Entropy and measures of quantum non-stabilizerness**
* Generative modeling of quantum states
* Transformer architectures for high-dimensional generative modeling

---

## Author

**Fansen Funata**

Research project investigating the use of generative flow models for **SRE-conditioned quantum-state generation**.
