# 🧠 Neural Networks from Zero to Hero

> Knowledge is light, and this man [Andrej Karpathy]—by the grace of Allah—was instrumental in illuminating many concepts in my mind. May Allah enlighten his insight and guide him 
> 
> Truly, from the very first half-hour, this course stood out as completely different from any other I had taken previously. After completing the Machine Learning and Deep Learning specializations with Andrew Ng on Coursera, I felt I understood concepts like backpropagation; however, I realized I had only grasped the surface. By the grace of God, this course has significantly deepened my understanding and insight into how things actually work. 
> 
> From start to finish, the course focuses on practical application and a deep dive into the fundamentals—delivered in a truly enjoyable and amazing way.
> 
> *O Allah, teach us what benefits us, benefit us with what You have taught us, and increase us in knowledge.*

---

A deep-dive journey into building modern deep learning frameworks from the absolute ground up. This repository tracks my implementation of foundations taught by Andrej Karpathy, moving from scalar-valued autograd engines to full character-level language models. 

Instead of treating frameworks like PyTorch as a black box, this repository houses from-scratch implementations to master the underlying calculus, graph tracking, and architectural micro-components of deep neural networks.

---

## 🛠️ Repository Roadmap & Architecture

### 1. `micrograd/` — The Scalar Autograd Engine
A tiny, pure-Python scalar-valued automatic gradient engine implementing backpropagation along a dynamically built Directed Acyclic Graph (DAG). 
* **Core Engine:** Custom `Value` class tracking mathematical operations (`+`, `*`, `pow`, `exp`, `tanh`, `relu`).
* **Graph Management:** Automated **Topological Sorting** to accurately structure the sequence of the backward pass execution.
* **Bug Safety:** Proper gradient accumulation (`+=`) handling multi-use variable branches to prevent mathematical divergence.
* **Neural Blocks:** Modular constructions for single `Neuron` definitions, parallel `Layer` groupings, and stacked Multi-Layer Perceptrons (`MLP`).

### 2. `makemore/` — Character-Level Language Models
Stepping from scalars to multi-dimensional data arrays, this module maps, processes, and evaluates the `names.txt` dataset to generate fresh, realistic human names.
* **Bigram Count Matrix:** Creating statistical frequency distributions using manual indexing and dictionary lookups.
* **The Neural Network Approach:** Transitioning counts into a single-layer linear model via PyTorch.
* **Key Mechanisms:** Forward pass operations featuring dynamic **One-Hot Encoding**, vectorized Matrix Multiplication (`@`), and stable **Softmax** probability calculations (`logits.exp()`).
* **Optimization:** Evaluation via Negative Log-Likelihood (NLL) loss tracked and minimized through Stochastic Gradient Descent (SGD).

---

## 🚀 Key Mathematical Takeaways Implemented

* **The Chain Rule:** Hardcoded analytical local derivatives within closure functions (`_backward()`) to pass downstream gradients sequentially:
  $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x}$$
* **Gradient Flow Disruption:** Debugged and engineered safeguards against common numerical stability limits (e.g., handling explosive outputs causing `NaN`/`inf` probability distributions by tracking proper learning rate weight steps).
* **Framework Alignment:** Verified custom implementations directly alongside industry-standard `torch.Tensor` architectures to confirm bit-perfect alignment across gradients.

---

## 🎯 The Optimization Loop Layout

Every model trained inside this workspace adheres strictly to the fundamental 4-step deep learning cycle:
1. **Forward Pass:** Compute predictions (`ypred`) and capture current error margins via loss functions.
2. **Zero Gradients:** Reset accumulated partial derivatives across all active parameter nodes (`p.grad = 0.0`).
3. **Backward Pass:** Trigger recursive backpropagation starting from the final scalar loss node (`loss.backward()`).
4. **Parameter Step:** Execute Gradient Descent optimization to update parameters against the direction of ascent (`p.data -= lr * p.grad`).

---

## 💻 Tech Stack & Environment
* **Language:** Python 3.x
* **Libraries:** PyTorch (for verification and tensor exploration)
* **Environment:** Linux Ubuntu / Jupyter Notebooks

---
*Developed as a journey to master Deep Learning mechanics from first principles.*
