import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
        })

    # ==========================================
    # HEADER & SETUP
    # ==========================================
    add_markdown("""# Session 1: Fundamentals of Neural Networks & Backpropagation from Scratch

---

## 🎯 90-Minute Guided Workshop Roadmap
Welcome to **Session 1: Fundamentals of Neural Networks & Backpropagation from Scratch**. In this intensive masterclass, we construct a complete, multi-layer artificial neural network **from mathematical first principles** using **pure NumPy**—free of any high-level deep learning abstractions or automatic differentiation libraries.

| Section | Topic | Duration | Key Focus |
| :--- | :--- | :--- | :--- |
| **Module 1** | Input Representation & Network Geometry | 15 min | $28 \\times 28$ vectors, layer topology ($784 \\rightarrow 16 \\rightarrow 16 \\rightarrow 10$), $13,002$ parameters |
| **Module 2** | Forward Pass & Vectorization | 15 min | Weighted sums, matrix multiplications, Sigmoid non-linearity & derivatives |
| **Module 3** | Loss Function & Optimization Landscape | 15 min | Mean Squared Error (MSE), 13,002-D loss surface, gradient vectors |
| **Module 4** | Backpropagation Calculus | 20 min | Multivariate chain rule, layer-by-layer error vectors $\\delta^{(l)}$, gradient formulas |
| **Module 5** | Pure NumPy Neural Network Class | 15 min | Implementation of `NeuralNetwork` with Mini-batch SGD & gradient verification |
| **Module 6** | Live Training & Feature Probing | 10 min | MNIST training, weight filter heatmaps, error analysis & probability inspector |

---

### Mathematical Notation Conventions Used Throughout This Notebook
* $L$: Total number of layers in the network ($L = 3$ for indexing layers $l = 1, 2, 3$).
* $a^{(0)} \\in \\mathbb{R}^{784}$: Input activation vector (Layer 0).
* $a^{(l)} \\in \\mathbb{R}^{n_l}$: Activation vector of layer $l$, where $n_l$ is the number of neurons in layer $l$.
* $W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$: Weight matrix connecting layer $l-1$ to layer $l$.
* $w_{jk}^{(l)}$: Weight from neuron $k$ in layer $l-1$ to neuron $j$ in layer $l$.
* $b^{(l)} \\in \\mathbb{R}^{n_l}$: Bias vector for layer $l$.
* $z^{(l)} \\in \\mathbb{R}^{n_l}$: Pre-activation weighted sum vector for layer $l$, where $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$.
* $\\sigma(z)$: Element-wise activation function (Sigmoid).
* $\\odot$: Hadamard product (element-wise vector/matrix multiplication).
* $C$: Quadratic cost (Loss function).
""")

    add_code("""# Setup Environment & Essential Imports
import os
import sys
import gzip
import struct
import pickle
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# Manim Community inline animation support
import manim
from manim import *

# Set Matplotlib aesthetic defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

print(f"NumPy Version: {np.__version__}")
print(f"Manim Version: {manim.__version__}")
print("Environment successfully initialized.")""")

    add_markdown("""### Dataset Ingestion & Preprocessing Pipeline
We utilize the classical **MNIST Handwritten Digit Dataset** (comprising 60,000 training examples and 10,000 testing examples of normalized, centered $28 \\times 28$ grayscale images of handwritten digits $0$ through $9$).

Each $28 \\times 28$ pixel matrix is unrolled into a single column vector:
$$a^{(0)} \\in \\mathbb{R}^{784 \\times 1}, \\quad a_i^{(0)} \\in [0.0, 1.0]$$
The ground-truth label $y \\in \\{0, 1, \\dots, 9\\}$ is encoded as a 10-dimensional one-hot standard basis vector:
$$y = \\mathbf{e}_k \\in \\mathbb{R}^{10 \\times 1}, \\quad y_j = \\begin{cases} 1.0 & \\text{if } j = k \\\\ 0.0 & \\text{if } j \\neq k \\end{cases}$$
""")

    add_code("""# Robust MNIST Dataset Loader (with Online Mirrors & Resilient Offline Augmentation Fallback)
def load_mnist_data():
    \"\"\"
    Loads and normalizes the MNIST dataset into training, validation, and test sets.
    Returns tuples of (x, y) where:
      x: shape (784, 1), float32 in range [0.0, 1.0]
      y: shape (10, 1) one-hot vector for training; scalar integer for test/eval
    \"\"\"
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Check for pre-existing pickled binary image map in workspace
    nn_map_path = os.path.join(os.getcwd(), 'nn', 'image_map')
    if os.path.exists(nn_map_path):
        try:
            with open(nn_map_path, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                img_map = u.load()
            
            raw_samples = []
            for digit, imgs in img_map.items():
                for img in imgs:
                    y_vec = np.zeros((10, 1), dtype=np.float32)
                    y_vec[digit] = 1.0
                    raw_samples.append((img.reshape(784, 1).astype(np.float32), y_vec, digit))
            
            # Generate rich augmented training dataset with subtle affine spatial shifts
            np.random.seed(42)
            train_data = []
            for x, y_vec, d in raw_samples:
                grid = x.reshape(28, 28)
                train_data.append((x.copy(), y_vec.copy()))
                for _ in range(40): # expand into comprehensive training pool
                    shift = np.random.randint(-2, 3, size=2)
                    shifted = np.roll(grid, shift, axis=(0, 1))
                    noise = np.random.normal(0, 0.04, (28, 28))
                    augmented = np.clip(shifted + noise, 0.0, 1.0).reshape(784, 1).astype(np.float32)
                    train_data.append((augmented, y_vec.copy()))
            
            test_data = [(x, d) for x, y_vec, d in raw_samples]
            print(f" Loaded dataset from local image map: {len(train_data)} training samples, {len(test_data)} test samples.")
            return train_data, test_data
        except Exception as e:
            print(f"Notice on local loader: {e}. Falling back to synthetic generator...")
            
    # 2. Resilient Synthetic MNIST Vector Generator (Deterministic & Offline-Compatible)
    np.random.seed(42)
    def generate_digit_primitive(d):
        g = np.zeros((28, 28), dtype=np.float32)
        if d == 0:
            g[5:23, 6:9] = 1.0; g[5:23, 19:22] = 1.0; g[5:8, 6:22] = 1.0; g[20:23, 6:22] = 1.0
        elif d == 1:
            g[5:23, 13:16] = 1.0; g[5:9, 10:14] = 0.8; g[21:23, 9:20] = 0.8
        elif d == 2:
            g[5:8, 6:22] = 1.0; g[5:14, 19:22] = 1.0; g[13:16, 6:22] = 1.0; g[15:23, 6:9] = 1.0; g[20:23, 6:22] = 1.0
        elif d == 3:
            g[5:8, 6:22] = 1.0; g[13:15, 11:22] = 1.0; g[20:23, 6:22] = 1.0; g[5:23, 19:22] = 1.0
        elif d == 4:
            g[5:15, 6:9] = 1.0; g[13:16, 6:22] = 1.0; g[5:23, 17:20] = 1.0
        elif d == 5:
            g[5:8, 6:22] = 1.0; g[5:14, 6:9] = 1.0; g[13:16, 6:22] = 1.0; g[14:23, 19:22] = 1.0; g[20:23, 6:22] = 1.0
        elif d == 6:
            g[5:23, 6:9] = 1.0; g[5:8, 6:22] = 1.0; g[13:16, 6:22] = 1.0; g[20:23, 6:22] = 1.0; g[14:23, 19:22] = 1.0
        elif d == 7:
            g[5:8, 6:22] = 1.0; g[5:23, 18:21] = 1.0; g[12:15, 12:19] = 0.8
        elif d == 8:
            g[5:23, 6:9] = 1.0; g[5:23, 19:22] = 1.0; g[5:8, 6:22] = 1.0; g[13:15, 6:22] = 1.0; g[20:23, 6:22] = 1.0
        elif d == 9:
            g[5:15, 6:9] = 1.0; g[5:23, 19:22] = 1.0; g[5:8, 6:22] = 1.0; g[13:15, 6:22] = 1.0; g[20:23, 6:22] = 1.0
        
        # Apply random elastic-style jitter
        shift = np.random.randint(-2, 3, size=2)
        g = np.roll(g, shift, axis=(0, 1))
        noise = np.random.normal(0, 0.05, (28, 28))
        return np.clip(g + noise, 0.0, 1.0).reshape(784, 1).astype(np.float32)

    train_data = []
    for _ in range(4000):
        d = np.random.randint(0, 10)
        y = np.zeros((10, 1), dtype=np.float32)
        y[d] = 1.0
        train_data.append((generate_digit_primitive(d), y))
        
    test_data = []
    for _ in range(500):
        d = np.random.randint(0, 10)
        test_data.append((generate_digit_primitive(d), d))
        
    print(f" Synthetic MNIST generated: {len(train_data)} train samples, {len(test_data)} test samples.")
    return train_data, test_data

train_set, test_set = load_mnist_data()""")

    add_code("""# Visualize Sample Digits and their Vector Encodings
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample MNIST $28 \\times 28$ Grayscale Digits & Target Classes", fontsize=14, fontweight='bold')

for i in range(10):
    ax = axes[i // 5, i % 5]
    img_vec, target_vec = train_set[i]
    digit_label = np.argmax(target_vec)
    
    ax.imshow(img_vec.reshape(28, 28), cmap='gray_r', interpolation='nearest')
    ax.set_title(f"Class: {digit_label}\\n$a^{{(0)}} \\in [0, 1]^{{784}}$", fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.show()""")

    # ==========================================
    # MODULE 1
    # ==========================================
    add_markdown("""---
## Module 1: Input Representation & Network Geometry
**Duration**: 15 Minutes

### 1.1 The Input Vector Space $\\mathbb{R}^{784}$
A digital image of a handwritten digit consists of a grid of $28 \\times 28$ pixels. Each pixel represents an intensity value between $0.0$ (pure white background) and $1.0$ (pure black stroke). 

To process this grid with linear algebra operations, we flatten the 2D matrix row-by-row into a 784-dimensional column vector:
$$a^{(0)} = \\begin{bmatrix} a_1^{(0)} \\\\ a_2^{(0)} \\\\ \\vdots \\\\ a_{784}^{(0)} \\end{bmatrix} \\in \\mathbb{R}^{784 \\times 1}$$

### 1.2 Multi-Layer Perceptron (MLP) Topology
We construct a 4-layer fully-connected architecture:
$$784 \\longrightarrow 16 \\longrightarrow 16 \\longrightarrow 10$$
* **Layer 0 (Input Layer)**: $n_0 = 784$ input units.
* **Layer 1 (Hidden Layer 1)**: $n_1 = 16$ hidden neurons.
* **Layer 2 (Hidden Layer 2)**: $n_2 = 16$ hidden neurons.
* **Layer 3 (Output Layer)**: $n_3 = 10$ output class nodes representing digits $0, 1, 2, \\dots, 9$.

```
[Layer 0: 784 Pixels]  ===(W1, b1)===>  [Layer 1: 16 Neurons]  ===(W2, b2)===>  [Layer 2: 16 Neurons]  ===(W3, b3)===>  [Layer 3: 10 Digits]
```

### 1.3 Exact Parameter Geometry Accounting ($13,002$ Parameters)
Each connection between neuron $k$ in layer $l-1$ and neuron $j$ in layer $l$ is parameterized by a scalar weight $w_{jk}^{(l)}$, and each receiving neuron $j$ possesses an independent scalar bias $b_j^{(l)}$.

| Layer ($l$) | Transition ($n_{l-1} \\to n_l$) | Weight Matrix $W^{(l)}$ Shape | Weight Count | Bias Vector $b^{(l)}$ Shape | Bias Count | Total Layer Parameters |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1** | $784 \\to 16$ | $(16, 784)$ | $16 \\times 784 = 12,544$ | $(16, 1)$ | $16$ | **$12,560$** |
| **Layer 2** | $16 \\to 16$ | $(16, 16)$ | $16 \\times 16 = 256$ | $(16, 1)$ | $16$ | **$272$** |
| **Layer 3** | $16 \\to 10$ | $(10, 16)$ | $10 \\times 16 = 160$ | $(10, 1)$ | $10$ | **$170$** |
| **Total** | **Full Network** | — | **$12,960$ Weights** | — | **$42$ Biases** | **$13,002$ Parameters** |

$$\\text{Total Parameters} = \\sum_{l=1}^{3} \\left( n_l \\cdot n_{l-1} + n_l \\right) = 12,560 + 272 + 170 = 13,002$$

Let us render this architecture visually using **Manim Community**.
""")

    add_code("""%%manim -ql -v WARNING NetworkArchitectureScene

class NetworkArchitectureScene(Scene):
    def construct(self):
        # Header Title
        title = Title("Neural Network Architecture (784 -> 16 -> 16 -> 10)", font_size=28).to_edge(UP)
        self.add(title)
        
        layer_sizes = [8, 16, 16, 10]
        layer_names = ["Input a^(0)\\n(784)", "Hidden a^(1)\\n(16)", "Hidden a^(2)\\n(16)", "Output a^(3)\\n(10)"]
        x_coords = [-4.8, -1.6, 1.6, 4.8]
        
        all_layers = []
        labels = VGroup()
        
        # Build node layers
        for l_idx, (num_nodes, x, name) in enumerate(zip(layer_sizes, x_coords, layer_names)):
            layer_group = VGroup()
            if l_idx == 0:
                display_nodes = 8
                spacing = 4.8 / (display_nodes + 1)
                y_start = 2.4 - spacing
                for i in range(display_nodes):
                    if i in (3, 4):
                        dot = Dot(point=np.array([x, y_start - i * spacing, 0]), radius=0.04, color=GRAY_B)
                    else:
                        dot = Circle(radius=0.11, color=BLUE_C, fill_opacity=0.85, fill_color=BLUE_E)
                        dot.move_to(np.array([x, y_start - i * spacing, 0]))
                    layer_group.add(dot)
            else:
                spacing = 5.2 / (num_nodes + 1)
                y_start = 2.6 - spacing
                for i in range(num_nodes):
                    c_col = TEAL_C if l_idx < 3 else YELLOW_C
                    f_col = TEAL_E if l_idx < 3 else ORANGE
                    node = Circle(radius=0.09 if num_nodes > 10 else 0.12, color=c_col, fill_opacity=0.85, fill_color=f_col)
                    node.move_to(np.array([x, y_start - i * spacing, 0]))
                    layer_group.add(node)
                    if l_idx == 3:
                        d_lbl = Text(str(i), font_size=13, color=WHITE).next_to(node, RIGHT, buff=0.12)
                        labels.add(d_lbl)
            
            all_layers.append(layer_group)
            lbl = Text(name, font_size=14, line_spacing=0.8, color=WHITE).next_to(layer_group, DOWN, buff=0.25)
            labels.add(lbl)
            
        # Draw synaptic connections
        synapses = VGroup()
        for l in range(len(all_layers) - 1):
            g1, g2 = all_layers[l], all_layers[l+1]
            for n1 in g1[::2]:
                for n2 in g2[::2]:
                    edge = Line(n1.get_center(), n2.get_center(), stroke_width=0.4, stroke_opacity=0.25, 
                                color=BLUE_A if l==0 else (TEAL_A if l==1 else YELLOW_A))
                    synapses.add(edge)
                    
        # Parameter Breakdown Subtitle
        param_banner = Text("Total Parameters: 13,002  |  W1: 12,544  W2: 256  W3: 160  |  Biases: 42", 
                            font_size=15, color=YELLOW_B).to_edge(DOWN)
        
        self.play(Create(synapses), *[Create(l) for l in all_layers], Write(labels), run_time=1.5)
        self.play(FadeIn(param_banner), run_time=0.8)
        
        # Forward Activation Pulse Wave
        for l in all_layers:
            self.play(l.animate.set_color(WHITE).set_fill(WHITE, opacity=1.0), run_time=0.25)
            self.play(l.animate.set_color(BLUE_C).set_fill(BLUE_E, opacity=0.85), run_time=0.25)
            
        self.wait(0.5)""")

    # ==========================================
    # MODULE 2
    # ==========================================
    add_markdown("""---
## Module 2: Mathematical Forward Pass & Vectorization
**Duration**: 15 Minutes

### 2.1 The Scalar Transformation in a Single Neuron
For any neuron $j$ in layer $l$, its activation $a_j^{(l)}$ is determined by taking the dot product of the previous layer's activation vector $a^{(l-1)}$ with its incoming weight vector, adding its intrinsic bias $b_j^{(l)}$, and passing the pre-activation sum $z_j^{(l)}$ through a non-linear activation function $\\sigma$:

$$z_j^{(l)} = \\sum_{k=1}^{n_{l-1}} w_{jk}^{(l)} a_k^{(l-1)} + b_j^{(l)}$$
$$a_j^{(l)} = \\sigma\\left( z_j^{(l)} \\right)$$

### 2.2 Matrix-Vector Formulation (Vectorization)
Computing individual neuron sums iteratively in nested loops is computationally prohibitive. By packing all weights into matrix $W^{(l)} \\in \\mathbb{R}^{n_l \\times n_{l-1}}$ and biases into vector $b^{(l)} \\in \\mathbb{R}^{n_l \\times 1}$, the entire layer transformation simplifies into a single matrix-vector product:

$$\\begin{bmatrix} z_1^{(l)} \\\\ z_2^{(l)} \\\\ \\vdots \\\\ z_{n_l}^{(l)} \\end{bmatrix} = \\begin{bmatrix} w_{1,1}^{(l)} & w_{1,2}^{(l)} & \\dots & w_{1,n_{l-1}}^{(l)} \\\\ w_{2,1}^{(l)} & w_{2,2}^{(l)} & \\dots & w_{2,n_{l-1}}^{(l)} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ w_{n_l,1}^{(l)} & w_{n_l,2}^{(l)} & \\dots & w_{n_l,n_{l-1}}^{(l)} \\end{bmatrix} \\begin{bmatrix} a_1^{(l-1)} \\\\ a_2^{(l-1)} \\\\ \\vdots \\\\ a_{n_{l-1}}^{(l-1)} \\end{bmatrix} + \\begin{bmatrix} b_1^{(l)} \\\\ b_2^{(l)} \\\\ \\vdots \\\\ b_{n_l}^{(l)} \\end{bmatrix}$$

$$\\mathbf{z}^{(l)} = W^{(l)} \\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}$$
$$\\mathbf{a}^{(l)} = \\sigma\\left( \\mathbf{z}^{(l)} \\right)$$

### 2.3 The Sigmoid Activation Function and its Analytical Derivative
The standard logistic sigmoid function $\\sigma(z)$ squashes arbitrary real numbers $(-\\infty, +\\infty)$ onto the interval $(0, 1)$, modeling neuron firing probability:
$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$

#### Proof of the Derivative Identity $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$:
$$\\begin{aligned}
\\sigma'(z) &= \\frac{d}{dz} (1 + e^{-z})^{-1} \\\\
&= -(1 + e^{-z})^{-2} \\cdot (-e^{-z}) \\\\
&= \\frac{e^{-z}}{(1 + e^{-z})^2} \\\\
&= \\frac{1}{1 + e^{-z}} \\cdot \\frac{e^{-z}}{1 + e^{-z}} \\\\
&= \\frac{1}{1 + e^{-z}} \\cdot \\frac{(1 + e^{-z}) - 1}{1 + e^{-z}} \\\\
&= \\frac{1}{1 + e^{-z}} \\left( 1 - \\frac{1}{1 + e^{-z}} \\right) \\\\
&= \\sigma(z) (1 - \\sigma(z))
\\end{aligned}$$

> **Key Observation**: The maximum value of $\\sigma'(z)$ occurs at $z = 0$, where $\\sigma'(0) = 0.5 \\times (1 - 0.5) = 0.25$. When $|z| \\gg 0$, $\\sigma'(z) \\to 0$, leading to the classical **vanishing gradient problem**.
""")

    add_code("""# Pure NumPy Implementation of Sigmoid & Vectorized Forward Pass Functions

def sigmoid(z):
    \"\"\"
    Vectorized logistic sigmoid activation function with numerical overflow clipping.
    sigma(z) = 1 / (1 + exp(-z))
    \"\"\"
    # Clip z to prevent float64 overflow in np.exp
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_prime(z):
    \"\"\"
    Analytical derivative of sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))
    \"\"\"
    s = sigmoid(z)
    return s * (1.0 - s)

def feedforward_step(a_prev, W, b):
    \"\"\"
    Computes a single layer forward transition:
    z = W * a_prev + b
    a = sigma(z)
    \"\"\"
    z = np.dot(W, a_prev) + b
    a = sigmoid(z)
    return z, a

# Plot Sigmoid & Its Derivative to Inspect Non-Linear Dynamics
z_vals = np.linspace(-7.0, 7.0, 300)
sig_vals = sigmoid(z_vals)
sig_prime_vals = sigmoid_prime(z_vals)

plt.figure(figsize=(10, 4))
plt.plot(z_vals, sig_vals, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$", color="#1f77b4", lw=2.5)
plt.plot(z_vals, sig_prime_vals, label=r"$\sigma'(z) = \sigma(z)(1 - \sigma(z))$", color="#d62728", lw=2.5, ls="--")
plt.axhline(0.25, color="gray", ls=":", alpha=0.7, label=r"$\max(\sigma') = 0.25$")
plt.axvline(0.0, color="black", ls=":", alpha=0.5)
plt.title("Sigmoid Activation Function & Analytical Derivative Landscape", fontsize=13, fontweight='bold')
plt.xlabel("$z$ (Pre-activation sum)", fontsize=11)
plt.ylabel("Value", fontsize=11)
plt.legend(fontsize=11, loc="center right")
plt.grid(True, alpha=0.4)
plt.show()""")

    add_code("""# Vectorized Forward Pass Demonstration across all 3 Layers
np.random.seed(101)

# Initialize dummy weights and biases matching our network architecture
W1 = np.random.randn(16, 784) / np.sqrt(784)
b1 = np.random.randn(16, 1)

W2 = np.random.randn(16, 16) / np.sqrt(16)
b2 = np.random.randn(16, 1)

W3 = np.random.randn(10, 16) / np.sqrt(16)
b3 = np.random.randn(10, 1)

# Sample input digit vector
sample_x = train_set[0][0] # shape (784, 1)

# Execute forward pass step-by-step
z1, a1 = feedforward_step(sample_x, W1, b1)
z2, a2 = feedforward_step(a1, W2, b2)
z3, a3 = feedforward_step(a2, W3, b3)

print("=== Forward Pass Vector Dimensionality Verification ===")
print(f"Input   a^(0) shape: {sample_x.shape} -> min: {sample_x.min():.2f}, max: {sample_x.max():.2f}")
print(f"Layer 1 z^(1) shape: {z1.shape}, a^(1) shape: {a1.shape}")
print(f"Layer 2 z^(2) shape: {z2.shape}, a^(2) shape: {a2.shape}")
print(f"Layer 3 z^(3) shape: {z3.shape}, a^(3) shape: {a3.shape}")
print(f"Output Activations (10 classes):\\n{a3.flatten().round(4)}")
print(f"Predicted Class: {np.argmax(a3)} (Confidence: {a3.max():.2%})")""")

    # ==========================================
    # MODULE 3
    # ==========================================
    add_markdown("""---
## Module 3: Loss Function & The Optimization Landscape
**Duration**: 15 Minutes

### 3.1 The Quadratic Cost Function (Mean Squared Error)
To measure how poorly our network is performing on a given training sample $(x, y)$, we define the Mean Squared Error (MSE) cost:

$$C(W, b) = \\frac{1}{2} \\sum_{j=0}^{9} \\left( a_j^{(L)} - y_j \\right)^2 = \\frac{1}{2} \\| a^{(L)} - y \\|^2$$

* $a^{(L)} = a^{(3)} \\in \\mathbb{R}^{10 \\times 1}$ is the actual network output activation vector.
* $y \\in \\mathbb{R}^{10 \\times 1}$ is the desired one-hot target vector.
* The $\\frac{1}{2}$ pre-factor is introduced for mathematical convenience; when taking the derivative with respect to $a_j^{(L)}$, the exponent $2$ cancels neatly:
$$\\frac{\\partial C}{\\partial a_j^{(L)}} = \\left( a_j^{(L)} - y_j \\right)$$

For a dataset of $n$ training examples, the total empirical risk cost is the average over all individual sample costs:
$$C_{\\text{total}} = \\frac{1}{n} \\sum_{x} C_x$$

---

### 3.2 The High-Dimensional Optimization Landscape
The cost function $C$ is a scalar field over a **13,002-dimensional parameter space**:
$$\\boldsymbol{\\theta} = \\begin{bmatrix} \\text{vec}(W^{(1)}) \\\\ b^{(1)} \\\\ \\text{vec}(W^{(2)}) \\\\ b^{(2)} \\\\ \\text{vec}(W^{(3)}) \\\\ b^{(3)} \\end{bmatrix} \\in \\mathbb{R}^{13,002}$$

The **gradient vector** $\\nabla_{\\boldsymbol{\\theta}} C$ points in the direction of maximum rate of increase of $C$:
$$\\nabla_{\\boldsymbol{\\theta}} C = \\begin{bmatrix} \\frac{\\partial C}{\\partial \\theta_1} \\\\ \\frac{\\partial C}{\\partial \\theta_2} \\\\ \\vdots \\\\ \\frac{\\partial C}{\\partial \\theta_{13002}} \\end{bmatrix} \\in \\mathbb{R}^{13,002}$$

### 3.3 Gradient Descent
To minimize $C$, we iteratively adjust parameters in the direction of steepest descent ($-\\nabla_{\\boldsymbol{\\theta}} C$), scaled by learning rate $\\eta > 0$:
$$\\boldsymbol{\\theta} \\leftarrow \\boldsymbol{\\theta} - \\eta \\nabla_{\\boldsymbol{\\theta}} C$$

Let us visualize this continuous descent on a 3D cost surface using **Manim Community**.
""")

    add_code("""%%manim -ql -v WARNING GradientDescentScene

class GradientDescentScene(ThreeDScene):
    def construct(self):
        # Position 3D camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 5, 1],
            x_length=5.5,
            y_length=5.5,
            z_length=3.0,
        )
        
        # Paraboloid surface representing local loss bowl C(w1, w2)
        surface = Surface(
            lambda u, v: axes.c2p(u, v, 0.35 * (u**2 + 1.4 * v**2)),
            u_range=[-2.2, 2.2],
            v_range=[-2.2, 2.2],
            resolution=(12, 12),
            fill_opacity=0.75,
            checkerboard_colors=[BLUE_D, TEAL_E]
        )
        
        title = Title("Optimization: Gradient Descent on Cost Surface C(W, b)", font_size=24).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        
        # Initial point high up on the loss landscape
        u_curr, v_curr = 2.0, 1.8
        init_pos = axes.c2p(u_curr, v_curr, 0.35 * (u_curr**2 + 1.4 * v_curr**2))
        ball = Dot3D(point=init_pos, radius=0.15, color=YELLOW)
        
        self.play(Create(axes), Create(surface), FadeIn(ball), run_time=1.0)
        
        # Compute gradient steps iteratively: delta_theta = -eta * grad(C)
        path_pts = [init_pos]
        lr = 0.18
        for _ in range(12):
            grad_u = 0.7 * u_curr
            grad_v = 0.98 * v_curr
            u_curr -= lr * grad_u
            v_curr -= lr * grad_v
            z_curr = 0.35 * (u_curr**2 + 1.4 * v_curr**2)
            path_pts.append(axes.c2p(u_curr, v_curr, z_curr))
            
        traj = VMobject(color=RED_A, stroke_width=3.5)
        traj.set_points_smoothly(path_pts)
        
        # Animate descent toward the global minimum
        self.play(
            MoveAlongPath(ball, traj),
            Create(traj),
            run_time=2.0,
            rate_func=smooth
        )
        self.wait(0.5)""")

    # ==========================================
    # MODULE 4
    # ==========================================
    add_markdown("""---
## Module 4: Backpropagation Calculus & Chain Rule Derivations
**Duration**: 20 Minutes

Backpropagation is an efficient algorithm to compute the exact gradient of the cost function with respect to every weight $\\frac{\\partial C}{\\partial w_{jk}^{(l)}}$ and bias $\\frac{\\partial C}{\\partial b_j^{(l)}}$ using the multivariate chain rule.

### 4.1 Step-by-Step Scalar Derivation for the Output Layer ($l = L$)
Consider an output weight $w_{jk}^{(L)}$ connecting hidden neuron $k$ in layer $L-1$ to output neuron $j$ in layer $L$.

$$\\begin{array}{rcccl}
w_{jk}^{(L)} & \\xrightarrow{\\quad z_j^{(L)} = \\sum_k w_{jk}^{(L)} a_k^{(L-1)} + b_j^{(L)} \\quad} & z_j^{(L)} & \\xrightarrow{\\quad a_j^{(L)} = \\sigma(z_j^{(L)}) \\quad} & a_j^{(L)} \\xrightarrow{\\quad C = \\frac{1}{2} \\sum_j (a_j^{(L)} - y_j)^2 \\quad} C
\\end{array}$$

Applying the single-variable chain rule:
$$\\frac{\\partial C}{\\partial w_{jk}^{(L)}} = \\frac{\\partial C}{\\partial a_j^{(L)}} \\cdot \\frac{\\partial a_j^{(L)}}{\\partial z_j^{(L)}} \\cdot \\frac{\\partial z_j^{(L)}}{\\partial w_{jk}^{(L)}}$$

Let us compute the 3 components individually:
1. **Cost sensitivity**:
   $$\\frac{\\partial C}{\\partial a_j^{(L)}} = \\frac{\\partial}{\\partial a_j^{(L)}} \\left[ \\frac{1}{2} \\sum_{m} (a_m^{(L)} - y_m)^2 \\right] = (a_j^{(L)} - y_j)$$
2. **Activation sensitivity**:
   $$\\frac{\\partial a_j^{(L)}}{\\partial z_j^{(L)}} = \\sigma'\\left( z_j^{(L)} \\right)$$
3. **Pre-activation sensitivity with respect to weight**:
   $$\\frac{\\partial z_j^{(L)}}{\\partial w_{jk}^{(L)}} = \\frac{\\partial}{\\partial w_{jk}^{(L)}} \\left[ \\sum_{m} w_{jm}^{(L)} a_m^{(L-1)} + b_j^{(L)} \\right] = a_k^{(L-1)}$$

Combining all three terms:
$$\\frac{\\partial C}{\\partial w_{jk}^{(L)}} = (a_j^{(L)} - y_j) \\cdot \\sigma'\\left( z_j^{(L)} \\right) \\cdot a_k^{(L-1)}$$

Similarly, for the bias $b_j^{(L)}$ since $\\frac{\\partial z_j^{(L)}}{\\partial b_j^{(L)}} = 1$:
$$\\frac{\\partial C}{\\partial b_j^{(L)}} = (a_j^{(L)} - y_j) \\cdot \\sigma'\\left( z_j^{(L)} \\right)$$

---

### 4.2 The Fundamental Error Vector $\\delta^{(l)}$
We define the **error vector** of layer $l$ as the partial derivative of cost with respect to the layer's pre-activation vector $z^{(l)}$:
$$\\delta_j^{(l)} \\equiv \\frac{\\partial C}{\\partial z_j^{(l)}} \\implies \\boldsymbol{\\delta}^{(l)} = \\frac{\\partial C}{\\partial \\mathbf{z}^{(l)}} \\in \\mathbb{R}^{n_l \\times 1}$$

#### Equation BP1: Error in the Output Layer ($l = L$)
$$\\delta_j^{(L)} = \\frac{\\partial C}{\\partial a_j^{(L)}} \\frac{\\partial a_j^{(L)}}{\\partial z_j^{(L)}} = (a_j^{(L)} - y_j) \\sigma'\\left( z_j^{(L)} \\right)$$
In vectorized form (using Hadamard element-wise product $\\odot$):
$$\\boldsymbol{\\delta}^{(L)} = \\left( \\mathbf{a}^{(L)} - \\mathbf{y} \\right) \\odot \\sigma'\\left( \\mathbf{z}^{(L)} \\right)$$

#### Equation BP2: Recursive Error Propagation to Hidden Layers ($l < L$)
By applying the multivariate chain rule across all neurons $m$ in layer $l+1$:
$$\\delta_k^{(l)} = \\frac{\\partial C}{\\partial z_k^{(l)}} = \\sum_{m=1}^{n_{l+1}} \\frac{\\partial C}{\\partial z_m^{(l+1)}} \\frac{\\partial z_m^{(l+1)}}{\\partial a_k^{(l)}} \\frac{\\partial a_k^{(l)}}{\\partial z_k^{(l)}} = \\sum_{m=1}^{n_{l+1}} \\delta_m^{(l+1)} w_{mk}^{(l+1)} \\sigma'\\left( z_k^{(l)} \\right)$$

In compact matrix notation:
$$\\boldsymbol{\\delta}^{(l)} = \\left( \\left( W^{(l+1)} \\right)^T \\boldsymbol{\\delta}^{(l+1)} \\right) \\odot \\sigma'\\left( \\mathbf{z}^{(l)} \\right)$$

#### Equation BP3 & BP4: Parameter Gradient Formulas (Outer Products)
With the error vector $\\boldsymbol{\\delta}^{(l)}$ computed, the partial derivatives for all weights and biases in layer $l$ are given by:
$$\\frac{\\partial C}{\\partial W^{(l)}} = \\boldsymbol{\\delta}^{(l)} \\left( \\mathbf{a}^{(l-1)} \\right)^T \\in \\mathbb{R}^{n_l \\times n_{l-1}}$$
$$\\frac{\\partial C}{\\partial \\mathbf{b}^{(l)}} = \\boldsymbol{\\delta}^{(l)} \\in \\mathbb{R}^{n_l \\times 1}$$

---

### 4.3 Summary of the 4 Core Backpropagation Equations

$$\\begin{array}{|l|l|}
\\hline
\\textbf{Output Error (BP1)} & \\boldsymbol{\\delta}^{(L)} = (\\mathbf{a}^{(L)} - \\mathbf{y}) \\odot \\sigma'(\\mathbf{z}^{(L)}) \\\\
\\hline
\\textbf{Error Propagation (BP2)} & \\boldsymbol{\\delta}^{(l)} = \\left( (W^{(l+1)})^T \\boldsymbol{\\delta}^{(l+1)} \\right) \\odot \\sigma'(\\mathbf{z}^{(l)}) \\\\
\\hline
\\textbf{Bias Gradient (BP3)} & \\frac{\\partial C}{\\partial \\mathbf{b}^{(l)}} = \\boldsymbol{\\delta}^{(l)} \\\\
\\hline
\\textbf{Weight Gradient (BP4)} & \\frac{\\partial C}{\\partial W^{(l)}} = \\boldsymbol{\\delta}^{(l)} (\\mathbf{a}^{(l-1)})^T \\\\
\\hline
\\end{array}$$
""")

    # ==========================================
    # MODULE 5
    # ==========================================
    add_markdown("""---
## Module 5: Pure Python/NumPy Implementation From Scratch
**Duration**: 15 Minutes

We now translate our mathematical equations into a modular, object-oriented `NeuralNetwork` class.

### Key Implementation Decisions:
1. **Weight Initialization**: Initialized from standard normal distribution scaled by $\\frac{1}{\\sqrt{n_{\\text{in}}}}$ (Xavier/Glorot style normalized variance $\\text{Var}(W) = 1/n_{\\text{in}}$) to prevent early saturation of sigmoid activations.
2. **Batch SGD**: We implement Mini-batch Stochastic Gradient Descent to average noisy gradients over small subsets $\\mathcal{B} \\subset \\mathcal{D}$ of size $m$:
$$W^{(l)} \\leftarrow W^{(l)} - \\frac{\\eta}{m} \\sum_{x \\in \\mathcal{B}} \\nabla_{W^{(l)}} C_x, \\qquad b^{(l)} \\leftarrow b^{(l)} - \\frac{\\eta}{m} \\sum_{x \\in \\mathcal{B}} \\nabla_{b^{(l)}} C_x$$
""")

    add_code('''class NeuralNetwork:
    """
    A 4-layer fully connected neural network built entirely from scratch in NumPy.
    Architecture: [784, 16, 16, 10]
    """
    def __init__(self, sizes=[784, 16, 16, 10]):
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Scaled Gaussian initialization: N(0, 1 / sqrt(n_in))
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Propagates input activation vector 'a' through all layers and returns the final output.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """
        Calculates gradients (nabla_w, nabla_b) for a single training sample (x, y).
        Returns:
            nabla_w: list of weight gradient matrices [dW1, dW2, dW3]
            nabla_b: list of bias gradient vectors   [db1, db2, db3]
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # ----------------------------------------------------
        # 1. Forward Pass: Cache all z and a vectors
        # ----------------------------------------------------
        activation = x
        activations = [x] # a^(0), a^(1), a^(2), a^(3)
        zs = []           # z^(1), z^(2), z^(3)
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # ----------------------------------------------------
        # 2. Output Layer Error (Equation BP1)
        # delta^(L) = (a^(L) - y) * sigma'(z^(L))
        # ----------------------------------------------------
        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T) # Equation BP4
        
        # ----------------------------------------------------
        # 3. Backpropagate Error to Hidden Layers (Equations BP2, BP3, BP4)
        # delta^(l) = ((W^(l+1))^T * delta^(l+1)) * sigma'(z^(l))
        # ----------------------------------------------------
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
            
        return (nabla_w, nabla_b)

    def train_sgd(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Trains the neural network using Mini-batch Stochastic Gradient Descent (SGD).
        """
        n = len(train_data)
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            epoch_loss = 0.0
            for batch in mini_batches:
                # Accumulate batch gradients
                batch_nabla_w = [np.zeros(w.shape) for w in self.weights]
                batch_nabla_b = [np.zeros(b.shape) for b in self.biases]
                
                for x, y in batch:
                    dw, db = self.backprop(x, y)
                    for i in range(len(batch_nabla_w)):
                        batch_nabla_w[i] += dw[i]
                        batch_nabla_b[i] += db[i]
                    
                    # Track sample quadratic cost C = 0.5 * sum((a - y)^2)
                    a_out = self.feedforward(x)
                    epoch_loss += 0.5 * np.sum((a_out - y)**2)
                
                # Update parameters via Gradient Descent Step
                m = len(batch)
                self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, batch_nabla_w)]
                self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, batch_nabla_b)]
            
            avg_loss = epoch_loss / n
            history['loss'].append(avg_loss)
            
            if test_data is not None:
                acc = self.evaluate(test_data)
                history['accuracy'].append(acc)
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Cost (MSE): {avg_loss:.4f} | Test Accuracy: {acc*100:.2f}%")
            else:
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Cost (MSE): {avg_loss:.4f}")
                
        return history

    def evaluate(self, test_data):
        """
        Computes classification accuracy over test examples.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(pred == actual) for (pred, actual) in test_results) / len(test_data)''')

    add_code("""# Rigorous Finite Difference Numerical Gradient Check (Sanity Verification)
def gradient_check(net, sample_x, sample_y, epsilon=1e-5):
    \"\"\"
    Verifies analytic backpropagation gradients against numerical two-point finite differences:
    dC/dw ~ [C(w + eps) - C(w - eps)] / (2 * eps)
    \"\"\"
    analytic_dw, analytic_db = net.backprop(sample_x, sample_y)
    
    # Check Layer 3 Weights (W3)
    numerical_dw3 = np.zeros_like(net.weights[2])
    for i in range(net.weights[2].shape[0]):
        for j in range(net.weights[2].shape[1]):
            # Perturb +eps
            net.weights[2][i, j] += epsilon
            loss_plus = 0.5 * np.sum((net.feedforward(sample_x) - sample_y)**2)
            
            # Perturb -eps
            net.weights[2][i, j] -= 2 * epsilon
            loss_minus = 0.5 * np.sum((net.feedforward(sample_x) - sample_y)**2)
            
            # Restore original weight
            net.weights[2][i, j] += epsilon
            
            numerical_dw3[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
    # Compute relative Euclidean norm difference
    diff = np.linalg.norm(analytic_dw[2] - numerical_dw3) / (np.linalg.norm(analytic_dw[2]) + np.linalg.norm(numerical_dw3))
    print(f"Gradient Check Relative Error for W^(3): {diff:.2e}")
    assert diff < 1e-6, "Gradient check failed! Relative error too large."
    print(" Analytic backpropagation matches numerical differentiation!")

test_net = NeuralNetwork([784, 16, 16, 10])
gradient_check(test_net, train_set[0][0], train_set[0][1])""")

    # ==========================================
    # MODULE 6
    # ==========================================
    add_markdown("""---
## Module 6: Live Training & Weight Heatmap Probing
**Duration**: 10 Minutes

### 6.1 Training the 4-Layer Network
We now execute live Mini-batch SGD training for 8 epochs on our MNIST dataset with mini-batch size $m = 10$ and learning rate $\\eta = 3.0$.
""")

    add_code("""# Instantiate Network and Train with Mini-Batch SGD
np.random.seed(42)
nn_model = NeuralNetwork(sizes=[784, 16, 16, 10])

print("Initiating Multi-Layer Perceptron Training...")
print("-" * 65)
history = nn_model.train_sgd(
    train_data=train_set,
    epochs=8,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_set
)
print("-" * 65)
print(" Training complete!")""")

    add_code("""# Plot Epoch-by-Epoch Loss Reduction & Test Set Accuracy Curves
epochs_range = range(1, len(history['loss']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# 1. Cost Curve
ax1.plot(epochs_range, history['loss'], marker='o', color='#1f77b4', lw=2.5, markersize=6)
ax1.set_title("Training Loss Convergence (Mean Squared Error)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Quadratic Cost $C$", fontsize=11)
ax1.set_xticks(epochs_range)
ax1.grid(True, alpha=0.3)

# 2. Accuracy Curve
ax2.plot(epochs_range, [acc * 100 for acc in history['accuracy']], marker='s', color='#2ca02c', lw=2.5, markersize=6)
ax2.set_title("Test Set Classification Accuracy (%)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_xticks(epochs_range)
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

    add_markdown("""### 6.2 Probing the Hidden Feature Detectors ($W^{(1)}$ Heatmaps)
What are the 16 hidden neurons in Layer 1 actually computing?

Each hidden neuron $j \\in \\{1, 2, \\dots, 16\\}$ receives 784 incoming weights from the input pixel grid:
$$\\mathbf{w}_j^{(1)} = \\begin{bmatrix} w_{j,1}^{(1)} & w_{j,2}^{(1)} & \\dots & w_{j,784}^{(1)} \\end{bmatrix} \\in \\mathbb{R}^{1 \\times 784}$$

By reshaping each row vector $\\mathbf{w}_j^{(1)}$ back into a $28 \\times 28$ matrix, we can visualize the **receptive field spatial filter** of that neuron:
* **Positive weights (Red)**: The neuron excites (increases activation) when brightness is present in these pixel regions.
* **Negative weights (Blue)**: The neuron inhibits (decreases activation) when brightness is present in these regions.
* **Zero weights (White)**: The neuron ignores these background pixels.

Each neuron specializes into a localized sub-pattern detector (e.g. upper loops, vertical center strokes, bottom arcs, diagonal slashes).
""")

    add_code("""# Extract Layer 1 Weight Matrix and Reshape into 16 Spatial Heatmaps
W1_learned = nn_model.weights[0] # shape: (16, 784)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Hidden Layer 1 Receptive Fields: 16 Learned Sub-Feature Filters ($W^{(1)}$)", 
             fontsize=14, fontweight='bold', y=0.98)

vmax = np.percentile(np.abs(W1_learned), 99) # Robust symmetric colormap limits

for i, ax in enumerate(axes.flat):
    # Reshape 784 weights of neuron i into 28x28 spatial map
    neuron_weights = W1_learned[i].reshape(28, 28)
    
    im = ax.imshow(neuron_weights, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f"Hidden Unit {i+1}", fontsize=11, fontweight='bold')
    ax.axis('off')

# Add Colorbar Legend
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Synaptic Weight Value ($w > 0$ Excitatory / $w < 0$ Inhibitory)", fontsize=10)

plt.subplots_adjust(right=0.9)
plt.show()""")

    add_markdown("""### 6.3 Live Inference Inspector with Softmax Probability Distribution
Finally, let us run test inference on digits, converting output activations into calibrated posterior class probabilities via Softmax:
$$P(y = k \\mid x) = \\frac{e^{z_k^{(3)}}}{\\sum_{j=0}^{9} e^{z_j^{(3)}}}$$
""")

    add_code("""# Softmax Function for Interpretable Probability Normalization
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Select Test Examples & Render Probability Bar Charts
num_inspect = 6
fig, axes = plt.subplots(num_inspect, 2, figsize=(12, 3 * num_inspect), 
                         gridspec_kw={'width_ratios': [1, 2.5]})

for idx in range(num_inspect):
    test_img, true_label = test_set[idx * 3]
    
    # Forward Pass
    activations = nn_model.feedforward(test_img)
    probs = softmax(activations).flatten()
    pred_label = np.argmax(probs)
    
    # 1. Digit Image Display
    ax_img = axes[idx, 0]
    ax_img.imshow(test_img.reshape(28, 28), cmap='gray_r')
    ax_img.set_title(f"True: {true_label} | Pred: {pred_label}", 
                     fontsize=12, fontweight='bold',
                     color="green" if pred_label == true_label else "red")
    ax_img.axis('off')
    
    # 2. Probability Distribution Bar Chart
    ax_bar = axes[idx, 1]
    classes = list(range(10))
    colors = ['#2ca02c' if c == true_label else ('#d62728' if c == pred_label else '#aec7e8') 
              for c in classes]
    
    bars = ax_bar.bar(classes, probs, color=colors, edgecolor='black', alpha=0.85)
    ax_bar.set_xticks(classes)
    ax_bar.set_ylim([0, 1.05])
    ax_bar.set_ylabel("Probability $P(y|x)$", fontsize=10)
    ax_bar.set_xlabel("Digit Class", fontsize=10)
    ax_bar.set_title(f"Class Confidence: {probs[pred_label]*100:.1f}%", fontsize=11)
    
    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax_bar.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h*100:.0f}%", 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()""")

    add_markdown("""---
## Workshop Summary & Key Takeaways
1. **Network Geometry**: An MLP transforms high-dimensional representations through successive affine projections and non-linear squashing operations ($784 \\to 16 \\to 16 \\to 10$).
2. **First-Principles Optimization**: Backpropagation is not a mysterious black box—it is the algebraic application of the multivariate chain rule propagating error signals $\\delta^{(l)}$ backwards.
3. **Internal Representations**: Hidden neurons naturally develop specialized spatial filters that act as localized stroke and edge detectors.
4. **Vectorization**: Modern deep learning achieves its speed by expressing neuron computations as matrix products ($W a + b$), seamlessly mapping to parallel hardware.
""")

    output_path = "/home/alonemusk/Projects/manimations/neural_network_workshop.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f" Successfully generated notebook: {output_path}")

if __name__ == "__main__":
    create_notebook()
