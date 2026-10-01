import json
import os

def create_nn1_notebook():
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
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # -------------------------------------------------------------
    # 1. Header & Introduction
    # -------------------------------------------------------------
    add_markdown("""# Interactive Neural Network Digit Recognizer
### Live Drawing & Real-Time Activation Probing ($784 \\rightarrow 16 \\rightarrow 16 \\rightarrow 10$)

---

## 📌 Overview & Learning Objectives
In this notebook, we explore how a trained **Multi-Layer Perceptron (MLP)** perceives, transforms, and classifies handwritten digits in real time. 

Using **pretrained weights and biases** loaded from the `nn/` directory, we:
1. **Dissect the 4-Layer Architecture** ($784 \\rightarrow 16 \\rightarrow 16 \\rightarrow 10$) across all $13,002$ parameters.
2. **Visualize Hidden Layer Feature Detectors** ($W^{(1)}$ receptive fields) to understand how individual neurons detect localized edges, loops, and strokes.
3. **Build an Interactive Matplotlib 28x28 Drawing Canvas** that captures user mouse strokes and dynamically updates the 10 output class activation bars in real time.
4. **Deploy an Embedded Zero-Latency HTML5/JS Neural Canvas** directly inside the notebook to visualize the full propagation chain ($a^{(0)} \\to a^{(1)} \\to a^{(2)} \\to a^{(3)}$) while drawing.

---

### Mathematical Model Formulation
For an input image $a^{(0)} \\in [0, 1]^{784 \\times 1}$, activations propagate through 3 successive affine transformations and non-linear squashing functions:

$$\\begin{aligned}
\\mathbf{z}^{(1)} &= W^{(1)} \\mathbf{a}^{(0)} + \\mathbf{b}^{(1)} \\in \\mathbb{R}^{16 \\times 1}, & \\mathbf{a}^{(1)} &= \\sigma(\\mathbf{z}^{(1)}) \\\\
\\mathbf{z}^{(2)} &= W^{(2)} \\mathbf{a}^{(1)} + \\mathbf{b}^{(2)} \\in \\mathbb{R}^{16 \\times 1}, & \\mathbf{a}^{(2)} &= \\sigma(\\mathbf{z}^{(2)}) \\\\
\\mathbf{z}^{(3)} &= W^{(3)} \\mathbf{a}^{(2)} + \\mathbf{b}^{(3)} \\in \\mathbb{R}^{10 \\times 1}, & \\mathbf{a}^{(3)} &= \\sigma(\\mathbf{z}^{(3)})
\\end{aligned}$$

where $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ is the logistic sigmoid activation function.
""")

    # -------------------------------------------------------------
    # 2. Setup & Pretrained Weight Loading
    # -------------------------------------------------------------
    add_code("""# Setup Environment & Load Pretrained Model Weights
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from IPython.display import HTML, display

# Configure clean aesthetic visualization defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120

# Activation Functions
def sigmoid(z):
    \"\"\"Numerically stable sigmoid activation.\"\"\"
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

def softmax(z):
    \"\"\"Normalized softmax probability distribution.\"\"\"
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Load Pretrained Weights & Biases from the `nn/` directory
weights_file = os.path.join(os.getcwd(), 'nn', 'pretrained_weights_and_biases')
if not os.path.exists(weights_file):
    weights_file = os.path.join(os.path.dirname(os.getcwd()), 'nn', 'pretrained_weights_and_biases')

with open(weights_file, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    pretrained_weights, pretrained_biases = u.load()

print("=== Pretrained Neural Network Architecture ===")
for l, (w, b) in enumerate(zip(pretrained_weights, pretrained_biases), 1):
    print(f"Layer {l}: Weight Matrix W^({l}) Shape = {str(w.shape):12s} | Bias Vector b^({l}) Shape = {str(b.shape)}")

total_params = sum(w.size + b.size for w, b in zip(pretrained_weights, pretrained_biases))
print(f"\\nTotal Learnable Parameters: {total_params:,} (12,960 weights + 42 biases)")""")

    # -------------------------------------------------------------
    # 3. Forward Pass Engine & Benchmark
    # -------------------------------------------------------------
    add_markdown("""---
## 1. Forward Pass Engine & Verification on Benchmark Digits

Let us define a complete feedforward evaluator that extracts the activations across all layers:
$$[a^{(0)}, a^{(1)}, a^{(2)}, a^{(3)}]$$
We then test the network on real sample digits loaded from `nn/image_map`.
""")

    add_code("""# Feedforward Engine: Extracting Intermediate Activations
def get_layer_activations(x_input, weights=pretrained_weights, biases=pretrained_biases):
    \"\"\"
    Performs a complete forward pass and returns all layer activations:
    activations[0] = a^(0) [shape (784, 1)]
    activations[1] = a^(1) [shape (16, 1)]
    activations[2] = a^(2) [shape (16, 1)]
    activations[3] = a^(3) [shape (10, 1)]
    \"\"\"
    a = x_input.reshape(784, 1).astype(np.float64)
    activations = [a]
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = sigmoid(z)
        activations.append(a)
    return activations

def predict_digit(x_input):
    \"\"\"Returns predicted class, confidence, and output activation vector.\"\"\"
    activations = get_layer_activations(x_input)
    output_a = activations[-1].flatten()
    pred_class = int(np.argmax(output_a))
    confidence = float(output_a[pred_class])
    return pred_class, confidence, output_a

# Benchmark on sample digits from image_map
image_map_file = os.path.join(os.getcwd(), 'nn', 'image_map')
with open(image_map_file, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    image_map = u.load()

# Test network accuracy across all 100 benchmark digits (10 per digit class)
correct = 0
total = 0
for digit, samples in image_map.items():
    for img in samples:
        pred, _, _ = predict_digit(img)
        if pred == digit:
            correct += 1
        total += 1

print(f"Pretrained Model Accuracy on Benchmark MNIST Digits: {correct}/{total} ({correct/total*100:.1f}%)")""")

    add_code("""# Display Sample Benchmark Digits & Network Activation Bars
fig, axes = plt.subplots(5, 4, figsize=(14, 12))
fig.suptitle("Pretrained Network Recognition on Sample Benchmark Digits", fontsize=15, fontweight='bold', y=0.99)

for idx in range(10):
    digit_class = idx
    sample_img = image_map[digit_class][0] # take first sample for each digit
    pred_class, conf, output_activations = predict_digit(sample_img)
    
    # 1. Image Subplot
    row = (idx * 2) // 4
    col_img = (idx * 2) % 4
    ax_img = axes[row, col_img]
    ax_img.imshow(sample_img.reshape(28, 28), cmap='gray_r', interpolation='nearest')
    ax_img.set_title(f"Target: {digit_class} | Pred: {pred_class}", 
                     fontsize=11, fontweight='bold',
                     color="green" if pred_class == digit_class else "red")
    ax_img.axis('off')
    
    # 2. Activation Bar Chart Subplot
    col_bar = col_img + 1
    ax_bar = axes[row, col_bar]
    classes = np.arange(10)
    bar_colors = ['#2ca02c' if c == digit_class else ('#d62728' if c == pred_class else '#aec7e8') 
                  for c in classes]
    bars = ax_bar.bar(classes, output_activations, color=bar_colors, edgecolor='black', alpha=0.85)
    ax_bar.set_xticks(classes)
    ax_bar.set_ylim([0, 1.05])
    ax_bar.set_ylabel("Activation $a^{(3)}$", fontsize=9)
    ax_bar.set_title(f"Top Confidence: {conf:.1%}", fontsize=10)
    ax_bar.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()""")

    # -------------------------------------------------------------
    # 4. Hidden Feature Detectors
    # -------------------------------------------------------------
    add_markdown("""---
## 2. Probing Hidden Feature Detectors ($W^{(1)}$ Receptive Fields)

In Layer 1, each of the 16 hidden neurons computes a weighted sum over all 784 input pixels:
$$z_j^{(1)} = \\sum_{k=1}^{784} w_{jk}^{(1)} a_k^{(0)} + b_j^{(1)}$$

Reshaping the 784 weights of neuron $j$ into a $28 \\times 28$ image reveals the **spatial pattern** that excites ($w > 0$, Red) or inhibits ($w < 0$, Blue) that specific neuron.
""")

    add_code("""# Visualize Layer 1 Weight Heatmaps (Receptive Fields of 16 Hidden Neurons)
W1 = pretrained_weights[0] # Shape: (16, 784)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Hidden Layer 1 Receptive Fields: 16 Learned Sub-Stroke Detectors ($W^{(1)}$)", 
             fontsize=14, fontweight='bold', y=0.98)

vmax = np.percentile(np.abs(W1), 99) # Symmetric color bounds

for i, ax in enumerate(axes.flat):
    filter_img = W1[i].reshape(28, 28)
    im = ax.imshow(filter_img, cmap='coolwarm', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f"Neuron #{i+1} (Bias: {pretrained_biases[0][i, 0]:.2f})", fontsize=10, fontweight='bold')
    ax.axis('off')

# Colorbar Legend
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Synaptic Weight Value ($w > 0$ Excitatory / $w < 0$ Inhibitory)", fontsize=10)

plt.subplots_adjust(right=0.9)
plt.show()""")

    # -------------------------------------------------------------
    # 5. Interactive Matplotlib Drawing Canvas
    # -------------------------------------------------------------
    add_markdown("""---
## 3. Interactive Matplotlib 28x28 Drawing Canvas with Real-Time Activation Bars

The interactive application below connects mouse drag events to the $28 \\times 28$ pixel grid.
As soon as you start drawing with the mouse:
1. The canvas updates with a smooth anti-aliased Gaussian brush stroke.
2. The neural network computes the forward pass on the fly.
3. The 10 output activation bars and predicted class label update dynamically in real time.
4. The "Clear" button resets the canvas, and "Load Sample" tests preset digits.
""")

    add_code("""# Interactive Matplotlib Drawing & Recognition Application
class MatplotlibDigitDrawer:
    \"\"\"
    Interactive Matplotlib application allowing live drawing in a 28x28 grid
    with instantaneous updates to the 10 neural network activation bars.
    \"\"\"
    def __init__(self, weights=pretrained_weights, biases=pretrained_biases):
        self.weights = weights
        self.biases = biases
        self.grid = np.zeros((28, 28), dtype=np.float64)
        self.is_drawing = False
        self.brush_radius = 1.4
        
        # Create figure with 2 subplots: Drawing Canvas (Left) and Activation Bars (Right)
        self.fig = plt.figure(figsize=(12, 5))
        self.gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.4])
        
        self.ax_canvas = self.fig.add_subplot(self.gs[0, 0])
        self.ax_bars = self.fig.add_subplot(self.gs[0, 1])
        
        # 1. Drawing Canvas Display
        self.img_display = self.ax_canvas.imshow(
            self.grid, cmap='gray_r', vmin=0.0, vmax=1.0, interpolation='nearest'
        )
        self.ax_canvas.set_title("Draw a Digit Here (Click & Drag)", fontsize=12, fontweight='bold')
        self.ax_canvas.set_xticks(np.arange(0, 28, 5))
        self.ax_canvas.set_yticks(np.arange(0, 28, 5))
        self.ax_canvas.grid(True, color='gray', alpha=0.3, linestyle=':')
        
        # 2. Output Activation Bar Chart Display
        self.classes = np.arange(10)
        initial_acts = np.zeros(10)
        self.bars = self.ax_bars.bar(
            self.classes, initial_acts, color='#3498db', edgecolor='black', alpha=0.85
        )
        self.ax_bars.set_xticks(self.classes)
        self.ax_bars.set_ylim([0, 1.05])
        self.ax_bars.set_xlabel("Digit Class (0-9)", fontsize=11)
        self.ax_bars.set_ylabel("Output Layer Activation $a^{(3)}$", fontsize=11)
        self.title_text = self.ax_bars.set_title("Network Prediction: (Awaiting Input)", fontsize=12, fontweight='bold')
        self.ax_bars.grid(True, alpha=0.3, axis='y')
        
        # Connect mouse interaction events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Perform initial forward pass update
        self.update_prediction()
        self.fig.tight_layout()
        
    def apply_brush(self, center_x, center_y):
        \"\"\"Applies a soft Gaussian brush stroke around the mouse cursor.\"\"\"
        y_indices, x_indices = np.ogrid[:28, :28]
        dist_sq = (x_indices - center_x)**2 + (y_indices - center_y)**2
        # Gaussian intensity falloff
        stroke = np.exp(-dist_sq / (2.0 * (self.brush_radius**2)))
        self.grid = np.clip(self.grid + stroke * 0.9, 0.0, 1.0)
        
    def update_prediction(self):
        \"\"\"Runs forward pass and refreshes image and activation bars.\"\"\"
        # Forward pass
        acts = get_layer_activations(self.grid, self.weights, self.biases)
        output_a = acts[-1].flatten()
        pred = int(np.argmax(output_a))
        conf = float(output_a[pred])
        
        # Update image
        self.img_display.set_data(self.grid)
        
        # Update bar heights and colors
        for i, (bar, val) in enumerate(zip(self.bars, output_a)):
            bar.set_height(val)
            if i == pred and val > 0.1:
                bar.set_color('#2ecc71') # Green for top prediction
            else:
                bar.set_color('#3498db') # Standard blue
                
        if np.max(output_a) > 0.05:
            self.title_text.set_text(f"Predicted Digit: {pred}  |  Confidence: {conf:.1%}")
        else:
            self.title_text.set_text("Network Prediction: (Awaiting Input)")
            
        self.fig.canvas.draw_idle()
        
    def on_press(self, event):
        if event.inaxes == self.ax_canvas and event.button == 1:
            self.is_drawing = True
            self.apply_brush(event.xdata, event.ydata)
            self.update_prediction()
            
    def on_motion(self, event):
        if self.is_drawing and event.inaxes == self.ax_canvas:
            self.apply_brush(event.xdata, event.ydata)
            self.update_prediction()
            
    def on_release(self, event):
        self.is_drawing = False
        
    def clear(self):
        \"\"\"Clears the 28x28 grid.\"\"\"
        self.grid.fill(0.0)
        self.update_prediction()
        
    def load_preset(self, digit):
        \"\"\"Loads a sample benchmark digit into the drawing grid.\"\"\"
        if digit in image_map and len(image_map[digit]) > 0:
            self.grid = image_map[digit][0].reshape(28, 28).copy()
            self.update_prediction()

# Initialize Interactive Drawer
drawer = MatplotlibDigitDrawer()
plt.show()""")

    # -------------------------------------------------------------
    # 6. Embedded Zero-Latency HTML5/JS Live Drawing Canvas
    # -------------------------------------------------------------
    add_markdown("""---
## 4. Embedded Real-Time Neural Drawing Canvas (HTML5 + WebGL/JS)

For seamless interactive drawing directly within VS Code and web-based Jupyter environments with **60 FPS real-time responsiveness**, the cell below embeds an interactive HTML5 drawing widget. 

As you sketch on the $28 \\times 28$ pixel matrix:
* **All 3 Layers Compute Forward Passes Instantaneously** using the exact weights and biases loaded from `pretrained_weights_and_biases`.
* **Activations for Hidden Layer 1 ($16$ neurons), Hidden Layer 2 ($16$ neurons), and Output Class Bars ($10$ digits)** animate dynamically in real time.
""")

    add_code("""# Construct Self-Contained Interactive HTML5/JS Neural Drawing Widget
def create_interactive_js_widget(weights=pretrained_weights, biases=pretrained_biases):
    import json
    # Serialize weights and biases to JSON for client-side JS forward pass execution
    w1_json = json.dumps(weights[0].tolist())
    b1_json = json.dumps(biases[0].flatten().tolist())
    w2_json = json.dumps(weights[1].tolist())
    b2_json = json.dumps(biases[1].flatten().tolist())
    w3_json = json.dumps(weights[2].tolist())
    b3_json = json.dumps(biases[2].flatten().tolist())
    
    # Pre-encode sample digits for preset testing buttons
    presets = {str(d): image_map[d][0].flatten().round(3).tolist() for d in range(10)}
    presets_json = json.dumps(presets)

    html_code = f\"\"\"
    <div id="nn-app-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1e1e2e; color: #cdd6f4; padding: 24px; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); max-width: 950px; margin: 10px auto;">
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #313244; padding-bottom: 14px; margin-bottom: 20px;">
        <h2 style="margin: 0; color: #89b4fa; font-size: 20px; font-weight: 700;">Live Neural Network Digit Recognizer</h2>
        <span style="background: #a6e3a1; color: #11111b; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">784 -> 16 -> 16 -> 10</span>
      </div>

      <div style="display: grid; grid-template-columns: 300px 1fr; gap: 24px; align-items: start;">
        <!-- Left: Drawing Canvas & Controls -->
        <div style="display: flex; flex-direction: column; align-items: center; gap: 14px;">
          <div style="position: relative; border: 3px solid #89b4fa; border-radius: 12px; overflow: hidden; background: #000; box-shadow: 0 4px 16px rgba(137, 180, 250, 0.25);">
            <canvas id="paint-canvas" width="280" height="280" style="cursor: crosshair; display: block;"></canvas>
            <div id="canvas-hint" style="position: absolute; top: 120px; left: 0; width: 100%; text-align: center; color: #585b70; font-size: 14px; pointer-events: none;">Draw a digit (0-9) here</div>
          </div>
          
          <div style="display: flex; gap: 10px; width: 100%;">
            <button id="btn-clear" style="flex: 1; padding: 10px; background: #f38ba8; color: #11111b; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: 0.2s;">Clear Canvas</button>
            <button id="btn-smooth" style="flex: 1; padding: 10px; background: #cba6f7; color: #11111b; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: 0.2s;">Smooth Stroke</button>
          </div>

          <!-- Presets -->
          <div style="width: 100%; text-align: center;">
            <div style="font-size: 12px; color: #a6adc8; margin-bottom: 6px;">Test Benchmark Digits:</div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px;">
              {"".join([f'<button class="btn-preset" data-digit="{d}" style="padding: 6px; background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; font-weight: bold; cursor: pointer;">{d}</button>' for d in range(10)])}
            </div>
          </div>
        </div>

        <!-- Right: Neural Network Output & Activation Bars -->
        <div style="background: #181825; padding: 20px; border-radius: 12px; border: 1px solid #313244;">
          <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 16px;">
            <div style="font-size: 14px; color: #bac2de; font-weight: 600;">Predicted Output:</div>
            <div id="pred-badge" style="font-size: 26px; font-weight: 900; color: #a6e3a1;">-</div>
          </div>

          <!-- 10 Class Activation Bars -->
          <div id="bars-container" style="display: flex; flex-direction: column; gap: 8px;">
            {"".join([f'''
            <div style="display: flex; align-items: center; gap: 10px; font-size: 13px;">
              <span style="width: 16px; font-weight: bold; color: #cdd6f4;">{d}:</span>
              <div style="flex: 1; height: 16px; background: #313244; border-radius: 8px; overflow: hidden; position: relative;">
                <div id="bar-{d}" style="width: 0%; height: 100%; background: #89b4fa; transition: width 0.08s ease, background 0.2s;"></div>
              </div>
              <span id="val-{d}" style="width: 48px; text-align: right; font-family: monospace; color: #a6adc8;">0.00%</span>
            </div>
            ''' for d in range(10)])}
          </div>

          <!-- Intermediate Hidden Layer Firing Patterns -->
          <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #313244;">
            <div style="font-size: 12px; color: #bac2de; margin-bottom: 8px;">Hidden Layer Activations:</div>
            <div style="display: flex; gap: 16px; justify-content: space-around;">
              <div>
                <div style="font-size: 11px; color: #a6adc8; text-align: center; margin-bottom: 4px;">Layer 1 ($a^{(1)}$: 16 units)</div>
                <div id="h1-grid" style="display: grid; grid-template-columns: repeat(8, 12px); gap: 3px;">
                  {"".join([f'<div id="h1-dot-{i}" style="width: 12px; height: 12px; border-radius: 50%; background: #313244;"></div>' for i in range(16)])}
                </div>
              </div>
              <div>
                <div style="font-size: 11px; color: #a6adc8; text-align: center; margin-bottom: 4px;">Layer 2 ($a^{(2)}$: 16 units)</div>
                <div id="h2-grid" style="display: grid; grid-template-columns: repeat(8, 12px); gap: 3px;">
                  {"".join([f'<div id="h2-dot-{i}" style="width: 12px; height: 12px; border-radius: 50%; background: #313244;"></div>' for i in range(16)])}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const W1 = {w1_json};
      const b1 = {b1_json};
      const W2 = {w2_json};
      const b2 = {b2_json};
      const W3 = {w3_json};
      const b3 = {b3_json};
      const presets = {presets_json};

      const canvas = document.getElementById('paint-canvas');
      const ctx = canvas.getContext('2d');
      const hint = document.getElementById('canvas-hint');
      let isDrawing = false;
      let grid = new Float64Array(784);

      function sigmoid(z) {{
        return 1.0 / (1.0 + Math.exp(-Math.max(-500, Math.min(500, z))));
      }}

      function forwardPass(input784) {{
        // Layer 1
        const a1 = new Float64Array(16);
        for (let i = 0; i < 16; i++) {{
          let sum = b1[i];
          const wRow = W1[i];
          for (let j = 0; j < 784; j++) {{
            sum += wRow[j] * input784[j];
          }}
          a1[i] = sigmoid(sum);
        }}

        // Layer 2
        const a2 = new Float64Array(16);
        for (let i = 0; i < 16; i++) {{
          let sum = b2[i];
          const wRow = W2[i];
          for (let j = 0; j < 16; j++) {{
            sum += wRow[j] * a1[j];
          }}
          a2[i] = sigmoid(sum);
        }}

        // Layer 3 (Output)
        const a3 = new Float64Array(10);
        for (let i = 0; i < 10; i++) {{
          let sum = b3[i];
          const wRow = W3[i];
          for (let j = 0; j < 16; j++) {{
            sum += wRow[j] * a2[j];
          }}
          a3[i] = sigmoid(sum);
        }}

        return {{ a1, a2, a3 }};
      }}

      function updateUI() {{
        const res = forwardPass(grid);
        let maxVal = -1;
        let predClass = -1;
        for (let d = 0; d < 10; d++) {{
          if (res.a3[d] > maxVal) {{
            maxVal = res.a3[d];
            predClass = d;
          }}
        }}

        // Update 10 bars
        for (let d = 0; d < 10; d++) {{
          const pct = (res.a3[d] * 100).toFixed(1);
          const bar = document.getElementById('bar-' + d);
          const label = document.getElementById('val-' + d);
          bar.style.width = pct + '%';
          label.innerText = pct + '%';
          if (d === predClass && maxVal > 0.15) {{
            bar.style.background = '#a6e3a1';
            label.style.color = '#a6e3a1';
            label.style.fontWeight = 'bold';
          }} else {{
            bar.style.background = '#89b4fa';
            label.style.color = '#a6adc8';
            label.style.fontWeight = 'normal';
          }}
        }}

        // Update Badge
        const badge = document.getElementById('pred-badge');
        if (maxVal > 0.15) {{
          badge.innerText = predClass + ' (' + (maxVal * 100).toFixed(0) + '%)';
          badge.style.color = '#a6e3a1';
        }} else {{
          badge.innerText = '-';
          badge.style.color = '#585b70';
        }}

        // Update Hidden Dots
        for (let i = 0; i < 16; i++) {{
          const dot1 = document.getElementById('h1-dot-' + i);
          const val1 = res.a1[i];
          dot1.style.background = `rgba(203, 166, 247, ${{val1}})`;
          
          const dot2 = document.getElementById('h2-dot-' + i);
          const val2 = res.a2[i];
          dot2.style.background = `rgba(137, 220, 235, ${{val2}})`;
        }}
      }}

      function extractGridFromCanvas() {{
        // Downsample 280x280 canvas to 28x28 grid
        const imgData = ctx.getImageData(0, 0, 280, 280).data;
        for (let gy = 0; gy < 28; gy++) {{
          for (let gx = 0; gx < 28; gx++) {{
            let total = 0;
            for (let py = 0; py < 10; py++) {{
              for (let px = 0; px < 10; px++) {{
                const idx = ((gy * 10 + py) * 280 + (gx * 10 + px)) * 4;
                total += imgData[idx + 3] / 255.0; // alpha
              }}
            }}
            grid[gy * 28 + gx] = Math.min(1.0, total / 100.0);
          }}
        }}
        updateUI();
      }}

      function drawBrush(x, y) {{
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.beginPath();
        ctx.arc(x, y, 14, 0, Math.PI * 2);
        ctx.fill();
        extractGridFromCanvas();
      }}

      canvas.addEventListener('mousedown', (e) => {{
        isDrawing = true;
        hint.style.display = 'none';
        const rect = canvas.getBoundingClientRect();
        drawBrush(e.clientX - rect.left, e.clientY - rect.top);
      }});

      canvas.addEventListener('mousemove', (e) => {{
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        drawBrush(e.clientX - rect.left, e.clientY - rect.top);
      }});

      window.addEventListener('mouseup', () => {{ isDrawing = false; }});

      document.getElementById('btn-clear').addEventListener('click', () => {{
        ctx.clearRect(0, 0, 280, 280);
        grid.fill(0.0);
        hint.style.display = 'block';
        updateUI();
      }});

      document.getElementById('btn-smooth').addEventListener('click', () => {{
        // Apply simple blur on grid
        const newGrid = new Float64Array(784);
        for (let y = 1; y < 27; y++) {{
          for (let x = 1; x < 27; x++) {{
            let sum = 0;
            for (let dy = -1; dy <= 1; dy++) {{
              for (let dx = -1; dx <= 1; dx++) {{
                sum += grid[(y + dy) * 28 + (x + dx)];
              }}
            }}
            newGrid[y * 28 + x] = sum / 9.0;
          }}
        }}
        grid = newGrid;
        // redraw to canvas
        ctx.clearRect(0, 0, 280, 280);
        for (let y = 0; y < 28; y++) {{
          for (let x = 0; x < 28; x++) {{
            const val = grid[y * 28 + x];
            if (val > 0.01) {{
              ctx.fillStyle = `rgba(255, 255, 255, ${{val}})`;
              ctx.fillRect(x * 10, y * 10, 10, 10);
            }}
          }}
        }}
        updateUI();
      }});

      document.querySelectorAll('.btn-preset').forEach(btn => {{
        btn.addEventListener('click', (e) => {{
          const d = e.target.getAttribute('data-digit');
          const sample = presets[d];
          hint.style.display = 'none';
          ctx.clearRect(0, 0, 280, 280);
          for (let i = 0; i < 784; i++) {{
            grid[i] = sample[i];
            const gx = i % 28;
            const gy = Math.floor(i / 28);
            if (sample[i] > 0.01) {{
              ctx.fillStyle = `rgba(255, 255, 255, ${{sample[i]}})`;
              ctx.fillRect(gx * 10, gy * 10, 10, 10);
            }}
          }}
          updateUI();
        }});
      }});

      updateUI();
    }})();
    </script>
    \"\"\"
    return HTML(html_code)

# Display Embedded Real-Time Drawing Canvas
display(create_interactive_js_widget())""")

    # -------------------------------------------------------------
    # 7. Deep Dissection & Conclusion
    # -------------------------------------------------------------
    add_markdown("""---
## 5. Diagnostic Dissection: How the Network Recognizes Features

When a digit (e.g. $8$ vs $3$) is drawn, we can analyze the step-by-step layer activations:
1. **Input Grid ($a^{(0)}$)**: 784 pixel values activate based on where strokes were drawn.
2. **Hidden Layer 1 ($a^{(1)}$)**: Detects specific localized sub-components (such as the top loop of an $8$, the straight vertical stem of a $1$, or the bottom curved arc of a $3$).
3. **Hidden Layer 2 ($a^{(2)}$)**: Combines these primitive sub-components into composite structural shapes.
4. **Output Layer ($a^{(3)}$)**: Evaluates the composite pattern against the 10 digit archetypes and produces the final activation scores.
""")

    add_code("""# Detailed Numerical Dissection of a Sample Digit Prediction
def dissect_digit_prediction(sample_grid, target_digit=None):
    activations = get_layer_activations(sample_grid)
    a0, a1, a2, a3 = activations
    
    out_flat = a3.flatten()
    pred_digit = int(np.argmax(out_flat))
    conf = float(out_flat[pred_digit])
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})
    
    # 1. Input Image
    axes[0].imshow(sample_grid.reshape(28, 28), cmap='gray_r')
    axes[0].set_title(f"Input $a^{(0)}$ (784 px)\\nTarget: {target_digit if target_digit is not None else 'N/A'}", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Hidden Layer 1 Activations
    axes[1].barh(np.arange(16), a1.flatten(), color='#cba6f7', edgecolor='black')
    axes[1].set_yticks(np.arange(16))
    axes[1].set_yticklabels([f"H1_{i+1}" for i in range(16)], fontsize=8)
    axes[1].set_title("Layer 1: $a^{(1)}$ (16 units)", fontsize=11, fontweight='bold')
    axes[1].set_xlim([0, 1.05])
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # 3. Hidden Layer 2 Activations
    axes[2].barh(np.arange(16), a2.flatten(), color='#89dceb', edgecolor='black')
    axes[2].set_yticks(np.arange(16))
    axes[2].set_yticklabels([f"H2_{i+1}" for i in range(16)], fontsize=8)
    axes[2].set_title("Layer 2: $a^{(2)}$ (16 units)", fontsize=11, fontweight='bold')
    axes[2].set_xlim([0, 1.05])
    axes[2].grid(True, alpha=0.3, axis='x')
    
    # 4. Output Class Activations
    bar_cols = ['#2ecc71' if c == pred_digit else '#89b4fa' for c in range(10)]
    axes[3].bar(np.arange(10), a3.flatten(), color=bar_cols, edgecolor='black')
    axes[3].set_xticks(np.arange(10))
    axes[3].set_ylim([0, 1.05])
    axes[3].set_title(f"Output $a^{(3)}$: Pred = {pred_digit} ({conf:.1%})", fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# Run dissection on benchmark digit '8'
dissect_digit_prediction(image_map[8][0], target_digit=8)""")

    add_markdown("""---
## Summary & Key Takeaways
1. **Affine Projections & Activation Routing**: The neural network transforms continuous hand-drawn pixel arrays ($28 \\times 28$) into confidence distributions across 10 classes through sequential matrix multiplications and non-linear Sigmoid functions.
2. **Feature Hierarchies**: Layer 1 features act as stroke, loop, and edge filters; Layer 2 synthesizes higher-level structural components; Layer 3 outputs the final classification.
3. **Interactive Responsiveness**: Drawing on the $28 \\times 28$ grid and seeing real-time activation updates demystifies how neural networks evaluate handwritten input dynamically.
""")

    output_path = "/home/alonemusk/Projects/manimations/nn1.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f" Successfully generated notebook: {output_path}")

if __name__ == "__main__":
    create_nn1_notebook()
