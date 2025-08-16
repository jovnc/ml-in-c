# Machine Learning in C

This project will implement basic machine learning algorithms from scratch in C, starting from a single perceptron.

## Usage

To compile and run the project, use the following commands:

```bash
# MacOS
chmod +x run.sh
./run.sh # modify as needed to run program you want
```

## Core Concepts

### Perceptron

Perceptron is the most basic unit of a neural network. It takes one or more inputs, processes them, and produces an output.

Linear combination of inputs(x), weights (w), and a bias.

$$ y = w_1 (x_1) + w_2 (x_2) + ... + w_n (x_n) + b $$

**Gradient Descent**

Find correct values for weights (w) and bias (b) using smart guesses through gradient descent.

- Iterative optimization algorithm, used to minimise cost function, eg. MSE
- How it works: take derivative of function to determine which direction does function grows. To find minimum, move in opposite direction of gradient
- Uses finite differences to approximate gradients for simplification

$$f'(x) = \frac{f(x + h) - f(x)}{h}$$

- Why bias?
  - Models `y= mx + c` where `c` is the bias
  - Without bias, forced to go through origin (0,0)
  - Allows model to fit data better by shifting the line up/down

**Activation Function**

Determines how the neuron produces output ==after== computing linear combination of inputs, weights, and bias.

$$ y = f(w_1 (x_1) + w_2 (x_2) + ... + w_n (x_n) + b) $$

- Introduces non-linearity into the model
- eg. Sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$
  - Maps any real-valued number into the range (0, 1)
  - +: smooth gradient, differentiable (useful for backpropagation)
  - Useful for binary classification tasks

**Interesting Observations**

- Problem: The model from `v1.1-gradient-descent` tag under `gates.c` can learn the OR function, but struggles with XOR
  - Single-layer perceptrons can only learn linearly separable functions
  - Solution: multi-layer perceptrons (MLPs) can learn non-linear decision boundaries
    - XOR can be defined from AND, NAND, OR, each can be defined by single neuron
    - XOR = (X|Y) & ~(X&Y)

### Multi-Layer Perceptrons (MLPs)

MLPs are a class of feedforward artificial neural networks that consist of multiple layers of neurons, allowing them to learn complex patterns. Each neuron has its own set of weights and biases, enabling it to learn different features from the input data.

- MLP can also be used to represent simpler functions that can be represented by single-layer perceptrons
- Allows for flexibility, ie. don't need to know exactly how many layers or neurons are required in advance
