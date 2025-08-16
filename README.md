# Machine Learning in C

This project will implement basic machine learning algorithms from scratch in C, starting from a single perceptron.

## Core Concepts

### Artificial Neuron (Perceptron)

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
