import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        if activation == 'tanh':
            self.activation_fn = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'sigmoid':
            self.activation_fn = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation_fn = relu
            self.activation_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        # Store intermediate values for visualization
        self.hidden_output = None
        self.gradients = {}
    def forward(self, X):
        self.hidden_input = X @ self.weights_input_hidden + self.bias_hidden
        self.hidden_output = self.activation_fn(self.hidden_input)

        # Forward pass: Hidden layer to output
        self.output_input = self.hidden_output @ self.weights_hidden_output + self.bias_output
        self.output = sigmoid(self.output_input)  # Binary classification output

        return self.output

    def backward(self, X, y):
        output_error = self.output - y  # Loss gradient
        output_gradient = output_error * sigmoid_derivative(self.output_input)

        # Compute hidden layer gradients
        hidden_error = output_gradient @ self.weights_hidden_output.T
        hidden_gradient = hidden_error * self.activation_derivative(self.hidden_input)

        # Store gradients for visualization
        self.gradients = {
            'weights_input_hidden': X.T @ hidden_gradient,
            'bias_hidden': np.sum(hidden_gradient, axis=0, keepdims=True),
            'weights_hidden_output': self.hidden_output.T @ output_gradient,
            'bias_output': np.sum(output_gradient, axis=0, keepdims=True)
        }

        # Update weights and biases using gradient descent
        self.weights_input_hidden -= self.lr * self.gradients['weights_input_hidden']
        self.bias_hidden -= self.lr * self.gradients['bias_hidden']
        self.weights_hidden_output -= self.lr * self.gradients['weights_hidden_output']
        self.bias_output -= self.lr * self.gradients['bias_output']


# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.hidden_output
    if hidden_features.shape[1] >= 3:  # Ensure there are at least 3 dimensions
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                          c=y.ravel(), cmap='bwr', alpha=0.7)
        ax_hidden.set_title("Hidden Space")
    else:
        ax_hidden.text(0.5, 0.5, "Insufficient dimensions for 3D plot",
                       transform=ax_hidden.transAxes, ha="center", va="center")

    # TODO: Hyperplane visualization in the hidden space
    if hidden_features.shape[1] >= 3:
        ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                        c=y.ravel(), cmap='bwr', alpha=0.8)
        x_h = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 30)
        y_h = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 30)
        x_h, y_h = np.meshgrid(x_h, y_h)
        z_h = -(mlp.weights_hidden_output[0, 0] * x_h +
                mlp.weights_hidden_output[1, 0] * y_h +
                mlp.bias_output[0, 0]) / mlp.weights_hidden_output[2, 0]
        ax_hidden.plot_surface(x_h, y_h, z_h, alpha=0.3, color='purple')
        ax_hidden.set_title("Hidden Space at Step {}".format(frame))


    # TODO: Distorted input space transformed by the hidden layer
    if hidden_features.shape[1] >= 2:
        distorted_X = hidden_features[:, :2]  # Use first two dimensions for the distorted space
        ax_input.scatter(distorted_X[:, 0], distorted_X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.5, marker='x')

    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax_input.contourf(xx, yy, Z, alpha=0.8, levels=[0, 0.5, 1], cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title("Input Space at Step {}".format(frame))


    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    for i, gradient in enumerate(mlp.gradients['weights_input_hidden']):
        start_x, start_y = 0, 0
        end_x, end_y = gradient[0], gradient[1]
        ax_gradient.plot([start_x, end_x], [start_y, end_y], color='purple',
                         alpha=0.6, linewidth=max(0.1, np.linalg.norm(gradient)))
        ax_gradient.scatter(end_x, end_y, s=50, c='blue', alpha=0.6)
    ax_gradient.set_title("Gradients (Thickness ∝ Magnitude)")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)