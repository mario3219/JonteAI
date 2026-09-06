import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = os.path.abspath(
    os.path.join(os.getcwd(), '..', 'data')
)

X = np.load(os.path.join(DATA_DIR, 'x.npy'))
y = np.load(os.path.join(DATA_DIR, 'y.npy'))

W1 = np.load(os.path.join(DATA_DIR, 'W1.npy'))
W2 = np.load(os.path.join(DATA_DIR, 'W2.npy'))
b1 = np.load(os.path.join(DATA_DIR, 'b1.npy'))
b2 = np.load(os.path.join(DATA_DIR, 'b2.npy'))

# --------------------------------------------------
# Reconstruct matrix shapes saved from C++
# --------------------------------------------------

input_dim = X.shape[1]
hidden_layer_size = b1.size
num_classes = b2.size

W1 = W1.reshape(input_dim, hidden_layer_size)
W2 = W2.reshape(hidden_layer_size, num_classes)

print("X shape: ", X.shape)
print("y shape: ", y.shape)
print("W1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("W2 shape:", W2.shape)
print("b2 shape:", b2.shape)

# --------------------------------------------------
# Construct a dense 2D grid covering the input space
# --------------------------------------------------

x_min = X[:, 0].min() - 0.5
x_max = X[:, 0].max() + 0.5

y_min = X[:, 1].min() - 0.5
y_max = X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

# --------------------------------------------------
# Forward propagation using your trained C++ weights
# --------------------------------------------------

# Z1 = X W1 + b1
L1_in = np.dot(grid_points, W1) + b1

# Q1 = ReLU(Z1)
L1_out = np.maximum(0, L1_in)

# Z2 = Q1 W2 + b2
L2_in = np.dot(L1_out, W2) + b2

# Softmax
L2_shifted = L2_in - np.max(L2_in, axis=1, keepdims=True)

exp_scores = np.exp(L2_shifted)

probs = exp_scores / np.sum(
    exp_scores,
    axis=1,
    keepdims=True
)

# Predicted class for every grid location
predicted_class = np.argmax(probs, axis=1)

# Return flattened grid predictions to mesh shape
Z = predicted_class.reshape(xx.shape)

# --------------------------------------------------
# Plot decision regions
# --------------------------------------------------

plt.figure(figsize=(7, 6))

plt.contourf(
    xx,
    yy,
    Z,
    levels=np.arange(num_classes + 1) - 0.5,
    cmap=plt.cm.Spectral,
    alpha=0.6
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.Spectral,
    edgecolors='k',
    s=30
)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision boundary of C++ neural network")

plt.show()
