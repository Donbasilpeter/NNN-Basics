import cupy as cp

# Convert inputs, weights, and biases to CuPy arrays
inputs_cp = cp.array([[1, 2, 3], [1, 2.5, 3], [1.2, 1.4, 4.5]])
weights_cp = cp.array([[0.2, 0.2, 0.5], [0.8, -0.6, 0.8], [-0.5, -0.5, -0.7]])
biases_cp = cp.array([2, 3, -0.4])

# Perform the matrix multiplication on GPU
output_cp = cp.dot(inputs_cp, weights_cp.T) + biases_cp

# Transfer the result back to CPU if necessary
output = cp.asnumpy(output_cp)

print(output)
