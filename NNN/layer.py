import cupy as cp

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons, activation="ReLU"):
        """
        Initialize the layer with random weights and biases.

        Parameters:
        - n_inputs (int): Number of input neurons.
        - n_neurons (int): Number of neurons in the layer.
        - activation (str): Activation function to be used. Defaults to "ReLU".
                            Supported values: "ReLU", "SoftMax".
        """
        if activation not in ["ReLU", "SoftMax"]:
            raise ValueError("Invalid activation function")
            
        self.weights = 0.10 * cp.random.randn(n_inputs, n_neurons)
        self.biases = cp.zeros((1, n_neurons))
        self.activation = activation
        
    def forward(self, inputs):
        """
        Perform forward pass through the layer.

        Parameters:
        - inputs (cupy.ndarray): Input data.

        Returns:
        - outputs (cupy.ndarray): Output of the layer after applying activation.
        """
        self.output = cp.dot(inputs, self.weights) + self.biases
        self.output = self.__activation(self.output)
        return self.output

    def __activation(self, x):
        """
        Apply activation function to the input.

        Parameters:
        - x (cupy.ndarray): Input to apply activation function.

        Returns:
        - result (cupy.ndarray): Output after applying activation function.
        """
        if self.activation == "ReLU":
            return cp.maximum(0, x)
        elif self.activation == "SoftMax":
            return self.__softmax(x)
        
    def __softmax(self, x):
        """
        Apply SoftMax activation function to the input.

        Parameters:
        - x (cupy.ndarray): Input data.

        Returns:
        - result (cupy.ndarray): Output after applying SoftMax activation.
        """
        exp_x = cp.exp(x - cp.amax(x, axis=1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)
