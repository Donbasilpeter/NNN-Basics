
import cupy as cp

class Layer_Dense:
    
    def __init__(self,n_inputs,n_neurons) -> None:
        self.weights = 0.10 * cp.random.randn(n_inputs,n_neurons)
        self.biases = cp.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = cp.dot(inputs,self.weights) + self.biases
    
