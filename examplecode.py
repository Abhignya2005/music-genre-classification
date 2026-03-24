import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)  # ReLU activation function
    
    def forward_propagation(self, X):
        # Input to hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        
        # Hidden to output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output = self.relu(self.output_layer_input)
        
        return output

# Example usage:
nn = SimpleNeuralNetwork(input_size=3, hidden_size=5, output_size=1)
X_input = np.array([[1, 2, 3]])  # Example input
output = nn.forward_propagation(X_input)
print("Forward Propagation Output:", output)
