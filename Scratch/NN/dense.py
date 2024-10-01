import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        # Randomly initialize the weights and the bias
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        # Compute the output -> Wx = b
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Update parameters and return input gradient
        # Derivative of the error with respects to its weights
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Derivate of the error with respect to the input
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Update the weights ans bias with gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient       
        return input_gradient
    