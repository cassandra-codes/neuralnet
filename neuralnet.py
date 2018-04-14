from random import random, seed
from math import exp, log as ln


class InvalidArgumentException(Exception):
    pass


class NeuralNet:
    """
    Learning rate determines the magnitude of updates between each epoch
    Threshold sets the minimum value a neuron must obtain to activate
    """
    
    def __init__(self, threshold=0):
        self.layers = []
        self.threshold = threshold
    
    
    def set_layers(self, layers):
        """
        Create #l layers of #n nodes (#n is defined during Layer initialization).
        """
        if isinstance(layers, list):
            for layer in layers:
                if not isinstance(layer, Layer):
                    raise InvalidArgumentException("NeuralNet.set_layers only accepts a list of Layer objects")
                else:
                    self.layers.append(layer)
        self.__initialize()
                    
    
    def __initialize(self):
        """
        Before training, we need to assign random weights (-1, 1) to every neural pathway
        
        In a fully connected network, this means creating a weighted path between each neuron in one layer to each
        neuron in the next.
        """
        
        for l in xrange(0, len(self.layers) - 1):
            for n in xrange(0, len(self.layers[l].nodes)):
                self.layers[l].nodes[n].set_weights(
                    [random() * 2 - 1 for weight in xrange(0, len(self.layers[l + 1].nodes))]
                )
                
                
    def __ingest(self, data):
        """
        Loads data into the input layer.  The input layer will always have 1 extra node which serves as a bias
        """
        for d in range(0, len(data)):
            self.layers[0].nodes[d].value = data[d]
            
            
    def __forward_propogate(self):
        """
        Activates neurons in each network on forward pass if the sum of weights times values of the previous layer
        exceeds threshold
        """
        
        for l in xrange(1, len(self.layers)):
            if self.layers[l].activation_function is not "softmax":
                for n in xrange(0, len(self.layers[l].nodes)):
                    neuron = 0.0
                    for x in self.layers[l - 1].nodes:
                        neuron += x.weights[n] * x.value

                    self.layers[l].nodes[n].set_value(self.layers[l].activate(neuron - self.threshold))
            else:
                pass
        
        
    def __back_propogate(self, actual):
        """
        Distributes error to weights on backward pass
        """
        for o in xrange(0, len(self.layers[-1].nodes)):  # Output layer's error is simply y minus y_hat
            self.layers[-1].nodes[o].set_error(actual[o] - self.layers[-1].nodes[o].value)
        for l in xrange(len(self.layers) - 1, 0, -1):
            if self.layers[l].activation_function is not "softmax":
                for n in self.layers[l - 1].nodes:
                    error = 0.0
                    for w in xrange(0, len(n.weights)):
                        error += n.weights[w] * self.layers[l].nodes[w].error
                        n.set_error(self.layers[l].error_function(n.value) * error)
                    new_weights = []
                    for w in xrange(0, len(n.weights)):
                        new_weights.append(n.weights[w] + self.lr * self.layers[l].nodes[w].error * n.value)
                    n.set_weights(new_weights)
            else:
                pass
    
    
    def train(self, data, epochs=100, learning_rate=0.001, verbose=False):
        self.lr = learning_rate
        for e in xrange(0, epochs):
            for d in range(0, len(data)):
                self.__ingest(data[d][0])
                self.__forward_propogate()
                self.__back_propogate(data[d][1])
            if verbose:
                print("Epoch {} out of {}".format(e, epochs))
                
                
    def predict(self, data):
        self.__ingest(data[0])
        self.__forward_propogate()
        return [node.value for node in self.layers[-1].nodes]
    
            
    
class Layer:
    def __init__(self, role, nodes, activation_function=None):
        self.role = self.__set_role(role)
        self.nodes = self.__set_nodes(nodes)
        self.activation_function = activation_function # for reference by name
        self.activate = self.__set_activation(activation_function)
        self.error_function = self.__set_error_function(activation_function)
        
        
    def __set_role(self, role):
        """
        Establishes how a particular layer should behave within the net.  Input and Output layers have special
        conditions that must be taken into consideration.
        
        TODO: Implement convolutional, pooling, and dropout layers
        """
        return {
            'input': 'Input',
            'hidden': 'Hidden',
            'output': 'Output',
            #'convolutional': 'Convolutional',
            #'pooling': 'Pooling',
            #'dropout': 'Dropout'
        }[role]    
    
    
    def __set_activation(self, af):
        """
        Sets how information will be passed during forward propogation
        TODO:  Implement softmax function for output layer.
        """
        
        if self.role == "Input":
            return None
        return {
            "LReLU": lambda x: x if x > 0 else 0.01 * x,
            "softmax": lambda x: self.__softmax(x)
        }[af]
    
    
    def __set_error_function(self, af):
        """
        Sets how information will be passed during back propogation
        """
        if self.role == "Input":
            return None
        return {
            "LReLU": lambda x: 1 if x > 0 else 0.01,
            "softmax": lambda x: self.__cross_entropy(x)
        }[af]
    
    
    def __softmax(self):
        exp_n = [exp(node.value) for node in self.nodes]
        self.softmax = [n / sum(exp_n) for n in exp_n]
    
    
    def __cross_entropy(self, i):
        pass
    

    def __set_nodes(self, nodes):
        """
        Initialize nodes within a layer.  If layer is the input layer, add one extra node whose value will
        remain constant as 1.  This bias node prevents anchoring our data to 0 and allows for a better fit. 
        """
        
        if self.role is 'Input':
            return [Node() for node in xrange(0, nodes + 1)]
        else:
            return [Node() for node in xrange(0, nodes)]
    
    
class Node:
    def __init__(self):
        """
        Creates a node with a default value of 1
        """
        self.value = 1
        self.error = 1000
        
        
    def set_value(self, value):
        self.value = value
        
        
    def set_weights(self, weights):
        self.weights = weights
        
        
    def set_error(self, error):
        self.error = error