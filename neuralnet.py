################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
from random import sample


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    maxi = img.max(axis=1)
    mini = img.min(axis=1)
    img = ((img.T - mini)/(maxi-mini)).T
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    oh_labels = np.zeros((labels.size, num_classes)) # row-wise
    oh_labels[np.arange(labels.size),labels] = 1
    return oh_labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    x_max = x.max(axis=1)  ## Taking x = w0x0+w1x1+wnxn
    e = (x.T - x_max).T
    return np.exp(e)/ np.sum(np.exp(e), axis=1, keepdims=True)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
#        raise NotImplementedError("Sigmoid not implemented")
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
#        raise NotImplementedError("Tanh not implemented")
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
#        raise NotImplementedError("ReLu not implemented")
        self.x = x
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
#        raise NotImplementedError("Sigmoid gradient not implemented")
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
#        raise NotImplementedError("tanh gradient not implemented")
        return 1 - (self.tanh(self.x))**2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return np.where(self.x <= 0, 0, 1)
        


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        w = np.random.randn(in_units, out_units)
        w /= (np.std(w, axis=0) * np.sqrt(in_units)) # weights initailized to mean=0 and std = 1/sqrt(in_units)
        self.w = w    # Declare the Weight matrix
        self.b = np.zeros(out_units)                    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = np.dot(self.x,self.w) + self.b
        # x_ = np.hstack((np.ones(self.x.shape[0]),self.x))
        # w_ = np.vstack((self.b, self.w))
        # self.a = x_ @ w_
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        n = self.x.shape[0] # batch size
        self.d_w = np.dot(self.x.T, delta)/n
        self.d_x = np.dot(delta, self.w.T)/n
        self.d_b = delta / n
        return self.d_x


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        for layer in self.layers:
            a = layer(x)
            x = a
        return x
        # raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        t = targets
        self.last_targets = targets
        self.logits = logits
        y = softmax(logits)
        cross_entropy_loss = - np.sum( t*np.log(y) + (1-t)*np.log(1-y) ) /y.shape[0]
        self.final_loss = cross_entropy_loss
        return self.final_loss
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        lr = config['learning_rate']
        gamma = config['momentum_gamma']
        last_loss = self.logits*(self.last_targets - self.logits) # gradient before softmax layer
        for layer in self.layers[::-1]:
            last_loss = layer.backward(last_loss)
            if isinstance(layer, Layer):
                layer.w = gamma*layer.w + lr*layer.d_w 
                layer.b = gamma*layer.b + lr*layer.d_w

        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    best_weights = []
    best_loss = 1e10
    early_stop = config['early_stop']
    early_stop_threshold = config['early_stop_epoch']
    prev_loss = 1e10
    loss_increase_counter = 0
    
    for i in range(epochs):
        sample_indices = sample(range(x_train.shape[0]), batch_size)
        mini_batch_x = x_train[sample_indices, :]
        mini_batch_y = y_train[sample_indices, :]
        logits = model(mini_batch_x)
        loss = model.loss(logits, mini_batch_y)
        model.backward()
        val_logits = model(x_valid)
        val_loss = model.loss(val_logits, y_valid)
        print("Train loss at epoch %d: %.3f" % (i+1, loss))
        print("Validation loss at epoch %d: %.3f" % (i+1, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = []
            for layer in model.layers:
                best_weights.append(layer.w)
        if val_loss > prev_loss:
            loss_increase_counter += 1
        else:
            loss_increase_counter = 0
        if loss_increase_counter >= 5 and early_stop:
            break
        prev_loss = val_loss
    
    for j, layer in enumerate(model.layers):
        layer.w = best_weights[j]
#    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    
#    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    split_ratio = 0.9
    split_idx = int(split_ratio*len(x_train))
    x_valid, y_valid = x_train[split_idx:], y_train[split_idx:]
    x_train, y_train = x_train[:split_idx], x_train[:split_idx]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
