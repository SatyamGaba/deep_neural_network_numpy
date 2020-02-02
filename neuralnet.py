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
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm


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
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x = x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        return np.maximum(0, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
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
        # w = np.random.randn(in_units, out_units)
        # w /= (np.std(w, axis=0) * np.sqrt(in_units)) # weights initailized to mean=0 and std = 1/sqrt(in_units))
        w = np.random.normal(loc=0.0, scale=1.0/((in_units+out_units)**0.5), size=(in_units, out_units))
        self.w = w    # Declare the Weight matrix
        self.b = np.zeros(out_units)                    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.delta_w = np.zeros_like(self.w) # to include momentum term
        self.delta_b = np.zeros_like(self.b)

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x, epsilon=0):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.w += epsilon
        self.a = np.dot(self.x,self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_w = np.dot(self.x.T, delta)
        self.d_x = np.dot(delta, self.w.T)
        self.d_b = np.sum(delta, axis=0)
        
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
        
        self.lr = config['learning_rate']
        self.gamma = config['momentum_gamma']
        self.L2 = config['L2_penalty']

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

    def forward(self, x, targets=None, epsilon=0):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        for layer in self.layers:
            a = layer(x)
            x = a
        if targets is not None:
            return x , self.loss(x,targets) 
        return x

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        self.targets = targets
        self.y = softmax(logits)
        cross_entropy_loss = - np.mean( self.targets*np.log(self.y) )
        self.final_loss = cross_entropy_loss
        return self.final_loss

    def backward(self, epsilon=0):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        last_loss = self.targets - self.y # gradient before softmax layer
        for layer in self.layers[::-1]:
            last_loss = layer.backward(last_loss)
            if isinstance(layer, Layer):
                layer.w += epsilon
                layer.delta_w = (self.gamma * layer.delta_w) + (1)*self.lr*layer.d_w
                layer.delta_b = (self.gamma * layer.delta_b) + (1)*self.lr*layer.d_b
                layer.w = layer.w + layer.delta_w - self.L2*layer.w
                layer.b = layer.b + layer.delta_b - self.L2*layer.b


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    activation = config['activation']
    gamma = config['momentum_gamma']
    learning_rate = config['learning_rate']
    l2_penalty = config['L2_penalty']
    best_weights = []
    best_loss = 1e10
    early_stop = config['early_stop']
    early_stop_threshold = config['early_stop_epoch']
    prev_loss = 1e10
    loss_increase_counter = 0
    train_loss_array = []
    val_loss_array = []
    train_acc_array=[]
    val_acc_array=[]
    
    for i in tqdm(range(epochs)):
        sample_indices = sample(range(x_train.shape[0]), batch_size)
        mini_batch_x = x_train[sample_indices, :]
        mini_batch_y = y_train[sample_indices, :]
        logits = model(mini_batch_x)
        output = softmax(logits)
        pred = np.argmax(output, axis=1)
        labels = np.argmax(mini_batch_y, axis=1)
        correct = np.where(pred==labels, 1, 0)
        train_acc = (np.sum(correct) / mini_batch_y.shape[0])*100
        train_acc_array.append(train_acc)
        train_loss = model.loss(logits, mini_batch_y)
        train_loss_array.append(train_loss)
        model.backward()

        val_logits = model(x_valid)
        val_output = softmax(val_logits)
        val_pred = np.argmax(val_output, axis=1)
        val_labels = np.argmax(y_valid, axis=1)
        correct = np.where(val_pred==val_labels, 1, 0)
        val_acc = (np.sum(correct) / y_valid.shape[0])*100
        val_acc_array.append(val_acc)
        val_loss = model.loss(val_logits, y_valid)
        val_loss_array.append(val_loss)
        # print("Train loss = %.3f, " % (train_loss), end='')
        # print("Validation loss = %.3f" % (val_loss))
        if val_loss < best_loss:
            best_epoch = i
            best_loss = val_loss
            best_weights = []
            for layer in model.layers:
                if isinstance(layer, Layer):
                    best_weights.append(layer.w)
        if val_loss > prev_loss:
            loss_increase_counter += 1
        else:
            loss_increase_counter = 0
        if loss_increase_counter >= early_stop_threshold and early_stop:
            print("Early Stopping")
            break
        prev_loss = val_loss
    
    # store best weights in the model
    j=0
    for layer in enumerate(model.layers):
        if isinstance(layer, Layer):
            layer.w = best_weights[j]
            j += 1

    # plot the training an validation loss curve

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(train_acc_array)), train_acc_array, label="Training Accuracy")
    ax.plot(range(len(val_acc_array)), val_acc_array, label="Validation Accuracy")
    ax.plot(best_epoch, val_acc_array[best_epoch], marker="o", Label="Best weights")
    ax.set(title="Mini-Batch Stochastic Gradient Descent", xlabel="No. of epochs", ylabel="Accuracy") # manually update title for each run
    ax.legend()
    ax.grid()
    fig.savefig("./plots/Accuracy_%s_lr=%.4f_bsize=%d_g=%.2f_l2=%.2f.png"%(activation,learning_rate,batch_size,gamma,l2_penalty))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(train_loss_array)), train_loss_array, label="Training Loss")
    ax.plot(range(len(val_loss_array)), val_loss_array, label="Validation Loss")
    ax.plot(best_epoch, val_loss_array[best_epoch], marker="o", Label="Best weights")
    ax.set(title="Mini-Batch Stochastic Gradient Descent", xlabel="No. of epochs", ylabel="Cross Entropy Loss") # manually update title for each run
    ax.legend()
    ax.grid()
    fig.savefig("./plots/Loss_%s_lr=%.4f_bsize=%d_g=%.2f_l2=%.2f.png"%(activation,learning_rate,batch_size,gamma,l2_penalty))
    plt.show()

def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    logits = model(X_test)
    output = softmax(logits)
    pred = np.argmax(output, axis=1)
    expected = np.argmax(y_test, axis=1)
    
    correct = np.where(pred==expected, 1, 0)
    return (sum(correct) / y_test.shape[0])*100


def check_gradients(model, x_train, y_train):
    """
    To perform gradient checks
    """
    x_sub = np.zeros((10, x_train.shape[1]))
    y_sub = np.zeros((10, y_train.shape[1]))
    for i in range(10):
        for j,row in enumerate(y_train):
            if row[i] == 1:
                x_sub[i,] = x_train[j, ]
                y_sub[i,] = row
                break
    logits = model(x_sub)
    loss = model.loss(logits, y_sub)
    model.backward()
    
    calculated_losses = []
    for layer in model.layers[1:-1]:
        if isinstance(layer, Layer):
            print(layer.d_w[0])
            calculated_losses.append(np.mean(layer.d_w[0]))
    
    model.backward()
    
    logits = model.forward(x_sub, epsilon=1e-2)
    loss_plus = model.loss(logits, y_sub)
    for layer in model.layers[1:-1]:
        if isinstance(layer, Layer):
            print(layer.d_w[0])
            calculated_losses.append(np.mean(layer.d_w[0]))
    
    logits = model.forward(x_sub, epsilon=-1e-2)
    loss_minus = model.loss(logits, y_sub)
    
    slope_losses = 2*1e2*(loss_plus - loss_minus)
    
    print(slope_losses)
    print(calculated_losses)
    
    
if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")
    
    shuffle_indices = np.random.shuffle(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices][0]
    y_train = y_train[shuffle_indices][0]

    # Create splits for validation data here.
    split_ratio = 0.9
    split_idx = int(split_ratio*len(x_train))
    x_valid, y_valid = x_train[split_idx:], y_train[split_idx:]
    x_train, y_train = x_train[:split_idx], y_train[:split_idx]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)    

    test_acc = test(model, x_test, y_test)
    print("Test accuracy:", test_acc)
    
#    check_gradients(model, x_train, y_train)