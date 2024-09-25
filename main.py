import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import math

class Node:
    def __init__(self, bias, connections=None, activation=None, x_pos=None, y_pos=None):
        self.connections = connections
        self.bias = bias
        self.activation = activation
        self.x_pos = x_pos
        self.y_pos = y_pos

    def get_neighbours(self):
        return np.where(np.array(self.connections) != 0)[0]


class NeuralNetwork:
    '''
    Train_data will be given as several images but the class will
    work through one image at a time.
    Hence, initialise the class and then train the model by iterating
    through the dataset and using functions accordingly.
    Will have one hidden layer as well as input and output layer.
    Will return the accuracy of prediction against Test_data given
    '''
    def __init__(self, Train_data, Test_data, input_size, hidden_size, output_size):
        self.Train_data = Train_data
        self.Test_data = Test_data

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_nodes = self.output_size + self.hidden_size + self.input_size

    def initial_activations(self, image):
        '''
        This function will flatten the image given and set
        the pixel images as the initial activations of the
        first 784 nodes.
        '''
        pixels = []
        flattened_image = image.flatten().astype('float32') / 255.0
        flattened_image.tolist()
        for node_index in range(0, self.input_size):
            self.nodes[node_index].activation = flattened_image[node_index]

    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        return np.where(x > 0, 1, 0)

    def adj_matrix(self):
        '''
        Function creates the adjacency matrix.
        The weights are assigned since another function for this
        is not needed.
        Returns the adjacency matrix.
        '''
        adj_matrix = array_zeros = np.zeros((self.num_nodes, self.num_nodes))
        # graph is directed, hence [a, b] = c but [b, a] = 0
        for index_1 in range(0, self.input_size):
            for index_2 in range(self.input_size, self.hidden_size + self.input_size):
                adj_matrix[index_1, index_2] = np.random.randn() * 0.01
        for index_1 in range(self.input_size, self.hidden_size + self.input_size):
            for index_2 in range(self.hidden_size + self.input_size, self.num_nodes):
                adj_matrix[index_1, index_2] = np.random.randn() * 0.01
        self.adjacency_matrix = adj_matrix


    def create_network(self, adj_matrix): # adj_matrix must be obtained from self.adj_matrix
        nodes = []
        for value in range(0, self.num_nodes):
            bias = np.random.randn() * 0.01
            nodes.append(Node(bias, adj_matrix[value]))
        self.nodes = nodes

    def forwards_pass(self, layer):
        if layer == "hidden":
            for node_1 in range((self.input_size + self.hidden_size), self.num_nodes):
                activations = []
                for node_2 in range(self.input_size, (self.input_size + self.hidden_size)):
                    x = self.nodes[node_2].activation
                    w = self.adjacency_matrix[node_2, node_1]
                    b = self.nodes[node_1].bias
                    activations.append(x*w)
                self.nodes[node_1].activation = self.relu(sum(activations) + b)

        if layer == "input":
            for node_1 in range(self.input_size, (self.input_size + self.hidden_size)):
                activations = []
                for node_2 in range(0, self.input_size):
                    x = self.nodes[node_2].activation
                    w = self.adjacency_matrix[node_2, node_1]
                    b = self.nodes[node_1].bias
                    activations.append(x*w)
                self.nodes[node_1].activation = self.relu(sum(activations) + b)

        if layer == "output":
            outputs = np.array
            for value in range((self.input_size + self.hidden_size), self.num_nodes):
                outputs = np.append(outputs, self.nodes[value].activation)
            return outputs

    def loss_function(self, target, output):
        difference = target - output
        loss = []
        for value in difference:
            loss.append(value**2)
        return loss

    def plot(self, x_lim, y_lim):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        layers = [self.input_size, self.hidden_size, self.output_size] # must be in order of layers
        largest_layer = max(layers)
        r = y_lim / ((3*largest_layer) + 1)
        node_count = 0
        for layer in layers: # In order to place circles at different layers
            x_pos = x_lim * (layers.index(layer)/len(layers))
            node_count += layer # Needed to account for range beginning at zero when assigning nodes positions
            for value in range(0, layer):
                y_pos = (3 * r * value) - r
                circle = plt.Circle((x_pos, y_pos), r,edgecolor='black', facecolor='none')
                self.nodes[value + (node_count - layer)].x_pos = x_pos
                self.nodes[value + (node_count - layer)].y_pos = y_pos
                ax.add_patch(circle)
        node_indexes = [self.input_size, self.input_size + self.hidden_size, self.num_nodes]
        for index_1 in range(0, self.num_nodes):
            if 0 <= index_1 < node_indexes[0]:
                for index_2 in range(node_indexes[0], node_indexes[1]):
                    point_one = [self.nodes[index_1].x_pos, self.nodes[index_1].y_pos]
                    point_two = [self.nodes[index_2].x_pos, self.nodes[index_2].y_pos]
                    ax.plot([point_one[0], point_two[0]], [point_one[1], point_two[1]], 'k-', lw=0.1)
            elif node_indexes[0] <= index_1 < node_indexes[1]:
                for index_2 in range(node_indexes[1], node_indexes[2]):
                    point_one = [self.nodes[index_1].x_pos, self.nodes[index_1].y_pos]
                    point_two = [self.nodes[index_2].x_pos, self.nodes[index_2].y_pos]
                    ax.plot([point_one[0], point_two[0]], [point_one[1], point_two[1]], 'k-', lw=0.5)
        plt.show()

    def input_to_hidden_bp(self, row, col, input_index, hidden_index, output_index, weights, outputs):
        for i_value in range(hidden_index, output_index):
            a_i = self.nodes[i_value].activation
            y_i = outputs[i_value - hidden_index]
            b_i = self.nodes[row].bias
            b_k = self.nodes[col].bias
            activation = self.nodes[row].activation
            weight = weights[row, col]
            total_k = b_i
            total_j = b_k
            for k_value in range(input_index, hidden_index):
                a_k = self.nodes[k_value].activation
                w_ik = weights[row, k_value]
                total_k += a_k * w_ik
            for j_value in range(hidden_index, output_index):
                w_jk = weights[col, j_value]
                a_j = self.nodes[j_value].activation
                total_j += a_j * w_jk
            grad = self.derivative_relu(total_j) * self.derivative_relu(total_k) * (a_i - y_i) * activation * weight
        return -grad

    def back_propagation(self, desired_output, learning_rate):
        outputs = []
        for value in range(0, 10):
            outputs.append(0)
        outputs[desired_output] = 1
        desired_output = outputs
        weights = self.adjacency_matrix
        output_index = self.num_nodes
        hidden_index = self.hidden_size + self.input_size
        input_index = self.input_size
        # for hidden layer to output layer

        for row in range(input_index, hidden_index):
            for col in range(hidden_index, output_index):
                weight = weights[row, col]
                a_1 = self.nodes[row].activation
                a_2 = self.nodes[col].activation
                bias = self.nodes[row].bias
                error = a_2 - desired_output[col - hidden_index]
                grad = a_1 * self.derivative_relu(a_2) * error
                self.adjacency_matrix[row, col] = weight - learning_rate * grad
                self.nodes[col].bias += - learning_rate * self.derivative_relu(a_2) * error

        # for input layer to hidden layer
        for row in range(0, input_index):
            for col in range(input_index, hidden_index):
                weight = weights[row, col]
                grad = self.input_to_hidden_bp(row, col, input_index, hidden_index, output_index, weights, outputs)
                weights[row, col] = weight - learning_rate * grad


    def prediction(self, y_test, x_test): # test data given as x_test and y_test
        self.initial_activations(x_test)
        self.forwards_pass("input")
        self.forwards_pass("hidden")
        output = self.forwards_pass("output")
        print(y_test, output)
def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    network = NeuralNetwork((x_train, y_train), (x_test, y_test), 784, 128, 10)
    network.adj_matrix()
    network.create_network(network.adjacency_matrix)
    for i in range(0, 2):
        network.initial_activations(x_train[i])
        network.forwards_pass("input")
        network.forwards_pass("hidden")
        network.back_propagation(desired_output=y_train[i], learning_rate=0.01)
    for j in range(0, 10):
        network.prediction(y_test[j], x_test[j])

if __name__ == "__main__":
    main()