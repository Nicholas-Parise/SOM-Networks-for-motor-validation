# Nicholas Parise, 7242530
import sys
import numpy as np
import csv
import time
import matplotlib.pyplot as plt

SIZE = 5
weights = []

def init(input_size):
    global weights, biases
    weights = np.random.uniform(-0.25, 0.25, (SIZE, SIZE,input_size))

# read in file and setup input and output array
# This function also normalizes the inputs to be between 0 -> 1
def readFile(file):
    lines = file.read().strip().splitlines()
    rows, dataPoints = map(int, lines[0].split())

    temp_inputs = []
    temp_outputs = []

    for line in lines[1:rows + 1]:
        values = list(map(float, line.split()))
        output_value = values[0]
        input_values = values[1:]

        if output_value == 1:
            temp_outputs.append([1, 0])
        else:
            temp_outputs.append([0, 1])
        
        temp_inputs.append(input_values)

    inputs = np.array(temp_inputs)
    outputs = np.array(temp_outputs)

    # normalize
    min = np.min(inputs, axis=0)
    max = np.max(inputs, axis=0)
    denominator = np.where(max - min == 0, 1, max - min)
    inputs = (inputs - min) / denominator

    if inputs.shape != (rows, dataPoints):
        print(f"Error: Expected ({rows}, {dataPoints}) but got {inputs.shape}")

    return inputs, outputs


def distance(x):
    global weights
    # sum (wij - xi)^2
    diff = weights - x
    euclidean_distance = np.sum(diff**2, axis=2)

    row, col = np.unravel_index(np.argmin(euclidean_distance), euclidean_distance.shape)
    grid = np.arange(SIZE)
    # turn euclidean_distance to toridial distance
    row_distance = np.minimum(
        np.abs(grid[:, None] - row),
        SIZE - np.abs(grid[:, None] - row)
    )

    col_distance = np.minimum(
        np.abs(grid[None, :] - col),
        SIZE - np.abs(grid[None, :] - col)
    )

    toroidal_distance = row_distance**2 + col_distance**2
    return toroidal_distance, row, col


def training(inputs, T=2000, learning_rate=1.0, starting_radius = 1.0):
    global weights
    
    for t in range(T):
        
        # cerate array of random index's
        order = np.random.permutation(len(inputs))
        # only after doing each input once do we increase t
        for idx in order:

            random_input = inputs[idx]

            toroidal_distance, row, col = distance(random_input)

            lr = learning_rate * np.exp(-t / T)

            radius = starting_radius * np.exp(-t / T)

            #neighbourhood = np.exp(-radius * toroidal_distance)

            neighbourhood = np.exp(-(t / T) * toroidal_distance)

            neighbourhood = neighbourhood[:, :, None]

            # update weights. The slideshow says to do the first way, but it doesn't seem to work
            #weight_delta = lr * neighbourhood * (weights - random_input)
            weight_delta = lr * neighbourhood * (random_input - weights)
            weights += weight_delta

    return weights


def label_matrix(inputs, outputs):
    global weights
    
    radius = 4 * np.log(2)

    activation_graph = np.zeros((SIZE, SIZE))

    # turn [1,0] -> 1 and [0,1] -> -1
    labels = np.where(outputs[:, 0] == 1, 1, -1)

    for x, label in zip(inputs, labels):
        
        toroidal_distance, row, col = distance(x)
        
        # use same activation
        activation = np.exp(-radius * toroidal_distance)
        activation_graph += activation * label

    # make sure map is between -1 -> 1 by dividing by max value
    max_abs = np.max(np.abs(activation_graph))
    if max_abs > 0:
        activation_graph /= max_abs

    return activation_graph


def display_heatmap(matrix, title="SOM"):
    plt.figure(figsize=(6, 6))
    
    plt.imshow(matrix, cmap="bwr", vmin=-1, vmax=1)
    
    plt.colorbar(label="Activation Strength")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Print values inside cells
    for i in range(SIZE):
        for j in range(SIZE):
            plt.text(j, i, f"{matrix[i,j]:.2f}", 
                     ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=6)
    
    filename = sys.argv[1]
    print(filename)
    file = open(filename)

    inputs, outputs = readFile(file)

    input_size = inputs.shape[1]

    init(input_size)

    training(inputs, T=2000, learning_rate=1.0, starting_radius = 1.0)

    title = "File:"+filename+" Size:"+str(SIZE)

    matrix = label_matrix(inputs, outputs)
    print(title)
    print(matrix)

    display_heatmap(matrix, title)



