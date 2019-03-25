import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys


def hypothesis(thetas, row):
    """
        This function evaluates the current hypothesis given the different thetas

        args:
            thetas -> array of numbers
            row -> row of dataset to evaluate
    """
    acum = 0
    for j in range(len(thetas)):
        acum += thetas[j] * row[j]
    return acum

def cost_function(thetas, data):
    """
        This function evaluates the total error for the model

        args:
            thetas -> array of numbers
            data -> complete dataset
    """
    total_error = 0
    for i in range(len(data)):
        acum = 0
        for j in range(len(thetas)):
            acum += thetas[j] * data[i][j]
        total_error = (acum - data[i][-1])**2
    return total_error / float(len(data))


def step_gradient_descent(data, thetas, learning_rate):
    """
        This function evaluates one step of gradient descent and returns updated value

        args:
            data -> complete dataset
            thetas -> array of numbers
            learning_rate -> float
            
    """
    N = len(data)
    temp_thetas = np.zeros(len(thetas))
    for j in range(len(thetas)):
        acum = 0
        for i in range(N):
            acum += (hypothesis(thetas, data[i]) - data[i][-1])*data[i][j]
        temp_thetas[j] = thetas[j] - (learning_rate*(1/N)) * acum
    return temp_thetas


def gradient_descent(dataset, alfa, coefficients, y):
	newCoef = list(coefficients)
	for i in range(len(coefficients)):
		sum = 0
		for j in range(len(dataset)):
			prediction = hypothesis(dataset[j], coefficients)
			sum = sum + (prediction - y[j]) * dataset[j][i]
		# Update
		newCoef[i] = coefficients[i] - alfa * (1 / len(dataset)) * sum
		#print(newParams)
	return newCoef

if __name__ == '__main__':
    """
        This script implements linear regression with gradient descent
    """
    # Receive filename as argument
    filename = sys.argv[1]
    # Load dataset into numpy array
    data = np.genfromtxt(filename, delimiter=',')
    # Define number of thetas needed
    thetas = np.zeros(len(data[0]))
    # Hyperparameters
    learning_rate = 0.0003
    iterations = 1000
    # Insert new column for bias value
    data = np.insert(data, 0, 1, axis=1)
    # Apply gradient descent until number of iterations
    for k in range(iterations):
        # Update values until iteration ended
        thetas = step_gradient_descent(data, thetas, learning_rate)
    # Print results
    print("Thetas: ", thetas)
    print("Error: ", cost_function(thetas, data))

    # If data set is 2d, run certain steps
    if len(data[0]) == 3:
        plt.scatter(data[:, 1], data[:, 2])
        x_s = np.linspace(min(data[:, 1]), max(data[:, 1]))
        y_s = (thetas[1] * x_s) + thetas[0]
        plt.plot(x_s, y_s)
        plt.ylabel('some numbers')
        plt.show()

    # If dataset is 3d, do other things to plot
    elif len(data[0]) == 4:
        x_s = np.linspace(min(data[:, 1]), max(data[:, 1]))
        y_s = np.linspace(min(data[:, 2]), max(data[:, 2]))
        z_s = (thetas[2]*y_s) + (thetas[1] * x_s) + thetas[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 1], data[:, 2], data[:, 3])
        ax.plot(x_s, y_s, z_s)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)
    # Not valid
    else:
        print("Too many or not enough parameters")
