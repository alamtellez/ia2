# coding utf-8

# This is the implementation of logistic regression using gradient descent
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import csv


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
    return 1.0 / (1.0 + math.exp(-acum))

def split_columns(dataset, cols):
    """
        This function splits the dataset's Xs and Ys
    """
    return np.split(dataset, cols, axis=1)

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

def get_max_and_min(dataset):
    min_max = []
    for i in range(len(dataset[0])):
        values = []
        for j in range(len(dataset)):
            values.append(dataset[j][i])
        min_val = min(values)
        max_val = max(values)
        min_max.append([min_val, max_val])
    return min_max

def normalize_data(dataset, min_max):
    for i in range(len(dataset[0])):
        for j in range(len(dataset)):
            dataset[j][i] = (dataset[j][i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
    return dataset


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
    learning_rate = 0.03
    iterations = int(sys.argv[2])
    # data, y = split_y(data)
    # Normalize data
    min_max = get_max_and_min(data)
    data = normalize_data(data, min_max)
    # Insert new column for bias value
    data = np.insert(data, 0, 1, axis=1)
    
    # Apply gradient descent until number of iterations
    for k in range(iterations):
        # Update values until iteration ended
        thetas = step_gradient_descent(data, thetas, learning_rate)
    # # Print results
    print("Thetas: ", thetas)
    # print("Error: ", cost_function(thetas, data))
    file_test = 'test_titanic.csv'
    # Load dataset into numpy array
    test_1 = np.genfromtxt(file_test, delimiter=',')
    test = np.delete(test_1,[0], axis=1)
    min_max = get_max_and_min(test)
    test = normalize_data(test, min_max)
    test = np.insert(test, 0, 1, axis=1)
    with open('test_alam.csv', mode='w') as submission_test:
        res_w = csv.writer(submission_test, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        res_w.writerow(['PassengerId', 'Survived'])
        for i in range(len(test)):
            prediction = hypothesis(thetas, test[i])
            if (prediction > 0.5):
                res_w.writerow([int(test_1[i][0]), '1'])
            else:
                res_w.writerow([int(test_1[i][0]), '0'])