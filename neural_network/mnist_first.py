from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import sys
import math

np.random.seed(1)


def relu(x):
    return (x > 0) * x

def relu_der(output):
    return x > 0

def tanh(x):
	return np.tanh(x)


def tanh_der(output):
	return 1 - (output ** 2)


def softmax(x):
	temp = np.exp(x)
	return temp/np.sum(temp, axis=1, keepdims=True)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
	one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
	test_labels[i][l] = 1

alpha, hidden_nodes, iterations = (2, 100, 300)
pixels, labels = (28*28, 10)
batch_size = 100

weights_0_1 = 0.02*np.random.random((pixels, hidden_nodes)) - 0.01
weights_1_2 = 0.2*np.random.random((hidden_nodes, labels)) - 0.1


for j in range(iterations):
	count_correct = 0
	for i in range(int(len(images) / batch_size)):
		batch_start, batch_end = ((i * batch_size), ((i+1)*batch_size))
		layer_0 = images[batch_size*i:(i+1)*batch_size]
		layer_1 = tanh(np.dot(layer_0, weights_0_1))
		dropout_mask = np.random.randint(2, size=layer_1.shape)
		layer_1 *= dropout_mask * 2
		layer_2 = softmax(np.dot(layer_1, weights_1_2))

		for k in range(batch_size):
			count_correct += int(np.argmax(layer_2[k:k+1]) ==
			                     np.argmax(labels[batch_start+k:batch_start+k+1]))

		layer_2_delta = (labels[batch_start:batch_end] -
		                 layer_2) / (batch_size * layer_2.shape[0])
		layer_1_delta = layer_2_delta.dot(weights_1_2.T)
		layer_1_delta *= dropout_mask

		weights_1_2 += alpha * layer_2_delta.dot(layer_1.T)
		weights_0_1 += alpha * layer_1_delta.dot(layer_0.T)

	if j % 10 == 0:
		test_correct = 0

		for i in range(len(test_images)):
			layer_0 = test_images[i:i+1]
			layer_1 = layer_0.dot(weights_0_1)
			layer_2 = layer_1.dot(weights_1_2)

			test_correct += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
		train_acc = count_correct/float(len(images))
		test_acc = test_correct/float(len(test_images))
		print(f'Iteration {j}, Train Acc: {train_acc}, Test Acc : {test_acc}')
