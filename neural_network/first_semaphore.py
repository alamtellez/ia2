import numpy as np

np.random.seed(1)

def relu(x):
  return (x > 0) * x

def relu_der(x):
  return x>0

street_lights = np.array([[1,0,1],
                          [0,1,1],
                          [0,0,1],
                          [1,1,1]])
hidden_nodes = 4
alpha = 0.2
true = np.array([[1,1,0,0]]).T
weights_0_1 = 2*np.random.random((3,hidden_nodes)) -1
weights_1_2 = 2*np.random.random((hidden_nodes,1)) -1

for iteration in range(60):
  layer_2_error = 0
  for i in range(len(street_lights)):
    layer_0 = street_lights[i:i+1]
    layer_1 = relu(np.dot(layer_0, weights_0_1))
    layer_2 = np.dot(layer_1, weights_1_2)

    layer_2_error += np.sum((layer_2 - true[i:i+1])**2)

    delta_2 = layer_2 - true[i:i+1]
    delta_1 = delta_2.dot(weights_1_2.T)*relu_der(layer_1)
    weights_1_2 -= alpha * layer_1.T.dot(delta_2)
    weights_0_1 -= alpha * layer_0.T.dot(delta_1)

  if iteration % 9 == 0:

    print("Iter: ", iteration, "Error: ", layer_2_error)

option = input("Enter \'s\' to exit or anything else to continue: ")
while(option != 's'):

  
  inp = []
  for light in range(1,4):
    inp.append(int(input(f'Introduce light {light}: ')))
  
  inpnp = np.array(inp)
  layer_1 = relu(np.dot(inpnp, weights_0_1))
  layer_2 = np.dot(layer_1, weights_1_2)
  print(layer_2)
  option = input("Enter \'s\' to exit or anything else to continue: ")
