import numpy as np
from numpy.core.defchararray import array

## We take in a batch of inputs
inputs = [[1,2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5,2.7, 3.3, -0.8]]

#layer 1
weights = [[0.2,0.8,-0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27,0.17, 0.87]]

bias = [2.0, 3.0, 0.5]

# output_layer = []
# for i in range(len(weights)):
#   l = np.dot(inputs, weights[i]) + bias[i]  
#   output_layer.append(l)

# complicated process
output_layer1 = np.dot(inputs, np.array(weights).T) + bias
print(output_layer1)

# layer 2 (different no. of neurons)
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73,0.13]]

bias2 = [-1.0, 2, -0.5]

output_layer2 = np.dot(output_layer1, np.array(weights2).T) + bias2
print(output_layer2)
## inputs cannot change, but we can tweak the weights and bias.
# the way we tweak the W & B, using back propagation allows us to have a very good neural net.
