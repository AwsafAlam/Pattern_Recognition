import numpy as np

inputs = [1,2,3, 2.5]
weights = [[0.2,0.8,-0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27,0.17, 0.87]]

bias = [2.0, 3.0, 0.5]

output_layer = []
for i in range(len(weights)):
  l = np.dot(inputs, weights[i]) + bias[i]  
  output_layer.append(l)

## inputs cannot change, but we can tweak the weights and bias.
# the way we tweak the W & B, using back propagation allows us to have a very good neural net.

print(output_layer)