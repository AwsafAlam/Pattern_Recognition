import numpy as np

inputs = [1,2,3]
weights = [0.2,0.8,-0.5]

bias = 2.0

output = np.dot(inputs, weights) + bias

print(output)