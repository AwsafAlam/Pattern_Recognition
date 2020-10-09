import numpy as np
from math import exp
from random import seed
from random import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
  for epoch in range(n_epoch):
    sum_error = 0
    for row in train:
      outputs = forward_propagate(network, row)

      expected = [0 for i in range(n_outputs)]
      # print(expected, row[-1])
      expected[row[-1] - 1] = 1
      sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
      backward_propagate_error(network, expected)
      update_weights(network, row, l_rate)
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)

## New code
f = open("trainNN.txt", "r")
lines = f.readlines()
f.close()
# Features, Classes, TrainingSize = map(int, lines[0].split())
TrainingSize = len(lines)
Features = len(lines[0].split()) - 1
labels, input_matrix, class_labels = [], [], []
dataset = []
# init weight vector randomly
weight_vec = np.random.uniform(-1,1, (Features+1) )

for i in range(TrainingSize):
  data = lines[i].split()
  # print(data)
  class_name = data[Features] # last element of data array
  labels.append(int(class_name))
  # input_matrix.append(np.array(data[: Features]).astype(float))
  input_matrix.append([float(item) for item in data[: Features]])
  if class_name not in class_labels: # saving features for each distinct classes
      class_labels.append(class_name)
  
  temp = []
  for j in range(Features):
    temp.append(float(data[j]))
  temp.append(int(data[Features]))
  dataset.append(temp)

# input_matrix = np.array(input_matrix)

# dataset = [[2.7810836,2.550537003,2.550537003,0],
# 	[1.465489372,2.362125076,2.550537003,0],
# 	[3.396561688,4.400293529,2.550537003,0],
# 	[1.38807019,1.850220317,2.550537003,0],
# 	[3.06407232,3.005305973,2.550537003,0],
# 	[7.627531214,2.759262235,2.550537003,1],
# 	[5.332441248,2.088626775,2.550537003,1],
# 	[6.922596716,1.77106367,2.550537003,1],
# 	[8.675418651,-0.242068655,2.550537003,1],
# 	[7.673756466,3.508563011,2.550537003,1]]

print(dataset)
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
print(n_inputs, n_outputs)
network = initialize_network(n_inputs, Features, n_outputs)
# print(network)
train_network(network, dataset, 0.01, 200, n_outputs)
for layer in network:
	print(layer)


#### -================================================================

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Test making predictions with the network
# testSet = [[2.7810836,2.550537003,2.550537003,0],
# 	[1.465489372,2.362125076,2.550537003,0],
# 	[3.396561688,4.400293529,2.550537003,0],
# 	[1.38807019,1.850220317,2.550537003,0],
# 	[3.06407232,3.005305973,2.550537003,0],
# 	[7.627531214,2.759262235,2.550537003,1],
# 	[5.332441248,2.088626775,2.550537003,1],
# 	[6.922596716,1.77106367,2.550537003,1],
# 	[8.675418651,-0.242068655,2.550537003,1],
# 	[7.673756466,3.508563011,2.550537003,1]]
testSet = [[31.90940103,	48.05728779,	63.6388212,	80.05600141,	4],
[24.83063375,	35.47480561,	47.82455924,	59.29045973,	3],
[16.36396876,	23.39697268,	31.57090289,	40.71147665,	2],
[8.411710149,	11.62201491,	16.50827155,	19.67045197,	1],
[32.05301368,	48.91718339,	65.12858166,	79.77842406,	4],
[31.98499451,	48.51954534,	64.30375593,	79.88348911,	4],
[31.89526512,	47.60798576,	62.86400978,	79.92167416,	4],
[23.4010657,	35.88602978,	49.2341497,	60.01694302,	3],
[8.754763846,	11.44464276,	16.63778633,	19.16319289,	1],
[7.860036888,	12.14537674,	16.31005136,	19.93786884,	1],
[7.458888116,	11.39116189,	15.63892132,	19.49041692,	1],
[7.710456931,	11.52473815,	16.6210287,	20.13495786,	1],
[24.14173358,	35.58050801,	48.1384058,	60.16150608,	3],
[7.531975424,	11.40541313,	15.03253226,	20.43639664,	1],
[23.49392584,	36.02336894,	48.04155546,	60.04560814,	3]]

for row in testSet:
	prediction = predict(network, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))