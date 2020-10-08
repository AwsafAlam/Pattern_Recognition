# single layer network

# Add a Layer class
# Create an instance for eah layer, and take the structure defined in the text file

# weights will be hard coded for now

# There will be 4 activation algo.
# 1. Sigmoid 2, unit step func. 3. RELU 4. softmax 5. study other literature

# create a forward pass, 64 samples batch, and train the network
# repeat until all the training samples are exhausted.

# test using the test file and calculate the accuracy for hard-coded weights and bias.

# --------- 

# calculate the error rate, and based on that, implement the gradient descent algo.
# using the gradient descent, adjust the weights and bias to get maximum fit of data.
# ==========================================================

import numpy as np

np.random.seed(21) # fixed seed for random distribution of weight vector

Threshold = 100
Classes, Features = 0, 0
TrainingSize, TestSize = 0, 0
Learning_Rate = 0.01

dataset = []
input_matrix = []
labels = []
class_labels = []
weight_vec = []

# ## We take in a batch of inputs
# X = [[1,2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5,2.7, 3.3, -0.8]]
# Reading training data
f = open("trainNN.txt", "r")
lines = f.readlines()
f.close()
# Features, Classes, TrainingSize = map(int, lines[0].split())
TrainingSize = len(lines)
Features = len(lines[0].split()) - 1

# init weight vector randomly
weight_vec = np.random.uniform(-1,1, (Features+1) )

for i in range(TrainingSize):
  data = lines[i].split()
  # print(data)
  class_name = data[Features] # last element of data array
  labels.append(int(class_name))
  input_matrix.append(np.array(data[: Features]).astype(float))
  if class_name not in class_labels: # saving features for each distinct classes
      class_labels.append(class_name)
        
  temp = []
  for j in range(Features):
    temp.append(float(data[j]))
  temp.append(int(data[Features]))
  dataset.append(temp)

# Initializing no. of classes
Classes = len(class_labels)

def test(weight_vec, report = False):
  global Features, TrainingSize, TestSize, Classes
  # TODO: needs changing
  f = open("testNN.txt", "r")
  lines = f.readlines()
  f.close()

  # wr = open("Report.txt", "w")
  test_dataset = []
  sample_count, correct = 0 , 0

  for line in lines:
    sample_count += 1
    temp = line.rstrip()
    temp = temp.split()
    class_name = int(temp[Features])
    inputs = np.array(temp[: Features]).astype(float)
      
    data = []
    for i in range(Features):
        data.append(float(temp[i]))
    data.append(int(temp[Features]))

    input_vec = np.array(data)
    input_vec[Features] = 1
    # input_vec = np.array(input_vec)
    input_vec = input_vec.reshape((Features+1),1)
    # dot_prod = np.dot(weight_vec[1:],inputs) + weight_vec[0]
    dot_prod = np.dot(weight_vec, input_vec)[0]
    if dot_prod > 0:
        predicted = 1
    else:
        predicted = 2

    if predicted == class_name:
        correct += 1
    else:
      if report:
        print("[Sample-{}] {} - {} - {}\n".format(sample_count, inputs, class_name, predicted))
      pass

  TestSize = sample_count
  print("Accuracy :",float((correct/float(sample_count))*100))
  
  # returns no. of misclassified
  return (sample_count - correct)


class Layer_dense:
  """
  docstring
  """
  def __init__(self, n_inputs, n_neurons):
    """
    docstring
    """
    # Gausian distribution bounded around 0
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  
  def forward(self, inputs):
    """
    docstring
    """
    self.output = np.dot(inputs, self.weights) + self.biases

# 4 features, and 5 neurons
layer1 = Layer_dense(Features, 5)

# 5 putputs of l1, and 2 neurons
layer2 = Layer_dense(5, 2)

layer1.forward(input_matrix)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
