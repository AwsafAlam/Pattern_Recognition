import numpy as np

np.random.seed(21) # fixed seed for random distribution of weight vector

MAXEPOCH = 1000
Classes, Features, Layers = 0, 0, 0
TrainingSize, TestSize = 0, 0
Learning_Rate = 0.01
dataset = []
input_matrix = []
labels, target_output = [], []
class_labels = []
structure, weight_vec, network = [], [], []

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

input_matrix = np.array(input_matrix)

# Featurewise normalization
# for i in range(Features):
#   input_matrix[:, i] = (input_matrix[: , i] - np.mean(input_matrix[:, i]))/np.std(input_matrix[:, i])

# Initializing no. of classes
Classes = len(class_labels)
labels = np.array(labels)
labels = labels.reshape(TrainingSize,1)
# print(labels.shape)

for i in range(len(labels)):
  temp = np.zeros(Classes)
  temp = list(temp)
  temp[int(labels[i]) - 1] = 1.0
  target_output.append(temp)

target_output = np.array(target_output)
# print(target_output)
# ==============================================

class Layer:
  """
  docstring
  """
  def __init__(self, n_inputs, n_neurons):
    """
    docstring
    """
    # Gausian distribution bounded around 0
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    # self.weights = np.random.uniform(-1,1,n_inputs, n_neurons )
    
    self.biases = np.zeros((1, n_neurons))
    self.errors = np.zeros((1, n_neurons))
    self.delta = np.zeros((1, n_neurons))

  
  def forward(self, inputs, func):
    """
    docstring
    """
    self.output = np.dot(inputs, self.weights) + self.biases
    self.output = np.array(self.output)
    self.output = self.activation_func(self.output, func)
    self.output = np.array(self.output)

  def backward(self, outputs):
    self.delta = self.errors * self.differential(outputs,0)

  def activation_func(self, X, func = 0):
    if func == 0:
      return 1.0 / (1.0 + np.exp(-X)) # sigmoid
    elif func == 1:
      return np.tanh(X) #tanh
    elif func == 2:
      return np.maximum(0, X) #ReLu

  def differential(self, X, func = 0):
    if func == 0:
        return (X * ( 1 - X ))
    elif func == 1:
        return (1 - np.square(X))
    elif func == 2:
        return (1.0)*(X>0)+(0.1)*(X<0)

def read_network_structure():
  """
  docstring
  """
  global Layers, structure, Features, class_labels
  f = open("structureNN.txt", "r")
  lines = f.readlines()
  f.close()
  Layers = int(lines[0])
  print("Constructing Neural Network with {} Hidden Layers".format(Layers))
  structure.append(Features)
  for i in range(Layers):
    structure.append(int(lines[i+1].split()[1]))

  structure.append(len(class_labels))

  # Creating the Network
  for i in range(len(structure) - 1):  # input layer is layer0
    layer = Layer(structure[i], structure[i+1])
    network.append(layer)

  return structure

def create_network():
  """
  docstring
  """
  global Layers, structure, Features, class_labels
  print("Enter No. of Layers")
  Layers = input()
  print("Enter No. of Neurons/Layer")
  neurons_per_layer = input()
  print("Constructing Neural Network with {} Hidden Layers".format(Layers))
  structure.append(Features)
  for i in range(Layers):
    structure.append(int(neurons_per_layer))

  structure.append(len(class_labels))

  # Creating the Network
  for i in range(len(structure) - 1):  # input layer is layer0
    layer = Layer(structure[i], structure[i+1])
    network.append(layer)

  return structure

def back_propagation():
  """
  Back Propagation
  """
  global network, TrainingSize, target_output
  i = len(network) - 1
  for i in reversed(range(len(network))):
    # print(network[i].output)
    errors = []
    if i != len(network) - 1:
      network[i].errors = np.dot(network[i + 1].delta, network[i + 1].weights.T)
      network[i].backward(network[i].output)
    else:
      network[i].errors = (0.5)*(np.square((target_output - network[i].output)))
      network[i].backward(network[i].output)

    network[i].weights -= Learning_Rate * np.dot(network[i-1].output.T , network[i].delta)

def forward_pass(X):
  """
  Forward pass throught Network
  """
  global network
  # Forward Propagation
  for i in range(len(network)):
    if i == 0:
      network[i].forward(X, 0)
      
    else:
      network[i].forward(network[i-1].output, 0)

  return network[ len(network) - 1].output

def train():
  """
  Training the NN
  """
  global input_matrix, structure, network, class_labels
  for epoch in range(MAXEPOCH):
    
    # Forward Propagation
    for i in range(len(network)):
      if i == 0:
        network[i].forward(input_matrix, 0)
        
      else:
        network[i].forward(network[i-1].output, 0)
        
    # print(network[Layers].output)
    # Calculate Mean Squared Error
    output_prediction = network[len(network)-1].output
    # print(output_prediction - target_output)
    error_out = (0.5)*(np.square((target_output - output_prediction)))
    cost = error_out.sum()
    # print(error_out.sum())

    back_propagation()
          
    # Print the cost every 100 iterations 
    if epoch % 100 == 0: 
      print ("Cost after iteration {}: {}".format(epoch, cost)) 


def test(report = False):
  global Features, TrainingSize, TestSize, Classes
  f = open("testNN.txt", "r")
  lines = f.readlines()
  f.close()

  # wr = open("Report.txt", "w")
  sample_count, correct = 0 , 0

  for line in lines:
    sample_count += 1
    temp = line.rstrip()
    temp = temp.split()
    class_name = int(temp[Features])
    inputs = np.array(temp[: Features]).astype(float)
      
    # Run Forward pass
    output = forward_pass(inputs)
    predicted = np.argmax(output, axis=1)[0] + 1
    # print(output)
    # print(predicted)
    
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



if __name__ == "__main__":
  print("Enter Network Structure :\n1-> File\n2-> Console")
  q = input()
  if q == 1:
    read_network_structure()
  else:
    create_network()  

  print("Network structure :")
  print(structure)
  train()
  test(True)

  