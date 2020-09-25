import numpy as np
import pandas as pd

np.random.seed(21) # fixed seed for random distribution of weight vector

Threshold = 100
Pocket_Iter = 10
Classes, Features = 0, 0
TrainingSize, TestSize = 0, 0
Learning_Rate = 0.01

dataset = []
input_matrix = []
labels = []
weight_vec = []

# Reading training data
f = open("train.txt", "r")
lines = f.readlines()
f.close()
Features, Classes, TrainingSize = map(int, lines[0].split())

# init weight vector randomly
weight_vec = np.random.uniform(-1,1, (Features+1) )

for i in range(TrainingSize):
  data = lines[i + 1].split()
  # print(data)
  class_name = data[Features] # last element of data array
  labels.append(int(class_name))
  input_matrix.append(np.array(data[: Features]).astype(float))

  temp = []
  for j in range(Features):
    temp.append(float(data[j]))
  temp.append(int(data[Features]))
  dataset.append(temp)


def test(weight_vec, report = False):
  global Features, TrainingSize, TestSize, Classes

  f = open("test.txt", "r")
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
    input_vec = np.array(input_vec)
    input_vec = input_vec.reshape(Features+1,1)
    # dot_prod = np.dot(weight_vec[1:],inputs) + weight_vec[0]
    dot_prod = np.dot(weight_vec, input_vec)[0]
    predicted = -1
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


def train_basic_perceptron():
  """
  Basic Perceptron
  """
  global weight_vec, dataset, Threshold, TrainingSize, Learning_Rate, Features
  
  for i in range(Threshold):
    print("Iter {} => {}".format(i))
    misclassified = []
    delX = []
    for j in range(TrainingSize):
        input_vec = np.array(dataset[j])
        class_name = input_vec[Features]
        input_vec[Features] = 1
        input_vec = input_vec.reshape(Features+1,1)
        dot_product = np.dot(weight_vec,input_vec)[0]
        if (class_name == 2 and dot_product > 0):
            misclassified.append(input_vec)
            delX.append(1)
        elif (class_name ==1 and dot_product < 0):
            misclassified.append(input_vec)
            delX.append(-1)
        else:
            pass
    
    sum = np.zeros(Features+1)   
    for j in range(len(misclassified)):
        sum += delX[j] * misclassified[j].transpose()[0]
    
    weight_vec = weight_vec - Learning_Rate * sum
    if len(misclassified) == 0:
      break        
  
def activation_func(inputs, weights):
  summation = np.dot(inputs, weights[1:]) + weights[0]
  if summation > 0: # Belongs to W1
    activation = 1
  else: # Belongs to W2
    activation = 0            
  return activation

def train_pocket():
  """
  Pocket ALgorithm
  """
  global weight_vec, dataset, Threshold, TrainingSize, Learning_Rate, Features

  # train_basic_perceptron()
  w = np.copy(weight_vec)
  misclassified = test(w)
  print(misclassified)
  if misclassified == 0:
    return
  # count = TestSize
  count = misclassified
  for iter in range(Threshold):
    print("Pocket Iteration {} ==========".format(iter))
    train_basic_perceptron() # W is updated according to basic perceptron algo
    misclassified = test(weight_vec)
    print("Misclassified {}".format(misclassified))
    
    if misclassified < count:
      count = misclassified
      w = np.copy(weight_vec)

    if misclassified == 0:
        break
  
  weight_vec = np.copy(w)
  

def train_reward_punish():
  """
  Reward Punish Algorithm
  """
  global weight_vec, dataset, Threshold, TrainingSize, Learning_Rate, Features
  
  for i in range(Threshold):
    print("Iteration {} ==========".format(i))
    misclassified = 0
    for j in range(TrainingSize):
      input_vec = np.array(dataset[j])
      class_name = input_vec[Features]
      input_vec[Features] = 1
      input_vec = input_vec.reshape((Features + 1) ,1)
      dot_product = np.dot(weight_vec,input_vec)[0]
      if class_name == 1 and dot_product <= 0: # class1 misclassified.
        # Wi = Wi + n*d*input -> d = 1
        misclassified += 1
        weight_vec = weight_vec + Learning_Rate * input_vec

      elif class_name == 2 and dot_product > 0: # class2 misclassified
        misclassified += 1
        weight_vec = weight_vec - Learning_Rate * input_vec
      else:
          pass
    print(misclassified)
    if misclassified == 0:
        break        
  

if __name__ == "__main__":
  
  ## read_training_dataset()
  # train_basic_perceptron()
  # train_pocket()
  train_reward_punish()

  print('Final Weight : ', weight_vec)
  test(weight_vec, True)
