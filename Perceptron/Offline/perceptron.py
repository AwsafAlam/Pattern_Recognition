import numpy as np
import pandas as pd

np.random.seed(21) # fixed seed for random distribution of weight vector

Threshold = 100
Classes, Features = 0, 0
TrainingSize = 0
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


def test(weight_vec):
  global Features, TrainingSize, Classes

  f = open("test.txt", "r")
  lines = f.readlines()
  f.close()

  # wr = open("Report_coding.txt", "w")
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

    x = np.array(data)
    x[Features] = 1
    x = np.array(x)
    x = x.reshape(Features+1,1)
    # dot_prod = np.dot(weight_vec[1:],inputs) + weight_vec[0]
    dot_prod = np.dot(weight_vec, x)[0]
    predicted = -1
    if dot_prod > 0:
        predicted = 1
    else:
        predicted = 2

    if predicted == class_name:
        correct += 1
    else:
      print("[Sample-{}] {} - {} - {}\n".format(sample_count, inputs, class_name, predicted))
      # pass

  print("Accuracy :",float((correct/float(sample_count))*100))
  
  return float(correct/sample_count)


def train_basic_perceptron():
  """
  Basic Perceptron
  """
  global weight_vec, dataset, Threshold, TrainingSize, Learning_Rate, Features
  weight_vec = np.random.uniform(-1,1, (Features+1) )
  
  for i in range(Threshold):
    Y = []
    arr_dx = []
    for j in range(TrainingSize):
        x = np.array(dataset[j])
        class_name = x[Features]
        x[Features] = 1
        x = x.reshape(Features+1,1)
        dot_product = np.dot(weight_vec,x)[0]
        if (class_name == 2 and dot_product > 0):
            Y.append(x)
            arr_dx.append(1)
        elif (class_name ==1 and dot_product < 0):
            Y.append(x)
            arr_dx.append(-1)
        else:
            pass
    
    sum = np.zeros(Features+1)
    
    for j in range(len(Y)):
        sum += arr_dx[j]*Y[j].transpose()[0]
    
    
    weight_vec = weight_vec - Learning_Rate * sum
    print("Iter {} => {}".format(i,"---"))
    if len(Y) == 0:
        break        
  

def activation_func(inputs, weights):
  summation = np.dot(inputs, weights[1:]) + weights[0]
  if summation > 0: # Belongs to W1
    activation = 1
  else: # Belongs to W2
    activation = 0            
  return activation



if __name__ == "__main__":
  
  ## read_training_dataset()
  train_basic_perceptron()

  print('Final Weight : ', weight_vec)
  test(weight_vec)
