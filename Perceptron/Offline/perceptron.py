import numpy as np
import pandas as pd

np.random.seed(21) # fixed seed for random distribution of weight vector

Threshold = 1000
Classes = 0,
Features = 0
TrainingSize = 0

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

weight_vec = np.random.uniform(-1,1, (Features+1) )

def test(weight_vec):
  global Features, TrainingSize, Classes

  f = open("test.txt", "r")
  lines = f.readlines()
  f.close()

  # wr = open("Report_coding.txt", "w")

  test_dataset = []
  for line in lines:
    temp = line.split()
    data = []
    for i in range(Features):
        data.append(float(temp[i]))
    data.append(int(temp[Features]))
    test_dataset.append(data)

  count_accurate = 0
  for data in test_dataset:
      x = np.array(data)
      group = x[Features]
      x[Features] = 1
      x = np.array(x)
      x = x.reshape(Features+1,1)
      dot_product = np.dot(weight_vec,x)[0]
      predicted = -1
      if dot_product >= 0:
          predicted = 1
      else:
          predicted = 2

      if predicted==group:
          count_accurate += 1

  print("Accuracy :",float((count_accurate/len(test_dataset))*100))
  
  return float(count_accurate/len(test_dataset))


def train_basic_perceptron():
  """
  Basic Perceptron
  """
  global weight_vec, dataset, Threshold, TrainingSize, Classes, Features
  learning_rate = 0.025
  t = 0
  for i in range(Threshold):
      Y = []
      arr_dx = []
      for j in range(TrainingSize):
          x = np.array(dataset[j])
          group = x[Features]
          x[Features] = 1
          x = x.reshape(Features+1,1)
          dot_product = np.dot(weight_vec,x)[0]
          if(group == 2 and dot_product>0):
              Y.append(x)
              arr_dx.append(1)
          elif(group ==1 and dot_product<0):
              Y.append(x)
              arr_dx.append(-1)
          else:
              pass
      
      sum = np.zeros(Features+1)
      
      for j in range(len(Y)):
          sum += arr_dx[j]*Y[j].transpose()[0]
      
      
      weight_vec = weight_vec - learning_rate*sum
      print("Iter {} => {}".format(i,"---"))
      if len(Y) == 0:
          break        
  

def train_reward_punish():
  """
  Reward punishment perceptron
  """
  

if __name__ == "__main__":
  
  ## read_training_dataset()
  train_basic_perceptron()
  print('Final Weight : ', weight_vec)
  test(weight_vec)
