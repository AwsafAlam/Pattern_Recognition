import numpy as np
import pandas as pd

MAXEPOCH = 500

file = open("Train.txt")

lines = file.readlines()

numClass, numFeature, datasetLen = 0, 0, 0

dataset = []
count = 0
for line in lines:
    if count == 0:
        var = line.split()
        numFeature = int(var[0])
        numClass   = int(var[1])
        datasetLen = int(var[2])
    else:
        var = line.split()
        data = []
        for i in range(numFeature):
          data.append(float(var[i]))
        data.append(int(var[numFeature]))
        dataset.append(data)
 
    count += 1

file = open("Test.txt")

lines = file.readlines()
test_dataset = []
np.random.seed(21)
wp = np.random.uniform(-1,1,numFeature+1)
w = wp

for line in lines:
    var = line.split()
    data = []
    for i in range(numFeature):
        data.append(float(var[i]))
    data.append(int(var[numFeature]))
    test_dataset.append(data)

def test(dataset,w):
    count_accurate = 0
    for data in dataset:
        x = np.array(data)
        group = x[numFeature]
        x[numFeature] = 1
        x = np.array(x)
        x = x.reshape(numFeature+1,1)
        dot_product = np.dot(w,x)[0]
        predicted = -1
        if dot_product >= 0:
            predicted = 1
        else:
            predicted = 2

        if predicted==group:
            count_accurate += 1

    print("Accuracy :",float((count_accurate/len(dataset))*100))
    
    return float(count_accurate/len(dataset))


def train_basic_perceptron():
  """
  docstring
  """
  global w, dataset, MAXEPOCH, datasetLen
  learning_rate = 0.01
  t = 0

  for i in range(MAXEPOCH):
      Y = []
      arr_dx = []
      for j in range(datasetLen):
          x = np.array(dataset[j])
          group = x[numFeature]
          x[numFeature] = 1
          x = x.reshape(numFeature+1,1)
          dot_product = np.dot(w,x)[0]
          if(group == 2 and dot_product>0):
              Y.append(x)
              arr_dx.append(1)
          elif(group ==1 and dot_product<0):
              Y.append(x)
              arr_dx.append(-1)
          else:
              pass
      
      sum = np.zeros(numFeature+1)
      
      for j in range(len(Y)):
          sum += arr_dx[j]*Y[j].transpose()[0]
      
      
      w = w - learning_rate*sum
      print("Iter {} => {}".format(i,"---"))
      if len(Y) == 0:
          break        
  
def train_pocket():
  """
  docstring
  """
  global w, wp, dataset, MAXEPOCH, datasetLen
  learning_rate = 0.01
  t = 0
  train_basic_perceptron()
	# test(dataset,w)
  misclassification = test(dataset,w) * len(dataset)
  ##
  for i in range(MAXEPOCH):
    count = 0 
    for i in range(datasetLen):
        original = dataset[i]
        x = np.array(dataset[i])
        group = x[numFeature]
        x[numFeature] = 1
        x = x.reshape(numFeature+1,1)
        dot_product = np.dot(w,x)[0]
        if dot_product<=0 and group == 1:
            count += 1
            w = w + np.array(original)
        elif dot_product >0 and group == 2:
            w = w - np.array(original)
            count += 1
        else:
            pass
    
    if count< misclassification:
        misclassification = count
        wp = w
    
    if count == 0:
        break
	

if __name__ == "__main__":
  train_pocket()
  print('Final Weight : ', w)
  print(wp)
  test(test_dataset,wp)