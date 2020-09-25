import numpy as np
import pandas as pd

MAXEPOCH = 1000

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

np.random.seed(41)
w = np.random.uniform(-10,10,numFeature+1)

learning_rate = 0.025
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
        
print('Final Weight : ', w)
test(test_dataset,w)