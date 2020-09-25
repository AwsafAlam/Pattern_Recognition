import numpy as np
import pandas as pd
import math
from perceptron import Perceptron

Features = 0
Classes = 0
TrainingSize = 0
correct = 0

dataset = []
input_matrix = []
labels = []

def read_dataset():
    global Features, Classes, TrainingSize, dataset, input_matrix, labels

    f = open("./train.txt", "r")
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


def test_accuracy(perceptron):
    global Features, TrainingSize, correct

    f = open("./test.txt", "r")
    lines = f.readlines()
    f.close()

    # wr = open("Report_coding.txt", "w")

    sample_count = 0
    for line in lines:
      sample_count += 1

      temp = line.rstrip()
      temp = temp.split()
      # for i in range(len(temp)):
      #     t = temp[i].strip()
      #     if len(t):
      #         test_vector.append(float(t))

      class_name = int(temp[Features])
      inputs = np.array(temp[: Features]).astype(float)
      output = perceptron.predict(inputs)
      
      if output == class_name:
        correct += 1
      else:
        #incorrect
        # print("[Sample - {}] {} {} {}\n".format(sample_count, inputs, class_name, output))
        pass
    
    acc = (correct / float(TrainingSize)) * 100.0
    print("accuracy: {} / {} = {}%".format(correct,TrainingSize,acc))
    # wr.close()


def test(w):
    global Features, dataset
    count_accurate = 0
    sample_count = 0
    for data in dataset:
        sample_count += 1
        x = np.array(data)
        class_name = x[Features]
        x[Features] = 1
        x = np.array(x)
        x = x.reshape(Features+1,1)
        dot_product = np.dot(w,x)[0]
        predicted = -1
        if dot_product >= 0:
            predicted = 1
        else:
            predicted = 2

        if predicted==class_name:
            count_accurate += 1
        else:
          print("[Sample - {}] {} {} {}\n".format(sample_count, x, class_name, predicted))

    print("Accuracy :",float((count_accurate/len(dataset))*100))
    
    return float(count_accurate/len(dataset))


if __name__ == "__main__":
    read_dataset()
    # train_basic_peceptron()
    # perceptron = Perceptron(4, threshold=10, learning_rate=1)
    perceptron = Perceptron(TrainingSize,Features)
    perceptron.train(input_matrix, labels, dataset)
    perceptron.print_weight_vec()

    # inputs = np.array([2.09894733, 3.927346913, 5.126590034, 7.219977249]).astype(float)
    # output = perceptron.predict(inputs)
    # print(output) # expected class = 1
    
    # test_accuracy(perceptron)
    test(perceptron.getWeight())