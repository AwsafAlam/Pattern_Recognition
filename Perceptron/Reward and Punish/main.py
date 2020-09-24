import numpy as np
import math
from perceptron import Perceptron

Features = 0
Classes = 0
TrainingSize = 0
Object_Dictionary = {}
correct = 0

weight_vec = []

input_matrix = []
labels = []

def read_dataset():
    global Features, Classes, TrainingSize, Object_Dictionary

    f = open("./train.txt", "r")
    lines = f.readlines()
    f.close()

    Features, Classes, TrainingSize = map(int, lines[0].split())
    for i in range(TrainingSize):
        data = lines[i + 1].split()
        # print(data)
        class_name = data[Features] # last element of data array
        labels.append(int(class_name))
        # if class_name not in Object_Dictionary: # saving features for each distinct classes
        #     Object_Dictionary[class_name] = Object(class_name)

        # Object_Dictionary[class_name].features.append(data[: Features])
        input_matrix.append(np.array(data[: Features]).astype(float))

    # print(Object_Dictionary)


def test_accuracy(perceptron):
    global Features, Object_Dictionary, TrainingSize, correct

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
        # print("Class: {} [Sample - {}] -------\n".format(class_name, sample_count))
        inputs = np.array(temp[: Features]).astype(float)
        output = perceptron.predict(inputs)
        if output == class_name or (output == 0 and class_name == 2):
          correct += 1
        else:
          #incorrect
          print("[Sample - {}] {} {} {}\n".format(sample_count, inputs, class_name, output))
    
    acc = (correct / float(TrainingSize)) * 100.0
    print("accuracy: {} / {} = {}%".format(correct,TrainingSize,acc))
    # wr.close()


if __name__ == "__main__":
    read_dataset()
    # perceptron = Perceptron(4, threshold=10, learning_rate=1)
    perceptron = Perceptron(Features)
    perceptron.train(input_matrix, labels)
    print("Weight Vectors")
    perceptron.print_weight_vec()

    # inputs = np.array([2.09894733, 3.927346913, 5.126590034, 7.219977249]).astype(float)
    # output = perceptron.predict(inputs)
    # print(output) # expected class = 1
    
    test_accuracy(perceptron)