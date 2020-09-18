import numpy as np
import math

Features = 0
Classes = 0
TrainingSize = 0
Object_Dictionary = {}
correct = 0

class Object:
    def __init__(self, class_name):
        self.class_name = class_name
        self.features = []
        self.mean = []
        self.sd = []
        self.transpose = []
        self.covariance = []
        self.determinent = 1.0
        self.inv_covar = []

    def get_mean(self, feature_no):
        sz = len(self.features)
        temp = np.array([i[feature_no] for i in self.features]).astype(float)

        return np.mean(temp)

    def get_standard_deviation(self, feature_no):
        sz = len(self.features)
        temp = np.array([i[feature_no] for i in self.features]).astype(float)

        return np.std(temp)

    def calc_mean(self):
        global Features
        for i in range(Features):
            mu = self.get_mean(i)
            self.mean.append(mu)

    def calc_standard_deviation(self):
        global Features
        for i in range(Features):
            sigma = self.get_standard_deviation(i)
            self.sd.append(sigma)

    def covariance_mat(self):
        pass
    
    def print_features(self):
        print(len(self.features))
        print(self.features)
        print("----------------------")


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

        if class_name not in Object_Dictionary: # saving features for each distinct classes
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(data[: Features])

    # print(Object_Dictionary)


def train():
    global Classes, Features, Object_Dictionary, TrainingSize
  # 1. Initialize w' with random number length of (number of features + 1)

  # 2. let l = learning rate set to a fixed value.

  # 3. set t = 0

  # 4. while misclassified_set is not empty: misclassified <- {} for k <- len(training_data): vector_class = get_class(training_data[k]) discreminent = dot_product(w', training_data[k]) if vector_class is misclassified: misclassified_set <- training_data[k] cost = cost_function(misclassified_set) w' = w - l*cost
  # Here cost_function = summation of all misclassified vectors with their respective

    for key in Object_Dictionary: # obtain mean and std. store in the object itself
        obj = Object_Dictionary[key]
        obj.calc_mean()
        obj.calc_standard_deviation()
        obj.print_features()


def test_accuracy():
    global Features, Object_Dictionary, TrainingSize, correct

    f = open("./test.txt", "r")
    lines = f.readlines()
    f.close()

    # wr = open("Report_coding.txt", "w")

    sample_count = 0
    for line in lines:
        sample_count += 1

        test_vector = []
        temp = line.rstrip()
        temp = temp.split()

        for i in range(len(temp)):
            t = temp[i].strip()
            if len(t):
                test_vector.append(float(t))

        class_name = temp[Features]
        print(test_vector,class_name)
    
    # acc = (correct / float(TrainingSize)) * 100.0
    # print("accuracy: {} / {} = {}%".format(correct,TrainingSize,acc))
    # wr.close()


if __name__ == "__main__":
    read_dataset()
    train()
    test_accuracy()