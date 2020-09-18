import numpy as np
import math

number_of_features = 0
number_of_classes = 0
dataset_size = 0
Object_Dictionary = {}

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
        global number_of_features
        for i in range(number_of_features):
            mu = self.get_mean(i)
            self.mean.append(mu)

    def calc_standard_deviation(self):
        global number_of_features
        for i in range(number_of_features):
            sigma = self.get_standard_deviation(i)
            self.sd.append(sigma)

    def covariance_mat(self):
        pass
    
    def print_features(self):
        print(self.features)
        print("----------------------")


def read_dataset():
    global number_of_features, number_of_classes, dataset_size, Object_Dictionary

    f = open("./train.txt", "r")
    lines = f.readlines()
    f.close()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].split())

    for i in range(dataset_size):
        data = lines[i + 1].split()

        class_name = data[number_of_features] # last element of data array

        if class_name not in Object_Dictionary: # saving features for each distinct classes
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(data[: number_of_features])
    
    print(Object_Dictionary)


if __name__ == "__main__":
    read_dataset()