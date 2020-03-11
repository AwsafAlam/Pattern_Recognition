import numpy as np
import math

datasetSize = 0
noOfFeatures = 0
noOfClasses = 0
Processed_DataSet = {}

class DataStore:
    def __init__(self, cls):
        self.cls = cls
        self.features = []
        self.mean = []
        self.sd = []
        self.transpose = []
        self.covariance = []
        self.determinent = 1.0
        self.inv_covar = []

    def mean(self, noOfFeatures):
        sz = len(self.features)
        temp = np.array([i[noOfFeatures] for i in self.features]).astype(float)
        return np.mean(temp)

    def classwiseMean(self):
        global noOfFeatures
        for i in range(noOfFeatures):
            mu = self.get_mean(i)
            self.mean.append(mu)

    def std(self, noOfFeatures):
        sz = len(self.features)
        temp = np.array([i[noOfFeatures] for i in self.features]).astype(float)

        return np.std(temp)

    def classwiseStd(self):
        global noOfFeatures
        for i in range(noOfFeatures):
            sigma = self.get_standard_deviation(i)
            self.sd.append(sigma)

    def print_features(self):
        print(self.features)
        print("----------------------")


def fileRead():
    f = open("Train.txt")
    test_data = f.readlines()
    f.close()
    noOfFeatures , noOfClasses , datasetSize = map(int, test_data[0].split())
    for line in test_data[1:]:
        print(line)
        data = line.split()
        givenCls = data[noOfFeatures] # last element of array
        if givenCls not in Processed_DataSet: # saving features for each distinct classes
            Processed_DataSet[givenCls] = DataStore(givenCls)

        Processed_DataSet[givenCls].features.append(data[: noOfFeatures])
    


if __name__ == "__main__":
    fileRead()