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


def fileRead():
    f = open("Test.txt")
    test_data = f.readlines()
    f.close()
    for line in test_data:
        print(line)


if __name__ == "__main__":
    fileRead()