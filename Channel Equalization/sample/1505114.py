import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

TEST_FILE="test.txt"
TRAIN_FILE="train.txt"
PARAMETER_FILE="parameter.txt"
OUT1="Out1.txt"
OUT2="Out2.txt"
TRAIN_DATA = []
TEST_DATA = []
INF = 999999999999

mean, variance = 0, 0
n_mean, n_var = 0, 0
transition_prob = []
noOfClusters = 8
received_bits = []
cluster_means, Prior_Probability, cluster_covariances, h = [], [], [], []


def read_dataset():
    global TRAIN_DATA, TRAIN_FILE

    file = open(TRAIN_FILE, 'r')
    str = file.readline()
    
    temp = []
    temp[:0]= str
    map_object = map(int, temp)

    TRAIN_DATA = list(map_object)

    # print(TRAIN_DATA) 
    return TRAIN_DATA


def read_test_data():
    """
    read test data
    """
    global TEST_FILE, TEST_DATA

    file = open(TEST_FILE, 'r')
    str = file.readline()
    
    temp = []
    temp[:0]= str
    map_object = map(int, temp)

    TEST_DATA = list(map_object)
    print(TEST_DATA)
    return TEST_DATA
    





if __name__ == "__main__":
    
    f = open(PARAMETER_FILE, 'r')
    h = list(map(float, f.readline().split()))
    n_var = float(f.readline())
    f.close()
    print(h,n_var)

    print("Starting channel equalization")
    I = read_dataset()
    
    # Determining the prior and transition probabilities
    # find_transition_prob()
    # train(I)
    # test_data = read_test_data()

    # y = equalizer_method_1(test_data)
    # calculate_accuracy(y, OUT1)

    # y = equalizer_method_2(test_data)
    # calculate_accuracy(y, OUT2)
