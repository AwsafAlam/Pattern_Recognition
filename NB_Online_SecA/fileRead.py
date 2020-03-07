from collections import Counter 
import math
import csv
import random

def countDistinct(arr): 
  
    # counter method gives dictionary of elements in list 
    # with their corresponding frequency. 
    # using keys() method of dictionary data structure 
    # we can count distinct values in array 
    return len(Counter(arr).keys())     

def mean(arr):
    sum = 0.0
    for i in range(len(arr)):
        sum += arr[i]
    return sum/len(arr)

def variance(arr , mean):
    sum = 0.0
    for i in range(len(arr)):
        sum += (mean - arr[i])*(mean - arr[i])
    return sum/len(arr)

def standardDeviation(arr, mean):
    return math.sqrt(variance(arr,mean))

def p_x_given_y(x, mean_y, variance_y):
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return p

def predict(test_feature_1, test_feature_2):
    prob_one_given_feature = prob_one*p_x_given_y(test_feature_1, feature1_mean_one, feature1_var_one)*p_x_given_y(test_feature_2, feature2_mean_one, feature2_var_one)
    prob_two_given_feature = prob_two*p_x_given_y(test_feature_1, feature1_mean_two, feature1_var_two)*p_x_given_y(test_feature_2, feature2_mean_two, feature2_var_two)
    if(prob_one_given_feature>prob_two_given_feature):
        return 1
    else:
        return 2

def preprocess():
    testFile = open('./during_coding/Test.txt', 'r') 
    Lines = testFile.readlines() 
    dataSize = len(Lines)
    numberOfData = len(Lines[0].split())
    features = [[] for i in range(numberOfData-1)]
    featClass = []

    count = 0
    for line in Lines: 
        # print("{}-{}".format(count, line.strip())) 
        count += 1
        data = line.split()
        # print("---------")
        for i in range(numberOfData - 1):
            features[i].append(data[i])
            # print(data[i])
        featClass.append(data[numberOfData - 1])
        # print("---------")

    print("Testing for Dataset Size : {}, features: {}; class: {}".format(dataSize,numberOfData-1, countDistinct(featClass)))

# # for ft in features[0]:
# #     print(ft) 

# print("Classification starting ... ")
# # print(featClass)

# correct = 0
# for i in range(len(test_feature1)):
#     #print("comes")
#     if(predict(test_feature1[i], test_feature2[i]) != test_label[i]):
#         print(str(test_feature1[i]) + " " + str(test_feature2[i]))
#     else:
#         correct = correct + 1
# print(correct/len(test_label))

if __name__ == "__main__":
    preprocess()    