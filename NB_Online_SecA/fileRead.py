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

def pf_given_c(f, meanC, varC):
    prob = 1/(math.sqrt(2* math.pi*varC)) * math.exp((- (math.pow(f - meanC, 2))/(2 * varC)))
    return prob

def predict(test_feature_1, test_feature_2):
    prob_one_given_feature = prob_one*p_x_given_y(test_feature_1, feature1_mean_one, feature1_var_one)*p_x_given_y(test_feature_2, feature2_mean_one, feature2_var_one)
    prob_two_given_feature = prob_two*p_x_given_y(test_feature_1, feature1_mean_two, feature1_var_two)*p_x_given_y(test_feature_2, feature2_mean_two, feature2_var_two)
    if(prob_one_given_feature > prob_two_given_feature):
        return 1
    else:
        return 2

def preprocess(Lines, numberOfData, totalClass):
    featClass = []
    matrix = [ [[] for col in range(numberOfData-1)] for col in range(totalClass)] 
    count = 0
    for line in Lines: 
        # print("{}-{}".format(count, line.strip())) 
        count += 1
        data = line.split()
        # print(data, numberOfData)
        for i in range(numberOfData - 1):
            if data[numberOfData-1] == '1':
                matrix[0][i].append(float(data[i]))
            elif data[numberOfData-1] == '2':
                matrix[1][i].append(float(data[i]))
            elif data[numberOfData-1] == '3':
                matrix[2][i].append(float(data[i]))
            else:
                print("Undermined class")
            # features[i].append(data[i])
            # print(data[i])
        featClass.append(data[numberOfData - 1])
        # print("---------")
    return matrix,featClass

def model(matrix, numberOfData, totalClass):
    cls_feat = []
    for j in range(totalClass):
        sum = 1.0
        feat = []
        for i in range(numberOfData -1):
            print("Class: {} - F {}".format(j+1,i+1))
            avg = mean(matrix[j][i])
            print(str(avg))
            var = variance(matrix[j][i],avg)
            print(str(var))
            feat.append({var: avg})
            # for k in expression_list:
            #     pass
            # prob = pf_given_c(matrix[j][i] , avg, std)
            # print(prob)
            # sum *= prob
            # print(sum)
            print("---------------------")
        cls_feat.append(feat)
    return cls_feat
#  Methodology :
#  First separate the features for each classes.

testFile = open('./during_coding/Test.txt', 'r') 
TestLines = testFile.readlines() 

trainFile = open('./during_coding/Train.txt', 'r') 
TrainLines = trainFile.readlines() 

testSize = len(TestLines)
dataSize = len(TrainLines)
numberOfData = len(TrainLines[0].split())

train_matrix, trainClass = preprocess(TrainLines, numberOfData , 3) 
test_matrix, testClass = preprocess(TestLines, numberOfData , 3) 

# totalClass = countDistinct(featClass)  
print("Testing for Dataset Size : {}, features: {}; class: {}".format(dataSize,numberOfData-1,3 ))
print("Class 1 : {}, Class 2: {}; Class 3: {}\n--------".format(len(train_matrix[0][0]),len(train_matrix[0][1]),len(train_matrix[0][2]) ))

class_count = []
for cl in train_matrix[0]:
    class_count.append(len(cl))

print(class_count)
prior_probability = [ float(x/dataSize) for x in class_count ]
print("Prior probability : {}".format(prior_probability))

trained_data = model(train_matrix, numberOfData, 3)
# print(testClass , len(testClass))
print(trained_data)

for i in range(3):
    print("For Class : {}\n".format(i+1))
    j = 0
    cls_prob = []
    for mt in test_matrix[i]:
        print(mt)
        print("Len: {}".format(len(mt)))
        avg = next(iter(trained_data[i][j].items()))[1]
        var = next(iter(trained_data[i][j].items()))[0]
        # print(next(iter(trained_data[i][j])))
        f1 = []
        for feat_val in mt:
            prob = pf_given_c(feat_val,avg,var)
            f1.append(prob)
            pass
        j += 1
        cls_prob.append(f1)
    sum = 1.0
    for k in range((j-1)):
        for p in range(len(cls_prob)):
            sum *= cls_prob[p][k]

    print("SUM : "+str(sum) + "----------------")

# for ft in matrix[1]:
#     print(ft) 

# correct = 0
# for i in range(len(test_feature1)):
#     #print("comes")
#     if(predict(test_feature1[i], test_feature2[i]) != test_label[i]):
#         print(str(test_feature1[i]) + " " + str(test_feature2[i]))
#     else:
#         correct = correct + 1
# print(correct/len(test_label))

