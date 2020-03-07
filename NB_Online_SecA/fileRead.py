from collections import Counter 
  
def countDistinct(arr): 
  
    # counter method gives dictionary of elements in list 
    # with their corresponding frequency. 
    # using keys() method of dictionary data structure 
    # we can count distinct values in array 
    return len(Counter(arr).keys())     
  

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

# for ft in features[0]:
#     print(ft) 

print("Classification starting ... ")
# print(featClass)
