

TEST_DATA="test.txt"
TRAIN_DATA="train.txt"


# Python code to convert string to list character-wise 
def Convert(string): 
    list1=[] 
    list1[:0]=string 
    return list1 


file = open('train.txt', 'r')
str = file.readline()
# print(str)

a_list=[] 
a_list[:0]= str
# a_list = str.split('')
map_object = map(int, a_list)

list_of_integers = list(map_object)

print(list_of_integers) 

# if __name__ == "__main__":
#     print("Starting channel equalization")