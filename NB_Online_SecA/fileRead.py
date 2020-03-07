# Using readlines() 
file1 = open('./during_coding/Test.txt', 'r') 
Lines = file1.readlines() 
  
count = 0
# Strips the newline character 
for line in Lines: 
    print("{}-{}".format(count, line.strip())) 
    count += 1