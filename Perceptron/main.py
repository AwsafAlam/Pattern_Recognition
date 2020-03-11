

def fileRead():
    f = open("Test.txt")
    test_data = f.readlines()
    f.close()
    for line in test_data:
        print(line)


if __name__ == "__main__":
    fileRead()