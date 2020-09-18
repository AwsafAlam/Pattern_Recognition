import numpy as np
import math

number_of_features = 0
number_of_classes = 0
dataset_size = 0
Object_Dictionary = {}
correct = 0

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
        print(len(self.features))
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
        # print(data)
        class_name = data[number_of_features] # last element of data array

        if class_name not in Object_Dictionary: # saving features for each distinct classes
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(data[: number_of_features])

    # print(Object_Dictionary)


def train():
    global number_of_classes, number_of_features, Object_Dictionary, dataset_size

    for key in Object_Dictionary: # obtain mean and std. store in the object itself
        obj = Object_Dictionary[key]
        obj.calc_mean()
        obj.calc_standard_deviation()
        obj.print_features()


def test_accuracy():
    global number_of_features, Object_Dictionary, dataset_size, correct

    f = open("./test.txt", "r")
    lines = f.readlines()
    f.close()

    # wr = open("Report_coding.txt", "w")

    sample_count = 0
    for line in lines:
        sample_count += 1

        test_vector = []
        temp = line.rstrip()
        temp = temp.split()

        for i in range(len(temp)):
            t = temp[i].strip()
            if len(t):
                test_vector.append(float(t))

        class_name = temp[number_of_features]
        print(test_vector,class_name)
        
        # max = 0
        # predicted_class = ''
        # for key in Object_Dictionary:
        #     obj = Object_Dictionary[key]
        #     prior_prob = len(obj.features) / dataset_size # p ( ci )

        #     co_var = []
        #     # Finding co-variant matrix
        #     for i in range(number_of_features):
        #         tmp = arr[i] - obj.mean[i]
        #         co_var.append(tmp)
            
        #     co_var_array = np.array([i for i in co_var]).astype(float)
        #     # co_var_array = np.array(co_var)
        #     co_var_array = co_var_array.reshape(-1,co_var_array.size)
        #     trans = co_var_array.reshape(co_var_array.size, -1)
        #     sigma = np.dot(co_var_array,trans) # sigma is the new 3 x 3 arr
        #     sigma_inv = np.linalg.inv(sigma) 
        #     det = np.linalg.det(sigma)
        #     d = sigma.shape[0]

        #     # print(co_var_array , co_var_array.shape, d) 
        #     # print(trans , trans.shape)
        #     # print("----------\n",sigma, sigma.shape,"\n______")

        #     for i in range(number_of_features):
        #         # p(c)p(Fi | c) for all features
        #         # prior_prob *= get_prob_f_given_c(arr[i] , obj.sd[i], obj.mean[i])
        #         # get_prob(arr[i])
        #         x_minus_mu = arr[i] - obj.mean[i]
        #         ex = math.exp(-0.5 * x_minus_mu *sigma_inv[0][0] * x_minus_mu)
        #         tmp = 1 / (2 * math.pi *math.pow(det, 0.5))
        #         prior_prob *= (tmp * ex)
        #         # print("Prob : "+str(tmp * ex))

        #     if max <= prior_prob:
        #         predicted_class = obj.class_name
        #         max = prior_prob

        # if predicted_class == class_name:
        #     correct += 1
        # else:
        #     print("sample no: " + str(sample) + ", feat:" + str(arr[: number_of_features]) \
        #              + ", actual-class:" + str(class_name) + ", predicted-class: " + str(predicted_class) + "\n")

    acc = (correct / float(dataset_size)) * 100.0
    print("accuracy: {} / {} = {}%".format(correct,dataset_size,acc))
    # wr.close()


if __name__ == "__main__":
    read_dataset()
    train()
    test_accuracy()