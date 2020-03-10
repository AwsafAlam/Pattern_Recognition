import numpy as np
import math

number_of_features = 0
number_of_classes = 0
dataset_size = 0
Processed_DataSet = {}

class DataStore:
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

    def findMeanArr(self):
        global number_of_features
        for i in range(number_of_features):
            mu = self.get_mean(i)
            self.mean.append(mu)

    def get_standard_deviation(self, feature_no):
        sz = len(self.features)
        temp = np.array([i[feature_no] for i in self.features]).astype(float)

        return np.std(temp)

    def calc_standard_deviation(self):
        global number_of_features
        for i in range(number_of_features):
            sigma = self.get_standard_deviation(i)
            self.sd.append(sigma)

    def covariance_mat():
        pass
    
    def print_features(self):
        print(self.features)
        print("----------------------")

def get_prob_f_given_c(feature , std, mean):
    
    det = math.sqrt(2 * math.pi * std * std)
    p = math.pow(((feature - mean)/std) ,2 )
    return (1 / det) * math.exp(-(0.5 *p))

def get_x_minus_mu(x , mu):
    print("________ X = {} - mu = {} _____ ".format(x, mu))

def training():
    global number_of_features, number_of_classes, dataset_size, Processed_DataSet

    f = open("./during evaluation/Train.txt", "r")
    lines = f.readlines()
    f.close()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].split())

    for i in range(dataset_size):
        data = lines[i + 1].split()

        class_name = data[number_of_features] # last element of data array

        if class_name not in Processed_DataSet: # saving features for each distinct classes
            Processed_DataSet[class_name] = DataStore(class_name)

        Processed_DataSet[class_name].features.append(data[: number_of_features])
    
    # print(Processed_DataSet)

def find_mean_var():
    global number_of_classes, number_of_features, Processed_DataSet, dataset_size

    for key in Processed_DataSet: # obtain mean and std. store in the object itself
        obj = Processed_DataSet[key]
        obj.findMeanArr()
        obj.calc_standard_deviation()
        # obj.print_features()



correct = 0


def predict():
    global number_of_features, Processed_DataSet, dataset_size, correct

    f = open("./during evaluation/Test.txt", "r")
    lines = f.readlines()
    f.close()

    report = open("Report.txt", "w")

    sample = 0
    for line in lines:
        sample += 1

        arr = []
        temp = line.rstrip()
        temp = temp.split()

        for i in range(len(temp)):
            t = temp[i].strip()
            if len(t):
                arr.append(float(t))

        class_name = temp[number_of_features]
        print(arr)
        
        max = 0
        predClass = ''
        for key in Processed_DataSet:
            obj = Processed_DataSet[key]
            prior_prob = len(obj.features) / dataset_size # p ( ci )
            print("Starting for class ... {} - sample: {}".format(key,sample))
            x_minus_mu_arr = []

            # Finding co-variant matrix
            for i in range(number_of_features):
                tmp = arr[i] - obj.mean[i]
                x_minus_mu_arr.append(tmp)
            
            # co_var_array = np.array([i for i in x_minus_mu_arr]).astype(float)
            co_var_array = np.array(x_minus_mu_arr)
            co_var_array = co_var_array.reshape(-1,co_var_array.size)
            trans = co_var_array.reshape(co_var_array.size, -1)
            
            final = []
            for elem1 in x_minus_mu_arr:
                row = []
                for elem2 in x_minus_mu_arr:
                    row.append((elem1 * elem2)/dataset_size)
                final.append(row)
            
            sigma = np.array(final)
            # sigma.reshape(number_of_features, number_of_features)
            print("Final ",sigma, sigma.shape)
            # sigma = np.dot(trans , co_var_array)/dataset_size # sigma is the new 2 x 2 arr
            
            det = np.linalg.det(sigma)
            if det == 0:
                print("Determinent 0, inverse not possible ", det,"\n\n")
                continue
            
            sigma_inv = np.linalg.inv(sigma) 
            d = sigma.shape[0]

            exp_mat = np.dot(co_var_array , sigma_inv)
            exp_mat = np.dot(exp_mat, trans)

            ex = math.exp(-0.5 * exp_mat[0][0])
            # print(ex , det)
            tmp = 1 / (2 * math.pi * math.sqrt(abs(det)))
            prior_prob *= (tmp * ex)
            
            print(co_var_array , co_var_array.shape, d) 
            print(trans , trans.shape)
            print("S ------\n",sigma, sigma.shape,"\n______")
            print("S inv -----\n",sigma_inv,"\n______")
            print("E -----\n",exp_mat,exp_mat.shape,"\n__________")

            # for i in range(number_of_features):
            #     # p(c)p(Fi | c) for all features
            #     # prior_prob *= get_prob_f_given_c(arr[i] , obj.sd[i], obj.mean[i])
            #     # get_prob(arr[i])
            #     x_minus_mu = arr[i] - obj.mean[i]
            #     ex = math.exp(-0.5 * x_minus_mu *sigma_inv[0][0] * x_minus_mu)
            #     tmp = 1 / (2 * math.pi *math.pow(det, 0.5))
            #     prior_prob *= (tmp * ex)
            #     print("Prob : "+str(tmp * ex))

            if max <= prior_prob:
                predClass = obj.class_name
                max = prior_prob

        if predClass == class_name:
            correct += 1
        else:
            report.write(str(sample) + " :" + str(arr[: number_of_features]) + ", Actual: " + str(class_name) + ", Predicted: " + str(predClass) + "\n")

    acc = (correct / dataset_size) * 100
    print("accuracy : " + str(acc))
    report.write("accuracy : " + str(acc))
    report.close()


if __name__ == "__main__":
    training()
    find_mean_var()
    predict()