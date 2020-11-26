
import numpy as np
import math

TEST_FILE="test.txt"
TRAIN_FILE="train.txt"
PARAMETER_FILE="parameter.txt"
TRAIN_DATA = []
TEST_DATA = []

mean, variance = 0, 0
n_mean, n_var = 0, 0
transition_prob = []
noOfClusters = 8

clusters_means, prior_prob, clusters_covariances, h = [], [], [], []
        

def find_transition_prob():
    """
    Transition probabilities for the 8 clusters w1 -> w8
    """
    global transition_prob, noOfClusters

    for i in range(noOfClusters):
        temp = []
        for j in range(noOfClusters):
            if (i == 0 or i == 1) and (j == 0 or j == 4):
                temp.append(0.5)
            elif (i == 2 or i == 3) and (j == 1 or j == 5):
                temp.append(0.5)
            elif (i == 4 or i == 5) and (j == 2 or j == 6):
                temp.append(0.5)
            elif (i == 6 or i == 7) and (j == 3 or j == 7):
                temp.append(0.5)
            else:
                temp.append(0)
        transition_prob.append(temp)
    print("Transition Probabilities :")
    print(transition_prob)
    
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

def covar_matrix(data):
    means = np.mean(data, axis=0)
    n = data.shape[0]
    temp = np.matrix(data[0, :]) - np.matrix(means)
    temp_t = temp.transpose()
    covariance_matrix = temp_t * temp
    for i in range(1, n):
        temp = np.matrix(data[i, :]) - np.matrix(means)
        temp_t = temp.transpose()
        covariance_matrix = covariance_matrix + (temp_t * temp)
    covariance_matrix /= n
    return covariance_matrix

def train(I):
    """
    docstring
    """
    global h, noOfClusters, transition_prob, n_mean, n_var, clusters_covariances, clusters_means, prior_prob
    
    # generate channel noise usign normal dist.
    noise_list = np.random.normal(n_mean, n_var, len(I))

    x = len(I)*[0]

    clusters = []

    for j in range(noOfClusters):
        clusters.append([])

    # Determining the observation probabilities
    for k in range(1, len(I)):
        if k == 1:
            x[k] = h[0] * I[k] + noise_list[k]
            cluster_no = I[k]*4
        else:
            x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
            cluster_no = I[k]*4 + I[k - 1]*2 + I[k - 2]*1

        clusters[cluster_no].append([x[k], x[k - 1]])

    print("Bits received after passing channel :")
    print(clusters)
    total_datapoints = 0
    for j in range(noOfClusters):
        print(j)
        cluster_size = float(len(clusters[j]))
        total_datapoints += cluster_size
        prior_prob.append(cluster_size)

        clusters[j] = np.array(clusters[j])
        print(clusters[j].shape)
        clusters_means.append(np.mean(clusters[j], axis=0))
        clusters_covariances.append(covar_matrix(clusters[j]))

    prior_probas = [x / total_datapoints for x in prior_prob]
    print(prior_probas)
    

def test():
    """
    docstring
    """
    f = open(TEST_FILE, 'r')
    str = f.readline()
    
    a_list=[] 
    a_list[:0]= str
    map_object = map(int, a_list)

    list_of_integers = list(map_object)

    pass


def calculate_accuracy():
    """
    test data accuracy
    """
    pass


if __name__ == "__main__":
    
    f = open(PARAMETER_FILE, 'r')
    h = list(map(float, f.readline().split()))
    variance = f.readline()
    f.close()
    print(h,variance)

    print("Starting channel equalization")
    I = read_dataset()
    # Defining the states (e.g., clusters along with their centroids). 
    # hi’s and n and l indicate their number.

    # Determining the prior and transition probabilities
    find_transition_prob()
    train(I)

