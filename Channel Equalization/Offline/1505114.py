
import numpy as np
import math

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
        

def find_transition_prob():
    """
    Transition probabilities for the 8 clusters w1 -> w8
    """
    global transition_prob, noOfClusters
    
    prob = []
    for i in range(noOfClusters):
        arr = [0 for i in range(noOfClusters)]
        prob.append(arr)
    
    start = 0
    for i in range(noOfClusters):
        # print(i,start)
        prob[i][start] = 0.5
        prob[i][start+1] = 0.5
        start = (start+2) % noOfClusters
    
    transition_prob = np.array(prob).T.tolist()
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


def covar_matrix(list):
    """
    calculate covariance matrix
    """
    list = np.array(list)
    n = list.shape[0]
    mu = np.mean(list, axis=0)
    temp = np.array(list[0, :]) - np.array(mu)
    transposed_vec = np.array(temp).T
    covariance_matrix = transposed_vec * temp
    
    for i in range(1, n):
        temp = np.matrix(list[i, :]) - np.matrix(mu)
        transposed_vec = temp.transpose()
        covariance_matrix = covariance_matrix + (transposed_vec * temp)
    covariance_matrix /= n
    return covariance_matrix


def channel(I):
    """
    Pass through channel
    """
    global h, n_mean, n_var, received_bits

    noise_list = np.random.normal(n_mean, n_var, len(I))
    x = [[0] for _ in range(len(I))]

    for k in range(1, len(I)):
        x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
        # if k == 1:
        #     x[k] = h[0] * I[k] + noise_list[k]
        # else:
        #     x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
    received_bits = x
    return x

# def bin_to_dec(arr):
#     """
#     convert binary to decimal
#     """
#     sum = 0.0
#     for i in range(len(arr)):
#         sum += arr[i] * math.pow(2,i)
#     return sum

def train(I):
    """
    Using training sample to determine cluster mean & covariance matrix
    """
    global h, noOfClusters, transition_prob, n_mean, n_var, cluster_covariances, cluster_means, Prior_Probability
    
    # generate channel noise usign normal dist.
    noise_list = np.random.normal(n_mean, n_var, len(I))

    x = [0 for _ in range(len(I))]
    clustersList = [[] for _ in range(noOfClusters)]

    
    # Determining the observation probabilities
    for k in range(1, len(I)):
        if k == 1:
            x[k] = h[0] * I[k] + noise_list[k]
            idx = I[k]*4
        else:
            x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
            # convert binary to decimal
            sum = 0
            for i in range(3):
                sum += I[k + i - 2] * math.pow(2,i)
            idx = int(sum)

        clustersList[idx].append([x[k], x[k - 1]])

    print("Clusters: {}".format(len(clustersList)))
    print("Calculating cluster centers ...")
    # print(x)

    datasetLen = 0.0
    for j in range(noOfClusters):
        print("For cluster {} - length: {}".format(j, len(clustersList[j])))
        bits_per_cluster = len(clustersList[j])
        datasetLen += bits_per_cluster
        Prior_Probability.append(bits_per_cluster)

        # find the mean and covariance matrix for each cluster
        clustersList[j] = np.array(clustersList[j])
        cluster_means.append(np.mean(clustersList[j], axis=0))
        cluster_covariances.append(covar_matrix(clustersList[j]))
        # cluster_covariances.append(np.cov(clustersList[j]))

    Prior_Probability = [float(x) / float(datasetLen) for x in Prior_Probability]
    print("Found prior probabilities:")
    print(Prior_Probability)


def multivariate_normal(x, mu, covariance_mat):
    d = len(x)
    # calculating probability density function
    pdf = math.sqrt(math.pow(2 * math.pi, d) * math.fabs(np.linalg.det(covariance_mat)))
    x_minus_mu = np.matrix(x) - np.matrix(mu)
    temp = -0.5 * x_minus_mu * np.linalg.inv(covariance_mat) * x_minus_mu.transpose()
    exponent_val = math.exp(temp)
    pdf =  exponent_val / pdf
    return pdf
    

def cost_function(x, w_ik, w_ik_1):
    global transition_prob, Prior_Probability, cluster_means, cluster_covariances

    if transition_prob[w_ik_1][w_ik] == 0:
        return -INF
    elif w_ik_1 == -1:
        norm = multivariate_normal(x, cluster_means[w_ik], cluster_covariances[w_ik])
        d = Prior_Probability[w_ik] * norm
        if d == 0:
            return -INF
        return math.log(d, math.e)
    else:
        norm = multivariate_normal(x, cluster_means[w_ik], cluster_covariances[w_ik])
        d = transition_prob[w_ik_1][w_ik] * norm
        if d == 0:
            return -INF
        return math.log(d, math.e)


def D_max(w_ik, x, k, backtrack_lst, D):
    """
    Recursive function for calculating the distance matrix for DP
    """
    if k == 1:
        return cost_function(x[k], w_ik, -1)
    if D[0][k - 1] == -1:
        max_value = D_max(0, x, k - 1, backtrack_lst, D) + cost_function(x[k], w_ik, 0)
    else:
        max_value = D[0][k - 1] + cost_function(x[k], w_ik, 0)
    
    maxIdx = 0
    for wik_1 in range(1, noOfClusters):
        if D[wik_1][k - 1] == -1:
            value = D_max(wik_1, x, k - 1, backtrack_lst, D) + cost_function(x[k], w_ik, wik_1)
        else:
            value = D[wik_1][k - 1] + cost_function(x[k], w_ik, wik_1)
        
        if value > max_value:
            max_value = value
            maxIdx = wik_1
    
    backtrack_lst[w_ik][k] = maxIdx
        
    return max_value


def equalizer_method_1(I):
    """
    Equalizer to obtain output using the vitterbi Algo
    """
    global h, n_mean, n_var, noOfClusters
    
    noise_list = np.random.normal(n_mean, n_var, len(I))

    # x = channel(I)
    x = [0 for _ in range(len(I))]
    y = [0 for _ in range(len(I))]
    X = [0]

    # Initialize D and backtracking path
    D = [[-1 for _ in range(len(x))] for _ in range(noOfClusters)]
    backtrack_lst = [[-1 for _ in range(len(x))] for _ in range(noOfClusters)]

    for k in range(1, len(I)):
        x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
        X.append([x[k], x[k - 1]])
        for i in range(noOfClusters):
            D[i][k] = D_max(i, X, k, backtrack_lst, D)


    max_D = D[0][len(I) - 1]
    maxIdx = 0
    for i in range(1, noOfClusters):
        if D[i][len(I) - 1] > max_D:
            max_D = D[i][len(I) - 1]
            maxIdx = i

    for i in range(len(I) - 1, 0, -1):
        if maxIdx in range(4):
            y[i] = 0
        else:
            y[i] = 1
        maxIdx = backtrack_lst[maxIdx][k]
    
    return y


def equalizer_method_2(I):
    """
    Using euclidean distances only
    """
    global h, n_mean, n_var, noOfClusters, received_bits
    
    x = received_bits
    y = [0 for _ in range(len(I))]
    X = [0]

    # Initialize D and backtracking path
    D = [[-1 for _ in range(len(x))] for _ in range(noOfClusters)]

    for k in range(1, len(I)):
        X.append([x[k], x[k - 1]])

        # for i in range(noOfClusters):
        #     D[i][k] = Euclidean(i, X, k, backtrack_lst, D)
    return y


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
    

def calculate_accuracy(y, out_file):
    """
    test data accuracy
    """
    global TEST_DATA

    count = 0.0
    inferd_string = ""
    
    for i in range(len(TEST_DATA)):
        if y[i] == TEST_DATA[i]:
            count += 1.0
        inferd_string = inferd_string + str(y[i])

    f = open(out_file, 'w')
    f.write(inferd_string)
    acc = count * 100/ len(TEST_DATA)
    print("Accuracy = {}".format(str(acc)))
    f.close()

    return acc


if __name__ == "__main__":
    
    f = open(PARAMETER_FILE, 'r')
    h = list(map(float, f.readline().split()))
    n_var = float(f.readline())
    f.close()
    print(h,n_var)

    print("Starting channel equalization")
    I = read_dataset()
    
    # Determining the prior and transition probabilities
    find_transition_prob()
    train(I)
    test_data = read_test_data()

    y = equalizer_method_1(test_data)
    calculate_accuracy(y, OUT1)

    # y = equalizer_method_2(test_data)
    # calculate_accuracy(y, OUT2)
