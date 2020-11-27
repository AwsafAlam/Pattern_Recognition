
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

clusters_means, prior_prob, clusters_covariances, h = [], [], [], []
        

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
        start = (start+2)%noOfClusters
    
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


def multivariate_normal(x, means, covariance_mat):
    d = len(x)
    temp1 = math.sqrt(math.pow(2 * math.pi, d) * math.fabs(np.linalg.det(covariance_mat)))
    temp2 = np.matrix(x) - np.matrix(means)
    temp3 = -0.5 * temp2 * np.linalg.inv(covariance_mat) * temp2.transpose()
    temp4 = math.exp(temp3)
    result = temp4 / temp1
    return result


def channel(I):
    """
    Pass through channel
    """
    global h, n_mean, n_var, noOfClusters

    noise_list = np.random.normal(n_mean, n_var, len(I))
    x = len(I)*[0]
    clusters = [[] for i in range(noOfClusters)]

    for k in range(1, len(I)):
        if k == 1:
            x[k] = h[0] * I[k] + noise_list[k]
            cluster_no = I[k]*4
        else:
            x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
            cluster_no = I[k]*4 + I[k - 1]*2 + I[k - 2]*1

        clusters[cluster_no].append([x[k], x[k - 1]])

    return x


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

    prior_prob = [x / total_datapoints for x in prior_prob]
    print(prior_prob)


def distance_proba(x, w_after, w_before):
    global transition_prob, prior_prob, clusters_means, clusters_covariances
    if transition_prob[w_before][w_after] == 0:
        return -INF
    elif w_before == -1:
        d = prior_prob[w_after] * multivariate_normal(x, clusters_means[w_after], clusters_covariances[w_after])
        if d == 0:
            return -INF
        return math.log(d, math.e)
    else:
        d = transition_prob[w_before][w_after] * multivariate_normal(x, clusters_means[w_after],
                                                                                clusters_covariances[w_after])
        if d == 0:
            return -INF
        return math.log(d, math.e)


def D_max(wik, x, k, from_list, D):
    if k == 1:
        return distance_proba(x[k], wik, -1)
    if D[0][k - 1] == -1:
        max_value = D_max(0, x, k - 1, from_list, D) + distance_proba(x[k], wik, 0)
    else:
        max_value = D[0][k - 1] + distance_proba(x[k], wik, 0)
    max_from = 0
    for wik_1 in range(1, noOfClusters):
        if D[wik_1][k - 1] == -1:
            value = D_max(wik_1, x, k - 1, from_list, D) + distance_proba(x[k], wik, wik_1)
        else:
            value = D[wik_1][k - 1] + distance_proba(x[k], wik, wik_1)
        if value > max_value:
            max_value = value
            max_from = wik_1
    from_list[wik][k] = max_from
    return max_value


def equalizer(I):
    """
    Equalizer to obtain output
    """
    global h, n_mean, n_var, noOfClusters
    
    noise_list = np.random.normal(n_mean, n_var, len(I))

    x = len(I)*[0]
    y = len(I)*[0]
    X = [0]

    D = [[-1 for _ in range(len(x))] for _ in range(noOfClusters)]
    from_list = [[-1 for _ in range(len(x))] for _ in range(noOfClusters)]

    for k in range(1, len(I)):
        x[k] = h[0]*I[k] + h[1]*I[k - 1] + noise_list[k]
        X.append([x[k], x[k - 1]])

        for i in range(noOfClusters):
            D[i][k] = D_max(i, X, k, from_list, D)

    max = D[0][len(I) - 1]
    cluster_max = 0
    for i in range(1, noOfClusters):
        D_val = D[i][len(I) - 1]
        if D_val > max:
            max = D_val
            cluster_max = i

    for k in range(len(I) - 1, 0, -1):
        if cluster_max in range(4):
            y[k] = 0
        else:
            y[k] = 1
        cluster_max = from_list[cluster_max][k]
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
    
    for k in range(len(TEST_DATA)):
        if y[k] == TEST_DATA[k]:
            count += 1.0
        inferd_string = inferd_string + str(y[k])

    f = open(out_file, 'w')
    f.write(inferd_string)
    acc = count * 100/ len(TEST_DATA)
    print("Accuracy = {}".format(str(acc)))
    f.close()

    return acc


if __name__ == "__main__":
    
    f = open(PARAMETER_FILE, 'r')
    h = list(map(float, f.readline().split()))
    n_var = int(f.readline())
    f.close()
    print(h,n_var)

    print("Starting channel equalization")
    I = read_dataset()
    
    # Determining the prior and transition probabilities
    find_transition_prob()
    train(I)
    test_data = read_test_data()
    y = equalizer(test_data)
    calculate_accuracy(y, OUT1)

