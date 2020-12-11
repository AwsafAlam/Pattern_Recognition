import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

INPUT_FILE="blobs.txt"
ITERATIONS=100
#Define label for differnt point group
NOISE = 0
UNASSIGNED = 0
core=-1
edge=-2

dataset = []

def read_dataset():
  """
  Reading dataset
  """
  global INPUT_FILE, dataset
  f = open(INPUT_FILE, "r")
  lines = f.readlines()
  
  for i in range(len(lines)):
    data = lines[i].split()
    dataset.append(list(map(float, data)))
    # print(data)

  f.close()
  pass

def calc_distance(X1, X2):
    return(sum((X1 - X2)**2))**0.5

def findClosestCentroids(ic, X):
    assigned_centroid = []
    for i in X:
        distance=[]
        for j in ic:
            distance.append(calc_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid

def calc_centroids(clusters, X):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])],
                      axis=1)
    for c in set(new_df['cluster']):
        current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

def k_means(k):
  """
  1. Initialize centroids – This is done by randomly choosing K no of points, the points can be present in the dataset or also random points.
  2. Assign Clusters – The clusters are assigned to each point in the dataset by calculating their distance from the centroid and assigning it to the centroid with minimum distance.
  3. Re-calculate the centroids – Updating the centroid by calculating the centroid of each cluster we have created.
  """
  global dataset
  X = np.array(dataset)

  init_centroids = random.sample(range(0, len(dataset)), 3)

  centroids = []
  for i in init_centroids:
      centroids.append(dataset[i])
  print(centroids)

  # converting to 2D - array
  centroids = np.array(centroids)
  get_centroids = findClosestCentroids(centroids, X)

  # # computing the mean of separated clusters
  # cent={}
  # for k in range(K):
  #     cent[k+1]=np.array([]).reshape(2,0)

  # # assigning of clusters to points
  # for k in range(m):
  #     cent[minimum[k]]=np.c_[cent[minimum[k]],X[k]]
  # for k in range(K):
  #     cent[k+1]=cent[k+1].T

  # # computing mean and updating it
  # for k in range(K):
  #      centroids[:,k]=np.mean(cent[k+1],axis=0)
  prev_centr = []
  for i in range(ITERATIONS):
    get_centroids = findClosestCentroids(centroids, X)
    centroids = calc_centroids(get_centroids, X)
    print(centroids)

    plt.figure()
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
    plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
    plt.show()



if __name__ == "__main__":
  print("Reading file")
  read_dataset()
  # find_nearest_neighbour(4)
  #Set EPS and Minpoint
  epss = [5,10]
  minptss = [5,10]
  k_means(2)
  # Find ALl cluster, outliers in different setting and print resultsw
  # for eps in epss:
  #   for minpts in minptss:
  #     DBSCAN_start(eps, minpts)