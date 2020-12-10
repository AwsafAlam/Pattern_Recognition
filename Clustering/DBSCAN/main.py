import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

INPUT_FILE="blobs.txt"

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

def find_nearest_neighbour(k):
  """
  Nearest neighbour
  """
  global dataset
  nearest_neighbors = NearestNeighbors(n_neighbors=k)
  nearest_neighbors.fit(dataset)
  distances, indices = nearest_neighbors.kneighbors(dataset)
  distances = np.sort(distances, axis=0)[:, 1]
  # print(distances, indices)
  plt.plot(distances)
  plt.show()


if __name__ == "__main__":
  print("Reading file")
  read_dataset()
  find_nearest_neighbour(4)
