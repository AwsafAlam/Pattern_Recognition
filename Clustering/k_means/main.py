import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import queue
import collections


INPUT_FILE="blobs.txt"
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

def dist(point1, point2):
    """Euclid distance function"""
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
  # create the points
    p1 = (x1 - x2)**2
    p2 = (y1 - y2)**2
    return np.sqrt(p1 + p2)


#function to find all neigbor points in radius
def neighbor_points(data, pointIdx, radius):
    points = []
    for i in range(len(data)):
        #Euclidian distance using L2 Norm
        # if np.linalg.norm(data[i] - data[pointIdx]) <= radius:
        if dist(data[i], data[pointIdx]) <= radius:
            points.append(i)
    return points

#DB Scan algorithom
def dbscan(data, Eps, MinPt):
    '''
    - Eliminate noise points
    - Perform clustering on the remaining points
      > Put an edge between all core points which are within Eps
      > Make each group of core points as a cluster
      > Assign border point to one of the clusters of its associated core points

    '''
    #initilize all pointlable to unassign
    pointlabel  = [UNASSIGNED] * len(data)
    pointcount = []
    #initilize list for core/noncore point
    corepoint=[]
    noncore=[]
    
    #Find all neigbor for all point
    for i in range(len(data)):
        pointcount.append(neighbor_points(dataset,i,Eps))
    
    #Find all core point, edgepoint and noise
    for i in range(len(pointcount)):
        if (len(pointcount[i])>=MinPt):
            pointlabel[i]=core
            corepoint.append(i)
        else:
            noncore.append(i)

    for i in noncore:
        for j in pointcount[i]:
            if j in corepoint:
                pointlabel[i]=edge

                break
            
    #start assigning point to cluster
    cl = 1
    #Using a Queue to put all neigbor core point in queue and find neigboir's neigbor
    for i in range(len(pointlabel)):
        q = queue.Queue()
        if (pointlabel[i] == core):
            pointlabel[i] = cl
            for x in pointcount[i]:
                if(pointlabel[x]==core):
                    q.put(x)
                    pointlabel[x]=cl
                elif(pointlabel[x]==edge):
                    pointlabel[x]=cl
            #Stop when all point in Queue has been checked   
            while not q.empty():
                neighbors = pointcount[q.get()]
                for y in neighbors:
                    if (pointlabel[y]==core):
                        pointlabel[y]=cl
                        q.put(y)
                    if (pointlabel[y]==edge):
                        pointlabel[y]=cl            
            cl=cl+1 #move to next cluster
           
    return pointlabel,cl

    
#Function to plot final result
def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    print("Plotting for {} points".format(nPoints))
    scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
    for i in range(clusterNum):
        if (i==0):
            #Plot all noise point as blue
            color='blue'
        else:
            color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j][0])
                y1.append(data[j][1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='.')


def DBSCAN_start(eps, minpts):
  """
  docstring
  """
  global dataset

  print('Set eps = ' +str(eps)+ ', Minpoints = '+str(minpts))
  pointlabel,cl = dbscan(dataset,eps,minpts)
  print(cl)
  print(pointlabel)
  plotRes(dataset, pointlabel, cl)
  plt.show()
  print('number of cluster found: ' + str(cl-1))
  counter=collections.Counter(pointlabel)
  print(counter)
  outliers  = pointlabel.count(0)
  print('numbrer of outliers found: '+str(outliers) +'\n')

if __name__ == "__main__":
  print("Reading file")
  read_dataset()
  # find_nearest_neighbour(4)
  #Set EPS and Minpoint
  epss = [5,10]
  minptss = [5,10]

  # Find ALl cluster, outliers in different setting and print resultsw
  for eps in epss:
    for minpts in minptss:
      DBSCAN_start(eps, minpts)