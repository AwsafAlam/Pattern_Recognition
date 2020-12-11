import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import queue
import collections
import pandas as pd


INPUT_FILE="blobs.txt"
ITERATIONS=10
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
	plt.savefig(INPUT_FILE+'_Nearest_Neighbour.png')
	# plt.show()

def euclidean_dist(point1, point2):
	"""
		Euclid distance function
	"""
	x1 = point1[0]
	x2 = point2[0]
	y1 = point1[1]
	y2 = point2[1]
	# create the points
	p1 = (x1 - x2)**2
	p2 = (y1 - y2)**2
	return np.sqrt(p1 + p2)

def neighbor_points(dataset, pointIdx, radius):
	'''
	find all neigbor points in radius from a given point.
	'''
	points = []
	for i in range(len(dataset)):
		# Calculating distance btn points
		if euclidean_dist(dataset[i], dataset[pointIdx]) <= radius:
			points.append(i)
	return points

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
	neighbourhood_arr = []
	#initilize list for core/noncore point
	core_pts=[]
	non_core_pts=[]
	
	#Find all neigbor for all point
	for i in range(len(data)):
		neighbourhood_arr.append(neighbor_points(dataset,i,Eps))
	
	#Find all core point, edgepoint and noise
	for i in range(len(neighbourhood_arr)):
		# A point is a core point if it has more than a specified number of points (MinPts) within Eps 
		if (len(neighbourhood_arr[i]) >= MinPt):
			pointlabel[i]=core
			core_pts.append(i)
		else:
			non_core_pts.append(i)

	for i in non_core_pts:
		for j in neighbourhood_arr[i]:
			if j in core_pts:
				pointlabel[i]=edge
				break
			
	#start assigning point to cluster
	cluster_no = 1
	
	#Using a Queue to put all neigbor core point in queue and find neigboir's neigbor
	for i in range(len(pointlabel)):
		q = queue.Queue()
		if (pointlabel[i] == core):
			pointlabel[i] = cluster_no
			for x in neighbourhood_arr[i]:
				if(pointlabel[x]==core):
					q.put(x)
					pointlabel[x]= cluster_no
				elif(pointlabel[x]==edge):
					pointlabel[x] = cluster_no
			
			#Stop when all point in Queue has been checked   
			while not q.empty():
				neighbors = neighbourhood_arr[q.get()]
				for y in neighbors:
					if (pointlabel[y]==core):
						pointlabel[y]=cluster_no
						q.put(y)
					if (pointlabel[y]==edge):
						pointlabel[y]=cluster_no            
			cluster_no = cluster_no+1 #move to next cluster
			 
	return pointlabel,cluster_no


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
		# print(centroids)

	plt.figure()
	plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
	plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
	plt.savefig(INPUT_FILE+'_k_means.png')
	# plt.show()


#Function to plot final result
def plotRes(data, clusterRes, clusterNum):
	nPoints = len(data)
	print("Plotting for {} points".format(nPoints))
	plt.figure()
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
	# print(pointlabel)
	plotRes(dataset, pointlabel, cl)
	plt.savefig(INPUT_FILE+'_DBSCAN.png')
	# plt.show()
	print('number of cluster found: ' + str(cl-1))
	counter=collections.Counter(pointlabel)
	print(counter)
	outliers  = pointlabel.count(0)
	print('numbrer of outliers found: '+str(outliers) +'\n')
	
	return cl

if __name__ == "__main__":
	print("Reading file")
	read_dataset()
	find_nearest_neighbour(4)
	#Set EPS and Minpoint
	epss = [5]
	minptss = [5]

	noOfClusters = 2
	# Find ALl cluster, outliers in different setting and print resultsw
	for eps in epss:
		for minpts in minptss:
			noOfClusters = DBSCAN_start(eps, minpts)

	k_means(noOfClusters-1)