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
CORE_PT = -1
EDGE_PT = -2

dataset = []
noOfClusters = 0

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

def plotClusters(dataset, labels, noOfClusters, file):
	total_points = len(dataset)
	print("Plotting for {} points".format(total_points))
	plt.figure()

	# Color array for clusters
	scatterColors = ['black', 'green', 'brown', 'red', 'purple', 'orange', 'yellow']
	for i in range(noOfClusters):
		if (i==0):
			#Plot all noise point as blue
			color='blue'
		else:
			color = scatterColors[i % len(scatterColors)]
		
		x = [];  y = []
		for j in range(total_points):
			if labels[j] == i:
				x.append(dataset[j][0])
				y.append(dataset[j][1])
		plt.scatter(x, y, c=color, alpha=1, marker='.')
	
	# plt.show()
	plt.savefig(file)

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
	global dataset, noOfClusters
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
			pointlabel[i] = CORE_PT
			core_pts.append(i)
		else:
			non_core_pts.append(i)

	for i in non_core_pts:
		for j in neighbourhood_arr[i]:
			if j in core_pts:
				pointlabel[i] = EDGE_PT
				break
			
	#start assigning point to cluster
	cluster_no = 1

	# Put all neigbor core point in queue and find neigboir's neigbor
	for i in range(len(pointlabel)):
		q = queue.Queue()
		if (pointlabel[i] == CORE_PT):
			pointlabel[i] = cluster_no
			for j in neighbourhood_arr[i]:
				if(pointlabel[j] == CORE_PT):
					q.put(j)
					pointlabel[j]= cluster_no
				elif(pointlabel[j] == EDGE_PT):
					pointlabel[j] = cluster_no
			
			# checking queue
			while not q.empty():
				neighbors = neighbourhood_arr[q.get()]
				for n in neighbors:
					if (pointlabel[n] == CORE_PT):
						pointlabel[n]=cluster_no
						q.put(n)
					if (pointlabel[n] == EDGE_PT):
						pointlabel[n]=cluster_no            
			
			cluster_no = cluster_no + 1
	
	noOfClusters = cluster_no
	return pointlabel


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

	init_centroids = random.sample(range(0, len(dataset)), k)

	centroids = []
	for i in init_centroids:
		centroids.append(dataset[i])
	print(centroids)

	# converting to 2D - array
	centroids = np.array(centroids)
	get_centroids = findClosestCentroids(centroids, X)

	prev_centr = []
	for i in range(ITERATIONS):
		get_centroids = findClosestCentroids(centroids, X)
		centroids = calc_centroids(get_centroids, X)
		# print(centroids)

	plotClusters(dataset, centroids, k, INPUT_FILE+'_k_means.png')
	# plt.figure()
	# plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
	# plt.scatter(X[:, 0], X[:, 1], alpha=0.1)
	# plt.savefig(INPUT_FILE+'_k_means.png')
	# plt.show()


def DBSCAN_start(eps, minpts):
	"""
	docstring
	"""
	global dataset, noOfClusters

	print("Starting DBSCAN for EPS: {} | Minpts: {}".format(eps, minpts))
	labels = dbscan(dataset,eps,minpts)
	
	plotClusters(dataset, labels, noOfClusters, INPUT_FILE+'_DBSCAN.png')
	outliers  = labels.count(0)

	print("No. of Clusters: {}".format(noOfClusters))
	print("Outliers: {}".format(outliers))
	
	return noOfClusters

if __name__ == "__main__":
	
	print("Reading file")
	read_dataset()
	find_nearest_neighbour(4)
	
	#Set EPS and Minpoint
	epss = [5]
	minptss = [5]

	k = 0
	# Find ALl cluster, outliers in different setting and print resultsw
	for eps in epss:
		for minpts in minptss:
			k = DBSCAN_start(eps, minpts)

	k_means(k)