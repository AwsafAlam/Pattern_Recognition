import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import queue

training_files = ["bisecting.txt","blobs.txt","moons.txt"]
INPUT_FILE="blobs.txt"
ITERATIONS=50
#Define label for differnt point group
UNASSIGNED = 0
CORE_PT = -1
EDGE_PT = -2

dataset = []
noOfClusters = 0

def read_dataset(INPUT_FILE):
	"""
	Reading dataset
	"""
	global dataset
	f = open(INPUT_FILE, "r")
	lines = f.readlines()
	
	for i in range(len(lines)):
		data = lines[i].split()
		dataset.append(list(map(float, data)))

	print("Total dataset = {} points".format(len(dataset)))
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
	plt.grid()
	plt.plot(distances)
	# plt.savefig(INPUT_FILE+'_Nearest_Neighbour.png')
	plt.show()

def plotClusters(dataset, labels, noOfClusters, file):
	total_points = len(dataset)
	print("Plotting for {} points".format(total_points))
	plt.figure()

	# Color array for clusters
	scatterColors = ["blue","green","red","cyan","brown","indigo", "pink", "royalblue",
									"orange","yellow","black","olive", "gold", "orangered", "skyblue", "teal" ]

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
	
	plt.grid()
	plt.savefig(file)
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
	DBSCAN Algorithm
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

def DBSCAN_start(eps, minpts):
	"""
	docstring
	"""
	global dataset, noOfClusters

	print("Starting DBSCAN for EPS: {} | Minpts: {}".format(eps, minpts))
	labels = dbscan(dataset,eps,minpts)
	
	plotClusters(dataset, labels, noOfClusters, INPUT_FILE+'_DBSCAN.png')
	outliers  = labels.count(0)

	print("No. of Clusters: {}".format(noOfClusters-1))
	print("Outliers: {}".format(outliers))
	
	return noOfClusters - 1


def calc_distance(X1, X2):
	return (sum((X1 - X2)**2))**0.5

def assign_clusters(centroids, X):
	assigned_cluster = []
	for i in X:
		distance=[]
		for j in centroids:
			distance.append(calc_distance(i, j))
		# print(distance)
		# print(np.argmin(distance))
		# print("--------------------------------")
		assigned_cluster.append(np.argmin(distance)) # idx of minimum element
		# print(assigned_cluster)
	return assigned_cluster


def calc_centroids(clusters_lables, k):
	global dataset
	points_per_cluster = [[] for _ in range(k)]
	for i in range(len(clusters_lables)):
		points_per_cluster[clusters_lables[i]].append(dataset[i])
	
	centroids = []
	for i in range(k):
		centroids.append(np.mean(points_per_cluster[i], axis=0))

	return centroids

def match_centroids(c_new, c_old):
	return (np.array(c_new) == np.array(c_old)).all()


def k_means(k):
	"""
	K-Means clustering algorithm
	"""
	global dataset

	print("Running k-Means for {} clusters..".format(k))
	X = np.array(dataset)

	init_centroids = random.sample(range(0, len(dataset)), k)

	centroids, cluster_labels = [], []
	for i in init_centroids:
		centroids.append(dataset[i])
	
	# converting to 2D - array
	# centroids = np.array(centroids)
	# get_centroids = assign_clusters(centroids, X)

	prev_centroids = centroids.copy()
	for i in range(ITERATIONS):
		print("For iteration {}: ".format(i))
		prev_centroids = np.array(prev_centroids)
		cluster_labels = assign_clusters(prev_centroids, X)
		centroids = calc_centroids(cluster_labels, k)
		
		# print(prev_centroids)
		print(centroids)
		if match_centroids(centroids,prev_centroids):
			print("Converged ...")
			break
		else:
			prev_centroids = centroids.copy()
			
	plotClusters(dataset, cluster_labels, k, INPUT_FILE+'_k_means.png')
	

if __name__ == "__main__":
	
	print("Choose Training file...")
	for i, item in enumerate(training_files, start=1):
		print(i,item)
	choice = int(input())
	INPUT_FILE=training_files[choice-1]
	read_dataset(INPUT_FILE)
	print("1. Plot k Nearest Neighbours\n2. Run Clustering Algorithms")
	choice = int(input())
	if choice == 1:
		print("Enter the value of k:")
		k = int(input())
		find_nearest_neighbour(k)
	else:
		print("Enter EPS value:")
		eps = float(input())

		print("Enter Minpts value:")
		minpts = int(input())
		
		k = DBSCAN_start(eps, minpts)
		k_means(k)

