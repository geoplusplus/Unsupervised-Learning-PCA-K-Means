###Exemplifies K-Means with classes generated from normal distributions

#Import modules
import aux_funcs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#Initializes the seed for the random numbers
np.random.seed(42)

#Sets parameters
nPoints=800 #sets total number of points
nClasses=10 #sets number of classes
s2=0.1 #sets variance for each class

#Generate random normally distributed points using the parameters above
[X,target]=aux_funcs.init_board_gauss(nPoints,nClasses,s2)

#Plots the data, find clusters and plots the centroids
plt.figure(3,figsize=(7,5))
plt.scatter(X[:,0],X[:,1],c=target,s=nPoints*[100]) #plots the data
model=KMeans(n_clusters=10).fit(X) #fits the K-Means model
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c="k",s=nClasses*[50]) #plots the centroids

