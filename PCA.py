### This example is based on the following page:
### http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

# Import modules
import aux_funcs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Import existing sklearn datasets for analysis
from sklearn import datasets

#Load the iris dataset
iris = datasets.load_iris()

#Obtains the raw data
X = iris.data

#Display the data in three dimensions
aux_funcs.display_Iris(data=X,labels=[])

#Display the distributions
aux_funcs.display_Iris_Dist(data=X,labels=[])

#Check mean of each attribute
print('\nMean for each attribute: \n', np.mean(X,axis=0))

#############################################
### Apply PCA methodology (Hard solution) ###
#############################################

#Scale the data
X_std = StandardScaler().fit_transform(X)

#Check mean of each attribute
print('\nMean for each attribute: \n', np.mean(X_std,axis=0))

#Obtain the covariance matrix (cov_mat1 should be nearly identical to cov_mat2)
cov_mat1 = np.cov(X_std.T)
cov_mat2 = np.corrcoef(X.T)
cov_mat=cov_mat2 #we can use either

#Check scatter of some combinations
plt.figure()
plt.scatter(X_std[:,0],X_std[:,3])

#Obtain eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' %eig_vals)                 
print('\nEigenvectors \n%s' %eig_vecs)

#Obtain variance 
var_exp = eig_vals/sum(eig_vals)*100
cum_var_exp = np.cumsum(var_exp)
print('\nCumulative of the proportion of variance:\n', cum_var_exp)

#Obtain projection matrix using the two first components (>95%)
matrix_w = eig_vecs[:,:2]
print('\nProjection Matrix W:\n', matrix_w)

#Project onto the new reduced space
Y = X_std.dot(matrix_w)

#Check covariance matrix of the data in the reduced dimensional space
Y_std = StandardScaler().fit_transform(Y)
cov_mat3 = np.cov(Y_std.T)
print('\nCovariance matrix in the reduced space: \n',cov_mat3)

#Plot outcome
aux_funcs.display_Iris_PCA(data=Y,labels=[])

###############################################
### Apply PCA methodology (Simple solution) ###
###############################################

#Creates the main PCA object
pca = PCA(n_components=0.95, svd_solver='full')

#Fits the data
pca_fit = pca.fit(X_std)
#Obtain the "scores" - projection onto the new reduced space
Y = pca_fit.transform(X_std)   
#Plot outcome
aux_funcs.display_Iris_PCA(data=Y,labels=[])

###Some exploratory visualization with the classes that we actually know

#Obtains CLass label IDs
y = iris.target
aux_funcs.display_Iris(data=X,labels=y)
aux_funcs.display_Iris_Dist(data=X,labels=y)
aux_funcs.display_Iris_PCA(data=Y,labels=y)

