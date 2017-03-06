###This example is based on the following page:
###http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#

#Import modules
import numpy as np
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

#Learn more about these metrics:
#http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f    %.3f'
          % (name, (time() - t0), 
             estimator.inertia_,
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean',sample_size=n_samples),
             metrics.calinski_harabaz_score(data, estimator.labels_)
             ))

#Initializes the seed for the random numbers
np.random.seed(42)

#Load a couple of data sets, we will be using the iris and the digits datasets
iris=load_iris()
digits = load_digits()

#Set the dataset names for the cycle
dataset_names = ['iris','digits']
for i in dataset_names:
    dataset = eval(i)
    data = scale(dataset.data) #obtain the data and scale it
    n_samples, n_features = data.shape #obtain number of samples and features
    n_classes = len(dataset.target_names)
    
    print("\n\nDataset: %s, n_classes: %d, n_samples: %d, n_features: %d" % (i,n_classes, n_samples, n_features))
    print(79 * '_')
    print('% 3s' % 'init'    '        time  inertia silhouette calinsky-harabaz')

    #Evaluate three different K-Means models
    bench_k_means(KMeans(init='random', n_clusters=n_classes, n_init=1),
                  name="random(1)",
                  data=data)
    
    bench_k_means(KMeans(init='random', n_clusters=n_classes, n_init=10),
                  name="random(10)",
                  data=data)
    
    bench_k_means(KMeans(init='k-means++', n_clusters=n_classes, n_init=10),
                  name="k-means++",
                  data=data)
    
    print(79 * '_')