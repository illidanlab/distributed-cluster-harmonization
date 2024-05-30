import numpy as np
from sklearn.cluster import KMeans
from ClusterComBat.clusterComBat import ClusterComBat
from DistributedClusterComBat.distributedClusterComBat import DistributedClusterComBat

# Generate Random Data
number_of_sites_train = 10
number_of_features = 10
number_of_samples_train = 100
number_of_biological_covariates = 3
number_of_samples_test = 30

# Train
data_train = np.random.rand(number_of_samples_train, number_of_features)
batch = np.random.randint(low = 0, high = number_of_sites_train -1, size = number_of_samples_train)
covars_train = np.random.rand(number_of_samples_train, number_of_biological_covariates)

# Test
data_test = np.random.rand(number_of_samples_test, number_of_features)
covars_test = np.random.rand(number_of_samples_test, number_of_biological_covariates)

# Clustering Algorithm
kmean = KMeans()

# Cluster ComBat
clusterComBat = ClusterComBat(kmean)
harmonized_data_train = clusterComBat.fit(data_train, covars_train)
harmonized_data_test = clusterComBat.harmonize(data_test)

# Distributed Cluster ComBat
distributedClusterComBat = DistributedClusterComBat(kmean)
harmonized_data_train = distributedClusterComBat.fit(data_train, covars_train, batch)
harmonized_data_test = distributedClusterComBat.harmonize(data_test, covars_test)