import numpy as np
from generateData import *
from plot import *
from sklearn.cluster import KMeans
import random
from DistributedClusterComBat.distributedClusterComBat import DistributedClusterComBat
import warnings
import matplotlib.pyplot as plt
import numpy as np
import json 

plt.rcParams.update({'font.size': 11})

# comment out the next line to see the warning
warnings.simplefilter('ignore')

def experiment(sites = 30, samples_per_site = 30, features = 30, num_biological_covariate = 10, sites_per_cluster = 3):

    # Generate
    y, biological_covariate, _, _ = generate(sites = sites, samples_per_site = samples_per_site, features = features, num_biological_covariate = num_biological_covariate, sites_per_cluster = sites_per_cluster)
    
    # Initialize data
    data = [y[i][j] for i in range(len(y)) for j in range(len(y[i]))]
    batch = [i+1 for i in range(len(y)) for j in range(len(y[i]))]
    covariate = [[biological_covariate[i][j][g] for i in range(len(y)) for j in range(len(y[i]))] for g in range(num_biological_covariate)]   
    
    # Get Parameters
    kmean = KMeans(n_clusters = sites // sites_per_cluster, random_state = 0)
    distributedClusterComBat = DistributedClusterComBat(kmean)
    distributedClusterComBat.fit(data, np.array(covariate).T, batch)
    parameters = distributedClusterComBat.parameters

    # plot 
    plot_PCA(data, batch, name = "Features-SiteID")
    plot_PCA(parameters, [i + 1 for i in range(sites)], name = "Parameter-SiteID")

with open("SyntheticDataConfig/parameter_distribution.json", "r") as read_file:
    configurations = json.load(read_file)

syntheticDataNumber = 1

for configuration in configurations:
    random.seed(1)
    np.random.seed(seed = 1)

    sites = configuration["sites"]
    samples_per_site = configuration["samples_per_site"]
    features = configuration["features"]
    sites_per_cluster = configuration["sites_per_cluster"]
    num_biological_covariate = configuration["num_biological_covariate"]
    
    experiment(sites = sites, samples_per_site = samples_per_site, features = features, num_biological_covariate = num_biological_covariate, sites_per_cluster = sites_per_cluster)
