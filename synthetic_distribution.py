import pandas as pd
import numpy as np
from generateData import *
from plot import *
import copy
from sklearn.cluster import KMeans
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from ComBat.combat import *
import copy
import json 
from ClusterComBat.clusterComBat import ClusterComBat

plt.rcParams.update({'font.size': 11})

# comment out the next line to see the warning
warnings.simplefilter('ignore')

def experiment(sites = 30, samples_per_site = 30, features = 30, num_biological_covariate = 10, sites_per_cluster = 3):
   
    # Generate and Preprocess data
    y, biological_covariate, expected, labels = generate(sites = sites, samples_per_site = samples_per_site, features = features, num_biological_covariate = num_biological_covariate, sites_per_cluster = sites_per_cluster, k = 10)
    
    for exp in range(1):

        # Clustering Algorithm
        kmean = KMeans(n_clusters = sites // sites_per_cluster, random_state = 0)

        # Split stratify by ground truth cluster
        ground_truth_cluster = [i//sites_per_cluster for i in range(len(y))]
        y_train, y_test, ground_truth_train, ground_truth_test, label_train, label_test, biological_covariate_train, biological_covariate_test =\
        train_test_split(y, expected, labels, biological_covariate, stratify = ground_truth_cluster, test_size = 0.3, random_state = exp)

        # Flatten Data
        ground_truth_train = [ground_truth_train[i][j] for i in range(len(ground_truth_train)) for j in range(len(ground_truth_train[i])) ]
        ground_truth_test = [ground_truth_test[i][j] for i in range(len(ground_truth_test)) for j in range(len(ground_truth_test[i])) ]
        ground_truth = list(ground_truth_train) + list(ground_truth_test)

        # Initialize data
        data = [y_train[i][j] for i in range(len(y_train)) for j in range(len(y_train[i]))]
        batch = [i+1 for i in range(len(y_train)) for j in range(len(y_train[i]))]
        label_train = [label_train[i][j] for i in range(len(y_train)) for j in range(len(y_train[i]))]
        covariate = [[biological_covariate_train[i][j][g] for i in range(len(y_train)) for j in range(len(y_train[i]))] for g in range(num_biological_covariate)]   
        all_batch = copy.deepcopy(batch)
    
        # Get Clusters ComBat and train cluster ComBat classifier
        clusterComBat = ClusterComBat(kmean)
        data_train = clusterComBat.fit(data, continuous_biological_covariates = np.array(covariate).T)
        cluster_combat_train = copy.deepcopy(data_train)

        # Get ComBat and train ComBat classifier
        covars = {'batch': batch}
        for g in range(num_biological_covariate):
            covars['covariate'+str(g)] = covariate[g]
        continuous_cols = ['covariate'+str(g) for g in range(num_biological_covariate)]
        data_train = neuroCombat(dat=np.array(data).T,
            covars=pd.DataFrame(covars),
            batch_col="batch",
            continuous_cols=continuous_cols)["data"].T
        original_combat_train = copy.deepcopy(data_train)   

        # Initialize for test
        original_combat, cluster_combat, unharmonized, labels_test = [], [], [], []
        
        # max batch index
        max_id = max(batch)
        current_batch = max_id

        for i in range(len(y_test)):
            data_test = []
            current_batch += 1

            # Testing data
            for j in range(len(y_test[i])):
                batch.append(max_id + 1)
                data.append(y_test[i][j])
                labels_test.append(label_test[i][j])
                data_test.append(y_test[i][j])
                all_batch.append(current_batch)
                label_train.append(label_test[i][j])
                for g in range(num_biological_covariate):
                    covariate[g].append(biological_covariate_test[i][j][g])

            # Retrain ComBat for Original ComBat
            covars = {'batch': batch}
            for g in range(num_biological_covariate):
                covars['covariate'+str(g)] = covariate[g]
            covars = pd.DataFrame(covars) 
            
            data_combat_original = list(neuroCombat(dat=np.array(data).T,
                covars=pd.DataFrame(covars),
                batch_col="batch",
                continuous_cols=continuous_cols)["data"].T)

            # Get Cluster ComBat    
            data_combat_cluster = list(clusterComBat.harmonize(data_test))
            
            # Add harmonized test data
            cluster_combat += data_combat_cluster
            original_combat += data_combat_original[-len(y_test[i]):]
            unharmonized += data[-len(y_test[i]):]
            
            # Remove Test Subject From Training for Next Test
            batch = batch[:-len(y_test[i])]
            data = data[:-len(y_test[i])]
            for g in range(num_biological_covariate):
                covariate[g] = covariate[g][:-len(y_test[i])]

        cluster_combat = list(cluster_combat_train) + cluster_combat
        original_combat = list(original_combat_train) + original_combat
        unharmonized = list(data) + unharmonized

        plot_PCA(ground_truth, all_batch, name = "GroundTruth-SiteID")
        plot_PCA(cluster_combat, all_batch, name = "ClusterComBat-SiteID")
        plot_PCA(original_combat, all_batch, name = "OriginalComBat-SiteID")
        plot_PCA(unharmonized, all_batch, name = "Unharmonization-SiteID")

        plot_PCA(ground_truth, label_train, index = "Label", name = "GroundTruth-Label")
        plot_PCA(cluster_combat, label_train, index = "Label", name = "ClusterComBat-Label")
        plot_PCA(original_combat, label_train, index = "Label", name = "OriginalComBat-Label")
        plot_PCA(unharmonized, label_train, index = "Label", name = "Unharmonization-Label")

with open("SyntheticDataConfig/synthetic_distribution.json", "r") as read_file:
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
