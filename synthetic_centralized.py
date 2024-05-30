from ComBat.combat import *
import pandas as pd
import numpy as np
from generateData import *
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
from sklearn.model_selection import train_test_split
from ClusterComBat.clusterComBat import ClusterComBat
import json 

warnings.simplefilter('ignore')

def experiment(sites = 30, samples_per_site = 30, features = 30, num_biological_covariate = 10, sites_per_cluster = 3, exps = 30):
    # Generate and Preprocess data
    y, biological_covariate, expected, labels = generate(sites = sites, samples_per_site = samples_per_site, features = features, num_biological_covariate = num_biological_covariate, sites_per_cluster = sites_per_cluster)
    
    accuracy_cluster, accuracy_original, accuracy_unharmonized, reconstruction_cluster, reconstruction_original, reconstruction_unharmonized = [], [], [], [], [], []

    for exp in range(exps):

        # Clustering Algorithm
        kmean = KMeans(n_clusters = sites // sites_per_cluster, random_state = 0)

        # Classifier
        clf_cluster, clf_original, clf_unharmonized = LogisticRegression(random_state=0), LogisticRegression(random_state=0), LogisticRegression(random_state=0)
       
        # Split stratify by ground truth cluster
        ground_truth_cluster = [i//sites_per_cluster for i in range(len(y))]
        y_train, y_test, ground_truth_train, ground_truth_test, label_train, label_test, biological_covariate_train, biological_covariate_test =\
        train_test_split(y, expected, labels, biological_covariate, stratify = ground_truth_cluster, test_size = 0.3, random_state = exp)

        # Flatten Data
        ground_truth_test = [ground_truth_test[i][j] for i in range(len(ground_truth_test)) for j in range(len(ground_truth_test[i])) ]

        # Initialize data
        data = [y_train[i][j] for i in range(len(y_train)) for j in range(len(y_train[i]))]
        batch = [i+1 for i in range(len(y_train)) for j in range(len(y_train[i]))]
        label_train = [label_train[i][j] for i in range(len(y_train)) for j in range(len(y_train[i]))]
        covariate = [[biological_covariate_train[i][j][g] for i in range(len(y_train)) for j in range(len(y_train[i]))] for g in range(num_biological_covariate)]   
       
        # Train unharmonized classifier
        clf_unharmonized.fit(data, label_train)

        # Get Clusters ComBat and train cluster ComBat classifier
        clusterComBat = ClusterComBat(kmean)
        data_train = clusterComBat.fit(data, continuous_biological_covariates = np.array(covariate).T)
        clf_cluster.fit(data_train, label_train)

        # Get ComBat and train ComBat classifier
        covars = {'batch': batch}
        for g in range(num_biological_covariate):
            covars['covariate'+str(g)] = covariate[g]
        continuous_cols = ['covariate'+str(g) for g in range(num_biological_covariate)]
        data_train = neuroCombat(dat=np.array(data).T,
            covars=pd.DataFrame(covars),
            batch_col="batch",
            continuous_cols=continuous_cols)["data"].T
        clf_original.fit(data_train, label_train)

        # Initialize for test
        original_combat, cluster_combat, unharmonized, labels_test = [], [], [], []
        
        # max batch index
        max_id = max(batch)
        
        for i in range(len(y_test)):
            # Testing data
            data_test = []
            for j in range(len(y_test[i])):
                batch.append(max_id + 1)
                data.append(y_test[i][j])
                labels_test.append(label_test[i][j])
                data_test.append(y_test[i][j])
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

        # Calculate metrics
        accuracy_cluster.append(accuracy_score(clf_cluster.predict(cluster_combat), labels_test))
        accuracy_original.append(accuracy_score(clf_original.predict(original_combat), labels_test))
        accuracy_unharmonized.append(accuracy_score(clf_unharmonized.predict(unharmonized), labels_test))
        
        reconstruction_unharmonized.append(mean_squared_error(unharmonized, np.array(ground_truth_test), squared = False))
        reconstruction_original.append(mean_squared_error(original_combat, np.array(ground_truth_test), squared = False))
        reconstruction_cluster.append(mean_squared_error(cluster_combat, np.array(ground_truth_test), squared = False))

    return accuracy_original, accuracy_cluster, accuracy_unharmonized, reconstruction_original, reconstruction_cluster, reconstruction_unharmonized

with open("SyntheticDataConfig/reconstruction_and_classification_task.json", "r") as read_file:
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
    
    accuracy_original, accuracy_cluster, accuracy_unharmonized, reconstruction_original, reconstruction_cluster, reconstruction_unharmonized = experiment(sites = sites, samples_per_site = samples_per_site, features = features, num_biological_covariate = num_biological_covariate, sites_per_cluster = sites_per_cluster)

    print("Synthetic data", syntheticDataNumber)
    print("Reconstruction:")
    print("Unharmonized: {:.2f}±{:.2f}".format(np.mean(reconstruction_unharmonized), np.var(reconstruction_unharmonized)))
    print("Original ComBat: {:.2f}±{:.2f}".format(np.mean(reconstruction_original), np.var(reconstruction_original)))
    print("Cluster Combat: {:.2f}±{:.2f}".format(np.mean(reconstruction_cluster), np.var(reconstruction_cluster)))
  
    print()
    print("Accuracy:")        
    print("Unharmonzied: {:.2f}±{:.2f}".format(np.mean(accuracy_unharmonized)*100, np.var(accuracy_unharmonized)*100))
    print("Original ComBat: {:.2f}±{:.2f}".format(np.mean(accuracy_original)*100, np.var(accuracy_original)*100))
    print("Cluster ComBat: {:.2f}±{:.2f}".format(np.mean(accuracy_cluster)*100, np.var(accuracy_cluster)*100))

    print()

    syntheticDataNumber += 1
