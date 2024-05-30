from __future__ import absolute_import, print_function
import pandas as pd
import numpy as np
from ComBat.combat import *
import pandas as pd
from sklearn.cluster import KMeans

class ClusterComBat:
    """
    A class to perform batch effect correction using ComBat and clustering algorithms.

    Attributes
    ----------
    clustering_algorithm : object
        The clustering algorithm to use. Defaults to KMeans if not provided.

    Methods
    -------
    fit(data, continuous_biological_covariates=None, categorical_biological_covariates=None):
        Fits the clustering algorithm to the data and performs ComBat harmonization.

    harmonize(data_test):
        Harmonizes new test data using the trained ComBat estimates.
    """
     
    def __init__(self, clustering_algorithm = None):
        """
        Initializes the ClusterComBat instance with a specified clustering algorithm.

        Parameters
        ----------
        clustering_algorithm : object, optional
            The clustering algorithm to use. Defaults to KMeans if not provided.
        """

        if clustering_algorithm is not None:
            self.clustering_algorithm = clustering_algorithm
        else:
            self.clustering_algorithm = KMeans()

    def fit(self, data, continuous_biological_covariates = None, categorical_biological_covariates = None):
        """
        Fits the clustering algorithm to the data and performs ComBat harmonization.

        Parameters
        ----------
        data : a numpy array
            Neuroimaging data to harmonize with shape = (samples, features) e.g. cortical thickness measurements, image voxels, etc
        continuous_biological_covariates : a numpy array, optional
            Continuous biological covariates with shape = (samples, covariates) to include in the ComBat harmonization.
        categorical_biological_covariates : a numpy array, optional
            Categorical biological covariates with shape = (samples, covariates) to include in the ComBat harmonization.

        Returns
        -------
        np.ndarray
            The harmonized data with shape = (samples, features).
        """

        self.data = np.array(data)
        
        # Fit the clustering algorithm to the data
        self.clustering_algorithm.fit(self.data)

        # Predict cluster indices for each data point
        self.cluster_index = self.clustering_algorithm.predict(data)

        self.continuous_cols = None
        self.categorical_cols = None

        # Preprocess continuous covariates
        if continuous_biological_covariates is not None:
            self.continuous_biological_covariates = np.array(continuous_biological_covariates)
            self.continuous_cols = []
            self.covars = {'cluster_index': self.cluster_index}
            for covariate_index in range(self.continuous_biological_covariates.shape[1]):
                self.covars['continuous_covariate{}'.format(covariate_index)] = []
                self.continuous_cols.append('continuous_covariate{}'.format(covariate_index))
                for sample in range(self.continuous_biological_covariates.shape[0]):
                    self.covars['continuous_covariate{}'.format(covariate_index)].append(self.continuous_biological_covariates[sample][covariate_index])
        
        # Preprocess categorical covariates
        if categorical_biological_covariates is not None:
            self.categorical_biological_covariates = np.array(categorical_biological_covariates)
            self.categorical_cols = []
            self.covars = {'cluster_index': self.cluster_index}
            for covariate_index in range(self.categorical_biological_covariates.shape[1]):
                self.covars['categorical_covariate{}'.format(covariate_index)] = []
                self.categorical_cols.append('categorical_covariate{}'.format(covariate_index))
                for sample in range(self.categorical_biological_covariates.shape[0]):
                    self.covars['categorical_covariate{}'.format(covariate_index)].append(self.categorical_biological_covariates[sample][covariate_index])

        # Perform ComBat harmonization
        ComBatResult =  neuroCombat(dat=self.data.T,
            covars=pd.DataFrame(self.covars),
            batch_col="cluster_index",
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols)
        
        # Store ComBat estimates
        self.estimates = ComBatResult["estimates"]
        
        # Retrun harmonized data
        return ComBatResult["data"].T

    def harmonize(self, data_test):
        """
        Harmonizes new test data from unseen sites.

        Parameters
        ----------
        data_test : a numpy array
            Neuroimaging data from unseen sites to harmonize with shape = (samples, features) e.g. cortical thickness measurements, image voxels, etc

        Returns
        -------
        np.ndarray
            The harmonized test data.
        """

        # Convert test data to numpy array
        data_test = np.array(data_test)
        
        # Predict cluster indices for the test data
        batch_test = [float(i) for i in list(self.clustering_algorithm.predict(data_test))]

        # Perform ComBat harmonization using the stored estimates
        data_combat_cluster = neuroCombatFromTraining(dat = data_test.T,
            batch = np.array(batch_test),
            estimates = self.estimates)["data"] 

        # Return harmonized test data
        return data_combat_cluster.T







