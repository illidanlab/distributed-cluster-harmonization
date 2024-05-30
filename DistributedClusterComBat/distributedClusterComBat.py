import pandas as pd
import numpy as np
from DistributedComBat import distributedCombat as dc
from sklearn.cluster import KMeans

class DistributedClusterComBat:
    def __init__(self, clustering_algorithm = None):
        if clustering_algorithm is not None:
            self.clustering_algorithm = clustering_algorithm
        else:
            self.clustering_algorithm = KMeans()
    
    def fit(self, data, biological_covariates, batch):
        self.parameters = self.getClusterDistributedComBatParameters(np.array(data).T, np.array(biological_covariates), batch)
        self.clustering_algorithm.fit(self.parameters)
        cluster_index = self.clustering_algorithm.predict(self.parameters)
        self.output, self.esitimates = self.getClusterDistributedComBat(np.array(data).T, biological_covariates, batch, cluster_index)
        return self.output
    
    def harmonize(self, data_test, biological_covariates_test):
        parameters_test = self.getClusterDistributedComBatParameters(np.array(data_test).T, np.array(biological_covariates_test), [0 for i in range(len(data_test))])
        cluster_index_test = self.clustering_algorithm.predict(parameters_test)
        sample_wise_cluster_index = [cluster_index_test[0] for i in range(len(data_test))]
        return self.ClusterDistributedCombatFromTraining(np.array(data_test).T,np.array(sample_wise_cluster_index), self.esitimates)["data"].T

    def getClusterDistributedComBat(self, data, covariate, batch, batch_c):
        dat = pd.DataFrame(data)
        mod = pd.DataFrame(np.array(covariate))
        batch_col = "batch"
        covars = pd.DataFrame({batch_col: batch})

        ### Distributed ComBat: No reference batch ####
        ## Step 1
        site_outs = []
        for b in sorted(covars[batch_col].unique()):
            s = list(map(lambda x: x == b, covars[batch_col]))
            df = dat.loc[:, s]
            bat = covars[batch_col][s]
            bat = bat.astype("category")

            x = mod.loc[s, :]
            f = "Pickle/site_out_" + str(b) + ".pickle"
            out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f)
            site_outs.append(f)

        central = dc.distributedCombat_central(site_outs)

        ### Step 2
        site_outs = []
        for b in sorted(covars[batch_col].unique()):
            s = list(map(lambda x: x == b, covars[batch_col]))
            df = dat.loc[:, s]
            bat = covars[batch_col][s]
            bat = bat.astype("category")

            x = mod.loc[s, :]
            f = "Pickle/site_out_" + str(b) + ".pickle"
            out = dc.distributedCombat_site(
                df, bat, x, verbose=True, central_out=central, file=f
            )
            site_outs.append(f)
        central = dc.distributedCombat_central(site_outs)

        output = None

        ## Compare distributed vs original
        site_outs = []
        error = []
        perror = []  # percent difference
        estimates = []

        for b in sorted(covars[batch_col].unique()):
            s = list(map(lambda x: x == b, covars[batch_col]))
            df = dat.loc[:, s]
            bat = covars[batch_col][s]
            bat = bat.astype("category")
            x = mod.loc[s, :]
            f =  "Pickle/site_out_"  + str(b) + ".pickle"
            out = dc.distributedCombat_site(
                df, bat, x, central_out=central, file=f
            )    
            estimates.append(out["estimates"])
            out = np.array(out["dat_combat"])
            out = out.T
            if output is None:
                output = out
            else:
                output = np.concatenate((output, out))        

        final_estimates = {
            "gamma_star": [],
            "delta_star": [],
            "stand_mean": estimates[0]["stand_mean"],
            "mod_mean": estimates[0]["mod_mean"],
            "var_pooled": np.expand_dims(estimates[0]["var_pooled"], axis = 1),
            "batches": []
        }

        for b in sorted(set(batch_c)):
            gamma_star = np.mean(np.array([estimates[i]["gamma_star"] for i in range(len(estimates)) if batch_c[i] == b]), axis = 0)
            delta_star = np.mean(np.array([estimates[i]["delta_star"] for i in range(len(estimates)) if batch_c[i] == b]), axis = 0)
            # grand_mean = 
            final_estimates["gamma_star"].append(gamma_star)
            final_estimates["delta_star"].append(delta_star)
            final_estimates["batches"].append(b)

        final_estimates["gamma_star"] = np.array(final_estimates["gamma_star"])
        final_estimates["delta_star"] = np.array(final_estimates["delta_star"])
        final_estimates["batches"] = np.array(final_estimates["batches"])

        return output, final_estimates

    def ClusterDistributedCombatFromTraining(self, dat,
                            batch,
                            estimates):
    
        batch = np.array(batch, dtype="str")
        new_levels = np.unique(batch)
        old_levels = np.array(estimates['batches'], dtype="str")
        missing_levels = np.setdiff1d(new_levels, old_levels)
        if missing_levels.shape[0] != 0:
            raise ValueError("The batches " + str(missing_levels) +
                            " are not part of the training dataset")


        wh = [int(np.where(old_levels==x)[0]) if x in old_levels else None for x in batch]

        var_pooled = estimates['var_pooled']
        stand_mean = estimates['stand_mean'][:, 0]
        mod_mean = estimates['mod_mean']
        gamma_star = estimates['gamma_star']
        delta_star = estimates['delta_star']
        n_array = dat.shape[1]   
        stand_mean = stand_mean+mod_mean.mean(axis=1)
        
        stand_mean = np.transpose([stand_mean, ]*n_array)
        bayesdata = np.subtract(dat, stand_mean)/np.sqrt(var_pooled)
        
        gamma = np.transpose(gamma_star[wh,:])
        delta = np.transpose(delta_star[wh,:])
        bayesdata = np.subtract(bayesdata, gamma)/np.sqrt(delta)
        
        bayesdata = bayesdata*np.sqrt(var_pooled) + stand_mean
        out = {
            'data': bayesdata,
            'estimates': estimates
        }
        return out

    def getClusterDistributedComBatParameters(self, data, covariate, batch):
        dat = pd.DataFrame(data)
        mod = pd.DataFrame(np.array(covariate))
        batch_col = "batch"
        covars = pd.DataFrame({batch_col: batch})

        batchid2parametersid = {}

        ### Distributed ComBat: No reference batch ####
        ## Step 1
        site_outs = []
        for b in covars[batch_col].unique():
            s = list(map(lambda x: x == b, covars[batch_col]))
            df = dat.loc[:, s]
            bat = covars[batch_col][s]
            bat = bat.astype("category")
            x = mod.loc[s, :]
            f = "Pickle/site_out_" + str(b) + ".pickle"
            out = dc.distributedCombat_site(df, bat, x, verbose=True, file=f)
            site_outs.append(f)
            batchid2parametersid[b] = len(site_outs) - 1
        
        parameters = dc.distributedCombat_central_parameters(site_outs)

        return parameters



