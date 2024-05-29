def generate(sites = 30, samples_per_site = 30, features = 30, num_biological_covariate = 10, sites_per_cluster = 3, k = 20):
    from scipy.stats import invgamma   
    from scipy.stats import uniform
    from scipy.stats import gamma
    from scipy.stats import norm
    from scipy.stats import halfcauchy
    import numpy as np
        
    # build multiplicative batch effect 
    lamda = list(gamma(50, 50).rvs(size = sites))
    v = list(gamma(50, 1).rvs(size = sites))
    delta = [list(gamma(lamda[i] * v[i], v[i]).rvs(size = features)) for i in range(sites)]
    delta = [[1/j for j in i] for i in delta]

    # build additive batch effect 
    Y = list(uniform(0, 0.1).rvs(size = sites))
    tau = list(invgamma(2, 0.5).rvs(size = sites))
    gamma = [list(norm(Y[i], tau[i]).rvs(size = features)) for i in range(sites)]

    # build feature parameter
    theta = [list(norm(0, 1).rvs(size = num_biological_covariate)) for g in range(features)]
    sigma = list(halfcauchy(0.2).rvs(size = features))
    alpha = list(uniform(0, 0.5).rvs(size = features))

    # make sure linearly separated between label
    labels = [[] for i in range(sites)]

    biological_covariate = [[[] for j in range(samples_per_site)] for i in range(sites)]
    for i in range(sites):
        for j in range(samples_per_site):
            labels[i].append(j%2)
            if j%2 == 0:
                biological_covariate[i][j] = list(norm(0.5, 0.5).rvs(size = num_biological_covariate))
            else:
                biological_covariate[i][j] = list(norm(-0.5, 0.5).rvs(size = num_biological_covariate))

    # build final output (unharmonized and groundtruth value)
    y = [[[norm(alpha[g] + np.dot(theta[g], biological_covariate[i][j]) + k*gamma[i//sites_per_cluster][g], (delta[i//sites_per_cluster][g]**2)*(sigma[g]**2)).rvs(size = 1)[0] for g in range(features)] for j in range(samples_per_site)] for i in range(sites)]
    ground_truth = [[[alpha[g] + np.dot(theta[g], biological_covariate[i][j]) for g in range(features)] for j in range(samples_per_site)] for i in range(sites)]
    
    return y, biological_covariate, ground_truth, labels