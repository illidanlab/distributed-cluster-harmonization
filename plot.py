import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure

def plot_PCA(y, batch, index = "Site", name = "fig1"):
    figure(figsize=(8, 8), dpi=300)
    plt.rcParams.update({'font.size': 15})

    # numpy
    y = np.array(y)
    batch = np.array(batch)

    # standardize
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(y)

    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_standardized)

    # plot by subject
    for b in list(set(batch)):
        plt.scatter(principal_components[batch == b, 0], principal_components[batch == b, 1], label = index+" "+str(b))
    
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend(loc = "upper right")
    plt.savefig("Figures/{}.png".format(name), dpi = 300, bbox_inches="tight")
    plt.close()