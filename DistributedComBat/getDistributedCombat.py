import pandas as pd
import numpy as np
from DistributedComBat import distributedCombat as dc

def getDistributedComBat(data,covariate, batch):
    dat = pd.DataFrame(data)
    mod = pd.DataFrame(np.array(covariate))
    batch_col = "batch"
    covars = pd.DataFrame({batch_col: batch})

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

    central = dc.distributedCombat_central(site_outs)

    ### Step 2
    site_outs = []
    for b in covars[batch_col].unique():
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

    site_outs = []
    for b in covars[batch_col].unique():
        s = list(map(lambda x: x == b, covars[batch_col]))
        df = dat.loc[:, s]
        bat = covars[batch_col][s]
        bat = bat.astype("category")
        x = mod.loc[s, :]
        f =  "Pickle/site_out_"  + str(b) + ".pickle"
        out = dc.distributedCombat_site(
            df, bat, x, central_out=central, file=f
        )    
        out = np.array(out["dat_combat"])
        out = out.T
        if output is None:
            output = out
        else:
            output = np.concatenate((output, out))        
    return output