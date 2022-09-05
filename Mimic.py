import numpy as np
from scipy.stats import multivariate_normal

def mimic_score(data_trf, grids, vals, fitted, fill_up, vals_only=False):  
    #  assign grid cell to data point and get corresponding entry in "fitted" or 0
    fitted_grid = np.zeros((len(data_trf), len(data_trf[0]))) # points x dims
    fill_grid = np.zeros((len(data_trf), len(data_trf[0]))) # points x dims
    vals_grid = np.zeros((len(data_trf), len(data_trf[0]))) # points x dims
    for d in range(len(data_trf[0])):
        # organize in grid cells: 0 = smaller; len(grids[0]) = larger
        grid_dim = np.digitize(data_trf[:,d], grids[d]) # points x dims
        map_to_fitted = np.vectorize(lambda idx: 0 if idx<=0 or idx>=len(grids[d]) else fitted[d][idx-1])
        map_to_fill = np.vectorize(lambda idx: 0 if idx<=0 or idx>=len(grids[d]) else fill_up[d][idx-1])
        map_to_vals = np.vectorize(lambda idx: 0 if idx<=0 or idx>=len(grids[d]) else vals[d][idx-1])
        fitted_grid[:, d] = map_to_fitted(grid_dim)
        fill_grid[:, d] = map_to_fill(grid_dim)
        vals_grid[:, d] = map_to_vals(grid_dim)
     
    if vals_only: return np.sum(np.log(vals_grid + 1), axis=1)
    s1 = np.sum(np.log(fitted_grid + 1), axis=1)  # fitted distribution
    s2 = np.sum(np.log(fill_grid + 1), axis=1)   # fill_up
    s = s1 + len(data_trf[0])*s2   # score as the sum of both (weighted?)
    s[np.sum(fill_grid, axis=1) == 0] = 0   # 0 score where we don't fill anything up
    s[np.prod(fitted_grid, axis=1) == 0] = 0   # 0 score for unprobable entries
    return s

# greedily add points and see if that improves likelihood
def add_points(data, grids, fitted, num_fill_up, score, data_rest, batchsize=10, num_restarts=10):
    add_idcs = np.array([], dtype=np.int32)
    p_model_given_data = P_model_given_data(data, grids, fitted)
    num_fill = int(min(max(num_fill_up), len(score>0)))
    batches = np.append([batchsize] * (num_fill // batchsize), [num_fill % batchsize])
    tries = 0
    for i in range(len(batches)):
        candidates = np.random.choice(range(len(score)), int(batches[i]), p=score).astype(int)
        d_new = np.vstack((data, data_rest[candidates]))
        P_new = P_model_given_data(d_new, grids, fitted)
        if P_new <= p_model_given_data: # stopping if likelihood gets worse
            if tries < num_restarts:
                i += -1 # try again!
                tries += 1
            else:
                tries = 0
            continue
        p_model_given_data = P_new
        add_idcs = np.append(add_idcs, candidates)
        score[add_idcs] = 0
        data = np.vstack((data, data_rest[add_idcs]))
        if np.sum(score) == 0: break
        score = score / np.sum(score)  #convert to probability distribution
        
    return add_idcs

def P_model_given_data(data, grids, fitted):
    p_data_given_model = P_data_given_model(data, grids, fitted)
    p_data = P_data(data, grids, fitted, s=3)
    p_model = 1 #same model for all datasets, so no need to calculate that!
    p_model_given_data = p_data_given_model - p_data
    #print("p_data_given_model - p_data = %.2f - %.2f = %.2f" % (p_data_given_model, p_data, p_model_given_data))
    return p_model_given_data

def P_data_given_model(data, grids, fitted):
    res = 0
    for d in range(len(data[0])):
        fitted[d][fitted[d]==0] = 0.00001  #use non-truncated results later!!
        hh = np.histogram(data[:,d], bins=grids[d])[0] #histogram heights
        f = fitted[d] / np.sum(fitted[d])
        tmp = hh * np.log(f)
        tmp[hh==0] = 0 # 0 times whatever should be 0
        res += np.sum(tmp)
    return res

def P_data(data, grids, fitted, s=3):
    mean = np.array([(g[0] + g[-1])/2.0 for g in grids])
    std = (mean - np.array([g[0] for g in grids])) / 3.0
    cov = np.zeros((len(grids), len(grids)))
    np.fill_diagonal(cov, std**2)
    
    mn = multivariate_normal(mean, cov)#.pdf(data)
    mids = [(np.array(grids[d][0:-1]) + np.array(grids[d][1:])) / 2 for d in range(len(grids))]
    cubesize = np.prod(np.array([g[1] - g[0] for g in grids]))
    
    data_in_bins = np.column_stack([np.digitize(data[:,i], grids[i])-1 for i in range(len(grids))])
    corresp_mids = np.column_stack([mids[i][data_in_bins[:,i]] for i in range(len(grids))])
    probs = mn.pdf(corresp_mids) * cubesize
    
    return np.sum(np.log(probs))

