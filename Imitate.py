import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FastICA, PCA
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tools.eval_measures import bic
from scipy.optimize import minimize
from scipy.integrate import dblquad
import itertools
from scipy.special import logsumexp, rel_entr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import copy
from statsmodels.nonparametric.kde import KDEUnivariate
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import scipy.optimize as optimization

# for debugging
from collections import Counter

# for data generation
from sklearn.datasets import make_classification
from math import ceil, log

# suppress warning: warnings are only thrown by ICA if ICA cannot converge. In this case,
# the data is already Gaussian (i.e., that's a positive outcome!)
np.seterr(divide='ignore', invalid='ignore')

'''Kernel Density Estimation (from Imitate paper)'''
class DE_kde():
    def __init__(self, num_bins):
        self.num_bins = num_bins
        
    def estimate(self, data, d_min, d_max, weights):
        d_range = (d_min - 0.5*(d_max-d_min), d_max + 0.5*(d_max-d_min))
        gridsize = (d_range[1] - d_range[0]) / self.num_bins

        self.grid = [(d_range[0] + i*gridsize) for i in range(self.num_bins+1)]
        self.mids = self.grid[:-1] + np.diff(self.grid)/2
        
        kde = KDEUnivariate(data)
        try:
            kde.fit(bw='silverman', kernel='gau', fft=False, weights=weights)
        except:
            kde.fit(bw=0.01, kernel='gau', fft=False, weights=weights)
        
        self.values = [kde.evaluate(i)[0] if kde.evaluate(i) > 0 else 0 for i in self.mids]
        
'''Scaled normal distribution (from Imitate paper)'''
class scaled_norm():
    def __init__(self, truncate_std=3, ends_zero=True, print_loss=False):
        self.ends_zero = ends_zero
        self.print_loss = print_loss
        self.truncate_std = truncate_std
        
    def func_trunc(x, scale, mu, sigma, trunc):
        res = scaled_norm.func(x, scale, mu, sigma)
        res[abs(x - mu) > trunc*sigma] = 0
        return res
        
    def func(x, scale, mu, sigma):
        return scale * norm(mu, sigma).pdf(x)
    
    def weighted_dist(weights, points_x, points_y, params):
        return (((scaled_norm.func(points_x, *params) - points_y)**2) * weights).sum()
    
    def constraint(points_x, points_y, params):
        return 2*points_y.sum() - scaled_norm.func(points_x, *params).sum()

    def fit(self, points_x, points_y, data, returnParams=False):
        d_mean = points_x[np.argmax(points_y)]  # highest bin
        d_std = max(0.0001, np.sqrt( np.sum((np.array(data) - d_mean)**2) / (len(data) - 1) ))
        d_scale = max(points_y) / max(scaled_norm.func(points_x, 1, d_mean, d_std))
        p0 = np.array([d_scale, d_mean, d_std])  # initial parameters
        weights = np.array(points_y) ** 2
        weights = [max(weights[i], 0.01*max(points_y)) for i in range(len(weights))]
        optimize_me = lambda p: scaled_norm.weighted_dist(weights, points_x, points_y, p)
        if self.ends_zero:
            weights[0] = weights[-1] = max(points_y)
        try:
            bounds = [[0.01, 2*d_scale], [points_x[0], points_x[-1]], [0.0001,(points_x[-1]-points_x[0])/2]]
            constr = lambda p: scaled_norm.constraint(np.array(points_x), np.array(points_y), p)
            res = minimize(optimize_me, p0, method='SLSQP', bounds=bounds, constraints={'type':'ineq', 'fun': constr})
            if constr(res.x) < 0:
                if returnParams:  return np.array(points_y), p0
                else: return np.array(points_y)
            if self.print_loss:
                print("final weighted loss:", optimize_me(res.x))
            if returnParams: return scaled_norm.func_trunc(points_x, *res.x, self.truncate_std), res.x
            else: return scaled_norm.func_trunc(points_x, *res.x, self.truncate_std)
        except:
            #print("pdf fitting with sigma was not successful")
            if returnParams:  return np.array(points_y), p0
            else: return np.array(points_y)
            
class scaled_norm_bounded():
    def __init__(self, truncate_std=3, bound_min=None, bound_max=None, ends_zero=True, ends_zero_strength=1, print_loss=False):
        self.ends_zero = ends_zero
        self.ends_zero_strength = ends_zero_strength
        self.print_loss = print_loss
        self.truncate_std = truncate_std
        self.b_min = bound_min
        self.b_max = bound_max
        
    def func_trunc(x, scale, mu, sigma, trunc):
        res = scaled_norm.func(x, scale, mu, sigma)
        res[abs(x - mu) > trunc*sigma] = 0
        return res
        
    def func(x, scale, mu, sigma):
        return scale * norm(mu, sigma).pdf(x)
    
    def weighted_dist(weights, points_x, points_y, params):
        return (((scaled_norm.func(points_x, *params) - points_y)**2) * weights).sum()
    
    def constraint(points_x, points_y, params):
        return 2*points_y.sum() - scaled_norm.func(points_x, *params).sum()

    def fit(self, points_x, points_y, data, returnParams=False):
        d_mean = points_x[np.argmax(points_y)]  # highest bin
        d_std = max(0.0001, np.sqrt( np.sum((np.array(data) - d_mean)**2) / (len(data) - 1) ))
        d_scale = max(points_y) / max(scaled_norm.func(points_x, 1, d_mean, d_std))
        p0 = np.array([d_scale, d_mean, d_std])  # initial parameters
        weights = np.array(points_y) ** 2
        weights = np.array([max(weights[i], 0.01*max(points_y)) for i in range(len(weights))])
        optimize_me = lambda p: scaled_norm.weighted_dist(weights, points_x, points_y, p)
        if self.ends_zero:
            weights[0] = weights[-1] = self.ends_zero_strength * max(points_y)
            if self.b_min is not None: weights[points_x <= self.b_min] = self.ends_zero_strength * max(points_y)
            if self.b_max is not None: weights[points_x >= self.b_max] = self.ends_zero_strength * max(points_y)
        try:
            bounds = [[0.01, 2*d_scale], [points_x[0], points_x[-1]], [0.0001,(points_x[-1]-points_x[0])/2]]
            constr = lambda p: scaled_norm.constraint(np.array(points_x), np.array(points_y), p)
            res = minimize(optimize_me, p0, bounds=bounds)#, method='SLSQP')#, constraints={'type':'ineq', 'fun': constr})
            if constr(res.x) < 0:
                if returnParams:  return np.array(points_y), p0
                else: return np.array(points_y)
            if self.print_loss:
                print("final weighted loss:", optimize_me(res.x))
            if returnParams: return scaled_norm.func_trunc(points_x, *res.x, self.truncate_std), res.x
            else: return scaled_norm.func_trunc(points_x, *res.x, self.truncate_std)
        except:
            print("pdf fitting with sigma was not successful")
            if returnParams:  return np.array(points_y), p0
            else: return np.array(points_y)
            
            
def remove_outliers_lof(data, k=10):
    k = min((len(data), k))
    lof = LocalOutlierFactor(n_neighbors=k)
    stays = lof.fit_predict(data)
    return np.array(data)[stays == 1]

# finds the number of bins that best describe the data (AICc)
def getBestNumBins(bins, d):
    min_aicc = np.Inf
    best_bins = 0
    
    for num_bins in bins:
        # get histogram
        d_range = (min(d) - 0.5*(max(d)-min(d)), max(d) + 0.5*(max(d)-min(d)))
        values, grid = np.histogram(d, bins=num_bins, density=True, range=d_range)
        mids = grid[:-1] + np.diff(grid)/2

        # calc MLE: ln(P[data | model])
        bin_per_p = np.digitize(d, grid)
        ln_L = sum(np.log( values[bin_per_p - 1] ))

        # calc aicc
        aicc = 2*num_bins*len(d) / (len(d)-num_bins-1) - 2*ln_L
        #bic =  log(len(d))*num_bins - 2*ln_L
        #bic =  2*num_bins - 2*ln_L # aic
        
        if aicc < min_aicc:
            min_aicc = aicc
            best_bins = num_bins

    return best_bins

def Imitate(data, plots=False, print_loss=False, num_bins=0, bounds=None, strength=1):
    grids = []
    vals = []
    fitted = []
    fill_up = []
    num_fill_up = []
    params = []

    # consider every dimension
    for line in range(len(data[0])):
        
        d = data[:,line]   # project onto line
        if num_bins==0: num_bins = getBestNumBins(range(min(int(len(d)/2),20), min(60, len(d)-2)), d)
        
        d_range = (min(d) - 0.5*(max(d)-min(d)), max(d) + 0.5*(max(d)-min(d)))
        grid = [(d_range[0] + i*((d_range[1] - d_range[0]) / num_bins)) for i in range(num_bins+1)]
        mids = grid[:-1] + np.diff(grid)/2
        kde = KDEUnivariate(d)
        try:
            kde.fit(bw='silverman', kernel='gau', fft=False)      
        except:
            kde.fit(bw=0.01, kernel='gau', fft=False)  
        values = np.array([kde.evaluate(i)[0] if kde.evaluate(i) > 0 else 0 for i in mids])
        values_scaled = (len(d) / sum(values)) * values    # scale to absolute values
        grids.append(grid)
        vals.append(values_scaled)

        if bounds is not None:
            SN = scaled_norm_bounded(bound_min=bounds[line][0], bound_max=bounds[line][1], ends_zero=True, ends_zero_strength=strength)
        else:
            SN = scaled_norm()
        fitted_, p = SN.fit(mids, values_scaled, d, returnParams=True) # fit Gaussian   
        params.append(p)  
        fitted.append(fitted_)

        diff = fitted_ - values_scaled    # decide where to fill up
        diff[diff < 1] = 0 # don't fill if we are not sure that we need the point
        fill_up.append(np.floor(diff).astype(int))

        num_fill_up.append(sum(np.floor(diff)))    # count how much needs to be filled
        
        if plots:
            plt.bar(mids, values_scaled, label='true data', width=(mids[1]-mids[0]))
            plt.bar(mids, diff, bottom=values_scaled, label='fill up', width=(mids[1]-mids[0]))
            plt.plot(mids, fitted_, label='fitted', c='red')
            plt.legend()
            plt.show()
    mean = np.array(params)[:,1]
    cov = np.zeros((len(data[0]), len(data[0])))
    np.fill_diagonal(cov, np.array(params)[:,2]**2)
    scale = np.array(params)[:,0]

    return grids, vals, fitted, fill_up, num_fill_up, scale, mean, cov