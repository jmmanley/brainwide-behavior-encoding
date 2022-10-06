"""
Shared Variance Component Analysis (SVCA, Stringer & Pachitariu et al. 2019 Science), a dimensionality
reduction method for neural timeseries data in order to identify latent dimensions of global shared variation.

Jason Manley, 2021
jmanley at rockefeller dot edu
"""

import numpy as np

def SVCA(X, ntrain=None, ntest=None, itrain=None, itest=None, n_randomized=None, shuffle=False, **kwargs):
    """
    Shared Variance Component Analysis
    Ported from MATLAB, originally by Stringer & Pachitariu et al 2018a
    https://github.com/MouseLand/stringer-pachitariu-et-al-2018a
    https://doi.org/10.1126/science.aav7893
      
    INPUTS: 
    X      : NxT neural data matrix
    ntrain : indices of first neural subset
    ntest  : indices of second neural subset
    itrain : indices of training timepoints
    itest  : indices of test timepoints
    
    OUTPUTS:
    sneur  : shared variance of each covariance component
    vneur  : total variance of each covariance component
    u      : left  eigenvectors of covariance matrix between ntrain and ntest during itrain timepoints
    v      : right eigenvectors of covariance matrix between ntrain and ntest during itrain timepoints

    Additional features added in this implementation:
    - n_randomized : if not None, use a randomized SVD algorithm to find n_randomized SVCs (default: None)
    - shuffle      : shuffling by circularly permuting each neuron's timeseries (default: False)
    """

    if shuffle:
        Xshuff = X.copy()
        for i in range(X.shape[0]):
            Xshuff[i,:] = np.roll(Xshuff[i,:], np.random.randint(X.shape[1])) # circularly permute - DOES NOT WORK FOR # NEURONS >> # TIMEPOINTS
        X = Xshuff

    if ntrain is None or ntest is None:
        ntrain, ntest = train_test_split_idx(X.shape[0])
        
    if itrain is None or itest is None:
        itrain, itest = train_test_split_idx(X.shape[1])

    cov = X[ntrain,:][:,itrain] @ X[ntest,:][:,itrain].T
    
    if n_randomized is None:
        # Perform full SVD
        u,s,vt = np.linalg.svd(cov)
    else:
        # Approximate SVD with randomized SVD
        from sklearn.utils.extmath import randomized_svd
        u,s,vt = randomized_svd(cov, n_components=n_randomized, **kwargs)

    if u.shape[1] != vt.shape[0]:
        # in case len(ntrain)!=len(ntest), keep maximum number of SVCs
        nsvc = min(u.shape[1],vt.shape[0])
        vt = vt[:nsvc,:]
        u = u[:,:nsvc]

    s1 = u.T @ X[ntrain,:][:,itest]
    s2 = vt  @ X[ntest, :][:,itest]

    sneur = np.sum(s1 * s2,axis=1)
    varneur = np.sum(s1**2 + s2**2,axis=1)/2
    
    return sneur, varneur, u, vt.T


### ADDITIONAL UTILITIES ###


def train_test_split_idx(N, train_frac=0.5, interleave=0, **kwargs):
    """
    Splits data into train and test sets.
    """

    from sklearn.model_selection import train_test_split

    if interleave>0:
        # Useful for time series data
        # Chunks data into blocks of length interleave and 
        # randomly assigns each chunk to train or test
        indx = np.ceil(np.arange(N) / interleave)
        Nblocks = int(np.max(indx))
        irand = np.random.permutation(Nblocks)
        Ntrain = int(np.ceil(train_frac * Nblocks))
        Ntest = Nblocks - Ntrain
        itrain = np.where(np.isin(indx, np.sort(irand[:Ntrain])))[0]
        itest  = np.where(np.isin(indx, np.sort(irand[Ntrain:])))[0]
    
        return itrain, itest
    else:    
        return train_test_split(np.arange(N), train_size=train_frac, **kwargs)


def checkerboard_centers(centers, checkerboard):
    """
    Returns indices of variables split into two sets according to a 
    checkerboard pattern based on the positions located in centers.

    centers : 2xN positions
    checkerboard : size of square in checkerboard
    """

    nbin_x = int(np.round((np.max(centers[0, :]) -
                     np.min(centers[0, :])) / checkerboard))

    if nbin_x < 2: nbin_x = 2

    nbin_y = int(np.round((np.max(centers[1, :]) -
                 np.min(centers[1, :])) / checkerboard))

    if nbin_y < 2: nbin_y = 2

    bin_x = np.linspace(np.min(centers[0, :])-1, np.max(
        centers[0, :]+1), num=nbin_x+1)
    bin_y = np.linspace(np.min(centers[1, :])-1, np.max(
        centers[1, :]+1), num=nbin_y+1)
    
    idx1 = []
    idx2 = []

    def is_odd(num):
        return num & 0x1

    for i in range(len(bin_x)-1):
        for j in range(len(bin_y)-1):
            ixx = np.where(np.all([centers[0, :] > bin_x[i], centers[0, :] < bin_x[i+1],
                           centers[1, :] > bin_y[j], centers[1, :] < bin_y[j+1]], axis=0))[0]
            if is_odd(i) == is_odd(j):
                idx1.append(ixx)
            else:
                idx2.append(ixx)

    idx1 = np.asarray([x for y in idx1 for x in y])
    idx2 = np.asarray([x for y in idx2 for x in y])

    return idx1, idx2
