"""
Tools for multivariate regression, primarily of predicting neural activity
from animal behavior, and vice versa.

Models implemented:
Reduced-rank linear regression
LSTM recurrent neural network

Jason Manley, 2022
jmanley at rockefeller dot edu
"""

import numpy as np


def LSTMRegression(xshape, yshape, optimizer='adam', loss='mse', **kwargs):
    """
    Regression using a simple recurrent neural network
    with a single LSTM layer and Dense layer.
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.LSTM(yshape[1], input_shape=(xshape[1], xshape[2]), dropout=0.1))
    model.add(tf.keras.layers.Dense(yshape[1], input_shape=(yshape[1],), activation=None))
    model.compile(optimizer=optimizer, loss=loss, **kwargs)
    
    return model


class RRRegression:
    """
    Reduced-rank regression, as performed by Stringer et al 2018a
    (https://github.com/MouseLand/stringer-pachitariu-et-al-2018a,
     https://doi.org/10.1126/science.aav7893)
     
    Mathematically like a canonical *covariation* analysis on the two
    sets of data X and Y, such that it finds the linear combinations of
    X that predict the most variance in Y.
    
    The rank can then be reduced at prediction time by approximating Y
    from only n projections of X.
    """
    
    def __init__(self, lam=0, rank=1):
        self.rank = rank
        self.lam = lam
    
    def fit(self, X, Y, **kwargs):
        Xn = X.shape[1]
        Yn = Y.shape[1]

        C = np.cov(np.hstack((X,Y)).T)

        CXX = C[:Xn,:Xn] + self.lam*np.eye(Xn)
        CYY = C[Xn:, Xn:]
        CYX = C[Xn:,:Xn]

        from scipy.linalg import fractional_matrix_power
        CXXMH = fractional_matrix_power(CXX, -0.5)

        M = CYX @ CXXMH
        M[np.isnan(M)] = 0

        # do SVD
        from sklearn.utils.extmath import randomized_svd
        u,s,vt = randomized_svd(M, n_components=self.rank, random_state=None, **kwargs)
        s = np.diag(s)

        b = CXXMH @ vt.T

        if np.sum(np.isnan(u[:,-1])) > 0:
            # remove nans from last col
            u = u[:,:-1]
            s = s[:-1,:][:,:-1]
            b = v[:, :-1]

        a = u @ s

        R2 = (np.diag(s) ** 2) / np.sum(np.var(Y),axis=0)
        
        self.a = a
        self.b = b
        self.R2 = R2
    
    def predict(self, X, rank=None):
        if rank is not None: 
            # potentially lower rank on fly
            if rank>self.rank:
                print('Cannot predict with rank', rank, 'as model was fit with rank', self.rank)
                rank = self.rank
        else:
            rank = self.rank
            
        return self.a[:,:rank] @ self.b[:,:rank].T @ X.T


### ADDITIONAL UTILITIES ###

def get_consecutive_chunks(t):
    """
    Finds consecutive sequences of integers in t.
    """
    
    diff = np.diff(t)
    shifts = np.where(diff>1)[0]+1
    shifts = np.concatenate((shifts, [len(t)]), axis=0)
    
    chunks = []
    
    for i in range(len(shifts)):
        if i == 0:
            start = 0
        else:
            start = shifts[i-1]
        
        chunks.append(t[start:shifts[i]])
    
    return chunks

def get_binned_sequences(X, Y, npre, npost, itrain, itest, navg=0, Y2=None):
    """
    Finds sequences of X with npre frames before and npost frames
    after instantaneous time, binned by averaging every navg frames.
    
    Removes any sequences overlapping in train and test timepoints (itrain, itest).
    """
    
    trains = get_consecutive_chunks(itrain)
    tests  = get_consecutive_chunks(itest)
    
    trainY = []
    trainY2 = []
    traints = []
    testts = []
    trainX = []
    testY = []
    testY2 = []
    testX = []

    for chunk in trains:
        curr = chunk[npre:-(npost+1)]
        if len(curr)>0:
            trainY.append(Y[curr,:])
            if Y2 is not None: trainY2.append(Y2[curr,:])
            traints.append(curr)
            trainX.append(np.stack([np.stack([np.mean(X[curr[ii]+x:curr[ii]+x+max(navg,1),:],axis=0) for x in range(-npre,npost+1-navg,max(navg,1))]) for ii in range(len(curr))]))

    trainY = np.concatenate(trainY)
    if Y2 is not None: trainY2 = np.concatenate(trainY2)
    trainX = np.concatenate(trainX)


    for chunk in tests:
        curr = chunk[npre:-(npost+1)]
        if len(curr)>0:
            testY.append(Y[curr,:])
            if Y2 is not None: testY2.append(Y2[curr,:])
            testts.append(curr)
            testX.append(np.stack([np.stack([np.mean(X[curr[ii]+x:curr[ii]+x+max(navg,1),:],axis=0) for x in range(-npre,npost+1-navg,max(navg,1))]) for ii in range(len(curr))]))

    testY = np.concatenate(testY)
    if Y2 is not None: testY2 = np.concatenate(testY2)
    testX = np.concatenate(testX)
    
    if Y2 is not None:
        return trainX, testX, trainY, testY, trainY2, testY2, traints, testts
    else:
        return trainX, testX, trainY, testY, traints, testts
