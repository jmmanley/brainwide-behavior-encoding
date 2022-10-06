"""
A variational autoencoder implementation in Tensorflow, utilized to test how
much further we can reduce the dimensionality of neural datasets utilizing
nonlinear dimensionality reduction.

Jason Manley, 2022
jmanley at rockefeller dot edu
"""

import numpy as np
import tensorflow as tf
import os
from scipy.stats import zscore
from pathlib import Path

from svca import SVCA, checkerboard_centers, train_test_split_idx
from run_svca_for_dataviz import Experiment

class VAE:
    """
    Basic implementation of a variational autoencoder.
    """

    def __init__(self, input_dim, output_dim,
                 optimizer='adamax', loss='mse'):
        
        # encoder architecture
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_dim))
        self.encoder.add(tf.keras.layers.Dense(output_dim, activation='relu'))
        
        # decoder architecture
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(output_dim))
        self.decoder.add(tf.keras.layers.Dense(input_dim, activation='relu'))
        self.decoder.add(tf.keras.layers.Dense(input_dim, activation='linear'))
        
        # compile model
        input_layer = tf.keras.layers.Input(input_dim)
        latent = self.encoder(input_layer)
        reconstructed = self.decoder(latent)
        
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=reconstructed)
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def fit(self, data, **kwargs):
        self.model.fit(x=data, y=data, **kwargs)
    
    def encode(self, data):
        return self.encoder.predict(data)
    
    def decode(self, latent):
        return self.decoder.predict(latent)


def compare_svca_vae_dimred(file, outpath,  checkerboard=60, nsvc_save=128, gpuId=None):
    """
    Runs SVCA on neural data.
    Then attempts to further reduce the dimensionality into nonlinear space using a
    variational autoencoder (VAE) approach.
    """

    if gpuId is not None: 
        # choose gpu to use
        os.environ['CUDA_VISIBLE_DEVICES']=str(gpuId)


    ### LOAD DATA ###

    expt = Experiment(file)

    expt.neurons = zscore(expt.neurons, axis=1)
    expt.motion = zscore(expt.motion, axis=1)


    ### RUN SVCA ###

    # split train, test sets
    ntrain, ntest = checkerboard_centers(expt.centers, checkerboard)
    itrain, itest = train_test_split_idx(expt.neurons.shape[1], train_frac=0.5, interleave=int(72 * expt.fhz))

    print('Running SVCA on', os.path.basename(file))
    sneur, varneur, u, v = SVCA(expt.neurons, ntrain=ntrain, ntest=ntest, itrain=itrain, itest=itest)

    projs1 = u.T @ expt.neurons[ntrain,:] # SVC projections from each set
    projs2 = v.T @ expt.neurons[ntest,:]

    # for better viz, flip SVCs that are strongly negative-going
    for i in range(projs1.shape[0]):
        if np.abs(np.min(projs1[i,:])) > np.abs(np.max(projs1[i,:])):
            projs1[i,:] = -projs1[i,:]
            projs2[i,:] = -projs2[i,:]


    ### FURTHER REDUCE DIMENSIONALITY OF SVCS USING VARIATIONAL AUTOENCODER ###

    input_dim = 2048
    output_dims = 2**np.arange(12)

    covneur_preds = np.zeros((len(output_dims), input_dim))
    varneur_preds = np.zeros((len(output_dims), input_dim))

    train_projs = projs1[:,itrain]

    valid_itrain, valid_itest = train_test_split_idx(train_projs.shape[1], train_frac=0.5, interleave=3*72)

    models = []

    projs2_preds = []

    from tqdm import tqdm
    for i in tqdm(range(len(output_dims))):
        
        model = VAE(input_dim, int(output_dims[i]))
        model.fit(train_projs[:input_dim,valid_itrain].T, epochs=1000,
                  validation_data=(train_projs[:input_dim,valid_itest].T,train_projs[:input_dim,valid_itest].T),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        
        models.append(model)
        
        projs2_pred = model.decode(model.encode(projs2[:input_dim,:].T)).T

        projs2_preds.append(projs2_pred[:128,:3000])
        
        covneur_preds[i,:] = np.sum(projs1[:input_dim,itest] * projs2_pred[:,itest],axis=1)
        varneur_preds[i,:] = np.sum(projs1[:input_dim,itest]**2 + projs2_pred[:,itest]**2,axis=1)/2

    projs1 = projs1[:128,:3000]
    projs2_preds = np.asarray(projs2_preds)

    outfile = os.path.join(outpath, Path(file).stem + '_' + str(nsvc_save) + 'SVC_VAE.npz')

    np.savez(outfile, sneur=sneur, varneur=varneur, projs1=projs1.astype('single'), projs2_preds=projs2_preds.astype('single'),
             sneur_vae=covneur_preds, varneur_vae=varneur_preds)


if __name__ == "__main__":

    file = '/vmd/jason_manley/stringer-pachitariu-etal-2018a/spont_M161025_MP030_2017-06-16.mat'
    outpath = '/vmd/jason_manley/stringer-pachitariu-etal-2018a/processed/'
    compare_svca_vae_dimred(file, outpath, gpuId=0)
