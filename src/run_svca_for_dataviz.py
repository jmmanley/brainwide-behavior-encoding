"""
Applies SVCA to data from Stringer & Pachitariu et al. 2019 Science, primarily to 
save features needed for this dataviz.

Jason Manley, 2022
jmanley at rockefeller dot edu
"""

import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import zscore
from svca import SVCA, train_test_split_idx, checkerboard_centers
import tensorflow as tf

class Experiment:
    """
    Class for loading neural and behavioral data from Stringer & Pachitariu et al.'s
    data (https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622)
    """

    def __init__(self, file):

        self.file = file
        data = loadmat(file)

        self.centers = data['med'].T # neuron locations in volume
        self.neurons = data['Fsp']   # neuron timeseries

        self.motion = data['beh'][0][0][0][0][0][0].T   # motion energy PCs
        self.motionMask = data['beh'][0][0][0][0][0][1] # motion energy PC masks

        self.pupil_area = data['beh'][0][0][3][0][0][0][:,0] # pupil area
        self.run_speed = data['beh'][0][0][4][:,0]           # running speed

        self.nplanes = data['db'][0][0][4][0][0] # number of imaging planes
        if self.nplanes == 10: self.fhz = 3      # volume rate, from Stringer & Pachitariu methods
        elif self.nplanes == 12: self.fhz = 2.5

        self.t = np.arange(len(self.run_speed)) / self.fhz


def run_svca_and_predict(file, outpath, checkerboard=60, predict=True, nsvc_save=128, gpuId=None):
    """
    Runs SVCA on neural data.
    Then if predict=True, predicts neural activity from behavioral motion energy PCs using both
    reduced-rank regression and LSTM neural network approaches.
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

    projs1 = projs1[:nsvc_save,:]
    projs2 = projs2[:nsvc_save,:]

    
    ### PREDICT NEURAL ACTIVITY FROM BEHAVIOR ###

    if predict is not None:
        from predict import RRRegression, LSTMRegression, get_binned_sequences

        nt_save = 3000 # number of prediction timepoints to save

        # LINEAR PREDICTIONS
        projs1_pred_linear = []
        projs2_pred_linear = []
        
        lams = [0.01,0.1,1]
        ranks = 2**np.arange(1,8)
        
        cov_res_beh_linear = np.zeros((nsvc_save,len(lams),len(ranks)))
        
        for l in range(len(lams)):
            model1 = RRRegression(lam=lams[l], rank=ranks[-1])
            model1.fit(expt.motion[:,itrain].T, projs1[:nsvc_save,:][:,itrain].T)
            
            model2 = RRRegression(lam=lams[l], rank=ranks[-1])
            model2.fit(expt.motion[:,itrain].T, projs2[:nsvc_save,:][:,itrain].T)
            
            currpreds1 = []
            currpreds2 = []
            
            for r in range(len(ranks)):
                currpreds1.append(model1.predict(expt.motion.T, rank=ranks[r]))
                currpreds2.append(model2.predict(expt.motion.T, rank=ranks[r]))
                
                # find residual covariance between neural sets after removing behavior predictions
                s1 = projs1[:nsvc_save,:][:,itest] - currpreds1[-1][:,itest]
                s2 = projs2[:nsvc_save,:][:,itest] - currpreds2[-1][:,itest]

                cov_res_beh_linear[:,l,r] = np.sum(s1 * s2, axis=1)

            projs1_pred_linear.append(currpreds1)
            projs2_pred_linear.append(currpreds2)


        projs1_pred_linear_vsrank = np.zeros((len(projs1_pred_linear[0]), projs1.shape[0], projs1.shape[1]))

        for k in range(projs1_pred_linear_vsrank.shape[0]):
            idx = np.argmin(cov_res_beh_linear[:,:,k],axis=1)
            for i in range(projs1.shape[0]):
                projs1_pred_linear_vsrank[k,i,:] = projs1_pred_linear[idx[i]][k][i,:]

        projs1_pred_linear_vsrank = projs1_pred_linear_vsrank[:,:,:nt_save]

        projs1_pred_linear_bestrank = np.zeros((projs1.shape[0], projs1.shape[1]))

        for i in range(projs1.shape[0]):
            l,k=np.unravel_index(np.argmin(cov_res_beh_linear[0,:]), cov_res_beh_linear.shape[1:])
            projs1_pred_linear_bestrank[i,:] = projs1_pred_linear[l][k][i,:]

        projs1_pred_linear = projs1_pred_linear_bestrank[:,:nt_save]
            
    
        # LSTM MULTI-TIMEPOINT PREDICTIONS
        projs1_pred_lstm = []
        projs2_pred_lstm = []
        
        pres = [int(np.round(x*expt.fhz)) for x in [3,6,9,12]]
        posts = [int(np.round(x*expt.fhz)) for x in [1,3,6,9]]
        navg  = int(np.round(expt.fhz))
        npc_behavior = 128
        
        cov_res_beh_lstm = np.zeros((nsvc_save,len(pres),len(posts)))
        
        for a in range(len(pres)):
            
            currpreds1 = []
            currpreds2 = []
            
            for p in range(len(posts)):
                
                trainX, testX, trainY, testY, trainY2, testY2, traints, testts = \
                      get_binned_sequences(expt.motion[:npc_behavior,:].T, projs1[:nsvc_save,:].T, pres[a], posts[p], itrain, itest, navg=navg, Y2=projs2[:nsvc_save,:].T)

                iitrain, iivalid = train_test_split_idx(trainX.shape[0], train_frac=0.9, interleave=100)
                tmp_model = 'model.h5'

                model1 = LSTMRegression(trainX.shape, trainY.shape)
                model1.fit(trainX[iitrain,:], trainY[iitrain,:],
                           batch_size=64, epochs=50,
                           validation_data=(trainX[iivalid,:], trainY[iivalid,:]),
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                             tf.keras.callbacks.ModelCheckpoint(tmp_model, save_best_only=True,
                                                                save_weights_only=True, save_frequency=1),
                             tf.keras.callbacks.ReduceLROnPlateau(patience=2)])

                model2 = LSTMRegression(trainX.shape, trainY2.shape)
                model2.fit(trainX[iitrain,:], trainY2[iitrain,:],
                           batch_size=64, epochs=50,
                           validation_data=(trainX[iivalid,:], trainY2[iivalid,:]),
                           callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                             tf.keras.callbacks.ModelCheckpoint(tmp_model, save_best_only=True,
                                                                save_weights_only=True, save_frequency=1),
                             tf.keras.callbacks.ReduceLROnPlateau(patience=2)])
                
                trainpreds1 = model1.predict(trainX)
                testpreds1 = model1.predict(testX)
                
                trainpreds2 = model2.predict(trainX)
                testpreds2 = model2.predict(testX)
                
                # find residual covariance between neural sets after removing behavior predictions
                s1 = testY  - testpreds1
                s2 = testY2 - testpreds2
                
                cov_res_beh_lstm[:,a,p] = np.sum(s1 * s2, axis=0)
                
                # add predictions to list
                curr1 = np.zeros(projs1[:nsvc_save,:].shape)+np.nan
                curr1[:,[x for y in traints for x in y]] = trainpreds1.T
                curr1[:,[x for y in testts for x in y]] = testpreds1.T
                currpreds1.append(curr1)
                
                curr2 = np.zeros(projs2[:nsvc_save,:].shape)+np.nan
                curr2[:,[x for y in traints for x in y]] = trainpreds2.T
                curr2[:,[x for y in testts for x in y]] = testpreds2.T
                currpreds2.append(curr2)    

            projs1_pred_lstm.append(currpreds1)
            projs2_pred_lstm.append(currpreds2)    

        projs1_pred_lstm_bestwindow = np.zeros((projs1.shape[0], projs1.shape[1]))

        for i in range(projs1.shape[0]):
            l,k=np.unravel_index(np.argmin(cov_res_beh_lstm[0,:]), cov_res_beh_lstm.shape[1:])
            projs1_pred_lstm_bestwindow[i,:] = projs1_pred_lstm[l][k][i,:]

        projs1_pred_lstm = projs1_pred_lstm_bestwindow[:,:nt_save]


    ### SAVE VARIABLES FOR DATAVIZ ###

    out = os.path.join(outpath, Path(file).stem + '_' + str(nsvc_save) + 'SVC_' + str(predict) + 'predict.npz')
    
    if predict:
        np.savez(out, sneur=sneur, varneur=varneur, projs1=zscore(projs1[:nsvc_save,:],axis=-1).astype('single'), motionpc1=expt.motion[0,:], 
                      pupil_area=zscore(expt.pupil_area), run_speed=zscore(expt.run_speed), t=expt.t, ex_neurons=zscore(expt.neurons[np.random.permutation(expt.neurons.shape[0])[:100],:].astype('single'),axis=1),
                 cov_res_beh_linear=cov_res_beh_linear, cov_res_beh_lstm=cov_res_beh_lstm, projs1_pred_linear=zscore(projs1_pred_linear,axis=-1).astype('single'),
                 projs1_pred_linear_vsrank=zscore(projs1_pred_linear_vsrank,axis=-1).astype('single'), projs1_pred_lstm=zscore(projs1_pred_lstm,axis=-1).astype('single'),
                 npc_behavior=npc_behavior, projs2=zscore(projs2[:nsvc_save,:],axis=-1).astype('single'))
    else:
        np.savez(out, sneur=sneur, varneur=varneur, projs1=zscore(projs1[:nsvc_save,:],axis=-1).astype('single'), motionpc1=expt.motion[0,:], 
                      pupil_area=zscore(expt.pupil_area), run_speed=zscore(expt.run_speed), t=expt.t, ex_neurons=zscore(expt.neurons[np.random.permutation(expt.neurons.shape[0])[:100],:].astype('single'),axis=1),
                      projs2=zscore(projs2[:nsvc_save,:],axis=-1).astype('single'))


if __name__ == "__main__":

    file = '/vmd/jason_manley/stringer-pachitariu-etal-2018a/spont_M150824_MP019_2016-04-05.mat'
    outpath = '/vmd/jason_manley/stringer-pachitariu-etal-2018a/processed/'
    run_svca_and_predict(file, outpath, gpuId=0)


