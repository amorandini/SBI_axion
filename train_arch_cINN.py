import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import os
import pickle
import argparse
import yaml

import pandas as pd

from feat_extractor import feature_extract
from architecture import CouplingNet, ConditionalCouplingLayer, Permutation, Summary, RealNVP_sum

for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
      

parser = argparse.ArgumentParser()

parser.add_argument('--counter', type=int, help="counter", default=0)                      # file counter to select the detector setup
parser.add_argument('--nobs', type=int, help="number of observed events (<=3)", default=3) # how many observed events to use as input
parser.add_argument('--epochs', type=int, help="Number of epochs", default=500)            # max number of epochs to use for training


args = parser.parse_args()

icounter = args.counter
nobs = args.nobs
num_epochs=args.epochs

# load the architecture and training parameters from some YAML files
with open('yamls/set'+str(icounter)+'.yaml', 'r') as stream: 
    meta_par = yaml.safe_load(stream) # architecture, training parameters and detector uncertainties saved here
sigs = meta_par['sigs']
 
# detector geometry, ALP mass and lifetimes, used to read out the file and manipulate smeared events
l_x, l_y, displ_y = 10, 35, 1.25, 1.25, 0
malp_min, malp_max = 0.1, 4.5
talp_min, talp_max = 0.1, 100

    
trainfile = "data/event_3_m_"+str(malp_min)+"_"+str(malp_max)+"_t_"+str(talp_min)+"_"+str(talp_max)+".csv"
feats=feature_extract(trainfile, sigs[0], sigs[1], sigs[2], sigs[3], Eres=meta_par["Eres"], lx=l_x, ly=l_y, disply=displ_y)

# add number of epochs and filename to saved hyperparameters
hyper = {"file": trainfile,  "n_epochs": num_epochs}
meta_par.update(hyper)

# folder where to save the weights, training curves, hyperparameter and scalers
modelfolder = "modelscINN/modelcINN"+str(nobs)+"_"+str(icounter)
os.mkdir(modelfolder)

# this is not saved, because these parameters are not varied for the different setups
# but it needs to be modified if we want to change activation function
arch_par = {
    'n_coupling_layers': meta_par['ncl'],
    's_args': {
        'units': meta_par['cinn_units'],
        'activation': 'relu',
        'initializer': 'glorot_uniform',
    },
    't_args': {
        'units': meta_par['cinn_units'],
        'activation': 'relu',
        'initializer': 'glorot_uniform',
    },
    'n_params': 2,
    'alpha': 1.0,
    'use_permutation': False # for 2D RANDOM permutation not important, checked with training
}
arch_par.update({"n_units_summary": meta_par['sum_units'],  "summary":  meta_par["summary_dim"]}) # summary can be different from input_dim in general
input_dim = arch_par['n_params']

# extract model parameters from file
malp, talp = feats.extract_model()
z = np.vstack((np.log10(malp), np.log10(talp/malp))).T 
zscaler=StandardScaler().fit(z) 
z=zscaler.transform(z)

features = np.vstack(([feats.extract_llo(iOBS) for iOBS in range(nobs)]))  
x = features.T   
xscaler=StandardScaler().fit(x) 
x=xscaler.transform(x)



with open(modelfolder+'/meta_par.pkl', 'wb') as file_t:
    pickle.dump(meta_par, file_t)
    
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    meta_par['learn_rate'],
    decay_steps=int(meta_par['decayepo']*0.8*len(x)/meta_par["batch_size"]), # 0.8 is validation split, this translates training steps in training epochs
    decay_rate=meta_par['decayr'],
    staircase=True)

NormFlowPost=RealNVP_sum(arch_par)
NormFlowPost.call([z, x]) 
NormFlowPost.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))

early= tf.keras.callbacks.EarlyStopping(
monitor="val_loss",
min_delta=0.001,
patience=50,
restore_best_weights=True,
)

history = NormFlowPost.fit(
    z, x, batch_size=meta_par['batch_size'], epochs=meta_par['n_epochs'],
    verbose=2, validation_split=0.2, callbacks = [early]
    
)
        
with open(modelfolder+'/xscaler.pkl', 'wb') as file_t:
    pickle.dump(xscaler, file_t)
    
with open(modelfolder+'/zscaler.pkl', 'wb') as file_t:
    pickle.dump(zscaler, file_t)    

with open(modelfolder+'/training.pkl', 'wb') as file_t:
    pickle.dump(history.history, file_t)

# I save only the weights for compatibility with older tensorflow versions, but the whole model can be saved if that is not a concern
NormFlowPost.save_weights(modelfolder+"/weights.h5", save_format="h5")
