import numpy as np
import random
import pandas as pd

# can comment out next three if not interested in parallelization
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

m_min, m_max = 0.1, 4.5     # GeV
tau_min, tau_max = 0.1, 100 # m

# specify number of sets and events to generate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nsets', type=int, help="total generated sets", default=100)
parser.add_argument('--nobs', type=int, help="number observed events per set", default=1)
args = parser.parse_args()

n_sets  = args.nsets
n_obs = args.nobs

# by default we parallelize the generation, can be turned off if needed
par = 1 

from ALP_decay import generate_events

if par:
    def processInput(i):
        np.random.seed() 
        return generate_events(1, n_obs, m_min, m_max, tau_min, tau_max, equal_weights = True)[0]
    events=Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(n_sets))
    
else:
    events = generate_events(n_sets, n_obs, m_min, m_max, tau_min, tau_max, equal_weights = True)

df = pd.DataFrame.from_records(events)
df_out = pd.concat([df['m_alp'], df['ctau_alp']], axis=1)


for iOBS in range(n_obs):
    # the event_list (which is a dictionary) is translated to a dataframe
    # change name of variables so to have a dataframe where all the columns have a single float entry (i.e. avoid columns with array entries for easier manipulation)
    df_g1 = pd.DataFrame(df["gamma_1_4momentum_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['E1','px1', 'py1', 'pz1']])
    df_g2 = pd.DataFrame(df["gamma_2_4momentum_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['E2','px2', 'py2', 'pz2']])
    df_ch1 = pd.DataFrame(df["gamma_1_calo_hit_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['chx1', 'chy1', 'chz1']])
    df_ch2 = pd.DataFrame(df["gamma_2_calo_hit_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['chx2', 'chy2', 'chz2']])
    df_V = pd.DataFrame(df["decay_position_"+str(iOBS)].to_list(), columns=[s + '_' +str(iOBS) for s in ['Vx', 'Vy', 'Vz']])


    df_out = pd.concat([df_out, df_g1, df_g2, df_ch1, df_ch2, df_V], axis=1) 

# save to csv
df_out.to_csv("event_"+str(n_obs)+"_m_"+str(m_min)+"_"+str(m_max)+"_t_"+str(tau_min)+"_"+str(tau_max)+".csv", index= False)
