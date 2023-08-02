# This code generates the ALP decay and returns the ALP decay position, the photon 4-momenta and the photon caloriemter hits
# The events are unweighted and sets of n_obs events can be generated
# Energies are in GeV and lengths in meters

import numpy as np
import random

Pbeam = 400 # in GeV
mp = 0.93827 # in GeV
Ebeam = np.sqrt(mp**2 + Pbeam**2) # in GeV

mB0 = 5.27963 # in GeV
mB = 5.27932 # in GeV
mK0 = 0.49761 # in GeV
mK = 0.49367 # in GeV
mK0star = 0.89555 # in GeV
mKstar = 0.89176 # in GeV

# The geometry has been hardcoded here, but can be modified if needed
# if we are interestd in evalauting several geometries it makes sense to pass the parameters to argparse

dump_x, dump_y, dump_z = 0, 0, 0        # dump position
z_min,  z_max, z_cal = 10, 35, 35       # if tracking chambers considered part of the decay volume z_max = z_cal
x_min, x_max = -1.25, 1.25              # calorimeter AND decay volume extension, 1, 3.5 for detector off-axis
y_min, y_max = -1.25, 1.25              # calorimeter AND decay volume extension, -1.25, 1.25 also for detector off-axis
dr_gg, E_min = 0.1, 1                   # photon separation (m) and minimum energy of each photon (GeV)


def lambda_abc(aa, bb, cc): # Phase-space factor for decay
    return (aa**2 - (bb + cc)**2) * (aa**2 - (bb - cc)**2)

raw_data = np.loadtxt("beauty_100kEvts_pp_8.2_400GeV_ptHat300MeV.txt")[:,2:6] # Pythia events
b_mesons = raw_data[(np.abs(raw_data[:,0])==511) | (np.abs(raw_data[:,0])==521)]     # Select B mesons 

def b_meson_decay(b_meson, m_alp): # Simulates a random decay of a given B meson into K + a. At the moment decays into K* + a are not implemented.

    id = b_meson[0]
    p_b = np.array([b_meson[1],b_meson[2],b_meson[3]])

    if(np.abs(id) == 511):
        massB = mB0
        massK = mK0
    else:
        massB = mB
        massK = mK

    pp2 = np.sum(p_b**2)

    energy = np.sqrt(massB**2 + pp2)

    # Determine Lorentz transformation from cms frame to lab frame

    gamma = energy/massB
    beta = p_b/energy

    Lambda = np.array([[ gamma, gamma*beta[0], gamma*beta[1], gamma*beta[2]], 
            [ gamma*beta[0], 1 + (gamma - 1)*beta[0]**2/(pp2/energy**2), (gamma - 1)*(beta[1]*beta[0])/(pp2/energy**2), (gamma - 1)*(beta[2]*beta[0])/(pp2/energy**2)],
            [ gamma*beta[1], (gamma - 1)*beta[0]*beta[1]/(pp2/energy**2), 1 + (gamma - 1)*(beta[1]**2)/(pp2/energy**2), (gamma - 1)*(beta[2]*beta[1])/(pp2/energy**2)],
            [ gamma*beta[2], (gamma - 1)*beta[0]*beta[2]/(pp2/energy**2), (gamma - 1)*(beta[1]*beta[2])/(pp2/energy**2), 1 + (gamma - 1)*(beta[2]**2)/(pp2/energy**2)]])

    # Generate decay in cms frame

    pa2cm = lambda_abc(massB, massK, m_alp)/(4*massB**2)

    thetaacm = np.arccos(random.uniform(-1,1))
    phiacm = random.uniform(0, 2*np.pi)

    p4acm = np.array([ np.sqrt(pa2cm + m_alp**2), np.sqrt(pa2cm)*np.sin(thetaacm)*np.cos(phiacm), np.sqrt(pa2cm)*np.sin(thetaacm)*np.sin(phiacm), np.sqrt(pa2cm)*np.cos(thetaacm)])

    # Boost to lab frame

    p4a = np.dot(Lambda, p4acm)

    return p4a



def alp_decay(p4a, m_alp, ctau_alp): # Simulates a random decay of a given ALP with mass m_alp and decay length ctau_alp into two photons
                                     # It is possible to force the decay to happen between z_min and z_max (at the cost of reducing the event weight)
       
        
    
    E_alp = p4a[0]
    p3_alp = p4a[1::]
    p_alp = np.sqrt(np.sum(p3_alp**2))

    l_min = p_alp / p3_alp[2] * (z_min - dump_z)
    l_max = p_alp / p3_alp[2] * (z_max - dump_z)

    decayLength_alp = ctau_alp * p_alp / m_alp
    weight = np.exp(-l_min/decayLength_alp)-np.exp(-l_max/decayLength_alp) # weight of the event = probability of decaying inside the detector

    decay_distance =   l_min - decayLength_alp * np.log(1 + (np.exp((l_min - l_max)/decayLength_alp) - 1) * random.uniform(0,1)) # Generates random draw from exponential distribution
    decay_position = [dump_x, dump_y, dump_z] + decay_distance / p_alp * p3_alp
    
    # discard events outside of the detector
            
    if ((decay_position[0]>x_max) | (decay_position[1]>y_max)  | (decay_position[0]<x_min) | (decay_position[1]<y_min)  ):
        event = {"event_weight": 0}
        return event


    # Determine Lorentz transformation from cms frame to lab frame

    gamma = E_alp/m_alp
    beta = p3_alp/E_alp

    Lambda = np.array([[ gamma, gamma*beta[0], gamma*beta[1], gamma*beta[2]], 
            [ gamma*beta[0], 1 + (gamma - 1)*beta[0]**2/(p_alp**2/E_alp**2), (gamma - 1)*(beta[1]*beta[0])/(p_alp**2/E_alp**2), (gamma - 1)*(beta[2]*beta[0])/(p_alp**2/E_alp**2)],
            [ gamma*beta[1], (gamma - 1)*beta[0]*beta[1]/(p_alp**2/E_alp**2), 1 + (gamma - 1)*(beta[1]**2)/(p_alp**2/E_alp**2), (gamma - 1)*(beta[2]*beta[1])/(p_alp**2/E_alp**2)],
            [ gamma*beta[2], (gamma - 1)*beta[0]*beta[2]/(p_alp**2/E_alp**2), (gamma - 1)*(beta[1]*beta[2])/(p_alp**2/E_alp**2), 1 + (gamma - 1)*(beta[2]**2)/(p_alp**2/E_alp**2)]])

    # Generate decay in cms frame

    E_gamma_cm = m_alp / 2

    thetaacm = np.arccos(random.uniform(-1,1))
    phiacm = random.uniform(0, 2*np.pi)

    p4_gamma1_cm = np.array([ E_gamma_cm, E_gamma_cm*np.sin(thetaacm)*np.cos(phiacm), E_gamma_cm*np.sin(thetaacm)*np.sin(phiacm), E_gamma_cm*np.cos(thetaacm)])
    p4_gamma2_cm = np.array([ E_gamma_cm, - E_gamma_cm*np.sin(thetaacm)*np.cos(phiacm), - E_gamma_cm*np.sin(thetaacm)*np.sin(phiacm), - E_gamma_cm*np.cos(thetaacm)])

    # Boost to lab frame

    p4_gamma1 = np.dot(Lambda, p4_gamma1_cm)
    p4_gamma2 = np.dot(Lambda, p4_gamma2_cm)
    
    # Discard events for ALP or photons with negative z momenta  
    if ((p4_gamma1[3] < 0) | (p4_gamma2[3] < 0) | (p4a[3] < 0)):
        event = {"event_weight": 0}
        return event
    
    # Discard events with low energy
    if ((p4_gamma1[0] < E_min) | (p4_gamma2[0] < E_min)  ):
        event = {"event_weight": 0}
        return event

    # Order photons such that E_gamma1 >= E_gamma2

    if(p4_gamma1[0] < p4_gamma2[0]):
        p4_gammatemp = p4_gamma1
        p4_gamma1 = p4_gamma2
        p4_gamma2 = p4_gammatemp

  # Propagate photons to z_cal

    travel_distance = z_cal - decay_position[2]

    hit_gamma1 = travel_distance / p4_gamma1[3] * p4_gamma1[[1,2,3]] + decay_position
    hit_gamma2 = travel_distance / p4_gamma2[3] * p4_gamma2[[1,2,3]] + decay_position
    
  # Discard events which do not hit the calorimeter
    if ((hit_gamma1[0]>x_max) | (hit_gamma1[1]>y_max) | (hit_gamma2[0]>x_max) | (hit_gamma2[1]>y_max) | (hit_gamma1[0]<x_min) | (hit_gamma1[1]<y_min) | (hit_gamma2[0]<x_min) | (hit_gamma2[1]<y_min) ):
        event = {"event_weight": 0}
        return event
        
 # Discard events where the two photons cannot be resolved separately
    if ( np.sqrt((hit_gamma1[0]-hit_gamma2[0])**2+(hit_gamma1[1]-hit_gamma2[1])**2) < dr_gg  ):
        event = {"event_weight": 0}
        return event    

    # Fill event dictionary

    event = {
        "gamma_1_calo_hit": hit_gamma1,
        "gamma_1_4momentum": p4_gamma1,
        "gamma_2_calo_hit": hit_gamma2,
        "gamma_2_4momentum": p4_gamma2,
        "decay_position": decay_position,
        "event_weight": weight
    }



    return event

def random_event(m_alp, ctau_alp, equal_weights): # Generates a random event for a randomly drawn ALP mass and decay length
    
    reject = True

    if equal_weights: 
        while(reject):  # For equal_weights = True the probability to accept an event is equal to its weight, such that all accepted events have equal weight
            b_meson = random.choice(b_mesons)
            p4a = b_meson_decay(b_meson, m_alp)
            event = alp_decay(p4a, m_alp, ctau_alp)
            acceptance = random.uniform(0,1)
            if acceptance < event["event_weight"]:
                event["event_weight"] = 1 
                reject = False

    else: 
        while(reject):  # return all non-0 weight events
            b_meson = random.choice(b_mesons)
            p4a = b_meson_decay(b_meson, m_alp)
            event = alp_decay(p4a, m_alp, ctau_alp)
            if event["event_weight"]>0:
                reject = False
    
    return event
 # Generates n_sets set of n_obs events                                                                                                                                                                                                              
def generate_events(n_sets, n_obs, m_alp_min, m_alp_max, ctau_min, ctau_max, equal_weights): 
  if m_alp_max > mB0 - mK0: # error in case you try to generate events which are unphysical
    raise ValueError('You are generating events with too massive ALPs: cannot be produced in this decay.\n  Lower m_alp_max to be smaller than {}'.format(mB0 - mK0))
    
  event_list = []
  while len(event_list) < n_sets:
    # this defines the prior/sampling region
    m_alp = np.exp(random.uniform(np.log(m_alp_min),np.log(m_alp_max)))   
    ctau_alp = np.exp(random.uniform(np.log(ctau_min), np.log(ctau_max)))
    # first we save the model parameters
    event_obs = {
    "m_alp": m_alp,
    "ctau_alp": ctau_alp}
    # then we generate n_obs (unweighted) events
    iOBS = 0
    while iOBS < n_obs:

        new_event = random_event(m_alp, ctau_alp, equal_weights)
        if new_event["event_weight"] > 0: # reject invalid events
            for k_old in ["gamma_1_calo_hit", "gamma_1_4momentum", "gamma_2_calo_hit", "gamma_2_4momentum", "decay_position", "event_weight"]: # new keys are assigned to new events
                new_event[k_old+"_"+str(iOBS)] = new_event.pop(k_old)
            event_obs.update(new_event)
            iOBS+=1
    event_list.append(event_obs)
  return event_list
