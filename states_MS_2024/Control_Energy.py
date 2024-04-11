#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#======================================================================
#                       Control Energy
#======================================================================

#@author: Tommy Broeders
#@email:  t.broeders@amsterdamumc.nl
#updated: 11 04 2024
#status: done
#to-do: -

#Review History
#Reviewed by -

# Description:
# - Calculate the minimum control energy associated with state-transitions
#
# - Prerequisites: The states need to have been defined
# - Input: fMRI time series, DTI connectivity matrices, a state sequence,
#          the included regions, 
# - Output: files obtained after running the script
#-----------------------------------------------------------------------
"""

## Load libraries
import scipy as sp
import scipy.io as io
import scipy.linalg as la
from scipy.linalg import svd, eig, expm
from scipy.stats import spearmanr,zscore
import numpy as np
from numpy import matmul as mm
from numpy import transpose as tp
import pandas as pd
import os
import pickle

#%%--------------------------------------------------------------------
# Imported functions
#----------------------------------------------------------------------
# These functions were derived from the nctpy toolbox (see references below)
def matrix_normalization(A, version=None, c=1):
    '''
    This function normalizes the ajacency matrix

    Args:
        A: np.array (n_parcels, n_parcels)
            adjacency matrix from structural connectome
        version: str
            options: 'continuous' or 'discrete'. default=None
            string variable that determines whether A is normalized for a continuous-time system or a discrete-time
            system. If normalizing for a continuous-time system, the identity matrix is subtracted.
        c: int
            normalization constant, default=1
    Returns:
        A_norm: np.array (n_parcels, n_parcels)
            normalized adjacency matrix

    '''

    if version == 'continuous':
        print("Normalizing A for a continuous-time system")
    elif version == 'discrete':
        print("Normalizing A for a discrete-time system")
    elif version == None:
        raise Exception("Time system not specified. "
                        "Please nominate whether you are normalizing A for a continuous-time or a discrete-time system "
                        "(see function help).")

    # singluar value decomposition
    u, s, vt = svd(A)

    # Matrix normalization for discrete-time systems
    A_norm = A / (c + s[0])

    if version == 'continuous':
        # for continuous-time systems
        A_norm = A_norm - np.eye(A.shape[0])

    return A_norm

def normalize_state(x):
    """
    This function will normalize a brain state's magnitude using its euclidean norm.
    
    Args:
        x (N, numpy array): brain state to be normalized.
    Returns:
        x_norm (N, numpy array): normalized brain state.
    """

    x_norm = x / np.linalg.norm(x, ord=2)

    return x_norm

def gramian(A,B,T,version=None,tol=1e-12):
    """
    This function computes the controllability Gramian.
    Args:
        A:             np.array (n x n)
        B:             np.array (n x k)
        T:             np.array (1 x 1)
        version:       str
            options: 'continuous' or 'discrete'. default=None
    Returns:
        Wc:            np.array (n x n)
    """

    # System Size
    n_parcels = A.shape[0]

    u,v = eig(A)
    BB = mm(B,np.transpose(B))
    n = A.shape[0]
    
    
    # If time horizon is infinite, can only compute the Gramian when stable
    if T == np.inf:
        # check version
        if version=='continuous':
            # If stable: solve using Lyapunov equation
            if(np.max(np.real(u)) < 0):
                return la.solve_continuous_lyapunov(A,-BB)
            else:
                print("cannot compute infinite-time Gramian for an unstable system!")
                return np.NAN
        elif version=='discrete':
            # If stable: solve using Lyapunov equation
            if(np.max(np.abs(u)) < 1):
                return la.solve_discrete_lyapunov(A,BB)
            else:
                print("cannot compute infinite-time Gramian for an unstable system!")
                return np.NAN

            
    # If time horizon is finite, perform numerical integration
    else:
        # check version
        if version=='continuous':
            ## Compute required number of steps for desired precision
           
            # Number of integration steps
            STEP = 0.001
            t = np.arange(0,(T+STEP/2),STEP)
            # Collect exponential difference
            dE = sp.linalg.expm(A * STEP)
            dEa = np.zeros((n_parcels,n_parcels,len(t)))
            dEa[:,:,0] = np.eye(n_parcels)
            # Collect Gramian difference
            dG = np.zeros((n_parcels,n_parcels,len(t)))
            dG[:,:,0] = mm(B,B.T)
            for i in np.arange(1, len(t)):
                dEa[:,:,i] = mm(dEa[:,:,i-1],dE)
                dEab = mm(dEa[:,:,i],B)
                dG[:,:,i] = mm(dEab,dEab.T)

            # Integrate
            if sp.__version__ < '1.6.0':
                G = sp.integrate.simps(dG,t,STEP,2)
            else:
                G = sp.integrate.simpson(dG,t,STEP,2)
            return G
        elif version=='discrete':
            Ap = np.eye(n)
            Wc = np.eye(n)
            for i in range(T):
                Ap = mm(Ap,A)
                Wc = Wc + mm(Ap,tp(Ap))
            return Wc

def minimum_energy_fast(A, T, B, x0_mat, xf_mat):
    """
    This function computes the minimum energy required to transition between all pairs of brain states
    encoded in (x0_mat,xf_mat)

     Args:
      A: numpy array (N x N)
            System adjacency matrix
      B: numpy array (N x N)
            Control input matrix
      x0_mat: numpy array (N x n_transitions)
             Initial states (see expand_states)
      xf_mat: numpy array (N x n_transitions)
            Final states (see expand_states)
      T: float (1 x 1)
           Control horizon

    Returns:
      E: numpy array (N x n_transitions)
            Regional energy for all state transition pairs.
            Notes,
                np.sum(E, axis=0)
                    collapse over regions to yield energy associated with all transitions.
                np.sum(E, axis=0).reshape(n_states, n_states)
                    collapse over regions and reshape into a state by state transition matrix.
    """
    if type(x0_mat[0][0]) == np.bool_:
        x0_mat = x0_mat.astype(float)
    if type(xf_mat[0][0]) == np.bool_:
        xf_mat = xf_mat.astype(float)

    G = gramian(A,B,T,version='continuous')
    delx = xf_mat - np.matmul(expm(A*T), x0_mat)
    E = np.multiply(np.linalg.solve(G, delx), delx)

    return E

#%%--------------------------------------------------------------------
# New function
#----------------------------------------------------------------------
def state_trans_energy(A,ts,stateseq,statenum,T):
    """ 
    Calculate the transition energy
    
    Args:
        A: Normalized adjacency Matrix (N x N)
        ts: Functional activity timeseries (N x T)
        stateseq: State assignment for each timepoint (T x 1)
        statenum: The number of unique states (1 x 1)
        T: The control horizon (1 x 1)
        
    Returns:
        Etrans: Global transition energies (S x 1)
        EtransREG: Transition energies per brain region (S x 1)
        transfreq: The number of transitions between states (S x 1)
    """
    #Set/restructure general input variables
    B = np.eye(A.shape[0]) #Control input matrix (whole brain)
    x0 = ts[:,0:ts.shape[1]-1] #initial states (=t)
    xf = ts[:,1:] #final states (=t+1)
    
    #Compute energy for each transition
    E = minimum_energy_fast(A, T, B, x0, xf)
    
    #compute the energy for the whole brain
    Eglob = np.sum(E,axis=0) #Whole brain minimum energy per transition
    
    #Calculate average whole brain minimum energy per state transition or state persistance (dwelling)
    Etrans=np.zeros([statenum,statenum])
    EtransREG=np.zeros([ts.shape[0],statenum,statenum])
    transfreq=np.zeros([statenum,statenum])
    for i, St in enumerate(stateseq[0:len(stateseq)-1]):
        Etrans[St-1,stateseq[i+1]-1]=Etrans[St-1,stateseq[i+1]-1] + Eglob[i]
        EtransREG[:,St-1,stateseq[i+1]-1]=np.sum([EtransREG[:,St-1,stateseq[i+1]-1],E[:,i]],axis=0)
        transfreq[St-1,stateseq[i+1]-1] = transfreq[St-1,stateseq[i+1]-1] + 1
        
    #Transform to relative values
    Etrans = np.divide(Etrans,transfreq, where=transfreq!=0)
    Etrans[transfreq==0]=np.NaN
    
    EtransREG=np.divide(EtransREG,transfreq, where=transfreq!=0)
    for i in range(EtransREG.shape[0]):
        EtransREG_tmp=EtransREG[i,:,:]
        EtransREG_tmp[transfreq==0]=np.NaN
        EtransREG[i,:,:]=np.array(EtransREG_tmp)
    
    return Etrans,EtransREG,transfreq

#%%--------------------------------------------------------------------
#                       Load data
#----------------------------------------------------------------------
#Load participant identifiers
fmri_participants = pd.read_csv('/fmri_participants.txt',header=None)[0] #txt file listing all identifiers of participants with fully processed fMRI data
numpart_fmri = fmri_participants.size

#Change to list based on dti sample
participants = pd.read_csv('/dti_participants.txt',header=None)[0] #txt file listing all identifiers of participants with fully processed DTI data
numpart = participants.size
included_regions= np.genfromtxt('/XXX/included_regions.txt',delimiter=' ')  #txt file indicating which regions to include (1=include)
included_regions[224]=0; #Exclude cerebellum
numreg=np.sum(included_regions,dtype='int64')

##Import state sequence
states = scipy.io.loadmat('StateDynamics.mat') #Output from EdgeTS_Kmeans_States.m script
states = states['cluster'][2][0]
states = np.reshape(states,[numpart_fmri,200])
states=states[fmri_participants.isin(participants),:]
statenum = states.max()

##Load fmri timeseries and structural connectivity matrices
timeseries=np.zeros([numpart,numreg,200])
StrucCon_norm=np.zeros([numpart,numreg,numreg])
for i, part in enumerate(participants):
    print(part)
    ##import timeseries
    timeseries_tmp = np.genfromtxt('/XXX/' + part + '/fmri_timeseries.txt',delimiter='  ')
    timeseries_tmp = timeseries_tmp[:,np.where(included_regions)[0]].transpose()
    timeseries[i,:,:] = zscore(timeseries_tmp,axis=1)
    
    ##import structural connectivity
    StrucCon = np.genfromtxt('/XXX/' + part + '/dti_conmatrix.csv',delimiter=',')
    StrucCon = StrucCon[np.where(included_regions)[0],:][:,np.where(included_regions)[0]]
    StrucCon_norm[i,:,:] = matrix_normalization(StrucCon, version='continuous')

#%%--------------------------------------------------------------------
#                   Determine optimal time horizon
#----------------------------------------------------------------------

#Initiate loop variables
horizon_range=np.linspace(0.001,2.501,6)
trans_cor = np.zeros([horizon_range.size,1])
trans_p = np.zeros([horizon_range.size,1])
Etrans = np.zeros([statenum,statenum,numpart,horizon_range.size])
EtransREG = np.zeros([numreg,statenum,statenum,numpart,horizon_range.size])
transfreq = np.zeros([statenum,statenum,numpart,horizon_range.size])
transfreq_nan = np.zeros([statenum,statenum,numpart,horizon_range.size])
mean_Etrans=np.zeros([statenum,statenum,horizon_range.size])

#Loop over horizon range
for i,horizon in enumerate(horizon_range):
    print('{0:.3f}'.format(horizon))
        
    #Determine the transition energy per participant
    for j,part in enumerate(participants):
        print(part)
        Etrans[:,:,j,i],EtransREG[:,:,:,j,i],transfreq[:,:,j,i] = transition_energy(StrucCon_norm[j,:,:],timeseries[j,:,:],states[j,:],statenum,horizon)
    
    #Determine which transitions are not observed
    transfreq_nan[:,:,:,i]=np.array(transfreq[:,:,:,i])
    transfreq_nan[np.isnan(Etrans)]=np.nan
    
    #Determine the mean transition energies across all participants and correlate with the transition frequency
    mean_Etrans[:,:,i] = np.nanmean(Etrans[:,:,:,i],axis=2)
    mean_transfreq = np.nanmean(transfreq_nan[:,:,:,i],axis=2)
    trans_cor[i],trans_p[i]= spearmanr(mean_Etrans[:,:,i].flatten(),mean_transfreq.flatten())
    print('Correlation: '+str(trans_cor[i]))
        
#%%--------------------------------------------------------------------
#                   Determine minimum energy
#----------------------------------------------------------------------

optimal_T_i=np.argmin(trans_cor[1:])
optimal_T = horizon_range[optimal_T_i]

Etrans_final=Etrans[:,:,:,optimal_T_i]
EtransREG_final=EtransREG[:,:,:,:,optimal_T_i]

HCparticipants=participants[participants.str.contains('^HC')]
numHCpart=HCparticipants.size

Etrans_HCref_mean = np.nanmean(Etrans_final[:,:,0:numHCpart],2)
Etrans_HCref_std = np.nanstd(Etrans_final[:,:,0:numHCpart],2,ddof=1)
Etrans_z = np.divide(np.subtract(Etrans_final,Etrans_HCref_mean[:,:,np.newaxis]),Etrans_HCref_std[:,:,np.newaxis])

#%%--------------------------------------------------------------------
#                   Determine minimum energy
#----------------------------------------------------------------------

CE_total = np.nanmean(Etrans_z,axis=(0,1))
CE_persist = np.nanmean(Etrans_z[np.dstack([np.identity(optimal_T,dtype=bool)]*numpart)].reshape((optimal_T,numpart)),axis=0)
CE_trans = np.nanmean(Etrans_z[~np.dstack([np.identity(optimal_T,dtype=bool)]*numpart)].reshape((optimal_T*optimal_T,numpart)),axis=0)

#%%--------------------------------------------------------------------
#                   Determine minimum energy
#----------------------------------------------------------------------

CE11_eff=Etrans_z[0,0,:]
CE12_eff=Etrans_z[0,1,:]
CE13_eff=Etrans_z[0,2,:]
CE14_eff=Etrans_z[0,3,:]

CE21_eff=Etrans_z[1,0,:]
CE22_eff=Etrans_z[1,1,:]
CE23_eff=Etrans_z[1,2,:]
CE24_eff=Etrans_z[1,3,:]

CE31_eff=Etrans_z[2,0,:]
CE32_eff=Etrans_z[2,1,:]
CE33_eff=Etrans_z[2,2,:]
CE34_eff=Etrans_z[2,3,:]

CE41_eff=Etrans_z[3,0,:]
CE42_eff=Etrans_z[3,1,:]
CE43_eff=Etrans_z[3,2,:]
CE44_eff=Etrans_z[3,3,:]

CE_all = np.stack((CE11_eff,CE12_eff,CE13_eff,CE14_eff,
                   CE21_eff,CE22_eff,CE23_eff,CE24_eff,
                   CE31_eff,CE32_eff,CE33_eff,CE34_eff,
                   CE41_eff,CE42_eff,CE43_eff,CE44_eff),axis=1)

with open('control_energy.pickle', 'wb') as f:
    pickle.dump([Etrans,EtransREG,transfreq,trans_cor,trans_p,trans_cor,horizon_range,optimal_T,CE_total,CE_persist,CE_trans,CE_all], f)

#%%--------------------------------------------------------------------
# References
#----------------------------------------------------------------------
# https://github.com/BassettLab/nctpy
