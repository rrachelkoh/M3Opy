### ORIGINAL ###

import numpy as np
# import simLake, sim

def run_ddp(sys_param):
    import simLake

    # %-- Initialization --
    disturbance   = sys_param['simulation']['q']
    initial_state = sys_param['simulation']['h_in']
    
    # %-- Run optimization --
    policy, sys_param = opt_ddp( disturbance, sys_param )
    # %-- Run simulation --
    Jflo, Jirr, _, _ = simLake.simLake( disturbance, initial_state, policy, sys_param );
    JJ = [Jflo, Jirr]    
    return JJ, policy


def opt_ddp(disturbance, sys_param):
    
    # %-- Initialization --
    discr_s = sys_param['algorithm']['discr_s']
    discr_q = sys_param['algorithm']['discr_q']
    
    min_rel = sys_param['algorithm']['min_rel']
    max_rel = sys_param['algorithm']['max_rel']
    
    Hend = sys_param['algorithm']['Hend']
    
    N   = len(disturbance);
    n_s = len(discr_s);
    
    
    # %-- Backward recursive optimization -- 
    # % note that 2 loops are run, with the first used for computing a penalty
    # % over the final state used in the second loop
    H = np.zeros([n_s, N+1]); #% create Bellman
    H[:,-1] = Hend;     #% initialize Bellman to penalty function
    for t in range(N,-2,1):
      for i in range (n_s): #% for each discretization level of state 
        # % calculate the min/max storage-discharge relationship. See README.md
        # % file for the description of 'min_rel' and 'max_rel' ;
        sys_param['simulation']['vv'] = np.interp( disturbance[t], discr_q , min_rel[i,:] );
        sys_param['simulation']['VV'] = np.interp( disturbance[t], discr_q , max_rel[i,:] );
        
        H[i,t], _ = Bellman_ddp( H[:,t+1], discr_s[i], disturbance[t], sys_param)
    # LOOP 2 #
    Hend = H[:,-1]
    H = np.zeros([n_s, N+1]) #% create Bellman *Note: Index 0 to N
    H[:,-1] = Hend;     # initialize Bellman to penalty function
    for t in range(N-1,-1,-1): #Index 0 to (N-1)
      for i in range (n_s): #% for each discretization level of state 
        # % calculate the min/max storage-discharge relationship. See README.md
        # % file for the description of 'min_rel' and 'max_rel' ;
        sys_param['simulation']['vv'] = np.interp( disturbance[t], discr_q , min_rel[i,:] );
        sys_param['simulation']['VV'] = np.interp( disturbance[t], discr_q , max_rel[i,:] );
        
        H[i,t], _ = Bellman_ddp( H[:,t+1], discr_s[i], disturbance[t], sys_param )
    return H, sys_param


def Bellman_ddp( H_ , s_curr , q_curr, sys_param ):
    
    import sim    

    # %-- Initialization --
    discr_s = sys_param['algorithm']['discr_s']
    discr_u = sys_param['algorithm']['discr_u']
    weights = sys_param['algorithm']['weights']
    
    VV = sys_param['simulation']['VV'] #% max release
    vv = sys_param['simulation']['vv'] #% min release
    
    delta = sys_param['simulation']['delta']
    
    # %-- Calculate actual release contrained by min/max release rate --
    R = np.minimum( VV , np.maximum( vv , discr_u ) ) ;
   
    # %==========================================================================
    # % Calculate the state transition; TO BE ADAPTAED ACCORDING TO
    # % YOUR OWN CASE STUDY
    s_next = s_curr + delta* ( q_curr - R )
    # h = s_next/sys_param['simulation']['A'] + sys_param['simulation']['h0']
    
    # %==========================================================================
    # % Compute immediate costs and aggregated one; TO BE ADAPTED ACCORDING TO
    # % YOUR OWN CASE STUDY
    g1, g2 = sim.immediate_costs( sim.storageToLevel(s_next,sys_param), R, sys_param) ;
    # g1, g2 = sim.immediate_costs( h, R, sys_param)
    G = g1*weights[0]+ g2*weights[1]
    # %-- Compute cost-to-go given by Bellman function --
    # % apply linear interpolation to update the Bellman value H_
    H_ = np.interp( s_next, discr_s , H_ )
    
    # %-- Compute resolution of Bellman value function --
    Q     = G + H_
    H     = min(Q)
    sens  = np.spacing(1)
    idx_u = np.flatnonzero( Q <= H + sens )
    
    return H, idx_u