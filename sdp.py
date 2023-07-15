import numpy as np
import numpy.matlib
# from scipy.stats import norm, lognorm

def run_sdp(sys_param):
    import simLake

    # %-- Initialization --
    disturbance   = sys_param['simulation']['q']
    initial_state = sys_param['simulation']['h_in']
    
    tol = -1    # accuracy level for termination 
    max_it = 10 # maximum iteration for termination 
    
    # %-- Run optimization --
    policy, sys_param = opt_sdp( tol, max_it, sys_param )
    
    # %-- Run simulation --
    Jflo, Jirr, _, _ = simLake.simLake( disturbance, initial_state, policy, sys_param );
    JJ = [Jflo, Jirr]    
    return JJ, policy


def opt_sdp(tol, max_it, sys_param):
    
    # %-- Initialization --
    discr_s = sys_param['algorithm']['discr_s']
    T = sys_param['algorithm']['T']
    n_s = len(discr_s)
    
    min_rel = sys_param['algorithm']['min_rel']
    max_rel = sys_param['algorithm']['max_rel']
    
    H = np.zeros([n_s, T]) #% create Bellman *Note: Index 0 to N
    
    # %-- Backward recursive optimization -- 
    diff_H = np.inf
    count  = 1
    
    while diff_H >= tol:
      H_ = H.copy()
      
      for i in range (n_s): #% for each discretization level of state 
        sys_param['simulation']['vv'] = min_rel[ i, : ]
        sys_param['simulation']['VV'] = max_rel[ i, : ]
        
        H[i],_ = Bellman_sdp( H_, discr_s[i], sys_param )
  
      diff_H = max( np.absolute( H_ - H ))
      # print(' count',count, ' diff_H', diff_H)
      count  = count+1
      
      if count > max_it:
          break   

    return H, sys_param


def Bellman_sdp( H_ , s_curr , sys_param ):
    
    import sim    

    # %-- Initialization --
    discr_s = sys_param['algorithm']['discr_s']
    discr_u = sys_param['algorithm']['discr_u']
    discr_q = sys_param['algorithm']['discr_q']
    p_diff = sys_param['algorithm']['p_diff']
    
    n_u = len(discr_u)
    n_q = len(discr_q)
    
    weights = sys_param['algorithm']['weights']
    gamma = sys_param['algorithm']['gamma']
    
    VV = np.array([sys_param['simulation']['VV']])
    VV = np.matlib.repmat(VV.T, 1 , n_u) #% max release
    vv = np.array([sys_param['simulation']['vv']])
    vv = np.matlib.repmat(vv.T, 1 , n_u) #% min release l
    # print('VV',VV[0],'vv',vv[0])
    
    delta = sys_param['simulation']['delta']
    
    
    # %-- Calculate actual release contrained by min/max release rate --
    discr_u = np.array([discr_u])
    discr_u = np.matlib.repmat(discr_u, n_q, 1)
    sys_param['algorithm']['discr_UU'] = discr_u
    R = np.minimum( VV , np.maximum( vv , discr_u ) ) 
    sys_param['algorithm']['R'] = R
    
    # %==========================================================================
    # % Calculate the state transition; TO BE ADAPTAED ACCORDING TO
    # % YOUR OWN CASE STUDY
    discr_q = np.array([discr_q])
    qq = np.matlib.repmat( discr_q.T, 1, n_u )
    s_next = s_curr + delta* ( qq - R )
    
    # %==========================================================================
    # % Compute immediate costs and aggregated one; TO BE ADAPTED ACCORDING TO
    # % YOUR OWN CASE STUDY
    g1, g2 = sim.immediate_costs( sim.storageToLevel(s_next, sys_param), R, sys_param ) ;
    G = g1*weights[0]+ g2*weights[1]
    sys_param['algorithm']['G'] = G
    # %-- Compute cost-to-go given by Bellman function --
    # % apply linear interpolation to update the Bellman value H_
    
    #____________________________________!!!!!!!!______#
    
    H_ = np.interp( s_next.flatten(), discr_s , H_.flatten() ) #Flatten to convert matrix into 1-D array
    sys_param['algorithm']['H_'] = H_
    H_ = np.reshape( H_, (n_q, n_u) )
    sys_param['algorithm']['H__'] = H_
    # print(H_.shape)
    
    
    # %-- Compute resolution of Bellman value function --
    # % compute the probability of occourence of inflow that falls within the
    # % each bin of descritized inflow level
  
    Q     = (G + gamma*H_).T .dot( p_diff ) #dot product between matrix and vector
    # Q = (G + gamma*H_) .dot( np.ones(len(p_diff) ))
    sys_param['algorithm']['Q'] = Q
    H     = min(Q)
    sens  = np.spacing(1)
    idx_u = np.flatnonzero( Q <= H + sens )
    # print('shape of idx_u in sdp Bellman', idx_u.shape, idx_u)
    
    return H, idx_u

