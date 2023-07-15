import numpy as np 
import matplotlib.pyplot as plt

import sim

# 'ddp','sdp','iso','mpc','emodps'
opt_mtd = 'mpc'

# =============================================================================

#Create nested reservoir data
grids = { 'discr_q':np.loadtxt('discr_q.txt', delimiter=","),
  'discr_s':np.loadtxt('discr_s.txt', delimiter=","),
  'discr_u':np.loadtxt('discr_u.txt', delimiter=",")}

#Create global nested dictionary
global sys_param
sys_param = { 'simulation': { 'q':np.loadtxt('inflow.txt'), 'h_in':0.6, 'w':370, 'hFLO':0.8, 
                             'h0':-0.5, 'A':145900000, 'r_min':0, 'r_max':518, 'delta':60*60*24,
                             'vv' : 0, 'VV' : 0} }

# =============================================================================

# ========Run Discrete Dynamic Programming ======== # 
if opt_mtd == 'ddp':
    import ddp
    # % Configure the parameters 
    vv, VV = sim.construct_rel_matrices(grids,sys_param) #compute daily min/max release matrices
    grids.update({'min_rel' : vv, 'max_rel' : VV})
    
    sys_param['algorithm'] = grids
    sys_param['algorithm'].update({'name' : opt_mtd, 'Hend' : 0})#Hend: penalty set to 0, T: period =1 (assume stationary conditions)
    
    # =============================================================================
    # weights for aggregation of objectives
    wts = [[1, 0], [.75, .25], [.5, .5], [.35, .65], [.2, .8], [.1, .9], [0, 1]]
    Nalt   = len(wts)
    
    
    JJ_ddp = np.nan * np.ones(shape = (Nalt,2))
    Hddp = [ [] for i in range(Nalt) ]
    
    for i in range(Nalt):
      sys_param['algorithm']['weights'] = wts[i]
      # import ddp
      JJ_ddp[i], Hddp[i] = ddp.run_ddp(sys_param)
      # print('weight=',sys_param['algorithm']['weights'])
    
    # plt.figure()
    plt.plot( JJ_ddp[:,0], JJ_ddp[:,1], 'kD' );
    plt.xlabel('flooding')
    plt.ylabel('irrigation')   
    

    
# =============================================================================

# ========Run Stochastic Dynamic Programming ======== # 
if opt_mtd == 'sdp':
    vv, VV = sim.construct_rel_matrices(grids, sys_param) #compute daily min/max release matrices
    
    sys_param['algorithm']= grids
    sys_param['algorithm'].update({'name': opt_mtd, 'Hend':0, 'T':1, 'gamma':1, 'min_rel' : vv, 'max_rel' : VV})
    
    logq = np.log(sys_param['simulation']['q'])#convert to log-normal dist
    
    from scipy.stats import lognorm
    discr_q = sys_param['algorithm']['discr_q']
    cdf_q = lognorm.cdf(discr_q, logq.std(), 0, np.exp(logq.mean())) #std & mean of log-norm dist
    p_q        = np.diff(cdf_q)                 
    p_diff_ini = 1-np.sum(p_q)
    p_diff     = np.append(p_diff_ini, p_q)    
    
    sys_param['algorithm']['p_diff'] = p_diff
    
    wts = [[1, 0], [.75, .25], [.5, .5], [.35, .65], [.2, .8], [.1, .9], [0, 1]]
    Nalt   = len(wts)
    JJ_sdp = np.nan * np.ones(shape = (Nalt,2))
    Hsdp = [ [] for i in range(Nalt) ]
    
    for i in  range(Nalt):
      print(wts[i])
      sys_param['algorithm']['weights'] = wts[i]
      import sdp
      JJ_sdp[i], Hsdp[i] = sdp.run_sdp(sys_param)
    
    ##########################################################################
    #Plot pareto front
    plt.figure()
    plt.plot( JJ_sdp[:,0], JJ_sdp[:,1], 'o' )
    plt.xlabel('flooding')
    plt.ylabel('irrigation')
       
    ################################################    
    plt.figure()
    for combi in range(len(wts)):
        BellFn = Hsdp[combi]          
        plt.plot(np.arange(BellFn.shape[0]), BellFn, linewidth=1.2, label=wts[combi])
    plt.xlabel('storage')
    plt.ylabel('cost')
    plt.legend()
    plt.title('SDP Bellman')

# =============================================================================

# ========Run Implicit Stochastic Optimization ======== # 
if opt_mtd == 'iso':
    import iso
    # % Configure the parameters 
    vv, VV = sim.construct_rel_matrices(grids,sys_param) #compute daily min/max release matrices
    grids.update({'min_rel' : vv, 'max_rel' : VV})
    
    sys_param['algorithm'] = grids
    sys_param['algorithm'].update({'name' : opt_mtd, 'Hend' : 0})#Hend: penalty set to 0
    
    # =============================================================================
    
    # Define regression method
    regressor = 'linear_spline'
    sys_param['algorithm']['regressorName'] = regressor

    # weights for aggregation of objectives
    # wts = [[1, 0], [.75, .25], [.5, .5], [.3, .7], [.2, .8], [.1, .9], [0, 1]]
    wts = [[1, 0], [.75, .25], [.35, .65], [.2, .8], [.1, .9], [0, 1]]
    Nalt   = len(wts)
    JJ_iso = np.nan * np.ones(shape = (Nalt,2))
    err_perc = np.nan * np.ones(shape = (Nalt,2))
    policy = [ ]
    
    # for i in range(0,1):
    for i in range(Nalt):
      sys_param['algorithm']['weights'] = wts[i]
      JJ_iso[i], _, err_perc[i] = iso.run_iso(regressor, sys_param)
    
    plt.figure()
    plt.plot( JJ_iso[:,0], JJ_iso[:,1], 'o' )
    plt.xlabel('flooding')
    plt.ylabel('irrigation')   
    
# =============================================================================
# ========Run Model Predictive Control ======== # 
if opt_mtd == 'mpc':
    import mpc
    sys_param['algorithm'] = {'name' : opt_mtd, 'P':3, 'mi_e':np.mean(sys_param['simulation']['q']), 
                              'sigma_e':np.std(sys_param['simulation']['q'])}    
    
    # % mpc_input  = sys_param.simulation.q;  % Candidate disturbance variable to be predicted for MPC
    errorLevel = 0   # Disturbance prediction error [%] 
    # wts = [[1, 0], [.75, .25], [.5, .5], [.35, .65], [.2, .8], [.1, .9], [0, 1]]
    wts = [[1, 0]]
    Nalt   = len(wts)
    JJ_mpc = np.nan * np.ones(shape = (Nalt,2))
    Ompc = [ [] for i in range(Nalt) ]
    
    for i in range(Nalt):
      sys_param['algorithm']['weights'] = wts[i]
      JJ_mpc[i], Ompc[i] = mpc.run_mpc(errorLevel, sys_param)
      print( JJ_mpc)
    
    plt.figure()
    plt.plot( JJ_mpc[:,0], JJ_mpc[:,1], 'bo' )
    plt.xlabel('flooding')
    plt.ylabel('irrigation') 
       
    
# =============================================================================  
 
# ======== Run Evolutionary Multi-Objective Direct Policy Search ======== # 
if opt_mtd == 'emodps':
    import emodps
    # % Define the parameterized class for the policy (i.e., standard operating
    # % policy)
    sys_param.update({'algorithm': { 'name':'emodps' } })
    pClass = 'stdOP'
    
    # % Define MOEA and its setting (i.e., NSGAII)
    moea_param = { 'name':'NSGAII', 'pop':40, 'gen':50 }
    
    JJ_emodps, Popt = emodps.run_emodps(pClass, moea_param, sys_param)
    
    # plt.figure()
    plt.plot( JJ_emodps[:,0], JJ_emodps[:,1], 'o' )
    plt.xlabel('flooding')
    plt.ylabel('irrigation') 
        