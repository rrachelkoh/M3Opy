import numpy as np

def run_mpc(errorLevel, sys_param):
    import sim
    
    h_in  = sys_param['simulation']['h_in']
    e = sys_param['simulation']['q']
    
    # % --- System Parameters
    S0 = sim.levelToStorage(h_in, sys_param) # Initial storage
    
    # % --- Simulation and Optimization
    # % Run MPC
    g_flo, g_irr, g, s, u, r ,v, V = sim_mpc_Test( e, S0, errorLevel, sys_param)
    
    # % Collect outputs
    Ompc = {'u':u, 'r':r, 's':s, 'v':v, 'V':V, 'g_flo':g_flo, 'g_irr':g_irr}
    
    # % Pareto front objectives
    Jflo = np.mean(g_flo)
    Jirr = np.mean(g_irr)
    
    JJ_mpc = [Jflo, Jirr]    
    return JJ_mpc, Ompc

def sim_mpc_Test(e, S0, errorLevel, sys_param):
    import sim
    
    # % -- Initialization --
    delta   = sys_param['simulation']['delta']
    
    p       = sys_param['algorithm']['P']
    weights = sys_param['algorithm']['weights']
    
    N = len(e) #% Length of the simulation
    
    u = np.nan * np.ones(shape = (N+1,1))
    r = np.nan * np.ones(shape = (N+1,1))
    s = np.nan * np.ones(shape = (N+1,1))
    
    v = np.zeros (shape = (N+1,1))
    V = np.zeros (shape = (N-p+1,1))
    g_flo = np.zeros (shape = (N-p+1,1))
    g_irr = np.zeros (shape = (N-p+1,1))
    g = np.zeros (shape = (N-p+1,1))
    
    # % define initial conditions and the integration time-step
    s[0] = S0
    
    # % mpc input trajectory for simulation
    e_sim = np.append(np.nan, e )
    
    # % -- Simulation --
    for t in range(N-p+1):
      # disp(num2str(t))
      
      # % prediction (b-a).*rand(1000,1) + a;
      if ((t+p) <= N)   : 
        e_pred = e[t:t+p]+errorLevel*(max(e)-min(e))*(2*np.random.rand(p,1)-1)
      else:
        e_pred = e[t:]+errorLevel*(max(e)-min(e))*(2*np.random.rand(len(e[t:]),1)-1)
      
      
      # % Define the initial condition (i.e. current state s(t))
      s1_init = s[t]
      
      # % Determine the trajectory of the optimal controls
      x  = mpc_Test(s1_init, e_pred, weights, sys_param).x
      u[t] = x[[0]]
      
      # % Compute release and mass balance equation
      v[t] = sim.min_release(s[t], sys_param)
      V[t] = sim.max_release(s[t], sys_param)
      
      r[t+1] = u[t]
      s[t+1] = s[t] + delta * ( e_sim[t+1] - r[t+1] );
      ht = sim.storageToLevel(s[t], sys_param);
      
      g_flo[t], g_irr[t] = sim.immediate_costs(ht, r[t+1], sys_param)
      g[t] = g_flo[t]*weights[0] + g_irr[t]*weights[1]
    
    return g_flo, g_irr, g, s, u, r, v, V

# =============================================================================
# MPC with DDP engine
def run_mpc_ddp(errorLevel, sys_param):
    import sim
    
    h_in  = sys_param['simulation']['h_in']
    e = sys_param['simulation']['q']
    
    # % --- System Parameters
    S0 = sim.levelToStorage(h_in, sys_param) # Initial storage
    
    # % --- Simulation and Optimization
    # % Run MPC
    g_flo, g_irr, g, s, u, r ,v, V = sim_mpc_Test_ddp( e, S0, errorLevel, sys_param)
    
    # % Collect outputs
    Ompc = {'u':u, 'r':r, 's':s, 'v':v, 'V':V, 'g_flo':g_flo, 'g_irr':g_irr}
    
    # % Pareto front objectives
    Jflo = np.mean(g_flo)
    Jirr = np.mean(g_irr)
    
    JJ_mpc = [Jflo, Jirr]    
    return JJ_mpc, Ompc

def sim_mpc_Test_ddp(e, S0, errorLevel, sys_param):
    import sim
    
    # % -- Initialization --
    delta   = sys_param['simulation']['delta']
    
    p       = sys_param['algorithm']['P']
    weights = sys_param['algorithm']['weights']
    
    N = len(e) #% Length of the simulation
    
    u = np.nan * np.ones(shape = (N+1,1))
    r = np.nan * np.ones(shape = (N+1,1))
    s = np.nan * np.ones(shape = (N+1,1))
    
    v = np.zeros (shape = (N+1,1))
    V = np.zeros (shape = (N-p+1,1))
    g_flo = np.zeros (shape = (N-p+1,1))
    g_irr = np.zeros (shape = (N-p+1,1))
    g = np.zeros (shape = (N-p+1,1))
    
    # % define initial conditions and the integration time-step
    s[0] = S0
    
    # % mpc input trajectory for simulation
    e_sim = np.append(e, np.nan)
    
    # % -- Simulation --
    for t in range(N-p+1):
      # disp(num2str(t))
      
      # % prediction (b-a).*rand(1000,1) + a;
      if ((t+p) <= N)   : 
        e_pred = e[t:t+p]+errorLevel*(max(e)-min(e))*(2*np.random.rand(p,1)-1)
      else:
        e_pred = e[t:]+errorLevel*(max(e)-min(e))*(2*np.random.rand(len(e[t:]),1)-1)
      
            
      # % Define the initial condition (i.e. current state s(t))
      s1_init = s[t]
      
      # % Determine the trajectory of the optimal controls
      _, x, _  = mpc_Test_ddp(e_pred[0], sys_param)
      # print('x=',x)
      u[t] = x[0]
      
      # % Compute release and mass balance equation
      v[t] = sim.min_release(s[t], sys_param)
      V[t] = sim.max_release(s[t], sys_param)
      
      r[t+1] = u[t]
      s[t+1] = s[t] + delta * ( e_sim[t+1] - r[t+1] );
      ht = sim.storageToLevel(s[t], sys_param);
      
      g_flo[t], g_irr[t] = sim.immediate_costs(ht, r[t+1], sys_param)
      g[t] = g_flo[t]*weights[0] + g_irr[t]*weights[1]
      
      
    
    return g_flo, g_irr, g, s, u, r, v, V


def mpc_Test(s1_init, e_pred, weights, sys_param):
    import sim
    # from scipy import optimize
    
    # % -- Initialization --
    w     = sys_param['simulation']['w']
    delta = sys_param['simulation']['delta']
    
    # % define the length of the prediction horizon, over which the optimal
    # % decision vector x must be optimized
    e_pred = np.append(np.nan, e_pred )
    H      = len(e_pred)
    
    # % initialize vectors
    r = np.nan * np.ones(shape = (H,1))
    s = np.nan * np.ones(shape = (H,1))
    v = np.nan * np.ones(shape = (H,1))
    V = np.nan * np.ones(shape = (H,1))
    
    # % define initial conditions and the integration time-step
    s[0]= s1_init
    
    # % minimum and maximum value of the control variables
    min_u = sys_param['simulation']['r_min']
    max_u = sys_param['simulation']['r_max']
    
    # % -- Optimization of the decision(s) vector --
    # % define constraints
    A = np.concatenate((np.identity(H-1), -np.identity(H-1)), axis=0)
    b = np.concatenate((max_u*np.ones(shape = (H-1,)), -min_u*np.ones(shape = (H-1,))), axis=0)
    
    # ===========================================================================
    # def constraint1(x):
    # #Constraint Ax<=b
    #     return A@x - b    
    # # ===========================================================================
    # def cons_f(x):
    #     return A@x
    
    # from scipy.optimize import BFGS
    # from scipy.optimize import NonlinearConstraint
    
    # nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, b, jac='2-point', hess=BFGS())
    # % Determine the initialization vector
    x0 = np.ones(shape = (H-1,))*w
        
    # ===========================================================================
    
    def optfun(x):       
        
    # % Nested function that computes the objective function
        
        # % simulation
        for t in range (H-1):
          # % compute release and mass balance equation
          v[t] = sim.min_release(s[t], sys_param)
          V[t] = sim.max_release(s[t], sys_param)
          r[t+1] = x[0]
          s[t+1] = s[t] + delta * ( e_pred[t+1] - r[t+1] )
        
        # % compute the step-costs over the simulation horizon and the aggregated
        # % cost, which correspond to the objective function
        g = step_cost_2( s.flatten(), r[1:], v[:-1], V[:-1], sys_param)
        
        f = np.mean(g)
        
        return f
    
    
    # cons = {'type':'ineq','fun':constraint1}
    
    from scipy.optimize import LinearConstraint
    linear_constraint = LinearConstraint(A, -np.inf*np.ones(shape = b.shape), b)
    
    # cons1 = ({'type': 'ineq',
             # 'fun' : lambda x: A*x - b})

    #method='L-BFGS-B', 
    # x = optimize.minimize(optfun, x0, args=(), method='BFGS', 
    #                       constraints=cons, 
    #                       options={'disp': None, 'maxls': 20, 'iprint': -1, 'gtol': 1e-05, 'eps': 1e-08, 
    #                                'maxiter': 2000, 'ftol': np.spacing(1), 'maxcor': 10, 'maxfun': 1000})
    
    
    from scipy.optimize import minimize
    x = minimize(optfun, x0, args=(), method='trust-constr', 
             hess=None, hessp=None, bounds=None, constraints=linear_constraint, tol=None, callback=None, 
             options={'maxiter': 2000})
    
    print(x)
    return x




def step_cost_2( s, r, v, V, sys_param):
    import sim
    
    w    = sys_param['simulation']['w']
    hFLO = sys_param['simulation']['hFLO']
    
    weights = sys_param['algorithm']['weights']
    
    # % compute step-costs:
    H = len(s)-1
    
    # pre-allocate the memory
    g_flo = np.zeros (shape = (H,1))
    g_irr = np.zeros (shape = (H,1))
    
    for i in range(H):
      ht = sim.storageToLevel(s[i], sys_param)
      g_flo[i], g_irr[i] = sim.immediate_costs(ht, r[i], sys_param)
    
    # % Cost for violation of constraint
    g_MaR = np.maximum( r - V ,0)
    g_MiR = np.maximum( v - r ,0)
    
    # % Penalties
    g_floP = 100*max(sim.storageToLevel(s[-1], sys_param) - hFLO, 0)
    g_irrP = max(w - sim.max_release(s[-1], sys_param), 0)
    
    # % aggregate step-costs:
    G = g_flo*weights[0] + g_irr*weights[1] \
        + g_floP*weights[0] + g_irrP*weights[1] \
            + g_MaR + g_MiR 

    return G

    
# =============================================================================
# def mpc_Test_ddp(e_pred, sys_param):
#     
#     # %-- Initialization --
#     discr_s = sys_param['algorithm']['discr_s']
#     discr_q = sys_param['algorithm']['discr_q']
#     
#     min_rel = sys_param['algorithm']['min_rel']
#     max_rel = sys_param['algorithm']['max_rel']
#     
#     Hend = 0#sys_param['algorithm']['Hend']
#     
#     N   = len(e_pred);
#     n_s = len(discr_s);
#     
#     
#     # % minimum and maximum value of the control variables
#     min_u = sys_param['simulation']['r_min']
#     max_u = sys_param['simulation']['r_max']
#     
#     # R = np.nan * np.ones(shape = (N,1))
#     R = np.zeros([n_s, 1])
#     
#     # %-- Backward recursive optimization -- 
#     # % note that 2 loops are run, with the first used for computing a penalty
#     # % over the final state used in the second loop
#     H = np.zeros([n_s, N+1]); #% create Bellman
#     H[:,-1] = Hend;     #% initialize Bellman to penalty function
#     for t in range(N,-2,1):
#       for i in range (n_s): #% for each discretization level of state 
#         # % calculate the min/max storage-discharge relationship. See README.md
#         # % file for the description of 'min_rel' and 'max_rel' ;
#         sys_param['simulation']['vv'] = np.interp( e_pred[t], discr_q , min_rel[i,:] );
#         print(sys_param['simulation']['vv'])
#         sys_param['simulation']['VV'] = np.interp( e_pred[t], discr_q , max_rel[i,:] );
#         
#         H[i,t], R[i,t],_ = Bellman_mpc( H[:,t+1], discr_s[i], e_pred[t], sys_param)
# # =============================================================================
#     print('disturbance=',e_pred)
# 	# LOOP 2 #
#     Hend = H[:,-1]
#     H = np.zeros([n_s, N+1]) #% create Bellman *Note: Index 0 to N
#     H[:,-1] = Hend;     # initialize Bellman to penalty function
#     for t in range(N-1,-1,-1): #Index 0 to (N-1)
#       for i in range (n_s): #% for each discretization level of state 
#         # % calculate the min/max storage-discharge relationship. See README.md
#         # % file for the description of 'min_rel' and 'max_rel' ;
#         sys_param['simulation']['vv'] = np.interp( e_pred[t], discr_q , min_rel[i,:] );
#         print(sys_param['simulation']['vv'])
#         sys_param['simulation']['VV'] = np.interp( e_pred[t], discr_q , max_rel[i,:] );
# # =============================================================================
#         
#         H[i,t], R[i], _ = Bellman_mpc( H[:,t+1], discr_s[i], e_pred[t], sys_param )
#     print('R=',R)
#     return H, R, sys_param
# 
# def Bellman_mpc(H_ , s_curr , q_curr, sys_param):
#     import sim    
#     
#     w    = sys_param['simulation']['w']
#     hFLO = sys_param['simulation']['hFLO']
# 
#     # %-- Initialization --
#     discr_s = sys_param['algorithm']['discr_s']
#     discr_u = sys_param['algorithm']['discr_u']
#     weights = sys_param['algorithm']['weights']
#     
#     VV = sys_param['simulation']['VV'] #% max release
#     vv = sys_param['simulation']['vv'] #% min release
#     print('VV=',VV)
#     
#     delta = sys_param['simulation']['delta']
#     
#     # %-- Calculate actual release contrained by min/max release rate --
#     R = np.minimum( VV , np.maximum( vv , discr_u ) ) 
#     # print('R from Bellman_mpc=',R)
#    
#     # %==========================================================================
#     # % Calculate the state transition; TO BE ADAPTAED ACCORDING TO
#     # % YOUR OWN CASE STUDY
#     # R = R[-1]
#     s_next = s_curr + delta* ( q_curr - R )
#     # h = s_next/sys_param['simulation']['A'] + sys_param['simulation']['h0']
#     
#     # %==========================================================================
#     # % Compute immediate costs and aggregated one; TO BE ADAPTED ACCORDING TO
#     # % YOUR OWN CASE STUDY
#     g_flo, g_irr = sim.immediate_costs( sim.storageToLevel(s_next,sys_param), R, sys_param) ;
#     
#     
#     print('s_next',s_next)
#     
#     # % Cost for violation of constraint
#     g_MaR = np.maximum( R - VV ,0)
#     g_MiR = np.maximum( vv - R ,0)
#     
#     # % Penalties
#     g_floP = 100*np.maximum(sim.storageToLevel(s_next, sys_param) - hFLO, 0)
#     
#     g_irrP = np.zeros(len(s_next))
#     # print(sim.max_release(s_next, sys_param))
#     # sys_param['algorithm']['mpc_bellman']= sim.max_release(s_next, sys_param)
#     for n in range(len(s_next)):
#     	g_irrP[n] = np.maximum(w - sim.max_release(s_next[n], sys_param), 0)
#     
#         # % aggregate step-costs:
#     G = g_flo*weights[0] + g_irr*weights[1] \
#         + g_floP*weights[0] + g_irrP*weights[1] \
#             + g_MaR + g_MiR 
#     
#     # %-- Compute cost-to-go given by Bellman function --
#     # % apply linear interpolation to update the Bellman value H_
#     H_ = np.interp( s_next, discr_s , H_ )
#     
#     # %-- Compute resolution of Bellman value function --
#     Q     = G + H_
#     print('Q',Q)
#     H     = min(Q)
#     sens  = np.spacing(1)
#     idx_u = np.flatnonzero( Q <= H + sens )
#     sys_param['algorithm']['r'] = R
#     # return
#     return H, R, idx_u
#     
# # =============================================================================
# =============================================================================
