 # from main_script import sys_param
import sim
import numpy as np
import numpy.matlib

def simLake( q, h_in, policy, sys_param ): 
    # global sys_param
    
    # % Simulation setting
    q_sim = np.append(np.nan, q )
    H = len(q_sim) - 1
    # % Initialization
    h = np.nan*np.ones(len(q_sim))
    s = np.nan*np.ones(len(q_sim))
    r = np.nan*np.ones(len(q_sim))
    u = np.nan*np.ones(len(q_sim))
    
    # % Start simulation
    h[0] = h_in
    # s[0] = sim.levelToStorage(h[0],sys_param)
    
    A  = sys_param['simulation']['A']
    h0 = sys_param['simulation']['h0']    
    s[0] = A*(h[0] - h0)
    
    for t in range(H):
# =============================================================================        
      # % Compute release decision
              
        if sys_param['algorithm']['name'] == 'rand':  
          r_min = sys_param['simulation']['r_min']
          r_max = sys_param['simulation']['r_max']
          
          uu = r_min + rand*(r_max - r_min)
          
# =============================================            
        elif sys_param['algorithm']['name'] == 'doe':
          uf = rand;
          r_min = sys_param['simulation']['r_min'];
          r_max = sys_param['simulation']['r_max'];
          w = sys_param['simulation']['w'];
          
          if uf < 0.2:
            uu = r_min + rand*(r_max*0.66 - r_min);
          elif uf < 0.40:
            uu = r_max*0.66 + rand*(r_max - r_max*0.66);
          else:
            uu = w
          
# =============================================           
        elif sys_param['algorithm']['name'] ==  'ddp':
          import ddp
          discr_s = sys_param['algorithm']['discr_s'];
          discr_q = sys_param['algorithm']['discr_q'];
          discr_u = sys_param['algorithm']['discr_u'];
          
          min_rel = sys_param['algorithm']['min_rel'];
          max_rel = sys_param['algorithm']['max_rel'];
          # print('simLake: min rel=',min_rel,'max rel=',max_rel)
          w = sys_param['simulation']['w'];
         
          idx_q  = np.argmin( np.absolute( discr_q - q_sim[t+1] ) );
          
          # % Minimum and maximum release for current storage and inflow:
          sys_param['simulation']['vv'] = np.interp( s[t], discr_s , min_rel[: , idx_q] )
          sys_param['simulation']['VV'] = np.interp( s[t], discr_s , max_rel[: , idx_q] )
         
          _ , idx_u  = ddp.Bellman_ddp( policy[:,t+1] , s[t] , q_sim[t+1] , sys_param)
          # % Choose one decision value (idx_u can return multiple equivalent decisions)          
          uu = sim.extractor_ref( idx_u , discr_u , w )
          
# =============================================           
        elif sys_param['algorithm']['name'] ==  'sdp':            
          import sdp
          discr_s = sys_param['algorithm']['discr_s'];
          discr_q = sys_param['algorithm']['discr_q'];
          discr_u = sys_param['algorithm']['discr_u'];
          
          min_rel = sys_param['algorithm']['min_rel'];
          max_rel = sys_param['algorithm']['max_rel'];
          
          w = sys_param['simulation']['w'];
          
          # % Minimum and maximum release for current storage and inflow:
          idx_q  = np.argmin( np.absolute( discr_q - q_sim[t+1] ) )
          
          v = np.interp( s[t], discr_s , min_rel[: , idx_q] )
          sys_param['simulation']['vv'] = np.matlib.repmat( v, 1, len(discr_q) ).flatten()
          
          V = np.interp( s[t], discr_s , max_rel[: , idx_q] )
          sys_param['simulation']['VV'] = np.matlib.repmat( V, 1, len(discr_q) ).flatten()
          # print(sys_param['simulation']['VV'].size)
          # print(sys_param['simulation']['VV'])
          # print('s[t]=',s[t])
          
          _ , idx_u  = sdp.Bellman_sdp( policy , s[t] , sys_param )
          # print('idx_u in simlake', idx_u)
    
          # % Choose one decision value (idx_u can return multiple equivalent
          # % decisions)
          uu = sim.extractor_ref( idx_u , discr_u , w )
          # print('uu1=',uu, type(uu))
          
  # =============================================          
        elif sys_param['algorithm']['name'] ==  'emodps':
          import emodps
          
          policy_class = sys_param['algorithm']['policy_class']
          
          if policy_class == 'stdOP':
             uu = emodps.std_operating_policy(h[t], policy, sys_param)
          else:
             raise Exception('Policy class not defined.\
            Please check or modify this function to use a different\
            class of parameterized functions')
          
  # =============================================          
        elif sys_param['algorithm']['name'] ==  'fqi':
          discr_s = sys_param['algorithm']['discr_s'];
          discr_u = sys_param['algorithm']['discr_u'];
          
          w = sys_param['simulation']['w'];
          
          _, idx_u = readQ(s[t], discr_s, discr_u, policy['Q']);
          uu = sim.extractor_ref( idx_u , discr_u , w );
          
# =============================================           
        elif sys_param['algorithm']['name'] ==  'ssdp':
          interp_foo = sys_param['algorithm']['interp_foo'];
          T          = sys_param['algorithm']['T'];
          discr_u    = sys_param['algorithm']['discr_u'];
          discr_s    = sys_param['algorithm']['discr_s'];
          discr_q    = sys_param['algorithm']['discr_q'];      
          min_rel    = sys_param['algorithm']['min_rel'];
          max_rel    = sys_param['algorithm']['max_rel'];
          esp_sample = sys_param['algorithm']['esp_sample'];
          
          t_idx = mod[t-1, T] + 1
          
          w = sys_param['simulation']['w']
          
          _, idx_q  = np.minimum(np.absolute( discr_q - q_sim[t+1] ))
          
          sys_param['simulation']['vv'] = interp_foo( discr_s, min_rel [: , idx_q] , s[t] )
          sys_param['simulation']['VV'] = interp_foo( discr_s, max_rel [: , idx_q] , s[t] )
          
          # idx_u = reopt_ssdp( policy.H(:,:,t_idx+1), s[t], esp_sample(t_idx,:) );
#           uu = sim.extractor_ref( idx_u , discr_u, w );
          
# =============================================           
        elif sys_param['algorithm']['name'] ==  'iso' :
          regressorName = sys_param['algorithm']['regressorName']
          
          if regressorName == 'linear_spline':
             uu = policy.predict(h[t])
          else:
             raise Exception('Regressor not defined. Please check or modify this \
                        function to use a different regression method')
          
# =============================================           
        else:
          uu = np.nan        
         
        u[t] = uu
        # print('u=',u)
       # % Hourly integration of mass-balance equation
        # print('s[0]=',s[0])
        s[t+1], r[t+1] = sim.massBalance( s[t], u[t], q_sim[t+1], sys_param )
        # print('s=',s)
        # h[t+1] = sim.storageToLevel(s[t+1],sys_param)
        h[t+1] = s[t+1]/A + h0
      
    
# # =============================================================================     
    # % Calculate objectives (daily average of immediate costs)
    # print(s)
    # return
    g_flo, g_irr = sim.immediate_costs(h[1:], r[1:], sys_param);
    
    Jflo = np.mean(g_flo)
    Jirr = np.mean(g_irr)    
    # print('h', h)
    
    sys_param['algorithm']['h'] = h
    sys_param['algorithm']['s'] = s
    sys_param['algorithm']['r'] = r
    sys_param['algorithm']['u'] = u
    
    # print(policy)
    # return

    return Jflo, Jirr, h, u#, r, g_flo, g_irr

