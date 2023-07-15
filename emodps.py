import numpy as np
import math

def run_emodps(policy_class, moea_param, sys_param):
    import nsga2

    # % -- POLICY SETTING --
    if (policy_class == 'stdOP') :
      sys_param['algorithm']['policy_class'] = policy_class 
    else:
      raise Exception('Policy class not defined. Please check or modify \
                      this function to use a different class of parameterized functions')
    
    
    # % -- OPTIMIZATION SETTING --
    M = 2 ;     #% objective number
    V = 4 ;     #% decision variables number (policy parameters)
    min_range = [ -.5, -.5, 0, 0 ] ;
    max_range = [ 1.25, 1.25, math.pi/2, math.pi/2 ] ;
    
    # % -- RUN OPTIMIZATION --
    if (moea_param['name'] == 'NSGAII') :
      #% The algorithm returns the initial (ch0) and the final population (chF)
      ch0, chF = nsga2.nsga_2(moea_param['pop'],moea_param['gen'],M,V,min_range,max_range, sys_param)
      
      # % Save Pareto approximate set (PS) and front (JJ)
      PS = chF[:, 1:V]
      JJ = chF[:, V:V+M]
    else:
      raise Exception('MOEA not defined. Please check or modify this function \
                      to run a different optimization algorithm.')
      
    return JJ, PS

def evaluate_objective(x, M, V, sys_param):
    
    # J1: performance for flooding objective
    # J2: performance for irrigation objective
    
    import simLake 
    
    x = x[:V]
    
    #% --------------------------------------
    #% insert here your function f = f(x):
    
    q    = sys_param['simulation']['q']
    h_in = sys_param['simulation']['h_in']
    
    #% -- Get policy parameters --
    policy = x
    
    #% -- Run simulation to collect the results --
    J1, J2, _, _  = simLake.simLake( q, h_in, policy, sys_param )
    f = [J1, J2]
    # print('eval obj f',f)
    #% --------------------------------------
    
    #% Check for error
    if ( len(f) != M ):
      raise Exception('Incorrect number of output objectives. Expecting to solve \
                      %d objectives formulation. Please check your objective function again')
      
    return f


def std_operating_policy(h, policy,sys_param):
    w = sys_param['simulation']['w']
    
    #% -- Get policy parameters --
    h1 = policy[0]
    h2 = policy[1]
    m1 = policy[2]
    m2 = policy[3]
    
    #% -- Construct the policy using piecewise linear functions --
    L1 = w + math.tan(m1) * ( h - h1 )
    L2 = w + math.tan(m2) * ( h - h2 )
    r  = max( [ min( L1 , w ) , L2 ] )
    
    r = max(0,r)
    
    return r