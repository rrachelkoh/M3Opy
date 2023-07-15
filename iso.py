def run_iso(regressor,sys_param):
    
    import ddp, simLake
    import pwlf
    import numpy as np
    
    # % -- Error check --   
    if regressor=='linear_spline':
        sys_param['regressor'] = regressor
    else:        
        raise Exception('Regressor not defined. Please check or modify this \
                        function to use a different regression method')
    
    sys_param['algorithm']['name'] = 'ddp'
    
    # % -- Initialization --
    q    = sys_param['simulation']['q']
    h_in = sys_param['simulation']['h_in']
    
    # % -- Run optimization to collect samples -- 
    policy,_ = ddp.opt_ddp( q , sys_param )
    # print(policy)
    JJ_DDP1, JJ_DDP2, h, u = simLake.simLake( q, h_in, policy, sys_param  )
    JJ_DDP = [JJ_DDP1, JJ_DDP2]
    # print('JJ_DDP', JJ_DDP)
    
    # % -- Run regression to find the approximate decision rules --
    X = h[1:] #% regressors
    Y = u[1:] #% regressand
    
    sys_param['algorithm']['X'] = X
    sys_param['algorithm']['Y'] = Y
    
    if regressor=='linear_spline':
      # % Fit a piecewise linear function to data
      
      # initialize piecewise linear fit with your x and y data
        policy = pwlf.PiecewiseLinFit(X[:-1], Y[:-1])
        
        knots = []        
        for i in range(6):
            knot = min(X)+(max(X)-min(X))/5*i
            knots.append(knot)
    
        policy.fit_with_breaks(knots)
        sys_param['algorithm']['policy_fit'] = policy
        
        # # predict for the determined points
        xHat = knots
        yHat = policy.predict(xHat)        
        
        # plot the results
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(X, Y, 'o')
        # plt.plot(xHat, yHat, '-')
      
    else:        
        raise Exception('Regressor not defined. Please check or modify this \
                        function to use a different regression method')

    
    # % -- Run simulation to collect performance --
    sys_param['algorithm']['name'] = 'iso';
    JJ1, JJ2, _, _ = simLake.simLake( q, h_in, policy, sys_param  )
    JJ = [JJ1, JJ2]
    # print('JJ', JJ)
    
    # % compute error
    err_perc = 100*(np.absolute(np.array(JJ_DDP) - np.array(JJ))/np.array(JJ_DDP))
    
    return JJ, policy, err_perc


# aa = a.values.tolist()
# h_new = []
# for i in range(731):
#     h = policy.predict(aa[i][0])[0]
#     h_new.append(h)