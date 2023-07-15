import numpy as np

def max_release(s,sys_param):
    h = storageToLevel(s,sys_param);

    if (h <= -0.5):
      q = 0.0
    elif (h <= -0.40):
      q = 1488.1*h + 744.05
    else:
      q = 33.37*(h + 2.5)**2.015
    
    V = q;
    return V

def min_release(s,sys_param):
    h = storageToLevel(s,sys_param);

    if (h <= 1.25):
      q = 0.0
    else:
      q = 33.37*(h + 2.5)**2.015
      
    v = q
    return v


def storageToLevel(s,sys_param):
    # from main_script import sys_param
    # global sys_param

    A  = sys_param['simulation']['A']
    h0 = sys_param['simulation']['h0']
    
    h = s/A + h0   
    return h

def levelToStorage(h,sys_param):
    # from main_script import sys_param
    # global sys_param
    
    A  = sys_param['simulation']['A']
    h0 = sys_param['simulation']['h0']
    
    s = A*(h - h0)
    return s


def massBalance( s, u, q, sys_param ):
    HH = 24;
    delta = 3600
    s_ = np.nan * np.ones(shape = (HH+1,1))
    r_ = np.nan * np.ones(shape = (HH+1,1))
    
    s_[0] = s;
    for i in range(HH):
      qm = min_release(s_[i],sys_param)
      qM = max_release(s_[i],sys_param)
      r_[i+1] = np.minimum( qM , np.maximum( qm , u ) )
      s_[i+1] = s_[i] + delta*( q - r_[i+1] )    
    
    s1 = s_[HH]; #Final storage
    r1 = np.mean(r_[1:]) #Average release over 24h
    return s1, r1



def construct_rel_matrices(discr,sys_param):
    discr_s = discr['discr_s']
    discr_q = discr['discr_q']
    
    # min release for each value of storage and inflow, assuming the release
    # decision is equal to 0
    vv = np.nan * np.ones(shape = (len(discr_s),len(discr_q)))
    
    for i in range (len(discr_s)):
      for j in range (len(discr_q)):
        _, r1 = massBalance( discr_s[i], 0, discr_q[j], sys_param )
        vv[i,j] = r1
    
    # max release for each value of storage and inflow, assuming the release
    # decision is equal to the maximum
    VV = np.nan * np.ones(shape = (len(discr_s),len(discr_q)))
    for i in range (len(discr_s)):
      for j in range (len(discr_q)):
        _, r1 = massBalance( discr_s[i], discr['discr_u'][-1], discr_q[j],sys_param )
        VV[i,j] = r1
    return vv, VV



def immediate_costs(ht, rt, sys_param):
    # from main_script import sys_param
    # global sys_param;

    hFLO = sys_param['simulation']['hFLO'];
    w = sys_param['simulation']['w'];
    
    g_flo = np.maximum( ht - hFLO, 0 )*100; # exceedance of water level above the flooding threshold
    g_irr = np.maximum( w-rt, 0 ); # deficit between the demand and supply
    
    return g_flo, g_irr


def extractor_ref( idx_U , discr_u , w ):
    u = discr_u[ idx_U ]

    if len( idx_U ) == 1:
        idx_u = idx_U
    else:
        dif = u - w  #difference btw decision and dd
        # print('dif1 in sim =', dif)
        if np.sum( dif >= 0 ):
            dif [dif < 0] = np.inf # penalize decisions that produce deficit
            # print('dif2 in sim =', dif)
        idx  = np.argmin( np.absolute( dif ) ) # find decision closest to the water demand
        u = u[ idx ]           
        idx_u = idx_U[idx]
        # print('idx_u in sim =', idx_u)
    # print('length of u=', len( u ), 'u=',u)  
    # print('dif3 in sim =', dif)
    return u#, idx_u


def interp_lin_scalar( X , Y , x ):
    # % extreme cases
    if x <= X[ 1 ]:
    	y = Y[ 1 ]
    elif x >= X[-1]:
    	y = Y[-1]
    
    # % otherwise
    
    # % Find index 'k' of subinterval [ X(k) , X(k+1) ] s.t. X(k) <= x < X(k+1)
    i = np.argmin( np.absolute( X - x ) ) ;
    
    # % If X( i ) = x     then   y = Y( i ) :
    if X[i] == x:
    	y = Y[i]
    
    # % Else :
    # % if X( i ) < x     then   k = i  
    # % if X( i ) > x     then   k = i - 1
    k = i - ( X( i ) > x ) ;       
    # % Line joining points ( X(k) , Y(k) ) and ( X(k+1) , Y(k+1) ) 
    Dy = Y( k + 1 ) - Y( k ) ;
    Dx = X( k + 1 ) - X( k ) ;
    m  = Dy / Dx             ; # slope
    # % Interpolate :
    y = Y( k ) +  m * ( x - X( k ) ) ;
    
    return y

# for i in dir(sim): print (i)