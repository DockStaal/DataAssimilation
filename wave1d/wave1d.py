#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
# 
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 0 1  2 3  4 5  6  7   # index in state vector
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
#= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
#= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

"""
    I am pretty sure they are just plotting the whole estuary. That is 100.000 m.
    In the end plots black is observed and blue is calculated.
"""

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn import ensemble
import timeseries
import dateutil.parser
import datetime

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.
path_to_figs = Path("../figures")
path_to_data = Path("../data")

def settings():
    s=dict() #hashmap to  use s['g'] as s.g in matlab
    # Constants
    s['g']=9.81 # acceleration of gravity
    s['D']=20.0 # Depth
    s['f']=1/(0.06*days_to_seconds) # damping time scale
    L=100.e3 # length of the estuary
    s['L']=L
    n=100 #number of cells
    s['n']=n    

    # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
    #      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx=L/(n+0.5)
    s['dx']=dx
    x_h = np.linspace(0,L-dx,n)
    s['x_h'] = x_h
    s['x_u'] = x_h+0.5    
    # initial condition
    #TODO: Adjust intitial condition for ensemble, needs to be random I think.
    s['h_0'] = np.zeros(n)
    s['u_0'] = np.zeros(n)    
    # time
    t_f=2.*days_to_seconds #end of simulation
    #t_f =   t_f * 100
    dt=10.*minutes_to_seconds
    s['dt']=dt
    reftime=dateutil.parser.parse("201312050000") #times in secs relative
    s['reftime']=reftime
    t=dt*np.arange(np.round(t_f/dt))
    s['t']=t
    #boundary (western water level)
    #1) simple function
    #s['h_left'] = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)
    #2) read from file
    (bound_times,bound_values)=timeseries.read_series(path_to_data / 'tide_cadzand.txt')
    bound_t=np.zeros(len(bound_times))
    for i in np.arange(len(bound_times)):
        bound_t[i]=(bound_times[i]-reftime).total_seconds()

    s['h_left'] = np.interp(t,bound_t,bound_values)
    T = 6 * 60 * 60
    s['alpha'] = np.exp(-dt/T)

    s['sigma_w'] = 0.2 * np.sqrt((1 - s['alpha']**2))
    s['ensemble_noise'] = s['sigma_w']
    s['ilocs_size'] = 5
    #[0.19947064 0.19788146 0.20207744 0.20950166 0.21756491] obtained with 5000 ensemble members, t_f = t_f * 20
    s['observation_cov'] = np.diag(np.array([0.19947064, 0.19788146, 0.20207744, 0.20950166, 0.21756491])**2)
    #s['observation_cov'] = np.eye(s['ilocs_size']) * 0.2**2
    s['N'] = 0

    #Enable Twin experiment
    #Adds an extra realization to the enseble, that will not be run through the kalman filter.
    s['ensemble_size'] = 50
    s['enable_state_plot'] = False
    s['enable_kalman'] = True
    s['enable_twin'] = True
    s['estimate_RMSE_bias']  = True
    s['ensemble_size'] = s['ensemble_size'] + s['enable_twin'] * 1
    s['h_left_zero'] = False
    #deactivate boundary noise
    s['zero_boundary_noise'] = False
    s['sigma_w'] = s['sigma_w'] * (1 - s['zero_boundary_noise'])

    s['h_left'] = s['h_left'] * (1 - s['h_left_zero'])
    if s['h_left_zero']:
        print('Bondary forcing around zero')
    s['ensemble_spread_determination'] = True
    s['spread_vs_kalman_var'] = False
    s['plot_kalman_gain'] = False

    #Cut off series from some index forward (including this index)
    s['cut_series'] = True
    s['series_cut_off_index'] = 1
                                           #hence, need to to change it in exercise 9!! 
    return s

def initialize(settings): #return (h,u,t) at initial time
    #compute initial fields and cache some things for speed
    h_0=settings['h_0']
    u_0=settings['u_0']
    n=settings['n']
    ensemble_size = settings['ensemble_size']

    x=np.zeros((2*n,ensemble_size)) #order h[0],u[0],...h[n],u[n]

    if settings['enable_twin'] == True:
        print('Twin experiment enabled.')
        print('Twin index is 0, not taken into account for Kalman filter')

    #NOTE: There was a mistake here. Was ordered differently!!!!
    #TODO: Add random element to initial condition of ensemble members.
    for i in range(ensemble_size):
        x[0::2,i]=h_0[:]
        x[1::2,i]=u_0[:]

    #Setting initial boundary conditions
    #x[0,:] +=  np.random.normal(0,settings['ensemble_noise'],size=(ensemble_size))

    #time
    t=settings['t']
    reftime=settings['reftime']
    dt=settings['dt']
    times=[]
    second=datetime.timedelta(seconds=1)
    for i in np.arange(len(t)):
        times.append(reftime+i*int(dt)*second)
    settings['times']=times
    #initialize coefficients
    # create matrices in form A*x_new=B*x+alpha 
    # A and B are tri-diagonal sparse matrices 
    Adata=np.zeros((3,2*n)) #order h[0],u[0],...h[n],u[n]  
    Bdata=np.zeros((3,2*n))
    #left boundary
    Adata[1,0]=1.
    #right boundary
    Adata[1,2*n-1]=1.
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
    #= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g=settings['g'];dx=settings['dx'];f=settings['f']
    temp1=0.5*g*dt/dx
    temp2=0.5*f*dt
    for i in np.arange(1,2*n-1,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0 + temp2
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0 - temp2
        Bdata[2,i+1]= -temp1
    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
    #= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D=settings['D']
    temp1=0.5*D*dt/dx
    for i in np.arange(2,2*n,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0
        Bdata[2,i+1]= -temp1    
    # build sparse matrix
    A=spdiags(Adata,np.array([-1,0,1]),2*n,2*n)
    B=spdiags(Bdata,np.array([-1,0,1]),2*n,2*n)
    A=A.tocsr()
    B=B.tocsr()
    settings['A']=A #cache for later use
    settings['B']=B

    return (x, t[0])

def timestep(x,i,settings): #return (h,u) one timestep later
    # take one timestep
    temp=x.copy() 
    A=settings['A']
    B=settings['B']
    sigma_w=settings['sigma_w']
    rhs=B.dot(temp) #B*x
    rhs[0,:]=settings['h_left'][i] #left boundary
    #apply forcing
    if sigma_w > 0:
        N_old = settings['N']
        N_new = N_old * settings['alpha']  + np.random.randn(1,np.shape(temp)[1]) * sigma_w
        if settings['enable_twin'] == True:
            rhs[0, 1:] = rhs[0, 1:] + N_new[:,1:]
        else:
            rhs[0,:]= rhs[0,:]  + N_new
        settings['N'] = N_new
    newx=spsolve(A,rhs)
    return newx.reshape((x.shape))

def plot_state(fig,x,i,s,stat_curves = None):
    #plot all waterlevels and velocities at one time
    #stat_curves: Contains mean and 2sigma bound of the ensemble (lower,mean,upper)
    fig.clear()

    xh=s['x_h']
    ax1=fig.add_subplot(211)
    #ax1.set_ylim(-4,4)
    for i in range(x.shape[1]):
        ax1.plot(xh,x[0::2,i],'k-',alpha=0.01)


    ax1.set_ylabel('h')
    #PLotting measurement lines
    xposition = xh[(s['ilocs']/2)[:5].astype(int)]

    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')

    xu=s['x_u']
    ax2=fig.add_subplot(212)
    for i in range(x.shape[1]):
        ax2.plot(xu,x[1::2,i])
    ax2.set_ylabel('u')

    if stat_curves is not None:
        ax1.plot(xh, stat_curves[0][0::2],'b--')
        ax1.plot(xh, stat_curves[1][0::2], 'g--')
        ax1.plot(xh, stat_curves[2][0::2], 'k--')
        ax1.plot(xh, stat_curves[3][0::2],'b--')
        ax1.plot(xh, stat_curves[4][0::2], 'g--')


        ax2.plot(xh, stat_curves[0][1::2],'b--')
        ax2.plot(xh, stat_curves[1][1::2], 'b--')
        ax2.plot(xh, stat_curves[2][1::2], 'k--')
        ax2.plot(xh, stat_curves[3][1::2],'g--')
        ax2.plot(xh, stat_curves[4][1::2], 'g--')

        if s['enable_twin'] == True:
            ax1.plot(xh, stat_curves[-1][0::2], 'r:')


    figname = "fig_map_%3.3d.png"%i
    plt.savefig(path_to_figs / figname)



    plt.draw()
    plt.pause(0.5)

def plot_K_gain(fig,K):
    fig.clear()
    plt.plot(K)
    plt.draw()
    plt.pause(0.5)

def plot_series(t,series_data,s,obs_data,stat_curves = None):
    # plot timeseries from model and observations
    loc_names=s['loc_names']
    nseries=len(loc_names)
    for i in range(nseries):
        fig,ax=plt.subplots()
        for j in range(series_data.shape[1]):#(s['ensemble_size']):
            ax.plot(t,series_data[i,j,:],linewidth=0.5,alpha=0.8) #blue is calculated
        if stat_curves is not None:
            ax.plot(t, stat_curves[0][i], 'r--')
            ax.plot(t, stat_curves[1][i], 'sr--')
            ax.plot(t, stat_curves[2][i], 'r--')

        ax.set_title(loc_names[i])
        ax.set_xlabel('time')
        ntimes=min(len(t),obs_data.shape[1])
        mean = np.average(series_data[i,:,:],axis = 0)
        #ax.plot(t, mean,'s:')
        ax.plot(t[0:ntimes],obs_data[i,0:ntimes],'k-') #black is observed
        figname = ("%s.png"%loc_names[i]).replace(' ','_')
        plt.savefig(path_to_figs / figname)

    
def simulate(identical_twin = False):
    # for plots
    plt.close('all')

    #fig2,ax2 = plt.subplots() #maps: all state vars at one time
    # locations of observations
    s=settings()

    if s['enable_state_plot']:
        fig1,ax1 = plt.subplots()

    L=s['L']
    dx=s['dx']
    xlocs_waterlevel=np.array([0.0*L,0.25*L,0.5*L,0.75*L,0.99*L])
    xlocs_velocity=np.array([0.0*L,0.25*L,0.5*L,0.75*L])
    ilocs=np.hstack((np.round((xlocs_waterlevel)/dx)*2,np.round((xlocs_velocity-0.5*dx)/dx)*2+1)).astype(int) #indices of waterlevel locations in x
    print(ilocs)
    loc_names=[]
    names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
    for i in range(len(xlocs_waterlevel)):
        loc_names.append('Waterlevel at x=%f km %s'%(0.001*xlocs_waterlevel[i],names[i]))
    for i in range(len(xlocs_velocity)):
        loc_names.append('Velocity at x=%f km %s'%(0.001*xlocs_velocity[i],names[i]))
    s['xlocs_waterlevel']=xlocs_waterlevel
    s['xlocs_velocity']=xlocs_velocity
    s['ilocs']=ilocs
    s['loc_names']=loc_names

    #Initializing
    (x,t0)=initialize(s)

    # Generating H-matrix for observations
    H = np.zeros((s['ilocs_size'],s['n']*2))
    H[(np.arange(5),ilocs[:5])] = 1

    s['H'] = H

    if s['enable_twin'] == True:
        #0 index is Twin
        x_twin = x[:,:1]
        twin_rmse_list = []
        observed_data_current_time = x_twin[s['ilocs'],:]
        observed_data = observed_data_current_time
    else:
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'tide_cadzand.txt')
        observed_data=np.zeros((len(ilocs),len(obs_times)))
        observed_data[0,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'tide_vlissingen.txt')
        observed_data[1,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'tide_terneuzen.txt')
        observed_data[2,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'tide_hansweert.txt')
        observed_data[3,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'tide_bath.txt')
        observed_data[4,:]=obs_values[:]

    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data=np.zeros((len(ilocs),s['ensemble_size'],len(t)))

    #Note: This is how it is originally, this means the series data is not filled at timestep 0.
    for i in np.arange(1,len(t)):
        print('timestep %d'%i)
        x=timestep(x,i,s)

        if s['enable_twin'] == True:
            x_twin = x[:,0]
            observed_data_current_time = x_twin[s['ilocs']].reshape(9,1)
            observed_data = np.hstack((observed_data,observed_data_current_time))

            x[:,1:], en_mean,en_var = Ensemble_Kalman_Filter(x[:,1:],observed_data_current_time[:5,0].reshape(5,1),s)

            # Kalman bounds
            x_upper_k  = en_mean  + np.sqrt(en_var)
            x_lower_k = en_mean - np.sqrt(en_var)

            #Actual observed bounds
            x_upper_m  = en_mean  + np.std(x[:,1:], axis=1,ddof=1).reshape(x[:,1:].shape[0],1)
            x_lower_m = en_mean - np.std(x[:,1:], axis=1,ddof=1).reshape(x[:,1:].shape[0],1)
            twin_rmse = np.linalg.norm(x_twin - en_mean.flatten())/np.sqrt(x_twin.size)
            twin_rmse_list.append(twin_rmse)


            if s['enable_state_plot']:
                plot_state(fig1, x[:,1:], None, s, stat_curves=(x_lower_k,x_lower_m, en_mean, x_upper_k,x_upper_m,x_twin))  # show spatial plot; nice but slow
        else:

            x, en_mean, en_var = Ensemble_Kalman_Filter(x, observed_data[:5, i].reshape(5, 1), s)
            # Kalman bounds
            x_upper_k  = en_mean  + np.sqrt(en_var)
            x_lower_k = en_mean - np.sqrt(en_var)

            #Actual observed bounds
            x_upper_m  = en_mean  + np.std(x, axis=1,ddof=1)
            x_lower_m = en_mean - np.std(x, axis=1,ddof=1)


            if s['enable_state_plot']:
                plot_state(fig1,x,None,s,stat_curves=(x_lower_k, x_lower_m,en_mean,x_upper_k,x_lower_m)) #show spatial plot; nice but slow
        if s['spread_vs_kalman_var'] and s['enable_kalman']:
            print('Discrepancy spread vs kalman uncertainty:', np.average((np.std(x[:-1,1:], axis=1,ddof=1)**2)[:]/en_var[:-1].flatten()))

        #Bounds:


        series_data[:,:,i]=x[ilocs,:]
    if s['enable_twin'] == True:
        series_data = series_data[:,1:,:]
        twin_rmse_list = np.asarray(twin_rmse)
        twin_rmse_avg = np.average(twin_rmse_list)
        print('State RMSE against twin:',twin_rmse_avg)

    if s['cut_series']:
        series_data = series_data[0:,0:,s['series_cut_off_index']:]
        observed_data = observed_data[0:,s['series_cut_off_index']:]
        t = t[s['series_cut_off_index']:]
        times = times[s['series_cut_off_index']:]

    #Estimate the RMSE and Bias
    if s['estimate_RMSE_bias']  == True:
        estimate_rmse(series_data,observed_data,t,s)

    if s['ensemble_spread_determination']:
        if s['enable_twin']:
            ensemble_series = series_data[:,1:,s['series_cut_off_index'] * s['cut_series']:][:5,:,:]
        else:
            ensemble_series = series_data[:5,:,s['series_cut_off_index'] * s['cut_series']:]
        obs_mean = np.average(ensemble_series,axis=1)
        obs_std = np.std(ensemble_series,axis=1)
        average_std = np.average(obs_std,axis=1)
        print('Ensemble standard deviation around observation locations averaged in time:', average_std)
        plt.figure('Ensemble_spread')
        plt.plot(obs_std.transpose())
        plt.show()




    #print('RMSE:',RMSE[:5])
    #print('Bias:',Bias[:5])
    if s['enable_kalman'] == True:
        s_avg = np.average(series_data,axis = 1)
        s_upper = np.average(series_data,axis = 1) + np.sqrt(en_var[s['ilocs']])
        s_lower = np.average(series_data,axis = 1) - np.sqrt(en_var[s['ilocs']])

        plot_series(times,series_data,s,observed_data,stat_curves=(s_lower,s_avg,s_upper))
    else:
        plot_series(times, series_data, s, observed_data)

def Ensemble_Kalman_Filter(ensemble_predicted,observations,settings):

    observation_cov = settings['observation_cov']
    member_dim, ensemble_size = ensemble_predicted.shape

    if settings['enable_kalman'] == False:
        print('Kalman deactivated')
        return ensemble_predicted, np.average(ensemble_predicted,axis = 1).reshape(member_dim,1), np.zeros(( member_dim, member_dim))


    #Note that ensemble is a (nx,N_esemble) array, thus each member is a col. vector.
    #Observations can be passed as (N_observation,1) column vector

    #Predcition step
    x_pertubed = np.copy(ensemble_predicted)
    m_pertubed = np.average(x_pertubed,axis = 1).reshape(member_dim,1)

    #Note: State is passed as a column vector
    C_pertubed = (x_pertubed - m_pertubed) @ (x_pertubed - m_pertubed).transpose()\
                 / (ensemble_predicted.shape[1] - 1)
    #Analysis step
    S = settings['H'] @ C_pertubed @ settings['H'].transpose() \
        + observation_cov
    #Compute Gain
    K = C_pertubed @ settings['H'].transpose() @ np.linalg.inv(S)

    #Pertube observations
    y_pertubed = observations + np.random.multivariate_normal\
        (np.zeros(settings['ilocs_size']),observation_cov,size=(ensemble_size)).transpose()

    #Update current state estimate in ensemble form
    x_update = (np.eye(member_dim) - K @ settings['H']) @ x_pertubed + K @ y_pertubed
    m_update =  np.average(x_update,axis = 1).reshape(member_dim,1)
    C_update = (x_update - m_update) @ (x_update - m_update).transpose()\
                 / (ensemble_predicted.shape[1] - 1)

    if  settings['plot_kalman_gain']:
        plt.plot(K)
        plt.show()

    return x_update,m_update , np.diag(C_update).reshape(member_dim,1)

def estimate_rmse(series_data, observed_data, t,settings):
    #load observations
    #Calculate RMSE
    #Note: All observered data has same amount of observations
    if settings['ensemble_size'] > 1:
        #Generating average observation data between ensembles.
        series_data = np.average(series_data,axis=1)
    else:
        series_data = series_data.reshape(series_data.shape[0],series_data.shape[2])

    print('Note: Only 5 locations are available in terms of observations, all water level')
    ntimes = min(len(t), observed_data.shape[1])
    RMSE = np.linalg.norm(series_data.reshape(series_data.shape[0],series_data.shape[1]) - observed_data[:,:ntimes],axis = 1) / np.sqrt(ntimes)
    RMSE_Global = np.linalg.norm(series_data.reshape(series_data.shape[0],series_data.shape[1])[:5].flatten() - observed_data[:5,:ntimes].flatten()) / np.sqrt(observed_data.size)
    Bias = np.average(series_data[:5],axis=1).flatten() - np.average(observed_data[:5,:ntimes],axis=1)



    print('Relevant RMSE:',RMSE[:5])
    print('Relevant Global RMSE:',RMSE_Global)
    print(Bias)



#main program
if __name__ == "__main__":
    simulate()
    plt.show()
