#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
# 
# staggered discretiztation in space and central in time
#

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

import timeseries
import dateutil.parser
import datetime

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.
path_to_figs = Path("../figures")
path_to_data = Path("../data")

def settings(ensemble_size = None):
    s=dict() #hashmap to  use s['g'] as s.g in matlab
    # Consants
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
    t_f =   t_f
    dt=10.*minutes_to_seconds
    s['dt']=dt
    reftime=dateutil.parser.parse("201312050000") #times in secs relative
    s['reftime']=reftime
    t=dt*np.arange(np.round(t_f/dt))
    s['t']=t
    #boundary (western water level)
    #1) simple function
    s['h_left'] = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)
    #print('NOTE: ALTERNATIVE BOUNDARY SELECTED')
    #2) read from file
    (bound_times,bound_values)=timeseries.read_series(path_to_data / 'tide_cadzand.txt')
    bound_t=np.zeros(len(bound_times))
    for i in np.arange(len(bound_times)):
        bound_t[i]=(bound_times[i]-reftime).total_seconds()

    #s['h_left'] = np.interp(t,bound_t,bound_values)
    T = 6 * 60 * 60
    s['alpha'] = np.exp(-dt/T)

    s['sigma_w'] = 0.9 * np.sqrt((1 - s['alpha']**2))
    s['ensemble_noise'] = s['sigma_w']
    s['ilocs_size'] = 5

    #Observation noise, subject to tuning.
    s['observation_noise'] = 0.2

    #Spread for ensemble initial condition
    s['initial_ensemble_spread'] = 0.2
    s['NP'] = s['sigma_w']**2

    s['boundary_noice'] = 0.63 + 0.29
    s['observation_cov'] = np.eye(s['ilocs_size']) * s['observation_noise']**2
    s['observation_cov'][0, 0] = 0.63 + 0.29
    s['P_0'] = np.eye(s['ilocs_size']) * s['initial_ensemble_spread']**2

    #Analysis lists
    s['innovation_sequence'] = []
    s['trace_comparison'] = []
    s['spread_list'] = []
    s['mean_rms'] = []
    s['mean_rms_spread_list'] = []
    s['mean_spread'] = []
    s['predicted_variance'] = []
    s['N_series'] = []
    s['N_original_series'] = []

    #True data for twin experiment if required
    s['true_data'] = None


    #Enable Twin experiment
    #Adds an extra realization to the enseble, that will not be run through the kalman filter.
    #Standard ensemble size
    if ensemble_size is None:
        s['ensemble_size'] = 500
    else:
        s['ensemble_size'] = ensemble_size
    s['enable_state_plot'] = False
    s['enable_innovation_plot'] = False
    s['innovation_sequence_analysis'] = False

    #Return values and plotting
    s['spread_rms_analysis'] = False
    s['spread_rmse_plot'] = False
    s['convergence_analysis'] = True
    s['collect_full_true_state'] = False
    if s['collect_full_true_state']:
        s['full_true_state'] = []


    #Run settings
    s['enable_kalman'] = True
    s['kalman_cutoff'] = 150 #If none no cutoff otherwise at given timestep
    s['enable_twin'] = False
    #Consistent twin data, obs. noise: 0.2, t_f = t_f
    s['consistent_synthetic_data'] =  s['enable_twin'] * False
    #Estimate RMSE and bias
    s['estimate_RMSE_bias']  = False
    #Ensemble size in case of twin experiment
    s['ensemble_size'] = s['ensemble_size'] + s['enable_twin'] * 1 * (1 - s['consistent_synthetic_data'])
    #Setting left boundary value to constant zero
    s['h_left_zero'] = False
    s['h_left'] = s['h_left'] * (1 - s['h_left_zero'])
    if s['h_left_zero']:
        print('Bondary forcing around zero')


    #deactivate boundary noise
    s['zero_boundary_noise'] = False

    s['sigma_w'] = s['sigma_w'] * (1 - s['zero_boundary_noise'])

    #Initial N state
    s['N'] = np.random.normal(0,s['sigma_w'],size =(1,s['ensemble_size']) )


    #Analysis stuff
    s['ensemble_spread_determination'] = False
    s['spread_vs_kalman_var'] = False
    s['plot_kalman_gain'] = False


    s['adapted_ic'] = True
    s['extended_state'] = True


    #Cut off series from some index forward (including this index)
    s['cut_series'] = True
    s['series_cut_off_index'] = 1

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
    if settings['enable_twin']==True and settings['adapted_ic']==True :
        x[:,1:] +=  np.random.normal(0,0.2,size=(x[:,1:].shape))

    if settings['enable_twin']==False and settings['adapted_ic']==True :
        x[1:,:] +=  np.random.normal(0,0.2,size=(x[1:2*settings['n'],:].shape))


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
        N_old = settings['N'].copy()
        N_new = N_old * settings['alpha'] #np.random.randn(1,np.shape(temp)[1]) * sigma_w
        N_new[:, :] += np.random.normal(0, scale=sigma_w, size=(1, temp.shape[1]))

        if settings['enable_twin'] == True:
            rhs[0, :] = rhs[0, :] + N_old
        else:
            rhs[0,:]= rhs[0,:]  + N_old
        settings['N'] = N_new.copy()
        settings['N_original_series'].append(N_new)
        #Avoiding changing linear system entirely
        newx = spsolve(A,rhs)
        if settings['extended_state'] == True:
            # Returning extended state for kalman filtering
            newx = np.vstack((newx, N_new.copy()))

        return  newx
    if sigma_w == 0 :
        newx = spsolve(A,rhs)



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
    for j in range(x.shape[1]):
        ax2.plot(xu,x[1::2,j])
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
    plt.pause(0.01)

def plot_K_gain(fig,K):
    fig.clear()
    plt.plot(K)
    plt.draw()
    plt.pause(0.5)

def plot_series(t,series_data,s,obs_data,stat_curves = None,true_data = None, plot_ensemble=False):
    # plot timeseries from model and observations
    loc_names=s['loc_names'] # contains the names of the titels of the last nine plots
    nseries=len(loc_names)
    print("kalman_cutoff is set to: ", s['kalman_cutoff'])
    for i in range(nseries):
        fig,ax=plt.subplots()
        if plot_ensemble == True:
            for j in range(series_data.shape[1]):#(s['ensemble_size']):
                ax.plot(t,series_data[i,j,:],linewidth=0.5,alpha=0.8) #blue is calculated
        if stat_curves is not None:
            ax.plot(t, stat_curves[0][i], 'r--')
            ax.plot(t, stat_curves[1][i], 'sr--')
            ax.plot(t, stat_curves[2][i], 'r--')



        ax.set_title(loc_names[i])
        ax.set_xlabel('time')
        ntimes=min(len(t),obs_data.shape[1])

        if true_data is not None:
                ax.plot(t[0:ntimes],true_data[i,0:ntimes],'k:',alpha=1.0)

        mean = np.average(series_data[i,:,:],axis = 0)
        #ax.plot(t, mean,'s:')
        if i is not 0:
            ax.plot(t[0:ntimes],obs_data[i,0:ntimes],'k-') #black is observed
        ax.plot(t[0:ntimes],mean,'b-')

        #RMSE calc :)
        if s['kalman_cutoff'] is None:
            RMSE = np.sum(np.linalg.norm(mean - obs_data[i,:ntimes]))/ np.sqrt(mean.size) 
        else:
            RMSE = np.sum(np.linalg.norm(mean[s['kalman_cutoff']:ntimes] - obs_data[i,s['kalman_cutoff']:ntimes]))/ np.sqrt(mean.size) 
        print("RMSE for %s is %f"%(loc_names[i],RMSE))
        figname = ("%s.png"%loc_names[i]).replace(' ','_')
        plt.savefig(path_to_figs / figname)

"""
    if s['ensemble_spread_determination']:
        if s['enable_twin']:
            ensemble_series = series_data[:,1:,0:][:5,:,:]
        else:
            ensemble_series = series_data[:5,:,:]
        obs_mean = np.average(ensemble_series,axis=1)
        obs_std = np.std(ensemble_series,axis=1)
        print(obs_std.shape)
        average_std = np.average(obs_std[:,100:],axis=1)
        print('Ensemble standard deviation around observation locations averaged in time:', average_std)
        names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
        plt.figure('Ensemble_spread')
        plt.plot(obs_std.transpose())
        plt.legend(names)
        plt.savefig(path_to_figs / 'ensemble_spread.png')
        plt.show()
"""
    
def simulate(verbose = False,ensemble_size = None):
    # for plots
    plt.close('all')
    #s=dict(settings())

    s = settings(ensemble_size = ensemble_size)
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

    if s['collect_full_true_state'] and s['enable_twin']:
        ts = x[:, 0]
        ts = np.append(ts, s['N'][0,0])
        s['full_true_state'].append(ts)

    # Generating H-matrix for observations
    H = np.zeros((s['ilocs_size'],s['n']*2 + s['extended_state']))
    H[(np.arange(5),ilocs[:5])] = 1

    s['H'] = H

    if s['enable_twin'] == True:
        #0 index is Twin
        #-1 index of second dimesnion is added state
        twin_rmse_list = []
        true_error_list = []
        spread_list = []
        mean_rms_spread_list = []
        if s['consistent_synthetic_data'] == True:
            full_true_data = np.genfromtxt('true_full_synthetic_0.2.txt')
            true_data = np.genfromtxt('true_synthetic_0.2.txt')

            #pertubed true data synthetically
            observed_data_current_time = true_data[:, i].reshape(9, 1).copy()
            observed_data_current_time[:s['ilocs_size']] += np.random.multivariate_normal(
                np.zeros(s['ilocs_size']),
                s['observation_cov']).reshape(s['ilocs_size'], 1)

            observed_data = np.copy(observed_data_current_time)


        if s['consistent_synthetic_data'] == False:
            x_twin = x[:2*s['n'],:1]

            true_data = x_twin[s['ilocs']]
            observed_data_current_time = np.copy(true_data)

            #Pertube data synthetically, according to observation noise
            observed_data_current_time[:s['ilocs_size']] +=  np.random.multivariate_normal(np.zeros(s['ilocs_size']),s['observation_cov'],1).transpose()
            observed_data = observed_data_current_time

    if s['enable_twin'] == False:
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'tide_cadzand.txt')
        observed_data=np.zeros((len(ilocs),len(obs_times)))
        observed_data[0,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'waterlevel_vlissingen.txt')
        observed_data[1,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'waterlevel_terneuzen.txt')
        observed_data[2,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'waterlevel_hansweert.txt')
        observed_data[3,:]=obs_values[:]
        (obs_times,obs_values)=timeseries.read_series(path_to_data / 'waterlevel_bath.txt')
        observed_data[4,:]=obs_values[:]

    t=s['t'][:] #[:40]
    times=s['times'][:] #[:40]
    series_data=np.zeros((len(ilocs),s['ensemble_size'],len(t)))

    #Note: This is how it is originally, this means the series data is not filled at timestep 0.
    for i in np.arange(1,len(t)):
        if verbose:
            print('timestep %d'%i)

        #Either extended state or not, depending on added noise and setting
        x=timestep(x,i,s)

        if s['collect_full_true_state'] and s['enable_twin']:
            s['full_true_state'].append(x[:, 0])

        #Note: EnKF returns only the physical state, in no case the extended state, it is appended in the timestep.

        if s['enable_twin'] == True:
            if s['consistent_synthetic_data'] == True:
                observed_data_current_time = true_data[:,i ].reshape(9,1).copy()
                observed_data_current_time[:s['ilocs_size'],:] += np.random.multivariate_normal(np.zeros(s['ilocs_size']),
                                                                            s['observation_cov']).reshape(s['ilocs_size'],1)
                observed_data = np.hstack((observed_data, observed_data_current_time))




            if s['consistent_synthetic_data'] == False:
                x_twin = x[:2*s['n'],0].reshape(2*s['n'],1)

                true_data_current_time = x_twin[s['ilocs']]
                true_data = np.hstack((true_data, true_data_current_time))

                # Pertube data synthetically, according to observation noise
                observed_data_current_time = np.copy(true_data_current_time)
                observed_data_current_time[:s['ilocs_size'],:] += np.random.multivariate_normal(np.zeros(s['ilocs_size']),
                                                                            s['observation_cov']).reshape(s['ilocs_size'],1)
                observed_data = np.hstack((observed_data,observed_data_current_time))





            #Note, only actual observations are passed
            x[:2*s['n'],1:], en_mean,en_var = Ensemble_Kalman_Filter(x[:,1:].copy(),observed_data_current_time[:5,0].reshape(5,1),s)
            x = x[:2*s['n'],:]

            # Kalman bounds
            x_upper_k  = en_mean  + np.sqrt(en_var) * 2
            x_lower_k = en_mean - np.sqrt(en_var) * 2

            #Actual observed bounds
            x_upper_m  = en_mean  + np.std(x[:,1:], axis=1,ddof=1).reshape(en_mean.size,1) * 2
            x_lower_m = en_mean - np.std(x[:,1:], axis=1,ddof=1).reshape(en_mean.size,1) * 2

            if s['spread_rms_analysis']:
                if s['consistent_synthetic_data'] == True:
                    x_twin = full_true_data[:,i].reshape(en_mean.shape[0] + 1,1)
                    true_error = (x_twin[:-1,:] - en_mean)
                    N = s['N'].copy().transpose()
                    N_error = x_twin[-1,:] - np.average(N[1:])
                    #Accounting for full state statistics
                    true_error_list.append(np.vstack((true_error,N_error)))


                if s['consistent_synthetic_data'] == False:
                    true_error = (x_twin - en_mean)
                    N = s['N'].copy().transpose()
                    N_error = N[0,:] - np.average(N[1:])
                    #Accounting for full state statistics
                    true_error_list.append(np.vstack((true_error,N_error)))
                    s['N_series'].append(s['N'].copy())

                spread_list.append(np.sqrt(en_var))


            if s['enable_state_plot'] * (1 - s['consistent_synthetic_data']):
                plot_state(fig1, x[:,1:], None, s, stat_curves=(x_lower_k,x_lower_m, en_mean, x_upper_k,x_upper_m,x_twin))  # show spatial plot; nice but slow

        if s['enable_twin'] == False:
            
            if s['kalman_cutoff'] is not None:
                if s['kalman_cutoff'] < i:
                    s['enable_kalman'] = False
                    s['alpha'] = 1
                    s['sigma_w'] = 0.01
            x, en_mean, en_var = Ensemble_Kalman_Filter(x, observed_data[:5, i].reshape(5, 1), s)
            # Kalman bounds
            x_upper_k  = en_mean  + np.sqrt(en_var) * 2
            x_lower_k = en_mean - np.sqrt(en_var) * 2

            #Actual observed bounds
            x_upper_m  = en_mean  + np.std(x[:,1:], axis=1,ddof=1).reshape(x[:,1:].shape[0],1) * 2
            x_lower_m = en_mean - np.std(x[:,1:], axis=1,ddof=1).reshape(x[:,1:].shape[0],1) * 2


            if s['enable_state_plot']:
                plot_state(fig1,x,None,s,stat_curves=(x_lower_k, x_lower_m,en_mean,x_upper_k,x_lower_m)) #show spatial plot; nice but slow

        if s['spread_vs_kalman_var'] and s['enable_kalman']:
            print('Discrepancy spread vs kalman uncertainty:', np.average((np.std(x[:-1,1:], axis=1,ddof=1)**2)[:]/en_var[:-1].flatten()))

        #Bounds:


        series_data[:,:,i]=x[ilocs,:]



    if s['enable_twin'] == True:
        series_data = series_data[:,1:,:]
        if s['collect_full_true_state'] == True:
            s_true = np.asarray(s['full_true_state']).transpose()
        if s['spread_rms_analysis'] == True:
            true_error_list = np.asarray(true_error_list)
            spread_list = np.asarray(spread_list)
            mean_rms_spread_list = np.asarray(s['mean_rms_spread_list']).reshape(len(s['mean_rms_spread_list']),1)

            mean_rms = np.linalg.norm(true_error_list,axis=1)/np.sqrt(true_error_list.shape[1])

            if s['spread_rmse_plot']:
                plt.figure('time_series')
                plt.plot(mean_rms_spread_list[10:])
                plt.plot(mean_rms[10:],alpha=0.1)
                plt.plot((np.ones_like(mean_rms) * np.average(mean_rms[10:]))[10:],'k--')

                rmse = np.linalg.norm(true_error_list, axis=0) / np.sqrt(true_error_list.shape[0])
                avg_spread = np.average(spread_list,axis=0)
                # N_a = np.asarray(s['N_series'])
                # N_o = np.asarray(s['N_original_series'])
                plt.figure('RMSE time')
                plt.plot(rmse[::2])
                plt.plot(avg_spread[::2])
                plt.show()

    if s['cut_series']:
        series_data = series_data[0:,0:,s['series_cut_off_index']:]
        observed_data = observed_data[0:,s['series_cut_off_index']:]
        if s['enable_twin']:
            true_data = true_data[:,s['series_cut_off_index']:]
        t = t[s['series_cut_off_index']:]
        times = times[s['series_cut_off_index']:]

    if s['enable_twin']:
        s['true_data'] = true_data

    #Estimate the RMSE and Bias
    if s['estimate_RMSE_bias']  == True:
        RMSE_loc = estimate_rmse(series_data,observed_data,t,s)


    if s['ensemble_spread_determination']:
        if s['enable_twin']:
            ensemble_series = series_data[:,1:,s['series_cut_off_index'] * s['cut_series']:][:5,:,:]
        else:
            ensemble_series = series_data[:5,:,s['series_cut_off_index'] * s['cut_series']:]
        obs_mean = np.average(ensemble_series,axis=1)
        obs_std = np.std(ensemble_series,axis=1,ddof=1)
        average_std = np.average(obs_std,axis=1)
        if s['estimate_RMSE_bias'] == True:
           print("RMSE_sum, spread: %lf %lf"%(np.sum(RMSE_loc[:5]),np.sum(average_std)))


        print('Ensemble standard deviation around observation locations averaged in time:', average_std)
        names=['Cadzand','Vlissingen','Terneuzen','Hansweert','Bath']
        plt.figure('Ensemble_spread')
        plt.plot(obs_std.transpose())
        plt.legend(names)
        plt.savefig(path_to_figs / 'ensemble_spread.png')
        plt.show()


    if s['enable_innovation_plot']:
        s['innovation_sequence'] = np.asarray(s['innovation_sequence'])
        s['trace_comparison'] = np.asarray(s['trace_comparison'])
        for k  in range(s['trace_comparison'].size):
            trace_estimated = np.trace(s['innovation_sequence'][k])
            diff = trace_estimated - s['trace_comparison'][k]
            print('trace difference',diff)

        print('success')

        #q = s['innovation_sequence']
        #q_a = np.average(q, axis=2)
        #print('average',np.average(q_a,axis=0))
        #plt.figure('Innovation sequence')
        #plt.plot(q_a,alpha=0.4)
        #plt.plot(np.average(q_a, axis=0).reshape(1, 5) * np.ones_like(q_a),'k--')


        #plt.plot(np.average(q_a,axis=0))
        #sequence = q_a.transpose() @ q_a * 1 / (q.shape[0] - 1)
        #t_1 = s['H'] @ s['P_f'] @ s['H'].transpose()
        #t_2 = s['observation_cov']
        #print('Trace comparison', np.trace(sequence) - np.trace(t_1) - np.trace(t_2))
        #plt.figure('Trace')
        #plt.plot(s['trace_comparison'])
        #plt.show()
        #print(np.average(np.average(s['innovation_sequence'], axis=0), axis=1))
        #plt.show()

    #print('RMSE:',RMSE[:5])
    #print('Bias:',Bias[:5])
    if s['enable_kalman'] == True:
        s_avg = np.average(series_data,axis = 1)
        s_upper = s_avg + np.std(series_data,axis=1,ddof=1) * 2
        s_lower =  s_avg - np.std(series_data,axis=1,ddof=1) * 2
        if s['enable_twin']:
            plot_series(times, series_data, s, observed_data, true_data = true_data,stat_curves=(s_lower, s_avg, s_upper),
                       plot_ensemble=False)
        else:
            plot_series(times,series_data,s,observed_data,stat_curves=(s_lower,s_avg,s_upper),plot_ensemble=False)
    if s['enable_kalman'] == False:
        plot_series(times, series_data, s, observed_data,plot_ensemble=True)
    if s['innovation_sequence_analysis'] == True:
        return np.asarray(s['innovation_sequence']), np.asarray(s['trace_comparison'])
    if  s['spread_rms_analysis'] and not s['convergence_analysis'] :
        return mean_rms_spread_list, mean_rms
    if s['convergence_analysis']:
        return estimate_rmse(series_data,observed_data,t,s)#np.average(np.average(np.asarray(s['predicted_variance']),axis=1))
    #if s['convergenve_analysis']:


def Ensemble_Kalman_Filter(ensemble_predicted,observations,settings):

    observation_cov = settings['observation_cov']
    member_dim, ensemble_size = ensemble_predicted.shape

    if settings['enable_kalman'] == False:
        print('Kalman deactivated')
        if settings['extended_state'] == True:

            avg = np.average(ensemble_predicted,axis = 1).reshape(member_dim ,1)
            avg = avg[:-1,:]
            ensemble_predicted = ensemble_predicted[:-1,:]
            C_return = np.zeros(( member_dim - 1,1))


        return ensemble_predicted, avg, C_return


    #Note that ensemble is a (nx,N_esemble) array, thus each member is a col. vector.
    #Observations can be passed as (N_observation,1) column vector

    #Predcition step
    x_pertubed = np.copy(ensemble_predicted)
    #x_pertubed[-1:] += np.random.normal(0,scale = settings['sigma_w'],size= (x_pertubed[-1:].shape)) * 10

    #x_pertubed[0:2*settings['n'],:] = x_pertubed[0:2*settings['n'],:] + np.random.normal(0,settings['sigma_w'],size = (ensemble_size,1))
    m_pertubed = np.average(x_pertubed,axis = 1).reshape(member_dim,1)

    #Note: State is passed as a column vector
    C_pertubed = (x_pertubed - m_pertubed) @ (x_pertubed - m_pertubed).transpose()\
                 / (ensemble_predicted.shape[1] - 1)
    #Analysis step
    S = settings['H'] @ C_pertubed @ settings['H'].transpose() \
        + observation_cov

    #Compute Gain
    K = C_pertubed @ settings['H'].transpose() @ np.linalg.inv(S)

    #Lecture formulation
    d = observations - settings['H'] @ x_pertubed

    #Adding Innovation
    settings['innovation_sequence'].append(observations - settings['H'] @ m_pertubed)
    settings['P_f'] = C_pertubed

    x_update = x_pertubed + K @ (d - np.random.multivariate_normal(np.zeros(settings['ilocs_size']),
                                                                   settings['observation_cov'],ensemble_size).transpose())
    #print(np.max(np.abs(settings['N'][:,1:].flatten() - x_update[-1])))
    m_update =  np.average(x_update,axis = 1).reshape(member_dim,1)

    C_update = (x_update - m_update) @ (x_update - m_update).transpose()\
                 / (ensemble_predicted.shape[1] - 1)

    if settings['spread_rms_analysis'] == True:
        mean_rms_spread = np.linalg.norm(np.linalg.norm(m_update - x_update, axis=0)) / (
            np.sqrt((m_update.size) * (ensemble_size - 1)))
        #mean_rmse_spread = np.linalg.norm(np.std())
        settings['mean_rms_spread_list'].append(mean_rms_spread.copy())
        # mean_rms_spread_list = n
    if settings['convergence_analysis']:
        settings['predicted_variance'].append(np.diag(C_update))

    if settings['innovation_sequence_analysis'] == True:
        settings['trace_comparison'].append(np.trace(S))


    if  settings['plot_kalman_gain']:
        plt.plot(K[::2])
        plt.show()
    if settings['extended_state']:
        #Returning only physical states
        settings['N'][:,settings['enable_twin']:] = (x_update[-1,:].reshape(1,ensemble_size)).copy()
        return x_update[:-1,:], m_update[:-1,:], np.diag(C_update)[:-1].reshape(member_dim - 1, 1)

    if settings['extended_state'] == False:
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
    if settings['enable_twin']:
        true_data = settings['true_data']
        RMSE_true = np.linalg.norm(series_data.reshape(series_data.shape[0],series_data.shape[1]) - true_data[:,:ntimes],axis = 1) / np.sqrt(ntimes)
        print('Relevant True against twin RMSE',RMSE_true)


    RMSE_Global = np.linalg.norm(series_data.reshape(series_data.shape[0],series_data.shape[1])[:5].flatten() - observed_data[:5,:ntimes].flatten()) / np.sqrt(observed_data.size)
    Bias = np.average(series_data[:5],axis=1).flatten() - np.average(observed_data[:5,:ntimes],axis=1).flatten()



    print('Relevant RMSE:',RMSE[:5])
    print('Relevant Global RMSE:',RMSE_Global)
    print(Bias)

    if settings['enable_twin'] == True:
        RMSE = RMSE_true
    return RMSE_Global



#main program
if __name__ == "__main__":
    simulate(verbose=True,ensemble_size= None) 
    plt.show()
