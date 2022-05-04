function settings=wave1d_settings();
%function settings=wave1d_settings();
    % settings is a structure in matlab, use for example s.g
    % Constants
    minutes_to_seconds=60.;
    hours_to_seconds=60.*60.;
    days_to_seconds=24.*60.*60.;

    settings.g=9.81; % acceleration of gravity
    settings.D=20.0; % Depth
    settings.f=1/(0.06*days_to_seconds); % damping time scale
    L=100.e3; % length of the estuary
    settings.L=L;
    n=100; %number of cells
    settings.n=n;    
    % Grid(staggered water levels at index 1 x=0 (boundary) dx 2dx ... (n-1)dx
    %      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx=L/(n+0.5);
    settings.dx=dx;
    x_h = linspace(0,L-dx,n);
    settings.x_h = x_h;
    settings.x_u = x_h+0.5;
    % initial condition
    settings.h_0 = zeros(n,1);
    settings.u_0 = zeros(n,1);   
    % time
    t_f=2.*days_to_seconds; %end of simulation
    dt=10.*minutes_to_seconds;
    settings.dt=dt;
    reftime=datenum('201312050000','yyyymmddhhMM'); %times in secs relative to this.
    settings.reftime=reftime;
    t=dt:dt:t_f;
    settings.t=t;
    %boundary (western water level)
    %1) simple function
    %settings.h_left = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t);
    %2) read from file
    [bound_times,bound_values]=wave1d_read_series('tide_cadzand.txt');
    bound_t=(bound_times-reftime)*days_to_seconds;
    settings.h_left = interp1(bound_t,bound_values,t);        
end
