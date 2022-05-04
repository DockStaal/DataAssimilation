%script wave1d_simulate
%
% 1d shallow water model
%
% solves
% dh/dt + D du/dx = 0
% du/dt + g dh/dx + f*u = 0
% 
% staggered discretiztation in space and central in time
%
% o -> o -> o -> o ->   # staggering
% L u  h u  h u  h  R   # element
% 1 2  3 4  5 6  7  8   # index in state vector
%
%  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
%= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
%  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
%= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

% for plots
close all
figure(1) %maps: all state vars at one time
% locations of observations
s=wave1d_settings();
L=s.L;
dx=s.dx;
xlocs_waterlevel=[0.0*L,0.25*L,0.5*L,0.75*L,0.99*L];
xlocs_velocity=[0.0*L,0.25*L,0.5*L,0.75*L];
ilocs=[round(xlocs_waterlevel/dx)*2+1,round(xlocs_velocity/dx)*2+2] %indices of waterlevel locations in x
loc_names={};
names={'Cadzand','Vlissingen','Terneuzen','Hansweert','Bath'};
for i=1:length(xlocs_waterlevel),
    loc_names{i}=sprintf('Waterlevel at x=%f km %s',0.001*xlocs_waterlevel(i),names{i});
end;
nn=length(loc_names);
for i=1:length(xlocs_velocity),
    loc_names{i}=sprintf('Velocity at x=%f km %s',0.001*xlocs_velocity(i),names{i});
end;
s.xlocs_waterlevel=xlocs_waterlevel;
s.xlocs_velocity=xlocs_velocity;
s.ilocs=ilocs;
s.loc_names=loc_names;
%%
[x,t0,s]=wave1d_initialize(s);
t=s.t;
times=s.times;
%series_data=np.zeros((len(ilocs),len(t)))
for i=1:length(t),
    fprintf(1,'timestep %d\n',i);
    x=wave1d_timestep(x,i,s);
    %wave1d_plotstate(1,x,i,s) %show spatial plot; nice but slow
    series_data(:,i)=x(ilocs);
end;
    
%load observations
[obs_times,obs_values]=wave1d_read_series('tide_cadzand.txt');
observed_data=zeros(length(ilocs),length(obs_times));
observed_data(1,:)=obs_values(:);
[obs_times,obs_values]=wave1d_read_series('tide_vlissingen.txt');
observed_data(2,:)=obs_values(:);
[obs_times,obs_values]=wave1d_read_series('tide_terneuzen.txt');
observed_data(3,:)=obs_values(:);
[obs_times,obs_values]=wave1d_read_series('tide_hansweert.txt');
observed_data(4,:)=obs_values(:);
[obs_times,obs_values]=wave1d_read_series('tide_bath.txt');
observed_data(5,:)=obs_values(:);

wave1d_plotseries(times,series_data,s,observed_data);
