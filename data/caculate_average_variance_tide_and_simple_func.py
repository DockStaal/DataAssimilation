import numpy as np

import dateutil.parser
import datetime

minutes_to_seconds=60.
hours_to_seconds=60.*60.
days_to_seconds=24.*60.*60.

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

series1 = 2.5 * np.sin(2.0*np.pi/(12.*hours_to_seconds)*t)

def read_series(filename):
    infile=open(filename,'r')
    times=[]
    values=[]
    for line in infile:
        #print(">%s<%d"%(line,len(line)))
        if line.startswith("#") or len(line)<=1:
            continue
        parts=line.split()
        times.append(dateutil.parser.parse(parts[0]))
        values.append(float(parts[1]))
    infile.close()
    return (times,values)

def shifted_data_variance(data):
    if len(data) < 2:
        return 0.0
    K = data[0]
    n = Ex = Ex2 = 0.0
    for x in data:
        n = n + 1
        Ex += x - K
        Ex2 += (x - K) * (x - K)
    variance = (Ex2 - (Ex * Ex) / n) / (n - 1)
    # use n instead of (n-1) if want to compute the exact variance of the given data
    # use (n-1) if data are samples of a larger population
    return variance

time, series2 = read_series('tide_cadzand.txt')

for j in range(len(series1)):
    series1[j] = series1[j] - series2[j]

print(shifted_data_variance(series1))
