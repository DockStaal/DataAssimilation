import numpy as np
from wave1d import *
import matplotlib.pyplot as plt



averaged_rmse_values = []
full_rmse_values = []
ensemble_sizes = [50,100,200,400,800,2000]

runs = 20

for k in ensemble_sizes:
    local_rms_values = []
    for p in range(runs):
        print('\n --run %d--\n'%k)
        average_rms = simulate(ensemble_size=k)
        local_rms_values.append(average_rms)
    full_rmse_values.append(local_rms_values)
    av_rms = np.average(np.asarray(local_rms_values))
    averaged_rmse_values.append(av_rms)
ensemble_sizes = np.asarray(ensemble_sizes)
averaged_rmse_values = np.asarray(averaged_rmse_values)
full_rmse_values = np.asarray(full_rmse_values)

plt.figure('convergence_results')
plt.plot(ensemble_sizes,averaged_rmse_values)
plt.plot(ensemble_sizes,averaged_rmse_values + averaged_rmse_values[-1]/np.sqrt(ensemble_sizes))
plt.plot(ensemble_sizes,full_rmse_values,alpha=0.2)
plt.show()