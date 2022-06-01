import numpy as np
from wave1d import *
import matplotlib.pyplot as plt

innovation_seq = []
traces = []
diff = []
mean_rmse_spread_list = 0.0
mean_rms = 0.0
runs = 10
for k in range(runs):
    print('\n --run %d--\n'%k)
    mean_rms_spread_new, mean_rms_new = simulate()
    mean_rms +=mean_rms_new
    mean_rmse_spread_list += mean_rms_spread_new

mean_rms = mean_rms / runs
mean_rmse_spread_list = mean_rmse_spread_list / runs
plt.figure('rms')
plt.plot(mean_rmse_spread_list)
plt.plot(mean_rms,alpha=0.2)
plt.plot(np.ones_like(mean_rms) * np.average(mean_rms[100:]))
plt.show()
