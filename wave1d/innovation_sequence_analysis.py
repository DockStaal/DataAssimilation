import numpy as np
from wave1d import *
import matplotlib.pyplot as plt

innovation_seq = []
traces = []
diff = []
for k in range(100):
    print('\n --run %d--\n'%k)
    ino, tr = simulate()
    innovation_seq.append(ino)
    traces.append(tr)
innovation_seq = np.asarray(innovation_seq)
traces = np.asarray(traces)
plt.figure('innovation_sequence')

innovation_seq = innovation_seq[:,:,:,0]
plt.plot(np.average(innovation_seq,axis=0))
plt.show()

#testing covariance trace
for l in range(innovation_seq.shape[1]):
    a = innovation_seq[:, l, :]
    t_observed = np.trace(np.cov(a.transpose()))
    difference = t_observed - tr[l]
    diff.append(difference)
    print('trace_difference',difference)
plt.figure('difference')
plt.plot(diff)
plt.show()

print('success')
