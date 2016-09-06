#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

datapoints = np.array([
        [-1.58, -2.01],
        [0.91, 3.98],
        [-0.73, 4.],
        [-4.22, 1.16],
        [4.19, -2.02],
        [-.33, 2.15]
        ])

def mor(arr):
    return (np.max(arr)+np.min(arr))/2.

split1 = mor(datapoints[:,0])
split2 = mor(datapoints[datapoints[:,0] < split1][:,1])

print "split1: ", split1
print "split2: ", split2

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(datapoints[:,0], datapoints[:,1], 'o')
for i,p in enumerate(datapoints):
    ax.annotate('P%d' % (i+1), xy=(p[0],p[1]), xytext=(0,10), textcoords='offset points')

plt.plot(-3, 1.5, 'ro')
ax.annotate('QP', xy=(-3,1.5), xytext=(0,10), textcoords='offset points')

plt.axvline(x=split1)
plt.axhline(y=split2, xmin=0, xmax=0.5)

plt.xlim(np.min(datapoints[:,0])-1, np.max(datapoints[:,0])+1)
plt.ylim(np.min(datapoints[:,1])-1, np.max(datapoints[:,1])+1)
plt.show()
