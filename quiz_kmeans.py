#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import math

datapoints = np.array([
            [-1.88, 2.05, -1],
            [-0.71, 0.42, -1],
            [2.41, -0.67, -1],
            [1.85, -3.8, -1],
            [-3.69, -1.33, -1]
        ])

centers = np.array([
            [2., 2., 0.],
            [-2.,-2., 1.]
        ])

def myplot(title):
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)

    for i,d in enumerate(datapoints):
        if d[2] == -1:
            plt.plot(d[0], d[1], 'bo', ms=5)
        if d[2] == 0:
            plt.plot(d[0], d[1], 'ro', ms=5)
        if d[2] == 1:
            plt.plot(d[0], d[1], 'go', ms=5)
        ax.annotate('P%d' % (i+1), xy=(d[0],d[1]), xytext=(0,10), textcoords='offset points')

    plt.plot(centers[0,0], centers[0,1], 'ro')
    plt.plot(centers[1,0], centers[1,1], 'go')

    plt.xlim((np.min(datapoints[:,0])-1, np.max(datapoints[:,0])+1))
    plt.ylim((np.min(datapoints[:,1])-1, np.max(datapoints[:,1])+1))
    plt.savefig("%s.png" % title)
    #plt.show()

def dist(p,lpc):
    ret = -1
    mind = 100000
    for i,c in enumerate(lpc):
        d = math.sqrt((c[0]-p[0])**2 + (c[1]-p[1])**2)
        if d < mind:
            mind = d
            ret = i
    return ret

def assignment_step():
    for i,d in enumerate(datapoints):
        d[2] = dist(d,centers)
        print "P%d: c%d" % (i,d[2])

def center_move():
    for i,c in enumerate(centers):
        point = np.array([0., 0., -1.])
        cnt = 0.
        for d in datapoints:
            if d[2] == i:
                cnt += 1.
                point += d
        print "was: ",c," - is now: ", point / cnt
        c[0:2] = (point / cnt)[0:2]

myplot("initial")
for j in range(10):
    print "------------------------- Step %d" % (j)
    assignment_step()
    myplot("assignment %d" % j)
    center_move()
    myplot("center_move %d" % j)
