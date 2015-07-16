# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Linear Regression

# <headingcell level=2>

# generate data

# <codecell>

import numpy as np
import matplotlib.pyplot as plt

from numpy import random

N = 200
M = 1
noiseSigma = 10
noise = np.random.normal(0,noiseSigma,N)


# true weight vector
w_true = np.array([-1.,2.])


def add_ones(X):
    return np.column_stack((np.ones((X.shape[0], 1)), X))


# generate data
X = np.random.uniform(-4,4,N*M).reshape((N,M))
X = add_ones(X)
y = X.dot(w_true) + noise

print w_true

# normal equations
XtX = np.dot(X.T,X)
Xty = np.dot(X.T,y)
invCov = np.linalg.inv(XtX)
w = np.linalg.solve(XtX,Xty)
print w



fig = plt.figure(figsize=(20,14))
x = X[:,1]
plt.scatter(x,y,c = 'blue',s = 30,alpha =0.3)


xData = np.arange(-4.0,4.0,8/20.0 )
xDataOnes = add_ones(xData)

yPred = xDataOnes.dot(w_true)
plt.plot(xData,yPred, color = "green",linewidth=2.0)
yPred = xDataOnes.dot(w)
plt.plot(xData,yPred, color = "red",label='regression')


# <headingcell level=2>

# linear regression: closed form solutions

# <codecell>

import numpy as np
from numpy import random

N = 10000
M = 5
noiseSigma = 4
noise = np.random.normal(0,noiseSigma,N)

w_true = np.array(range(1,M+2))
def add_ones(X):
    return np.column_stack((np.ones((X.shape[0], 1)), X))


X = np.random.randn(N,M)     #X = np.random.uniform(-4,4,N*M).reshape((N,M))
X = add_ones(X)
y = X.dot(w_true) + noise

print w_true
# normal equations
XtX = np.dot(X.T,X)
Xty = np.dot(X.T,y)
invCov = np.linalg.inv(XtX)
w = np.linalg.solve(XtX,Xty)
print w

# if XtX is NOT invertible use psuedo inverse
invXtX   = np.linalg.pinv(XtX)
w    = dot(invXtX, Xty)
print w

# using SVD
eps     = 1e-10
U, s, V = np.linalg.svd(XtX, full_matrices=True)
r       = (s > eps).nonzero()[0].shape[0]
inv_S   = array(np.diag([1.0/s[i] for i in xrange(r)]))
invXtX  = reduce(dot,[V[:r, :].T, inv_S , U[:, :r].T])
w       = dot(invXtX, Xty)
print w

# <codecell>

dof = N -M
yHat = np.dot(X,w)
residuals = y - yHat
SSE = np.dot(residuals,residuals)
sigma = np.sqrt(SSE/dof)

# coefficient standard errors
se = sigma*np.sqrt(invCov.diagonal())
print sigma
print se

# <headingcell level=2>

# gradient descent example

# <codecell>

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import numpy as np
import pylab as pl

def f(w):
    return w[0]**2 + 2*w[1]**2

def gradient(w):
    return np.array([2*w[0],4*w[1]])

w = np.array([-5,6])
z = f(w)

history = []
history.append([w,z])

eta=0.1
#eta=0.5

ITERS = 10  

for iter in range(ITERS):
    w = w - eta * gradient(w)
    z = f(w)
    history.append([w,z])
    

def build_mesh(lim,inc,f):
    X = np.arange(-lim, lim, inc)
    Y = np.arange(-lim, lim, inc)
    X, Y = np.meshgrid(X, Y)
    rowRange = range(X.shape[0])
    colRange = range(X.shape[1])
    Z= np.zeros((X.shape[0],X.shape[1]))
    for i in rowRange:
        for j in colRange:
            Z[i,j] = f(np.array([X[i,j],Y[i,j]]))    
    return X,Y,Z

X,Y,Z = build_mesh(8,0.25,f)  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xlist = np.array(map(lambda val: val[0][0],history))
ylist = np.array(map(lambda val: val[0][1],history))
zlist = np.array(map(lambda val: val[1],history))

jitter = True
if jitter:
    noise = 0.2
    xlist = xlist + np.random.normal(0, noise, size=len(xlist))
    ylist = ylist + np.random.normal(0, noise, size=len(ylist))
    zlist = zlist + np.random.normal(0, noise, size=len(zlist))

fig = plt.figure(figsize=(20,14))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
        linewidth=0, antialiased=False,cmap=cm.Blues,alpha=0.3)
ax.contour(X,Y,Z,zdir='z',offset=-2)

ax.plot(xlist,ylist,zlist, color='r', marker='o', label='Gradient decent')
#plt.savefig('/home/charles/testbed_course/gd_surface_toobigl.png')

plt.show()

fig = plt.figure(figsize=(20,14))
cs = pl.contour(X,Y,Z,10,zdir='z',offset=-2)
pl.clabel(cs,inline=1,fontsize=10)
pl.plot(xlist, ylist, color='r', marker='o',alpha=0.5)
pl.plot(xlist[0], ylist[0], 'go', alpha=0.5, markersize=10)
pl.text(xlist[0], ylist[0], '  start', va='center')
pl.plot(xlist[-1], ylist[-1], 'ro', alpha=0.5, markersize=10)
pl.text(xlist[-1], ylist[-1], '  stop', va='center')
#plt.savefig('/home/charles/testbed_course/gd_contour_toobig.png')
plt.show()

# <headingcell level=2>

# liear regression: gradient descent solution 

# <codecell>

import numpy as np
from numpy import random

N = 10000
M = 5
noiseSigma = 4
noise = np.random.normal(0,noiseSigma,N)

w_true = np.array(range(1,M+2))
def add_ones(X):
    return np.column_stack((np.ones((X.shape[0], 1)), X))


X = np.random.randn(N,M)     #X = np.random.uniform(-4,4,N*M).reshape((N,M))
X = add_ones(X)
y = X.dot(w_true) + noise

print w_true


def gradient(w):
    return (X.dot(w) - y).dot(X)
    
eta=0.1/float(N)
w = np.zeros(M+1)
ITERS = 100 

# gradient descent
for iter in range(ITERS):
    w =  w - eta * gradient(w)
    
print w

