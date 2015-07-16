# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Logistic Regression

# <headingcell level=2>

# generate data

# <codecell>

import numpy as np
import pylab as pl

def add_ones(X):
    return np.column_stack((np.ones((X.shape[0], 1)), X))

def logistic(t):
    """ logistic function - returns 1 / (1 + exp(-t))
    
    Using formula to calculate logistic in a stable fashion see
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    
    idx = t > 0
    out = np.empty(t.size, dtype=np.float)
    out[idx] = 1. / (1 + np.exp(-t[idx]))
    exp_t = np.exp(t[~idx])
    out[~idx] = exp_t / (1. + exp_t)
    return out

# generate X
N = 600
M = 2
X = np.random.randn(N,M)
X = add_ones(X)

w_true = np.array([0.0,2.0,1.4])
probs =  logistic(X.dot(w_true))
draws =  np.random.uniform(0,1,N)

# generate labels
y = np.array([1.0 if draw < prob else 0.0 for (draw,prob) in zip(draws,probs)])


# split in to train and test
r = round(0.66 * N)
X_train = X[0:r,:]
X_test  = X[r:-1,:]

y_train = y[0:r]
y_test = y[r:-1]

from matplotlib.colors import ListedColormap
marker = ListedColormap(['#FF0000', '#0000FF'])
pylab.scatter(X_train[:, 1], X_train[:, 2], c=y_train, cmap=marker, s = 15, alpha = 0.7)

# <headingcell level=2>

# simplified logistic regression

# <codecell>

import numpy as np
import pylab as pl

def logreg_cost(w,X,y,C=0.1):
    """ logistic regresion cost/loss function
    
    Using np.logaddexp for stability
    includes l2 regularization
    
    Arguments:
        w - modal weights
        X - data matrix
        y - vector of labels 0/1
        C - l2 regularaization hyperparameter
        
    Returns:
        scaler cost/loss   
    """    
    norm = C*0.5*(w.dot(w))
    Xw =X.dot(w)
    nll =  ( y.dot(np.logaddexp(0,-Xw))  + (1.0-y).dot(np.logaddexp(0,Xw)))
    return nll + norm

def logreg_gradient(w,X,y,C=0.1):
    """ logistic regression gradient function
        
    Arguments:
        w - modal weights
        X - data matrix
        y - vector of labels
        C - l2 regularaization hyperparameter
        
    Returns:
        gradient vector
    """        
    p   = logistic(X.dot(w))
    Xt  = X.T
    res = Xt.dot(p - y)
    return res + C*w

C = 1.0
cost     = lambda w: logreg_cost(w,X_train,y_train,C)
gradient = lambda w: logreg_gradient(w,X_train, y_train ,C)    

N, M   = X_train.shape
w     = np.zeros(M)
print "initial cost: ", cost(w)

iters = 0
MAX_ITERS = 10
step   = 0.01

# gradient descent 
while (iters < MAX_ITERS):
    iters += 1
    w = w - step * gradient(w)

print "final cost: ", cost(w)
   
print "w_true: ", w_true
print "w     : ", w

# <codecell>

def predict_logreg_prob(w,X):
    return logistic(X.dot(w))

def predict_logreg(w,X):
    return [1.0 if p > 0.5 else 0.0 for p in predict_logreg_prob(w,X)]

def error_rate(w,X,y,predict):
    y_predict = predict(w,X)
    return np.mean( y == y_predict )
    
print w    
print "\nTrain Accuracy: ", error_rate(w,X_train,y_train,predict_logreg)
print "Test Accuracy: ",error_rate(w,X_test,y_test,predict_logreg)

# <headingcell level=2>

# add line search

# <codecell>

def line_search(w, step_0, cost_func, grad_func):
    """ simple line search for gradient descent step size
        
    Arguments:
        w         - modal weights
        step_0    - initail step size
        cost func - cost function
        grad_func - gradient function
        
    Returns:
        step size, and number of iterations
    """ 
    
    iters    = 1
    step     = step_0
    cost     = cost_func(w) 
    gradient = grad_func(w)
    w_new    = w - step * gradient
    new_cost = cost_func(w_new) 

    while (new_cost > cost) :
        step /=2.0
        w_new = w - step * gradient
        new_cost = cost_func(w_new) 
        iters +=1
    return step,iters


N, M = X_train.shape

print N,"observations", M, "variables"

MAX_ITERS  = 10
PRINT_ITER = 1

w0 = np.zeros(M)

STEP_SIZE = 1.0
C         = 1.0

cost_func = lambda w: logreg_cost(w,X_train,y_train,C)
grad_func = lambda w: logreg_gradient(w,X_train, y_train ,C)    

iters = 0
w     = w0
print "\nTrain Accuracy before: ", error_rate(w,X_train,y_train,predict_logreg)
print "Test Accuracy before: ",error_rate(w,X_test,y_test,predict_logreg)

cost  = cost_func(w)
print "Initial cost: ",cost
print "\nTraining...."
print 0,cost
while (iters < MAX_ITERS):
    iters += 1
    step, line_iters = line_search(w, STEP_SIZE, cost_func, grad_func)
    #iters += line_iters
    print "step: ",step  
    w = w - step * grad_func(w)

    if iters%PRINT_ITER == 0:
        print iters,cost_func(w)  


#print "weight vector: ",w
print "\nTrain Accuracy: ", error_rate(w,X_train,y_train,predict_logreg)
print "Test Accuracy: ",error_rate(w,X_test,y_test,predict_logreg)
print
print "w_true: ", w_true
print "w: ", w

# <headingcell level=2>

# visualize classifier 

# <codecell>

from matplotlib.colors import ListedColormap

def plot_model(weights,X,y,predict,cm=plt.cm.RdBu,
         marker = ListedColormap(['#FF0000', '#0000FF']),
         alpha=0.8, step = 0.02, pad = 0.5,
         arrow_width =0.2,arrow_length = 0.4):
    """ visualize linear model along with contour sets
        
    """ 
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))
    m, n = np.c_[xx.ravel(), yy.ravel()].shape
    mesh = np.ones(shape=(m, 3))
    mesh[:, 1:3] = np.c_[xx.ravel(), yy.ravel()]    
    Z = predict(weights, mesh)
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    markerSize = 40
    ax.contourf(xx, yy, Z, cmap=cm, alpha=alpha)
    ax.contour(xx, yy, Z, [0.5], colors = 'y')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=marker,s = markerSize)
        
    b=w[0]
    w1=w[1]
    w2=w[2]
    
    blue_x= x_min + 0.6*(x_max - x_min)
    blue_y=(-b - w1*blue_x)/w2
    
    red_x= x_min + 0.4*(x_max - x_min)
    red_y=(-b - w1*red_x)/w2
     
    v2= arrow_length*np.array([w1,w2])

    arr1 = plt.Arrow(blue_x,blue_y, v2[0], v2[1],edgecolor='blue',width=arrow_width)
    ax.add_patch(arr1)
    arr1.set_facecolor('blue')

    arr2 = plt.Arrow(red_x,red_y, -v2[0], -v2[1],edgecolor='red',width=arrow_width)
    ax.add_patch(arr2)
    arr2.set_facecolor('red')
    
    plt.show()

plot_model(w,X_train[:,1:],y_train,predict_logreg_prob,alpha =0.9,pad=0.1,arrow_width=0.6)

# <codecell>


