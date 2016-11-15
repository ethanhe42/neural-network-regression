# coding: utf-8
from scipy.interpolate import spline
import sys
from data_utils import load_CIFAR10
from neural_net import *
import matplotlib.pyplot as plt
import time
start_time=time.time()
import numpy as np
from neural_net import *

def interp(y, x=None):
    assert x is None
    x = range(len(y))
    y=np.array(y).reshape(-1)
    xnew = np.linspace(0,len(y),300)
    power_smooth = spline(x,y,xnew)
    print xnew
    print power_smooth
    return xnew, power_smooth

def sinedataset(points=200):
    r = 1.
    left = -r*np.pi
    right = r*np.pi
    step = (right - left) / points
    x = np.arange(left, right, step)
    idxs = np.arange(points)
    y = np.sin(x)
    tridxs = np.where(idxs % 2)
    teidxs = np.where(idxs % 2 == 0)
    xtr = x[tridxs][:, np.newaxis] / right
    ytr = y[tridxs][:, np.newaxis]
    if 1:
        plt.scatter(xtr,ytr)
        #plt.show()
        plt.savefig('scatter')
    return xtr, ytr, x[teidxs][:, np.newaxis], y[teidxs][:,np.newaxis]

xtr, ytr, xte, yte = sinedataset()
print xtr.shape
print xte.shape

input_size = xtr.shape[1]
num_classes = ytr.shape[1]
hidden_size = 100

net = TwoLayerNet(input_size,
                  hidden_size,
                  num_classes,
                  1e-4,
                  activation='tanh')
stats = net.train(xtr, ytr, xte, yte,
                            num_iters=3000000, batch_size=2,
                            learning_rate=1e-2, learning_rate_decay=1.,
                            reg=0,
                  verbose=True,
                  update="momentum",
                  arg=0.95,
                  )
                  #activation='ReLU')

def acc(x,y):
    val_acc = sum((net.predict(x) - y)**2) / len(y) /2
    return val_acc

plt.cla()
#print net.predict(xtr)
plt.plot(net.predict(xtr))
plt.plot(ytr)
plt.savefig('fig')

print 'Train accuracy: ', acc(xtr,ytr)
print 'Validation accuracy: ', acc(xte,yte)

#print 'Test accuracy: ', val_acc

print "time",(time.time()-start_time)/60.0

# In[121]:

plt.cla()
#Plot the loss function and train / validation accuracies
plt.plot(*interp(stats['loss_history']))
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
#plt.savefig("dropout loss_history.eps")
#
plt.plot(*interp(stats['train_acc_history']), label='train')
plt.plot(*interp(stats['val_acc_history']), label='val')
plt.title('loss history')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()
#plt.savefig('dropout accuracy.eps')


