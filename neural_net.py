import numpy as np
import matplotlib.pyplot as plt
import time

def relu(x):
    x[x<0]=0
    return x

def drelu(x):
    dF = np.ones_like(x)
    dF[x==0.0]=0 #activation res a2 has been ReLUed
    return dF

def dtanh(x):
    return 1 - x**2

def dleaky(x):
    dF = np.ones_like(x)
    dF[a2<0.0]=0.01
    return dF


def tanh(x):
    return np.tanh(x)

def leaky(inp):
    return np.maximum(inp,.01*inp)

def l2(inp):
    return sum(inp**2) / 2 / len(inp)

def dl2(inp):
    return inp / len(inp)

def l1(inp):
    return sum(np.abs(inp)) / len(inp)

def dl1(inp):
    ret = inp.copy()
    ret[ret > 0] = 1.
    ret[ret < 0] = -1.
    return ret / len(ret)

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4,
    init_method="Normal", activation='relu'):
    """
    Initialize the model.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

    #special initialization
    if init_method=="i":
      self.params['W1']=np.random.randn(input_size,hidden_size)/np.sqrt(input_size)
      self.params['W2']=np.random.randn(hidden_size,output_size)/np.sqrt(hidden_size)
    elif init_method=="io":
      self.params['W1']=np.random.randn(input_size,hidden_size)*np.sqrt(2.0/(input_size+hidden_size))
      self.params['W2']=np.random.randn(hidden_size,output_size)*np.sqrt(2.0/(hidden_size+output_size))
    elif init_method=="ReLU":
      self.params['W1']=np.random.randn(input_size,hidden_size)*np.sqrt(2.0/input_size)
      self.params['W2']=np.random.randn(hidden_size,output_size)*np.sqrt(2.0/(hidden_size+output_size))

    self.activation = activation
    if activation=='leaky':
      self.acfunc = leaky
      self.dacfunc = dleaky
    elif activation == 'tanh':
      self.acfunc = tanh
      self.dacfunc = dtanh
    else:
      self.acfunc = relu
      self.dacfunc = drelu
    self.finalactivation = 0

  def loss(self, X, y=None, reg=0.0, dropout=0, dropMask=None):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None

    inp=X.dot(W1)+b1
    a2 = self.acfunc(inp)

    scores=a2.dot(W2)+b2 # z3

    if y is None:
      return scores

    # Compute the loss
    loss = None

    if self.finalactivation:
        a3 = tanh(scores)
    else:
        a3= scores #h(x)

    lossfunc = 'l2'
    if lossfunc == 'l1':
        lossf = l1
        dlossf = dl1
    elif lossfunc == 'l2':
        lossf = l2
        dlossf = dl2

    loss=lossf(a3 - y)+\
      0.5*reg*(np.sum(np.power(W1,2))+np.sum(np.power(W2,2)))

    # Backward pass: compute gradients
    grads = {}

    delta_3=dlossf(a3 - y)

    if self.finalactivation:
        delta_3 = delta_3 * (1 - a3**2)
    grads['W2']=a2.T.dot(delta_3)+reg*W2
    grads['b2']=np.sum(delta_3,0)


    dF = self.dacfunc(a2)

    delta_2=delta_3.dot(W2.T)*dF
    grads['W1']=X.T.dot(delta_2)+reg*W1
    grads['b1']=np.sum(delta_2,0)

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False,

            update="SGD",arg=.99,
            dropout=0):
    """
    Train this neural network using stochastic gradient descent.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    #### for tracking top model
    top_params=self.params.copy()
    cache_params=dict()
    top_acc=100000
    cache=dict()
    dropMask=dict()
    start_time=time.time()
    ####

    for it in xrange(num_iters):
      if it % (num_iters/3) == 0 and it != 0:
        learning_rate *=.1
        print learning_rate
      X_batch = None
      y_batch = None

      if num_train >= batch_size:
        rand_idx=np.random.choice(num_train,batch_size)
      else:
        rand_idx=np.random.choice(num_train,batch_size,replace=True)
      X_batch=X[rand_idx]
      y_batch=y[rand_idx]

      if dropout>1:
        for param in ['W2','b2']:
       	  dropMask[param]=np.random.randn(*self.params[param].shape)<(dropout-1)

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout=dropout,dropMask=dropMask)

      if np.isnan(grads['W1']).any() or np.isnan(grads['W2']).any() or \
        np.isnan(grads['b1']).any() or np.isnan(grads['b2']).any():
        continue
     #cache_params=self.params.copy()
      dx=None
      for param in self.params:
        if update=="SGD":
          dx=learning_rate*grads[param]
          #self.params[param]-=learning_rate*grads[param]

        elif update=="momentum":
          if not param in cache:
            cache[param]=np.zeros(grads[param].shape)
          cache[param]=arg*cache[param]-learning_rate*grads[param]
          dx=-cache[param]
          #self.params[param]+=cache[param]

        elif update=="Nesterov momentum":
          if not param in cache:
            cache[param]=np.zeros(grads[param].shape)
          v_prev = cache[param] # back this up
          cache[param] = arg * cache[param] - learning_rate * grads[param] # velocity update stays the same
          dx=arg * v_prev - (1 + arg) * cache[param]
          #self.params[param] += -arg * v_prev + (1 + arg) * cache[param] # position update changes form

        elif update=="rmsprop":
          if not param in cache:
            cache[param]=np.zeros(grads[param].shape)
          cache[param]=arg*cache[param]+(1-arg)*np.power(grads[param],2)
          dx=learning_rate*grads[param]/np.sqrt(cache[param]+1e-8)
          #self.params[param]-=learning_rate*grads[param]/np.sqrt(cache[param]+1e-8)


        elif update=="Adam":
          print "update error"

        elif update=="Adagrad":
          print "update error"

        else:
          # if have time try more update methods
          print "choose update method!"
        if dropout>1:
    	  if param == 'W2' or param == 'b2':
       	    dx*=dropMask[param]
        self.params[param]-=dx
      #Bug: learning rate should not decay at first epoch
      it+=1

      selectbest=0
      if not selectbest:
          for key in top_params:
            expratio=.9
            top_params[key] = expratio*top_params[key]+(1-expratio)*self.params[key]
      if verbose and it % (num_iters/500) == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
        loss_history.append(loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = loss
        val_acc = sum((self.predict(X) - y)**2)

        # Decay learning rate
        learning_rate *= learning_rate_decay


        ### update top model
        if selectbest and val_acc < top_acc:
          top_acc = val_acc
          top_params=self.params.copy()

    	if verbose:
          pass
          #print ('loss, %f train_acc %f, val_acc %f, time %d' % (loss, train_acc, val_acc,(time.time()-start_time)/60.0))

    self.params=top_params.copy()
    ### update params to top params finally

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict
    """
    y_pred = None

    #y_pred=relu(X.dot(self.params['W1'])+self.params['b1']).dot(self.params['W2'])+self.params['b2']
    y_pred=self.acfunc(X.dot(self.params['W1'])+self.params['b1']).dot(self.params['W2'])+self.params['b2']
    if self.finalactivation:
        y_pred = tanh(y_pred)

    return y_pred

  def accuracy(self,X,y):
    """
    compute the accuracy.
    """
    acc = (self.predict(X) == y).mean()

    return acc

  def gradient_check(self,X,y):
    realGrads=dict()
    _,grads=self.loss(X,y)
    keys=['W1','b1',
          'W2','b2']
    for key in keys:
      W1=self.params[key]
      W1_grad=[]
      delta=1e-4
      if len(np.shape(W1))==2:
        for i in range(np.shape(W1)[0]):
          grad=[]
          for j in range(np.shape(W1)[1]):
            W1[i,j]+=delta
            self.params[key]=W1
            l_plus,_=self.loss(X,y)
            W1[i,j]-=2*delta
            self.params[key]=W1
            l_minus,_=self.loss(X,y)
            grad.append((l_plus-l_minus)/2.0/delta)
            W1[i,j]+=delta
          W1_grad.append(grad)
      else:
        for i in range(len(W1)):
          W1[i]+=delta
          self.params[key]=W1
          l_plus,_=self.loss(X,y)
          W1[i]-=2*delta
          self.params[key]=W1
          l_minus,_=self.loss(X,y)
          W1_grad.append((l_plus-l_minus)/2.0/delta)
          W1[i]+=delta

      print(W1_grad)
      print(grads[key])
      print key,"error",np.mean(np.sum(np.power((W1_grad-grads[key]),2),len(np.shape(W1))-1)\
                        /np.sum(np.power((W1_grad+grads[key]),2),len(np.shape(W1))-1))
