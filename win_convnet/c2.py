import numpy
import sys
import getopt as opt
from util import *
from math import sqrt, ceil, floor
import os
from gpumodel import IGPUModel
import random as r
import numpy.random as nr
from convnet import ConvNet
from options import *

try:
    import pylab as pl
except:
    print "This script requires the matplotlib python library (Ubuntu/Fedora package name python-matplotlib). Please install it."
    sys.exit(1)

import pickle

with open('nbe_base.pickle', 'rb') as f:
    nbe = pickle.load(f)   

#with open('test_base.pickle', 'rb') as f: 
#with open('test128s.pickle', 'rb') as f: 
with open('test_145.pickle', 'rb') as f: 
     test_errors_base = pickle.load(f)   

#with open('train_base.pickle', 'rb') as f: 
#with open('train128s.pickle', 'rb') as f: 
with open('train_145.pickle', 'rb') as f: 
    train_errors_base = pickle.load(f)
    
with open('test.pickle', 'rb') as f: 
    test_errors = pickle.load(f)   

with open('train.pickle', 'rb') as f: 
    train_errors = pickle.load(f)    

numbatches = nbe[0]        
numepochs = nbe[1]

x = range(0, len(train_errors_base))
x1 = range(0, len(train_errors))
pl.plot(x, train_errors_base, 'k-', label='Training set 1')
pl.plot(x, test_errors_base, 'r-', label='Test set 1')
pl.plot(x1, train_errors, 'g-', label='Training set 2')
pl.plot(x1, test_errors, 'b-', label='Test set 2')
pl.legend()
ticklocs = range(numbatches, len(train_errors) - len(train_errors) % numbatches + 1, numbatches)
epoch_label_gran = int(ceil(numepochs / 20.)) # aim for about 20 labels
epoch_label_gran = int(ceil(float(epoch_label_gran) / 10) * 10) # but round to nearest 10
ticklabels = map(lambda x: str((x[1] / numbatches)) if x[0] % epoch_label_gran == epoch_label_gran-1 else '', enumerate(ticklocs))

pl.xticks(ticklocs, ticklabels)
pl.xlabel('Epoch')
#        pl.ylabel(self.show_cost)
pl.title('logreg')
pl.show()