#! /usr/bin/python

""" Collection of benchmark tasks for evoluationary 
    algorithms
"""

### IMPORTS ###
import random

# Libraries
import numpy as np

# Local


class VisualFieldTask(object):
    """ 2D version of the Visual Field task. 
    """
    
    def __init__(self, size=11, dims=2, trials=75):
        # Instance vars
        self.size   = size
        self.trials = trials
        self.dims   = dims
        # Build the cases 
        tgtsize = size // 3
        offset  = size // 2 
        self.cases = []
        if dims == 2:
            for x in xrange(size - tgtsize):
                for y in xrange(size - tgtsize):
                    # Generate pattern
                    ptrn = np.zeros((size, size))
                    # Add the big object
                    ptrn[x:x+tgtsize, y:y+tgtsize] = 1
                    # Add the distraction object
                    ptrn1 = ptrn.copy()
                    ptrn2 = ptrn.copy()
                    ptrn3 = ptrn.copy()
                    ptrn1[(x+offset) % size, (y+offset) % size] = 1 # Diag
                    ptrn2[(x+offset) % size, y % size] = 1 # Right
                    ptrn3[x % size, (y+offset) % size] = 1 # Bottom
                    self.cases.append((ptrn1, (x+tgtsize//2, y+tgtsize//2)))
                    self.cases.append((ptrn2, (x+tgtsize//2, y+tgtsize//2)))
                    self.cases.append((ptrn3, (x+tgtsize//2, y+tgtsize//2)))
        elif dims == 1:
            for x in xrange(size - tgtsize):
                ptrn = np.zeros(size)
                ptrn[x:x+tgtsize] = 1
                ptrn[(x+offset) % size] = 1
                self.cases.append((ptrn, (x+tgtsize//2, 0)))
        
    def evaluate(self, network):
        if not network.sandwich:
            raise Exception("Visual Field task should be performed by a sandwich net.")
        
        dist = 0.0
        self._history = []
        for pattern, (x,y) in random.sample(self.cases, min(self.trials, len(self.cases))):
            output = network.feed(pattern, add_bias=False)
            self._history.append((pattern, output))
            # output *= 1 + 0.01 * np.random.random(output.shape)
            if output.size != pattern.size:
                raise Exception("Network output size (%s) does not correspond to pattern size (%s)" % 
                                    (output.shape, pattern.shape))
            if self.dims == 2:
                mx = output.argmax()
                (x_, y_) = mx // self.size, mx % self.size
            elif self.dims == 1:
                x_ = output.argmax()
                y_ = 0
            dist += ((x-x_)**2 + (y-y_)**2)**0.5
        dist = dist / self.trials
        score = 1. / (1. + dist)
        return {'fitness': score}
        
    def solve(self, network):
        return self.do(network) > 0.99
        
    def visualize(self, network, filename):
        """ Visualize a solution strategy by the given individual. """
        import matplotlib
        matplotlib.use('Agg',warn=False)
        import matplotlib.pyplot as plt
        # Visualize
        # self.evaluate(network)
        # im = np.zeros((0,self.size*2 + 1))
        # for (i, o) in self._history[:5]:
        #     if self.dims == 1:
        #         i = np.tile(i, (1,1))
        #         o = np.tile(o, (1,1))
        #     trial = np.hstack((i, np.zeros((i.shape[0],1)), o))
        #     im = np.vstack((im, np.zeros((1, trial.shape[1])), trial))
        plt.imsave(filename, network.cm[network.cm.shape[0]/2:, :network.cm.shape[1]/2], cmap=plt.cm.hot, vmin=0)  
