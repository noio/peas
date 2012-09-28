""" Direct target task. Checks if the network resembles
    a target connection matrix.
"""

### IMPORTS ###
import math

# Libraries
import numpy as np
import scipy.misc
from PIL import Image

# Local
from ..networks.rnn import NeuralNetwork

### CLASSES ###

class TargetWeightsTask(object):
    
    def __init__(self, substrate_shape=(3,3), noise=0.1, 
                funcs=[(lambda c: c[0] < 0.5, lambda c: np.sin(c[1]*3))] 
                ):
        # Instance vars
        self.substrate_shape = substrate_shape
        # Build the connectivity matrix coords system
        cm_shape = list(substrate_shape) + list(substrate_shape)
        coords = np.mgrid[[slice(-1, 1, s*1j) for s in cm_shape]]
        cm = np.zeros(cm_shape)
        # Add weights
        for (where, what) in funcs:
            mask = where(coords)
            vals = what(coords)
            cm[mask] += vals[mask]
            
        # Add noise
        mask = np.random.random(cm.shape) < noise
        cm[mask] = np.random.random(cm.shape)[mask]
        self.target = cm.reshape(np.product(substrate_shape), np.product(substrate_shape))
                        
    def evaluate(self, network):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        if network.cm.shape != self.target.shape:
            raise Exception("Network shape (%s) does not match target shape (%s)." % 
                (network.cm.shape, self.target.shape))
        err = ((network.cm - self.target)**2).sum()
        score = 1 / (1 + err)
        return {'fitness': score}
        
    def solve(self, network):
        return self.do(network) > 0.9
        
    def visualize(self, network, filename):
        import matplotlib
        matplotlib.use('Agg',warn=False)
        import matplotlib.pyplot as plt
        cm = network.cm
        target = self.target
        error = (cm - target) ** 2
        plt.imsave(filename, np.hstack((network.cm, self.target, error)), cmap=plt.cm.RdBu)

        
if __name__ == '__main__':
    task = TargetWeightsTask()
    print task.target