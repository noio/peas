""" Direct target task. Checks if the network resembles
    a target connection matrix.
"""

### IMPORTS ###
import math
import os

# Libraries
import numpy as np
import scipy.misc
from PIL import Image

# Local
from ..networks.rnn import NeuralNetwork

### CLASSES ###

class TargetWeightsTask(object):
    
    def __init__(self, default_weight=0, substrate_shape=(3,3), noise=0, max_weight=3.0,
                funcs=[] 
                ):
        # Instance vars
        self.substrate_shape = substrate_shape
        self.max_weight      = max_weight
        if not (0 <= noise <= 1):
            raise Exception("Noise value has to be between 0 and 1.")
        # Build the connectivity matrix coords system
        cm_shape = list(substrate_shape) + list(substrate_shape)
        coords = np.mgrid[[slice(-1, 1, s*1j) for s in cm_shape]]
        cm = np.ones(cm_shape) * default_weight
        # Add weights
        for (where, what) in funcs:
            mask = where(coords) if callable(where) else (np.ones(cm.shape, dtype=bool))
            vals = what(coords, cm) if callable(what) else (what * np.ones(cm.shape))
            cm[mask] = vals[mask]
            
        # Add noise
        mask = np.random.random(cm.shape) < noise
        random_weights = np.random.random(cm.shape) * max_weight * 2 - max_weight
        cm[mask] = random_weights[mask]
        self.target = cm.reshape(np.product(substrate_shape), np.product(substrate_shape))
        
        # Clip
        self.target = np.clip(self.target, -max_weight, max_weight)
                        
    def evaluate(self, network):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        if network.cm.shape != self.target.shape:
            raise Exception("Network shape (%s) does not match target shape (%s)." % 
                (network.cm.shape, self.target.shape))
        err = np.abs(network.cm - self.target)
        score = ((2 * self.max_weight) - err).mean()
        correct = (err < (self.max_weight / 10.0)).mean()
        return {'fitness': 2**score, 'error':err.mean(), 'correct':correct}
        
    def solve(self, network):
        return self.evaluate(network)['correct'] > 0.8
        
    def visualize(self, network, filename):
        import matplotlib
        matplotlib.use('Agg',warn=False)
        import matplotlib.pyplot as plt
        cm = network.cm
        target = self.target
        error = (cm - target) ** 2
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.imsave(filename, np.hstack((network.cm, self.target, error)), cmap=plt.cm.RdBu)

        
if __name__ == '__main__':
    task = TargetWeightsTask()
    print task.target
