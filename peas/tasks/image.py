""" Direct target task. Checks if the network resembles
    a target connection matrix.
"""

### IMPORTS ###
import math

# Libraries
import numpy as np
from PIL import Image

### CLASSES ###

class ImageTask(object):
    
    def __init__(self, size=10, shape=['pyramid', 'waves_nw']):
        X, Y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        w = 0.4
        # self.target = ((-w < X) & (X < w) | (-w < Y) & (Y < w)) * 2.0 - 1.0
        self.target = np.zeros((size, size))
        if 'pyramid' in shape:
            self.target += abs(X) + abs(Y) - 1.0
        if 'waves_nw' in shape:
            waves = np.sin(5*(X + Y))
            waves[size/2:,:] = 0
            waves[:,size/2:] = 0
            self.target += waves
        # Normalize
        self.target -= self.target.min()
        self.target /= self.target.max()
                
    def evaluate(self, network):
        if network.cm.shape != self.target.shape:
            raise Exception("Network shape (%s) does not match target shape (%s)." % 
                (network.cm.shape, self.target.shape))
        err = ((network.cm - self.target)**2).mean()
        score = 1 / (1 + err)
        return {'fitness': score}
        
    def solve(self, network):
        return self.do(network) > 0.9
        
    def visualize(self, network, filename):
        import matplotlib
        matplotlib.use('Agg',warn=False)
        import matplotlib.pyplot as plt
        if network.cm.shape > self.target.shape:
            cm = network.cm[:self.target.shape[0],:self.target.shape[1]]
        else:
            cm = network.cm
        plt.imsave(filename, np.hstack((cm, self.target)), cmap=plt.cm.RdBu)


        
if __name__ == '__main__':
    task = ImageTask()
    print task.target