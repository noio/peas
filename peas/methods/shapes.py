""" Implements a 'wavelets' genotype, that encodes
    locations and scales of wavelets for building a connectivity
    matrix.
"""

### IMPORTS ###

import random

# Libs
import numpy as np
import scipy.misc

# Shortcuts
rand = random.random

### FUNCTIONS ###

def gabor(x, y, l=1.0, psi=np.pi/2, sigma=0.5, gamma=1.0):
    return np.exp(-(x**2 + (gamma*y)**2)/(2*sigma**2)) * np.cos(2*np.pi*x/l + psi)


def transform_meshgrid(x, y, mat):
    """ Transforms the given meshgrid (x, y) 
        using the given affine matrix 
    """
    coords = np.vstack([x.flat, y.flat, np.ones(x.size)])
    coords = np.dot(mat, coords)
    x = np.reshape(coords[0,:], x.shape)
    y = np.reshape(coords[1,:], y.shape)
    return (x, y)
    

### CLASSES ###
        

class WaveletGenotype(object):
    
    def __init__(self,
                 prob_add=0.1,
                 prob_modify=0.3,
                 stdev_mutate=0.3,
                 output_size=10):
        # Instance vars
        self.prob_add     = prob_add
        self.prob_modify  = prob_modify
        self.stdev_mutate = stdev_mutate
        self.output_size  = output_size
        self.wavelets     = [] # Each defined by an affine matrix.
    
    def mutate(self):
        """ Mutate this individual """
        if rand() < self.prob_add:
            th = rand() * np.pi
            cx, cy = rand() * 2 - 1, rand() * 2 - 1 
            r = rand() + 0.5
            mat = np.array([[np.cos(th) * r, -np.sin(th) * r, cx],
                            [np.sin(th) * r,  np.cos(th) * r, cy]])
            sigma = np.random.normal(0.5, 0.3)
            weight = np.random.normal(0, 0.3)
            wavelet = [weight, sigma, mat]
            self.wavelets.append(wavelet)
            
        for wavelet in self.wavelets:
            if rand() < self.prob_modify:
                wavelet[0] += np.random.normal(self.stdev_mutate/2)
                wavelet[1] += np.random.normal(self.stdev_mutate/2)
                wavelet[2] += np.random.normal(0, self.stdev_mutate, wavelet[2].shape)
                
        return self # for chaining
                
                
    def get_network_data(self):
        x, y = np.meshgrid(np.linspace(-1, 1, self.output_size), np.linspace(-1, 1, self.output_size))
        cm = np.zeros((self.output_size, self.output_size))
        
        for wavelet in self.wavelets:
            weight, sigma, mat = wavelet
            lx, ly = transform_meshgrid(x, y, mat)
            cm += weight * gabor(lx, ly, sigma=sigma)
            
        node_types = ['linear'] * cm.shape[0]
        return cm, node_types
        
    def __str__(self):
        return "%s with %d wavelets" % (self.__class__.__name__, len(self.wavelets))
        
if __name__ == '__main__':
    pass