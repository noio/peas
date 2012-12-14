""" Implements a 'wavelets' genotype, that encodes
    locations and scales of wavelets for building a connectivity
    matrix.
"""

### IMPORTS ###

import random

# Libs
import numpy as np
import scipy.misc

# Local
from ..networks.rnn import NeuralNetwork

# Shortcuts
rand = np.random.random

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
    
    def __init__(self, inputs,
                 prob_add=0.1,
                 prob_modify=0.3,
                 stdev_mutate=0.1,
                 initial=1):
        # Instance vars
        self.inputs       = inputs
        self.prob_add     = prob_add
        self.prob_modify  = prob_modify
        self.stdev_mutate = stdev_mutate
        self.wavelets     = [] # Each defined by an affine matrix.
        
        for _ in xrange(initial):
            self.add_wavelet()
        
    def add_wavelet(self):
        t = rand((2, 1)) * 2 - 1
        mat = rand((2, self.inputs))
        norms = np.sqrt(np.sum(mat ** 2, axis=1))[:, np.newaxis]
        mat /= norms
        mat = np.hstack((mat, t))
        sigma = np.random.normal(0.5, 0.3)
        weight = np.random.normal(0, 0.3)
        wavelet = [weight, sigma, mat]
        self.wavelets.append(wavelet)
    
    def mutate(self):
        """ Mutate this individual """
        if rand() < self.prob_add:
            self.add_wavelet()
        else:   
            for wavelet in self.wavelets:
                if rand() < self.prob_modify:
                    wavelet[0] += np.random.normal(self.stdev_mutate)
                    wavelet[1] += np.random.normal(self.stdev_mutate)
                    wavelet[2] += np.random.normal(0, self.stdev_mutate, wavelet[2].shape)
                
        return self # for chaining
                        
    def __str__(self):
        return "%s with %d wavelets" % (self.__class__.__name__, len(self.wavelets))
        
        
class WaveletDeveloper(object):
    """ Very simple class to develop the wavelet genotype to a
        neural network.
    """
    def __init__(self, substrate,
                       add_deltas=False,
                       min_weight=0.3,
                       weight_range=3.0,
                       node_type='tanh',
                       sandwich=False):
        # Fields
        self.substrate    = substrate
        self.add_deltas   = add_deltas
        self.min_weight   = min_weight
        self.weight_range = weight_range
        self.node_type    = node_type
        self.sandwich     = sandwich
        
    
    def convert(self, individual):
        cm = np.zeros((self.substrate.num_nodes, self.substrate.num_nodes))
        
        for (i,j), coords, conn_id in self.substrate.get_connection_list(self.add_deltas):
            # Add a bias (translation)
            coords = np.hstack((coords, [1]))
            weight = sum(weight * gabor(*(np.dot(mat, coords)), sigma=sigma) 
                        for (weight, sigma, mat) in individual.wavelets)
            cm[j,i] = weight
        
        # Rescale weights
        cm[np.abs(cm) < self.min_weight] = 0
        cm -= (np.sign(cm) * self.min_weight)
        cm *= self.weight_range / (self.weight_range - self.min_weight)
        
        # Clip highest weights
        cm = np.clip(cm, -self.weight_range, self.weight_range)
        net = NeuralNetwork().from_matrix(cm, node_types=[self.node_type])
        
        if self.sandwich:
            net.make_sandwich()

        return net
        
        
if __name__ == '__main__':
    pass
