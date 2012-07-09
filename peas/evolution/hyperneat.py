""" Implements HyperNEAT's conversion
    from genotype to phenotype.
"""

### IMPORTS ###
from itertools import product

# Libs
import numpy as np

# Local
from ..networks.rnn import NeuralNetwork



class HyperNEATDeveloper(object):
    
    def __init__(self, substrate=None, substrate_shape=None, 
                 sandwich=False, 
                 weight_range=3.0, 
                 min_weight=0.3,
                 node_type='sigmoid'):
        """ Constructor 

            :param substrate:      A list of node coordinates (tuples)
            :param substrate_shape: If a tuple is passed, a uniform NxN substrate is generated with coords [-1, 1]
            :param weight_range:   (min, max) of substrate weights
            :param min_weight:     The minimum CPPN output value that will lead to an expressed connection.
            :param sandwich:       Whether to turn the output net into a sandwich network.
            :param node_type:      What node type to assign to the output nodes.
        """
        self.substrate    = substrate
        self.sandwich     = sandwich
        self.weight_range = weight_range
        self.min_weight   = min_weight
        self.node_type    = node_type
        
        if substrate_shape is not None:
            self.substrate = list(product(np.linspace(-1.0, 1.0, substrate_shape[0]), 
                                          np.linspace(-1.0, 1.0, substrate_shape[1])))
                                          
        if self.substrate is None:
            raise Exception("You must pass either substrate or substrate_shape")
        
        
    def convert(self, network):
        # Cast input to a neuralnetwork if it isn't
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
            
        # Since Stanley mentions to "fully activate" the CPPN,
        # I assume this means it's a feedforward net, since otherwise
        # there is no clear definition of "full activation".
        # In an FF network, activating each node once leads to a stable condition. 
        network.make_feedforward()
        
        # Initialize connectivity matrix  
        cm = np.zeros((len(self.substrate), len(self.substrate)))
            
        for (i, fr), (j, to) in product(enumerate(self.substrate), repeat=2):
            weight = network.feed(fr + to)[-1]
            cm[j, i] = weight
            
        cm[np.abs(cm) < self.min_weight] = 0
        cm -= (np.sign(cm) * self.min_weight)
        cm *= self.weight_range / (self.weight_range - self.min_weight)
            
        cm = np.clip(cm, -self.weight_range, self.weight_range)
        net = NeuralNetwork().from_matrix(cm, node_types=[self.node_type])
        
        if self.sandwich:
            net.make_sandwich()
            
        return net
            
