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
    
    """ HyperNEAT developer object."""
    
    def __init__(self, substrate=None, substrate_shape=None, 
                 sandwich=False, 
                 add_deltas=False,
                 weight_range=3.0, 
                 min_weight=0.3,
                 activation_steps=10,
                 node_type='tanh'):
        """ Constructor 

            :param substrate:      A list of node coordinates (tuples)
            :param substrate_shape: If a tuple is passed, a uniform NxN substrate is generated with coords [-1, 1]
            :param weight_range:   (min, max) of substrate weights
            :param min_weight:     The minimum CPPN output value that will lead to an expressed connection.
            :param sandwich:       Whether to turn the output net into a sandwich network.
            :param node_type:      What node type to assign to the output nodes.
        """
        self.substrate        = substrate
        self.sandwich         = sandwich
        self.add_deltas       = add_deltas
        self.weight_range     = weight_range
        self.min_weight       = min_weight
        self.activation_steps = activation_steps
        self.node_type        = node_type
        self.substrate_shape  = substrate_shape
        
        if substrate_shape is not None:
            # Create coordinate grids
            self.substrate = np.mgrid[[slice(-1, 1, s*1j) for s in substrate_shape]]
            # Move coordinates to last dimension
            self.substrate = self.substrate.transpose(range(1,len(substrate_shape)+1) + [0])
            # Reshape to a N x nD list.
            self.substrate = self.substrate.reshape(-1, len(substrate_shape))
                                          
        if self.substrate is None:
            raise Exception("You must pass either substrate or substrate_shape")
        
        
    def convert(self, network):
        """ Performs conversion. 
            
            :param network: Any object that is convertible to a :class:`~peas.networks.NeuralNetwork`.
        """
        
        # Cast input to a neuralnetwork if it isn't
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
            
        # Since Stanley mentions to "fully activate" the CPPN,
        # I assume this means it's a feedforward net, since otherwise
        # there is no clear definition of "full activation".
        # In an FF network, activating each node once leads to a stable condition. 
        
        # Check if the network has enough inputs.
        cm_dims = 2 * len(self.substrate_shape)
        required_inputs = cm_dims + 1
        if network.cm.shape[0] < required_inputs:
            raise Exception("Network does not have enough inputs. Has %d, needs %d" %
                    (network.cm.shape[0], cm_dims+1))

        # Initialize connectivity matrix
        cm = np.zeros((len(self.substrate), len(self.substrate)))
            
        for (i, fr), (j, to) in product(enumerate(self.substrate), repeat=2):
            if not self.add_deltas:
                net_input = np.hstack((fr, to))
            else:
                deltas = np.array(to) - np.array(fr) 
                net_input = np.hstack((fr, to, deltas))
                
            if network.feedforward:
                weight = network.feed(net_input)[-1]
            else:
                network.flush()
                for _ in xrange(self.activation_steps):
                    weight = network.feed(net_input)[-1]
            cm[j, i] = weight
        
        # Rescale the CM
        cm[np.abs(cm) < self.min_weight] = 0
        cm -= (np.sign(cm) * self.min_weight)
        cm *= self.weight_range / (self.weight_range - self.min_weight)
        
        # Clip highest weights
        cm = np.clip(cm, -self.weight_range, self.weight_range)
        net = NeuralNetwork().from_matrix(cm, node_types=[self.node_type])
        
        if self.sandwich:
            net.make_sandwich()
            
        if not np.all(np.isfinite(net.cm)):
            raise Exception("Network contains NaN/inf weights.")
            
        return net
            
