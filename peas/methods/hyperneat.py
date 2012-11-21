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
                 connectivity=None,
                 max_connection_length=None,
                 weight_by_output=-1,
                 bias_by_output=None,
                 sandwich=False, 
                 add_deltas=False,
                 weight_range=3.0,
                 min_weight=0.3,
                 activation_steps=10,
                 node_type='tanh'):
        """ Constructor 

            :param substrate:      A list of node coordinates (tuples)
            :param substrate_shape: If a tuple is passed, a uniform NxN substrate is generated with coords [-1, 1]
            :param connectivity:   A list of tuples that determine all the connections that are evaluated.
                                    Pass None to evaluate *all* connections.
            :param max_connection_length: Evaluate only connections that are at most this length (manhattan distance).
            :param weight_range:   (min, max) of substrate weights
            :param min_weight:     The minimum CPPN output value that will lead to an expressed connection.
            :param sandwich:       Whether to turn the output net into a sandwich network.
            :param node_type:      What node type to assign to the output nodes.
        """
        self.substrate             = substrate
        self.connectivity          = connectivity
        self.max_connection_length = max_connection_length
        self.weight_by_output      = weight_by_output
        self.bias_by_output        = bias_by_output
        self.sandwich              = sandwich
        self.add_deltas            = add_deltas
        self.weight_range          = weight_range
        self.min_weight            = min_weight
        self.activation_steps      = activation_steps
        self.node_type             = node_type
        self.substrate_shape       = substrate_shape
        
        # If no substrate is passed, create a mesh grid of nodes.
        if substrate_shape is not None:
            # Create coordinate grids
            self.substrate = np.mgrid[[slice(-1, 1, s*1j) for s in substrate_shape]]
            # Move coordinates to last dimension
            self.substrate = self.substrate.transpose(range(1,len(substrate_shape)+1) + [0])
            # Reshape to a N x nD list.
            self.substrate = self.substrate.reshape(-1, len(substrate_shape))
            
        if type(self.substrate) is list:
            self.substrate = np.array(self.substrate)
        
        if self.substrate is None:
            raise Exception("You must pass either substrate or substrate_shape.")

        # If no connectivity is given, connect all nodes to all nodes,
        # within the max range given by the max_connection_length argument
        if self.connectivity is None:
            self.connectivity = []
            # Skip if outside max connection range, 
            # This behavior is used in O.J. Coleman's experiments.
            for (i, fr), (j, to) in product(enumerate(self.substrate), repeat=2):
                if (self.max_connection_length is None or 
                    np.all(np.abs(fr - to) <= self.max_connection_length)):
                    self.connectivity.append((i,j))
        
        self.connectivity = np.array(self.connectivity)
        
        # If only a single node is indicated as weight-determining node, 
        # repeat it for each connection
        if type(self.weight_by_output) == int:
            self.weight_by_output = [self.weight_by_output] * len(self.connectivity)
            
        if len(self.weight_by_output) != len(self.connectivity):
            raise Exception("""weight_by_output must be an array that indicates the node number to
                               use for each connection weight. Negative values are allowed.""")
        
        if (self.connectivity.min() < 0 or
            self.connectivity.max() >= len(self.substrate)):
            raise Exception("Connectivity pairs contain illegal index.")
        
        if (len(self.connectivity) != len(set(tuple(pair) for pair in self.connectivity))):
            raise Exception("Connectivity pairs contain duplicate.")
        
        
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
        cm_dims = 2 * len(self.substrate[0])
        required_inputs = cm_dims + 1
        if network.cm.shape[0] < required_inputs:
            raise Exception("Network does not have enough inputs. Has %d, needs %d" %
                    (network.cm.shape[0], cm_dims+1))

        # Initialize connectivity matrix
        cm = np.zeros((len(self.substrate), len(self.substrate)))
            
        for (i,j), weight_node in zip(self.connectivity, self.weight_by_output):
            fr = self.substrate[i]
            to = self.substrate[j]
            
            if not self.add_deltas:
                net_input = np.hstack((fr, to))
            else:
                deltas = np.array(to) - np.array(fr) 
                net_input = np.hstack((fr, to, deltas))
                
            if network.feedforward:
                weight = network.feed(net_input)[weight_node]
            else:
                network.flush()
                for _ in xrange(self.activation_steps):
                    weight = network.feed(net_input)[weight_node]
            cm[j, i] = weight
        
        # Rescale the CM
        cm[np.abs(cm) < self.min_weight] = 0
        cm -= (np.sign(cm) * self.min_weight)
        cm *= self.weight_range / (self.weight_range - self.min_weight)
        
        # Query the network for bias
        # Use the coordinates of each substrate node, and a set of zeros.
        if self.bias_by_output is not None:
            bias = [network.feed(np.hstack((coords, np.zeros(coords.shape))))[self.bias_by_output]
                    for coords in self.substrate]
            bias = np.array([bias]).T
            # Add a column of bias weights as first column
            cm = np.hstack((bias, cm))
            # Append a row of zeros to make the array square again
            cm = np.insert(cm, 0, 0, axis=0)
        
        # Clip highest weights
        cm = np.clip(cm, -self.weight_range, self.weight_range)
        net = NeuralNetwork().from_matrix(cm, node_types=[self.node_type])
        
        if self.sandwich:
            net.make_sandwich()
            
        if not np.all(np.isfinite(net.cm)):
            raise Exception("Network contains NaN/inf weights.")
            
        return net
            
