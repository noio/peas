""" Package with some classes to simulate neural nets.
"""

### IMPORTS ###

import sys
import numpy as np

# Libraries
import neat.chromosome

# Local
from ..methods.neat import NEATGenotype

# Shortcuts
inf  = float('inf')

### FUNCTIONS ###

# Node functions
def linear(x, clip=(-inf, inf)):
    return np.clip(x, *clip)

def gauss(x, mean=0.0, std=1.0):
    """ Returns the pdf of a gaussian.
    """
    z = (x - mean) / std
    return np.exp(-z**2/2.0) / np.sqrt(2*np.pi) / std
    
def sigmoid(x):
    """ Sigmoid function. 
        >>> s = sigmoid( np.linspace(-3, 3, 10) )
        >>> s[0] < 0.05 and s[-1] > 0.95
        True
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_alt(x):
    """ Alternative sigmoid function from NEAT paper. 
        >>> s = sigmoid( np.linspace(-3, 3, 10) )
        >>> s[0] < 0.05 and s[-1] > 0.95
        True
    """
    return 1 / (1 + np.exp(-4.8*x))
    

### CONSTANTS ###

ACTIVATION_FUNCS = {
    'sin': np.sin,
    'abs': np.abs,
    'linear': linear,
    'gauss': gauss,
    'sigmoid': sigmoid,
    'exp': sigmoid,
    'tanh': np.tanh,
    None : linear
}



### CLASSES ### 

class NeuralNetwork(object):
    """ A neural network. Can have recursive connections.
    """
    
    def from_matrix(self, matrix, node_types=['sigmoid']):
        """ Constructs a network from a weight matrix. 
        """
        # Initialize net
        self.original_shape = matrix.shape[:matrix.ndim//2]
        # If the connectivity matrix is given as a hypercube, squash it down to 2D
        n_nodes = np.prod(self.original_shape)
        self.cm  = matrix.reshape((n_nodes,n_nodes))
        self.node_types = [ACTIVATION_FUNCS[fn] for fn in node_types]
        if len(self.node_types) == 1:
            self.single_type = self.node_types[0]
            self.node_types *= n_nodes
        self.act = np.zeros(self.cm.shape[0])
        return self
        
    def from_neatchromosome(self, chromosome):
        """ Construct a network from a Chromosome instance, from
            the neat-python package. This is a connection-list
            representation.
        """
        # Typecheck
        if not isinstance(chromosome, neat.chromosome.Chromosome):
            raise Exception("Input should be a NEAT chromosome, is %r." % (chromosome))
        # Sort nodes: BIAS, INPUT, HIDDEN, OUTPUT, with HIDDEN sorted by feed-forward.
        nodes = dict((n.id, n) for n in chromosome.node_genes)
        node_order = ['bias']
        node_order += [n.id for n in filter(lambda n: n.type == 'INPUT', nodes.values())]
        if isinstance(chromosome, neat.chromosome.FFChromosome):
            node_order += chromosome.node_order
        else:
            node_order += [n.id for n in filter(lambda n: n.type == 'HIDDEN', nodes.values())]
        node_order += [n.id for n in filter(lambda n: n.type == 'OUTPUT', nodes.values())]
        # Construct object
        self.cm = np.zeros((len(node_order), len(node_order)))
        # Add bias connections
        for id, node in nodes.items():
            self.cm[node_order.index(id), 0] = node.bias
            self.cm[node_order.index(id), 1:] = node.response
        # Add the connections
        for conn in chromosome.conn_genes:
            if conn.enabled:
                to = node_order.index(conn.outnodeid)
                fr = node_order.index(conn.innodeid)
                # dir(conn.weight)
                self.cm[to, fr] *= conn.weight
        # Verify actual feed forward
        if isinstance(chromosome, neat.chromosome.FFChromosome):
            if np.triu(self.cm).any():
                raise Exception("NEAT Chromosome does not describe feedforward network.")
        node_order.remove('bias')
        self.node_types = [ACTIVATION_FUNCS[nodes[i].activation_type] for i in node_order]
        self.node_types = [linear] + self.node_types
        self.act = np.zeros(self.cm.shape[0])
        return self
    
    def __init__(self, source=None):
        # Set instance vars
        self.feedforward    = False
        self.sandwich       = False   
        self.cm             = None
        self.node_types     = None
        self.single_type    = None
        self.original_shape = None
        
        if source is not None:
            if isinstance(source, NEATGenotype):
                self.from_matrix(*source.get_network_data())
            # TODO: maybe remove this neat-python constructor
            elif isinstance(source, neat.chromosome.Chromosome):
                self.from_neatchromosome(source)
            else:
                raise Exception("Cannot convert from %s to %s" % (source.__class__, self.__class__))

    def make_sandwich(self):
        """ Turns the network into a sandwich network,
            a network with no hidden nodes and 2 layers.
        """
        self.sandwich = True
        self.cm = np.hstack((self.cm, np.zeros(self.cm.shape)))
        self.cm = np.vstack((np.zeros(self.cm.shape), self.cm))
        self.act = np.zeros(self.cm.shape[0])
        return self
        
    def make_feedforward(self):
        """ Zeros out all recursive connections. 
        """
        self.feedforward = True
        self.cm[np.triu_indices(self.cm.shape[0])] = 0
        return self
        
    def feed(self, input_activation, add_bias=True):
        """ Feed an input to the network, returns the entire
            activation state, you need to extract the output nodes
            manually.
            
            :param add_bias: Add a bias input automatically, before other inputs.
        """
        
        if add_bias:
            input_activation = np.hstack((1.0, input_activation))
        
        if input_activation.size >= self.act.size:
            raise Exception("More input values (%s) than input nodes (%s)." % (input_activations.shape, self.act.shape))
        
        input_size = min(self.act.size - 1, input_activation.size)
        
        # Feed forward nets reset the activation, then activate once
        # for every node in the network.
        if self.feedforward:
            self.act = np.zeros(self.cm.shape[0])
            self.act[:input_size] = input_activation.flat[:input_size]
            for i in range(input_size, self.act.size):
                self.act[i] = self.node_types[i](np.dot(self.cm[i], self.act))
        # Sandwich networks activate once globally, and need a single activation
        # type.
        elif self.sandwich:
            self.act[:input_size] = input_activation.flat[:input_size]
            self.act = np.dot(self.cm, self.act)
            self.act = self.single_type(self.act)
        # All other recursive networks only activate once too, upon feeding
        # this means that upon each feed, activation propagate by one step.
        else:
            self.act[:input_size] = input_activation.flat[:input_size]
            self.act = np.dot(self.cm, self.act)
            for i in range(len(self.node_types)):
                self.act[i] = self.node_types[i](self.act[i])
            
        # Reshape the output to 2D if it was 2D
        if self.sandwich:
            return self.act[self.act.size//2:].reshape(self.original_shape)      
        else:
            return self.act.reshape(self.original_shape)

    
    def visualize(self, filename, input_nodes=3, output_nodes=1):
        """ Visualize the network, stores in file. """
        if self.cm.shape[0] > 50:
            return
        import pygraphviz as pgv
        # Some settings
        node_dist = 1
        cm = self.cm.copy()
        # Sandwich network have half input nodes.
        if self.sandwich:
            input_nodes = cm.shape[0] // 2
            output_nodes = input_nodes
        # Clear connections to input nodes, these arent used anyway
        cm[:input_nodes, :] = 0
        G = pgv.AGraph(directed=True)
        mw = abs(cm).max()
        for i in range(cm.shape[0]):
            G.add_node(i)
            if self.single_type is not None:
                t = self.single_type.__name__
            else:
                t = self.node_types[i].__name__
            G.get_node(i).attr['label'] = '%d:%s' % (i, t[:3])
            for j in range(cm.shape[1]):
                w = cm[i,j]
                if abs(w) > 0.01:
                    G.add_edge(j, i, penwidth=abs(w)/mw*4, color='blue' if w > 0 else 'red')
        for n in range(input_nodes):
            pos = (node_dist*n, 0)
            G.get_node(n).attr['pos'] = '%s,%s!' % pos
            G.get_node(n).attr['shape'] = 'doublecircle'
            G.get_node(n).attr['fillcolor'] = 'steelblue'
            G.get_node(n).attr['style'] = 'filled'
        for i,n in enumerate(range(cm.shape[0] - output_nodes,cm.shape[0])):
            pos = (node_dist*i, -node_dist * 5)
            G.get_node(n).attr['pos'] = '%s,%s!' % pos
            G.get_node(n).attr['shape'] = 'doublecircle'
            G.get_node(n).attr['fillcolor'] = 'tan'
            G.get_node(n).attr['style'] = 'filled'
        
        G.node_attr['shape'] = 'circle'
        if self.sandwich: 
            # neato supports fixed node positions, so it's better for
            # sandwich networks
            prog = 'neato'
        else:
            prog = 'dot'
        G.draw(filename, prog=prog)
        

if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    
