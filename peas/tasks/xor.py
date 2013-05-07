""" Input/output relation task. Every input and output
    is explicitly defined. XOR is an example of this task.
"""

### IMPORTS ###
import random

# Libraries
import numpy as np

# Local
from ..networks.rnn import NeuralNetwork


class XORTask(object):
    
    # Default XOR input/output pairs
    INPUTS  = [(0,0), (0,1), (1,0), (1,1)]
    OUTPUTS = [(-1,), (1,), (1,), (-1,)]
    EPSILON = 1e-100
    
    def __init__(self, do_all=True):
        self.do_all = do_all
        self.INPUTS = np.array(self.INPUTS, dtype=float)
        self.OUTPUTS = np.array(self.OUTPUTS, dtype=float)
    
    def evaluate(self, network, verbose=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        network.make_feedforward()

        if not network.node_types[-1](-1000) < -0.95:
            raise Exception("Network should be able to output value of -1, e.g. using a tanh node.")
        
        pairs = zip(self.INPUTS, self.OUTPUTS)
        random.shuffle(pairs)
        if not self.do_all:
            pairs = [random.choice(pairs)]
        rmse = 0.0
        for (i, target) in pairs:
            # Feed with bias
            output = network.feed(i)
            # Grab the output
            output = output[-len(target):]
            err = (target - output)
            err[abs(err) < self.EPSILON] = 0;
            err = (err ** 2).mean()
            # Add error
            if verbose:
                print "%r -> %r (%.2f)" % (i, output, err)
            rmse += err 

        score = 1/(1+np.sqrt(rmse / len(pairs)))
        return {'fitness': score}
        
    def solve(self, network):
        return int(self.evaluate(network) > 0.9)
                
        