#!/usr/bin/env python
""" Testing module for PEAS package

Running this script will test if the SOME of the modules are
working properly. 
"""

### IMPORTS

# Python Imports
import os
import sys
import unittest

# Libraries
import numpy as np

# Package
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.networks import rnn
from peas.methods import neat
from peas.tasks import xor

### CONSTANTS

### CLASSES

class TestPEAS(unittest.TestCase):
        
    def test_rnn(self):
        net = rnn.NeuralNetwork().from_matrix(np.array([[0,0,0],[0,0,0],[1,1,0]]))
        output = net.feed(np.array([1,1]), add_bias=False)
        self.assertEqual(output[-1], rnn.sigmoid(1 + 1))

    def test_neat(self):
        task = xor.XORTask()
        genotype = lambda: neat.NEATGenotype(inputs=2, types=['tanh'])
        pop = neat.NEATPopulation(genotype)
        pop.epoch(task, 3)

    def test_rbfneat(self):
        def evaluate(network):
            cm, nt = network.get_network_data()
            if nt[-1] == 'rbfgauss':
                net = rnn.NeuralNetwork(network)
                e1 = net.feed(np.array([0, 0]), add_bias=False)[-1]
                mid = net.feed(np.array([2, -2]), add_bias=False)[-1]
                e2 = net.feed(np.array([4, -4]), add_bias=False)[-1]
                return {'fitness': mid - (e1 + e2)}
            else:
                return {'fitness': 0}


        genotype = lambda: neat.NEATGenotype(inputs=2, types=['tanh', 'rbfgauss'], max_nodes=5)
        pop = neat.NEATPopulation(genotype)
        pop.epoch(evaluate, 20)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPEAS)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":   
    run_tests()
