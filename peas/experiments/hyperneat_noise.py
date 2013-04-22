#!/usr/bin/env python

### IMPORTS ###
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np

# Local
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.methods.reaction import ReactionDeveloper
from peas.methods.evolution import SimplePopulation
from peas.methods.wavelets import WaveletGenotype, WaveletDeveloper
from peas.networks.rnn import NeuralNetwork, gauss
from peas.tasks.targetweights import TargetWeightsTask


def evaluate(individual, task, developer):
    stats = task.evaluate(developer.convert(individual))
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    elif isinstance(individual, WaveletGenotype):
        stats['nodes'] = sum(len(w) for w in individual.wavelets)
    return stats

def run(method, level, generations=500, popsize=500, visualize_individual=None):
                
    shape = (3,3)
    task = TargetWeightsTask(substrate_shape=shape, noise=level, fitnessmeasure='sqerr')
        
    substrate = Substrate()
    substrate.add_nodes(shape, 'l')
    substrate.add_connections('l', 'l')
    
    if method == 'hyperneat':
        geno = lambda: NEATGenotype(feedforward=True, inputs=len(shape)*2, weight_range=(-3.0, 3.0), 
                                       prob_add_conn=0.3, prob_add_node=0.03,
                                       types=['sin', 'linear', 'gauss', 'sigmoid', 'abs'])
                                   
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False)
        
    elif method == '0hn':
        
        t = [(i, 4) for i in range(4)]
        geno = lambda: NEATGenotype(feedforward=True, inputs=len(shape)*2, weight_range=(-3.0, 3.0), 
                                       prob_add_conn=0.0, prob_add_node=0.00, topology=t,
                                       types=['sin', 'linear', 'gauss', 'sigmoid', 'abs'])
                                   
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False)
        

    elif method == 'wavelet':
        geno = lambda: WaveletGenotype(inputs=len(shape)*2)
        pop = SimplePopulation(geno, popsize=popsize)
        developer = WaveletDeveloper(substrate=substrate, add_deltas=False, sandwich=False)
    

    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer)
                        )
    return results
    
if __name__ == '__main__':
    for method in ['0hn', 'wavelet', 'hyperneat']:
        for level in np.linspace(0, 1, 11):
            run(method, level)
