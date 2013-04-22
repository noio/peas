#!/usr/bin/env python

### IMPORTS ###
import sys, os
from functools import partial
from itertools import product

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.methods.reaction import ReactionDeveloper
from peas.methods.evolution import SimplePopulation
from peas.methods.wavelets import WaveletGenotype, WaveletDeveloper
from peas.tasks.shapediscrimination import ShapeDiscriminationTask
from peas.networks.rnn import NeuralNetwork

# Libs
import numpy as np

### SETUPS ###    

def evaluate(individual, task, developer):
    stats = task.evaluate(developer.convert(individual))
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    elif isinstance(individual, WaveletGenotype):
        stats['nodes'] = sum(len(w) for w in individual.wavelets)
    return stats
    
def solve(individual, task, developer):
    return task.solve(developer.convert(individual))

def run(method, setup, generations=250, popsize=100):
    # Create task and genotype->phenotype converter
    size = 11
    task_kwds = dict(size=size)
            
    if setup == 'big-little':
        task_kwds['targetshape'] = ShapeDiscriminationTask.makeshape('box', size//3)
        task_kwds['distractorshapes'] = [ShapeDiscriminationTask.makeshape('box', 1)]
    elif setup == 'triup-down':
        task_kwds['targetshape'] = np.triu(np.ones((size//3, size//3)))
        task_kwds['distractorshapes'] = [np.tril(np.ones((size//3, size//3)))]
        
    task = ShapeDiscriminationTask(**task_kwds)

    substrate = Substrate()
    substrate.add_nodes((size, size), 'l')
    substrate.add_connections('l', 'l')
    
    if method == 'wavelet':
        num_inputs = 6 if deltas else 4
        geno = lambda: WaveletGenotype(inputs=num_inputs)
        pop = SimplePopulation(geno, popsize=popsize)
        developer = WaveletDeveloper(substrate=substrate, add_deltas=True, sandwich=True)
    
    else:
        geno_kwds = dict(feedforward=True, 
                         inputs=6, 
                         weight_range=(-3.0, 3.0), 
                         prob_add_conn=0.1, 
                         prob_add_node=0.03,
                         bias_as_node=False,
                         types=['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'])
    
        if method == 'nhn':
            pass
        elif method == '0hnmax':
            geno_kwds['max_nodes'] = 7
        elif method == '1hnmax':
            geno_kwds['max_nodes'] = 8
    
        geno = lambda: NEATGenotype(**geno_kwds)
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
    	
        developer = HyperNEATDeveloper(substrate=substrate, 
                                       sandwich=True, 
                                       add_deltas=True,
                                       node_type='tanh')
                               
        # Run and save results
    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve, task=task, developer=developer),
                        )    
    
    return results

if __name__ == '__main__':
    # Method is one of  ['wvl', 'nhn', '0hnmax', '1hnmax']
    # setup is one of ['big-little', 'triup-down']
	run('nhn', 'big-little')
	    

