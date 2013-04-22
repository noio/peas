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

def area(coords, axis, offset):
    return coords[0] * axis[0] + coords[1] * axis[1] > offset
    
def split(coords, axis, flip, distance):
    return coords[axis] * flip >= distance
    
def slope(coords, offset, axis):
    return coords[0] * axis[0] + coords[1] * axis[1]
    
def random_direction_vector():
    theta = np.random.random() * np.pi*2
    return np.array([np.cos(theta), np.sin(theta)])

def run(method, splits, generations=500, popsize=500):
    complexity = 'half'
    splits = int(splits)
    
    funcs = []
    
    if complexity in ['half', 'flat', 'slope']:
        funcs.append((True, np.random.random() * 6. - 3))        
     
    for num in range(splits):
        axis = random_direction_vector()
        offset = np.random.random() - 0.2
        where = partial(area, axis=axis, offset=offset)
        
        if complexity == 'half':
            
            xs = 0 if num % 2 == 0 else 1
            mp = 1 if (num//2) % 2 == 0 else -1
            if num < 2:
                d = 0 
            elif num < 2 + 4:
                d = 0.5
            elif num < 2 + 4 + 4:
                d = 0.25
            elif num < 2 + 4 + 4 + 4:
                d = 0.75
            
            where = partial(split, axis=xs, flip=mp, distance=d)
            what = lambda c, v: v + np.random.random() * 6. - 3
                    
        funcs.append((where, what))
                
    task = TargetWeightsTask(substrate_shape=(8,), funcs=funcs, fitnessmeasure='sqerr', uniquefy=True)
        
    substrate = Substrate()
    substrate.add_nodes((8,), 'l')
    substrate.add_connections('l', 'l')
    
    if method == 'hyperneat':
        geno = lambda: NEATGenotype(feedforward=True, inputs=2, weight_range=(-3.0, 3.0), 
                                       prob_add_conn=0.3, prob_add_node=0.03,
                                       types=['sin', 'ident', 'gauss', 'sigmoid', 'abs'])
                                   
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False)
        
        
    elif method == '0hnmax':
        geno = lambda: NEATGenotype(feedforward=True, inputs=2, weight_range=(-3.0, 3.0), 
                                    max_nodes=3,
                                    types=['sin', 'ident', 'gauss', 'sigmoid', 'abs'])
                                   
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False)
        

    elif method == 'wavelet':
        geno = lambda: WaveletGenotype(inputs=2)
        pop = SimplePopulation(geno, popsize=popsize)
        developer = WaveletDeveloper(substrate=substrate, add_deltas=False, sandwich=False)
    
    
    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        )
    
    return results

if __name__ == '__main__':
    for method in ['hyperneat', 'wavelet', '0hnmax']:
        for splits in range(15):
            run(method, splits)
    