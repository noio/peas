#!/usr/bin/env python

### IMPORTS ###
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
np.seterr(invalid='raise')

# Local
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.evolution import SimplePopulation
from peas.methods.wavelets import WaveletGenotype, WaveletDeveloper
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.tasks.linefollowing import LineFollowingTask


def evaluate(individual, task, developer):
    phenotype = developer.convert(individual)
    stats = task.evaluate(phenotype)
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    elif isinstance(individual, WaveletGenotype):
        stats['nodes'] = sum(len(w) for w in individual.wavelets)
    print '~',
    sys.stdout.flush()
    return stats
    
def solve(individual, task, developer):
    phenotype = developer.convert(individual)
    return task.solve(phenotype)


### SETUPS ###    
def run(method, setup, generations=100, popsize=100):
    """ Use hyperneat for a walking gait task
    """
    # Create task and genotype->phenotype converter
    

    if setup == 'easy':
        task_kwds = dict(field='eight',
                         observation='eight',
                         max_steps=3000,
                         friction_scale=0.3,
                         damping=0.3,
                         motor_torque=10,
                         check_coverage=False,
                         flush_each_step=False,
                         initial_pos=(282, 300, np.pi*0.35))

    elif setup == 'hard':
        task_kwds = dict(field='eight',
                         observation='eight_striped',
                         max_steps=3000,
                         friction_scale=0.3,
                         damping=0.3,
                         motor_torque=10,
                         check_coverage=False,
                         flush_each_step=True,
                         force_global=True,
                         initial_pos=(282, 300, np.pi*0.35))

    elif setup == 'force':
        task_kwds = dict(field='eight',
                         observation='eight',
                         max_steps=3000,
                         friction_scale=0.1,
                         damping=0.9,
                         motor_torque=3,
                         check_coverage=True,
                         flush_each_step=True,
                         force_global=True,
                         initial_pos=(17, 256, np.pi*0.5))

    elif setup == 'prop':
        task_kwds = dict(field='eight',
                         observation='eight_striped',
                         max_steps=3000,
                         friction_scale=0.3,
                         damping=0.3,
                         motor_torque=10,
                         check_coverage=False,
                         flush_each_step=False,
                         initial_pos=(282, 300, np.pi*0.35))
    
    elif setup == 'cover':
        task_kwds = dict(field='eight',
                         observation='eight_striped',
                         max_steps=3000,
                         friction_scale=0.1,
                         damping=0.9,
                         motor_torque=3,
                         check_coverage=True,
                         flush_each_step=False,
                         initial_pos=(17, 256, np.pi*0.5))
                 
    task = LineFollowingTask(**task_kwds)

    # The line following experiment has quite a specific topology for its network:    
    substrate = Substrate()
    substrate.add_nodes([(0,0)], 'bias')
    substrate.add_nodes([(r, theta) for r in np.linspace(0,1,3)
                              for theta in np.linspace(-1, 1, 5)], 'input')
    substrate.add_nodes([(r, theta) for r in np.linspace(0,1,3)
                              for theta in np.linspace(-1, 1, 3)], 'layer')
    substrate.add_connections('input', 'layer',-1)
    substrate.add_connections('bias', 'layer', -2)
    substrate.add_connections('layer', 'layer',-3)
        
    if method == 'wvl':
        geno = lambda: WaveletGenotype(inputs=4, layers=3)
        pop = SimplePopulation(geno, popsize=popsize)
        developer = WaveletDeveloper(substrate=substrate, 
                                     add_deltas=False, 
                                     sandwich=False,
                                     node_type='tanh')
                
    else:
        geno_kwds = dict(feedforward=True, 
                         inputs=4,
                         outputs=3,
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
                                       add_deltas=False,
                                       sandwich=False,
                                       node_type='tanh')
                        
    
    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve, task=task, developer=developer), 
                        )

    return results

if __name__ == '__main__':
	# Method is one of METHOD = ['wvl', 'nhn', '0hnmax', '1hnmax']
    resnhn = run('nhn', 'hard')
