#!/usr/bin/env python

### IMPORTS ###
import sys, os
from functools import partial

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
# from peas.methods.neatpythonwrapper import NEATPythonPopulation
from peas.tasks.polebalance import PoleBalanceTask

# Create a factory for genotypes (i.e. a function that returns a new 
# instance each time it is called)
genotype = lambda: NEATGenotype(inputs=6, 
                                weight_range=(-50., 50.), 
                                types=['tanh'])

# Create a population
pop = NEATPopulation(genotype, popsize=150)
    
# Create a task
dpnv = PoleBalanceTask(velocities=True, 
                       max_steps=100000, 
                       penalize_oscillation=True)

# Run the evolution, tell it to use the task as an evaluator
pop.epoch(generations=100, evaluator=dpnv, solution=dpnv)
