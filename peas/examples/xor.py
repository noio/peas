#!/usr/bin/env python

### IMPORTS ###
import sys, os
from functools import partial
from collections import defaultdict

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
# from peas.methods.neatpythonwrapper import NEATPythonPopulation
from peas.tasks.xor import XORTask

# Create a factory for genotypes (i.e. a function that returns a new 
# instance each time it is called)
genotype = lambda: NEATGenotype(inputs=2, 
                                weight_range=(-3., 3.), 
                                types=['sigmoid2'])

# Create a population
pop = NEATPopulation(genotype, popsize=150)
    
# Create a task
task = XORTask()

nodecounts = defaultdict(int)

for i in xrange(100):
	# Run the evolution, tell it to use the task as an evaluator
	pop.epoch(generations=100, evaluator=task, solution=task)
	nodecounts[len(pop.champions[-1].node_genes)] += 1

print sorted(nodecounts.items())
