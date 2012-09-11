""" This module implements the different genotypes and evolutionary
    methods used in NEAT and HyperNEAT. It is meant to be a reference
    implementation, so any inneficiencies are copied as they are
    described.
"""

### IMPORTS ###
import sys
import random
from copy import deepcopy
from itertools import product

# Libs
import numpy as np
np.seterr(over='warn', divide='raise')

# Package

# Shortcuts
rand = random.random
inf  = float('inf')


### CLASSES ###
                
class SimplePopulation(object):
    
    def __init__(self, geno_factory, 
                       popsize = 100, 
                       stop_when_solved=False, 
                       tournament_selection_k=3,
                       verbose=True):
        # Instance properties
        self.geno_factory           = geno_factory
        self.popsize                = popsize
        self.stop_when_solved       = stop_when_solved
        self.tournament_selection_k = tournament_selection_k
        self.verbose                = verbose

        
    def _reset(self):
        """ Resets the state of this population.
        """
        self.population      = [] # List of species
                
        # Keep track of some history
        self.champions  = []
        self.generation = 0
        self.solved_at  = None
        self.stats = {}
        self.stats['fitness_avg'] = []
        self.stats['fitness_max'] = []
        self.stats['solved']      = []        
        
    def epoch(self, evaluator, generations, solution=None, reset=True, callback=None):
        """ Runs an evolutionary epoch 

            :param evaluator:    Either a function or an object with a function
                                 named 'evaluate' that returns a given individual's
                                 fitness.
            :param callback:     Function that is called at the end of each generation.
        """
        if reset:
            self._reset()
        
        for _ in xrange(generations):
            self._evolve(evaluator, solution, callback)
            if self.solved_at is not None and self.stop_when_solved:
                break
                
        return {'stats': self.stats, 'champions': self.champions}


    def _evolve(self, evaluator, solution=None , callback=None):

        ## INITIAL BIRTH
        while len(self.population) < self.popsize:
            individual = self.geno_factory()
            individual.neat_species = 0
            self.population.append(individual)


        ## EVALUATE
        for individual in self.population:
            if callable(evaluator):
                individual.fitness = evaluator(individual)
            elif hasattr(evaluator, 'evaluate'):
                individual.fitness = evaluator.evaluate(individual)
            else:
                raise Exception("Evaluator must be a callable or object" \
                                "with a callable attribute 'evaluate'.")
            if self.verbose:
                sys.stdout.write('#')
                sys.stdout.flush()
                
        ## CHAMPION
        self.champions.append(max(self.population, key=lambda ind: ind.fitness))
        
        ## SOLUTION CRITERION
        if solution is not None:
            if isinstance(solution, (int, float)):
                solved = (self.champions[-1].fitness >= solution)
            elif callable(solution):
                solved = solution(self.champions[-1])
            elif hasattr(solution, 'solve'):
                solved = solution.solve(self.champions[-1])
            else:
                raise Exception("Solution checker must be a threshold fitness value,"\
                                "a callable, or an object with a method 'solve'.")
            if solved and self.solved_at is None:
                self.solved_at = self.generation
        
        
        newpop = []
        for _ in xrange(self.popsize):
            # Perform tournament selection
            k = min(self.tournament_selection_k, len(self.population))
            winner = max(random.sample(self.population, k), key=lambda ind:ind.fitness)
            winner = deepcopy(winner).mutate()
            newpop.append(winner)
        
        self.population = newpop
        
        ## STATS
        self.stats['fitness_avg'].append(np.mean([ind.fitness for ind in self.population]))
        self.stats['fitness_max'].append(self.champions[-1].fitness)
        self.stats['solved'].append( self.solved_at is not None )
        
        if self.verbose:
            print "\n== Generation %d ==" % self.generation
            print "Best (%.2f): %s" % (self.champions[-1].fitness, self.champions[-1])
            print "Solved: %s" % (self.solved_at)
            
        if callback is not None:
            callback(self)
        
        self.generation += 1
