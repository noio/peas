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
from collections import defaultdict

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
                       elitism = True,
                       stop_when_solved=False, 
                       tournament_selection_k=3,
                       verbose=True):
        # Instance properties
        self.geno_factory           = geno_factory
        self.popsize                = popsize
        self.elitism                = elitism
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
        self.stats = defaultdict(list)
                
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
        """ Runs a single step of evolution.
        """
        
        pop = self._birth()
        self._evaluate_all(pop, evaluator)
        self._find_best(pop, solution) 
        pop = self._reproduce(pop)        
        self._gather_stats(pop)
        
        if self.verbose:
            self._status_report(pop)
            
        if callback is not None:
            callback(self)
        
        self.population = pop
        self.generation += 1

    def _birth(self):
        """ Creates a population if there is none, returns
            current population otherwise.
        """
        while len(self.population) < self.popsize:
            individual = self.geno_factory()
            self.population.append(individual)
        
        return self.population
        
    def _evaluate_all(self, pop, evaluator):
        """ Evaluates all of the individuals in given pop,
            and assigns their "stats" property.
        """
        for individual in pop:
            if callable(evaluator):
                individual.stats = evaluator(individual)
            elif hasattr(evaluator, 'evaluate'):
                individual.stats = evaluator.evaluate(individual)
            else:
                raise Exception("Evaluator must be a callable or object" \
                                "with a callable attribute 'evaluate'.")
            if self.verbose:
                sys.stdout.write('#')
                sys.stdout.flush()
    
    def _find_best(self, pop, solution=None):
        """ Finds the best individual, and adds it to the champions, also 
            checks if this best individual 'solves' the problem.
        """
        ## CHAMPION
        self.champions.append(max(pop, key=lambda ind: ind.stats['fitness']))
        
        ## SOLUTION CRITERION
        if solution is not None:
            if isinstance(solution, (int, float)):
                solved = (self.champions[-1].stats['fitness'] >= solution)
            elif callable(solution):
                solved = solution(self.champions[-1])
            elif hasattr(solution, 'solve'):
                solved = solution.solve(self.champions[-1])
            else:
                raise Exception("Solution checker must be a threshold fitness value,"\
                                "a callable, or an object with a method 'solve'.")
            if solved and self.solved_at is None:
                self.solved_at = self.generation
                
    def _reproduce(self, pop):
        """ Reproduces (and mutates) the best individuals to create a new population.
        """
        newpop = []

        if self.elitism:
            newpop.append(self.champions[-1])
            
        while len(newpop) < self.popsize:
            # Perform tournament selection
            k = min(self.tournament_selection_k, len(pop))
            winner = max(random.sample(pop, k), key=lambda ind:ind.stats['fitness'])
            winner = deepcopy(winner).mutate()
            newpop.append(winner)
            
        return newpop
        
    def _gather_stats(self, pop):
        """ Collects avg and max of individuals' stats (incl. fitness).
        """
        for key in pop[0].stats:
            self.stats[key+'_avg'].append(np.mean([ind.stats[key] for ind in pop]))
            self.stats[key+'_max'].append(np.max([ind.stats[key] for ind in pop]))
        self.stats['solved'].append( self.solved_at is not None )
        
    def _status_report(self, pop):
        """ Prints a status report """
        print "\n== Generation %d ==" % self.generation
        print "Best (%.2f): %s" % (self.champions[-1].stats['fitness'], self.champions[-1])
        print "Solved: %s" % (self.solved_at)
        