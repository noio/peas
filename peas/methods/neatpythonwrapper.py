""" This module contains a wrapper for the NEAT implementation
    called "neat-python", found at http://code.google.com/p/neat-python/
    
    It wraps the package's global config in object oriented
    variables. 
"""

### IMPORTS ###
from __future__ import absolute_import
import sys
import os
import pickle

import numpy as np

try:
    from neat import config, chromosome, genome, population
except ImportError:
    print "This module requires neat-python to be installed,\n"\
          "you can get it at http://code.google.com/p/neat-python/."
    raise

class NEATPythonPopulation(object):
    """ A wrapper class for python-neat's Population
    """
    def __init__(self, 
                 popsize = 200,
                 
                 input_nodes     = 2,
                 output_nodes    = 1,
                 fully_connected = 1,
                 w_range         = (-30, 30),
                 feedforward     = False,
                 nn_activation   = 'exp',
                 hidden_nodes    = 0,
                 weight_stdev    = 2.0,

                 prob_addconn          = 0.05,
                 prob_addnode          = 0.03,
                 prob_mutatebias       = 0.2,
                 bias_mutation_power   = 0.5,
                 prob_mutate_weight    = 0.9,
                 weight_mutation_power = 1.5,
                 prob_togglelink       = 0.01,
                 elitism               = 1,

                 compatibility_threshold = 3.0,
                 compatibility_change    = 0.0,
                 excess_coeficient       = 1.0,
                 disjoint_coeficient     = 1.0,
                 weight_coeficient       = 0.4,

                 species_size        = 10,
                 survival_threshold  = 0.2,
                 old_threshold       = 30,
                 youth_threshold     = 10,
                 old_penalty         = 0.2,
                 youth_boost         = 1.2,
                 max_stagnation      = 15,
                
                 stop_when_solved=False,
                 verbose=True):

        # Set config
        self.config = dict(
            pop_size = popsize,
            input_nodes = input_nodes,
            output_nodes = output_nodes,
            fully_connected = fully_connected,
            min_weight = w_range[0], max_weight = w_range[1],
            feedforward = feedforward,
            nn_activation = nn_activation,
            hidden_nodes = hidden_nodes,
            weight_stdev = weight_stdev,
            
            prob_addconn = prob_addconn,
            prob_addnode = prob_addnode,
            prob_mutatebias = prob_mutatebias,
            bias_mutation_power = bias_mutation_power,
            prob_mutate_weight = prob_mutate_weight,
            weight_mutation_power = weight_mutation_power,
            prob_togglelink = prob_togglelink,
            elitism = elitism,
            
            compatibility_threshold = compatibility_threshold,
            compatibility_change = compatibility_change,
            excess_coeficient = excess_coeficient,
            disjoint_coeficient = disjoint_coeficient,
            weight_coeficient = weight_coeficient,
            species_size = species_size,
            survival_threshold = survival_threshold,
            old_threshold = old_threshold,
            youth_threshold = youth_threshold,
            old_penalty = old_penalty,
            youth_boost = youth_boost,
            max_stagnation = max_stagnation
        )
        
        self.stop_when_solved = stop_when_solved
        self.verbose   = verbose                
        
    def epoch(self, evaluator, generations, solution=None):
        # Set config
        chromosome.node_gene_type = genome.NodeGene
        for k, v in self.config.iteritems():
            setattr(config.Config, k, v)

        # neat-python has a max fitness threshold, we can set it if
        # we want to stop the simulation there, otherwise set it to some
        # really large number
        if isinstance(solution, (int, float)) and self.stop_when_solved:
            config.Config.max_fitness_threshold = solution
        else:
            config.Config.max_fitness_threshold = sys.float_info.max
        
        self.pop = population.Population()
        
        def evaluate_all(population):
            """ Adapter for python-neat, which expects a function that
                evaluates all individuals and assigns a .fitness property
            """
            for individual in population:
                individual.fitness = evaluator(individual)
            return [individual.fitness for individual in population]
        
        population.Population.evaluate = evaluate_all
        self.pop.epoch(generations, report=self.verbose, save_best=True, checkpoint_interval=None)
        # Find the timestep when the problem was solved
        i = 0        
        self.champions = []
        while os.path.exists('best_chromo_%d' % (i)):
            # Load the champions and delete them
            f = open('best_chromo_%d' % (i), 'rb')
            self.champions.append(pickle.load(f))
            f.close()
            os.remove('best_chromo_%d' % (i))
            # Check if champion solves problem
            solved = False
            if solution is not None:
                if isinstance(solution, (int, float)):
                    solved = (self.champions[-1].neat_fitness >= solution)
                elif callable(solution):
                    solved = solution(self.champions[-1])
                elif hasattr(solution, 'solve'):
                    solved = solution.solve(self.champions[-1])
                else:
                    raise Exception("Solution checker must be a threshold fitness value,"\
                                    "a callable, or an object with a method 'solve'.")
            self.champions[-1].solved = solved
            i += 1
        
        self.stats = {}
        self.stats['fitness_max'] = np.array([individual.fitness for individual in self.champions])
        self.stats['fitness_avg'] = self.pop.stats[1]
        self.stats['solved'] = np.array([individual.solved for individual in self.champions])
        
        return {'stats': self.stats, 'champions': self.champions}
