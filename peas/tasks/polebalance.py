""" Implementation of the standard (multiple) pole
    balancing task.
"""

### IMPORTS ###
import random

# Libraries
import numpy as np

# Local
from ..networks.rnn import NeuralNetwork

class PoleBalanceTask(object):
    """ Double pole balancing task. 
    """
    
    def __init__(self, gravity=-9.81, cart_mass=1.0, 
                       pole_mass=[0.1, 0.01], 
                       pole_length=[0.5, 0.05],
                       track_limit=2.4,
                       failure_angle=0.628,
                       timestep=0.01,
                       force_magnitude=10.0,
                       start_random=False,
                       penalize_oscillation=True,
                       velocities=False,
                       max_steps=1000):
        """ Constructor for PoleBalanceTask 
        
            :param gravity:         Gravity constant.
            :param cartmass:        Mass of the cart.
            :param pole_mass:       List of the mass of each pole.
            :param pole_length:     List of the length of each pole, respectively.
            :param track_limit:     Length of the track that the cart is on.
            :param failure_angle:   Pole angle in radians at which the task is failed.
            :param timestep:        Length of each time step.
            :param force_magnitude: The force that is exerted when an action of 1.0 is performed.
            :param start_random:    Set the pole to a random position on each trial.
            :param penalize_oscillation: Uses the alternative fitness function described in (Stanley, 2002).
            :param velocities:      Known velocities make the task markovian.
            :param max_steps:       Maximum length of the simulation.
        """
        
        self.g  = gravity
        self.mc = cart_mass
        self.mp = np.array(pole_mass)
        self.l  = np.array(pole_length)
        self.h  = track_limit
        self.r  = failure_angle
        self.f  = force_magnitude
        self.t  = timestep
        self.velocities           = velocities
        self.start_random         = start_random
        self.penalize_oscillation = penalize_oscillation
        self.max_steps            = max_steps

    def _step(self, action, state):
        """ Performs a single simulation step. 
            The state is a tuple of (x, dx, (p1, p2), (dp1, dp2)), 
        """
        x, dx, theta, dtheta = state
        
        f = (min(1.0, max(-1.0, action)) - 0.5) * self.f * 2.0;
        
        # Alternate equations
        fi = self.mp * self.l * dtheta**2 * np.sin(theta) + (3.0/4) * self.mp * np.cos(theta) * self.g * np.sin(theta)
        mi = self.mp * (1 - (3.0/4) * np.cos(theta)**2)
        ddx = f + np.sum(fi) / (self.mc + np.sum(mi))
        ddtheta = (- 3.0 / (4 * self.l)) * (ddx * np.cos(theta) + self.g * np.sin(theta))
     
        # Equations from "THE POLE BALANCING PROBLEM"
        # _ni = (-f - self.mp * self.l * dtheta**2 * np.sin(theta))
        # m = self.mc + np.sum(self.mp)
        # _n = self.g * np.sin(theta) + np.cos(theta) * (_ni / m)
        # _d = self.l * (4./3. - (self.mp * np.cos(theta)**2) / m)
        # ddtheta = (_n / _d)
        # ddx = (f + np.sum(self.mp * self.l * np.floor(dtheta**2 * np.sin(theta) - ddtheta * np.cos(theta)))) / m
        
        x += self.t * dx
        dx += self.t * ddx        
        theta += self.t * dtheta
        dtheta += self.t * ddtheta
        
        return (x, dx, theta, dtheta)

    def _loop(self, network, max_steps, initial=None, verbose=False):
        if initial is None:
            x, dx  = 0.0, 0.0
            if self.start_random:
                theta = np.random.normal(0, 0.01, self.l.size)
            else:
                # Long pole starts at a fixed 1 degree angle.
                theta = np.array([0.017, 0.0])
            dtheta = np.zeros(self.l.size)
        else:
            (x, dx, theta, dtheta) = initial
        steps  = 0
        states = []
        actions = []
        while (steps < max_steps and
               np.abs(x) < self.h and
               (np.abs(theta) < self.r).all()):
            steps += 1
            if self.velocities:
                # Divide velocities by 2.0 because that is what neat-python does
                net_input = np.hstack((x/self.h, dx/2.0, theta/self.r, dtheta/2.0))
            else:
                net_input = np.hstack((x/self.h, theta/self.r))
            action = (network.feed( net_input )[-1] + 1) * 0.5
            (x, dx, theta, dtheta) = self._step(action, (x, dx, theta, dtheta))
            actions.append(action)
            states.append((x, dx, theta.copy(), dtheta.copy()))
            if verbose:
                print states[-1]
        
        return steps, states, actions
        
    def evaluate(self, network, verbose=False):
        """ Perform a single run of this task """
        # Convert to a network if it is not.

        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        steps, states, _ = self._loop(network, max_steps=self.max_steps, verbose=verbose)
        
        if network.node_types[-1].__name__ != 'tanh':
            raise Exception("Network output must have range [-1, 1]") 
        
        if self.penalize_oscillation:
            """ This bit implements the fitness function as described in
                Stanley - Evolving Neural Networks through Augmenting Topologies
            """
            f1 = steps/float(self.max_steps)
            if steps < 100:
                f2 = 0
            else:
                wiggle = sum(abs(x) + abs(dx) + abs(t[0]) + abs(dt[0]) for 
                                    (x, dx, t, dt) in states[-100:])
                wiggle = max(wiggle, 0.01) # Cap the wiggle bonus
                f2 = 0.75 / wiggle
            score = 0.1 * f1 + 0.9 * f2
        else:
            """ This is just number of steps without falling.
            """
            score = steps/float(self.max_steps)
            
        return {'fitness': score, 'steps': steps}

    def solve(self, network):
        """ This function should measure whether the network passes some
            threshold of "solving" the task. Returns False/0 if the 
            network 'fails' the task. 
        """
        # Convert to a network if it is not.
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        steps, _, _ = self._loop(network, max_steps=100000)
        if steps < 100000:
            print "Failed 100k test with %d" % steps
            return 0
        successes = 0
        points = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
        # The 1.35 and 0.15 were taken from the neat-python implementation.
        for x in points * self.h:
            for theta in points * self.r:
                for dx in points * 1.35:
                    for dtheta0 in points * 0.15:
                        state = (x, dx, np.array([theta, 0.0]), np.array([dtheta0, 0.0]))
                        steps, states, _ = self._loop(network, initial=state, max_steps=1000)
                        if steps >= 1000:
                            successes += 1
        # return random.random() < 0.5
        return int(successes > 100)
        
    def visualize(self, network, f):
        """ Visualize a solution strategy by the given individual
        """
        import matplotlib
        matplotlib.use('Agg',warn=False)
        import matplotlib.pyplot as plt
        # Convert to a network if it is not.
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        fig = plt.figure()
        steps, states, actions = self._loop(network, max_steps=1000)
        # TEMP STUFF
        actions = np.array(actions)
        print actions.size, np.histogram(actions)[0]
        ##
        x, dx, theta, dtheta = zip(*states)
        theta = np.vstack(theta).T
        dtheta = np.vstack(dtheta).T
        # The top plot (cart position)
        top = fig.add_subplot(211)
        top.fill_between(range(len(x)), -self.h, self.h, facecolor='green', alpha=0.3)
        top.plot(x, label=r'$x$')        
        top.plot(dx, label=r'$\delta x$')
        top.legend(loc='lower left', ncol=4, fancybox=True, bbox_to_anchor=(0, 0, 1, 1))
        # The bottom plot (pole angles)
        bottom = fig.add_subplot(212)
        bottom.fill_between(range(theta.shape[1]), -self.r, self.r, facecolor='green', alpha=0.3)
        for i, (t, dt) in enumerate(zip(theta, dtheta)):
            bottom.plot(t, label=r'$\theta_%d$'%i)
            bottom.plot(dt, ls='--', label=r'$\delta \theta_%d$'%i)
        bottom.legend(loc='lower left', ncol=4, fancybox=True, bbox_to_anchor=(0, 0, 1, 1))
        fig.savefig(f)
    
    def __str__(self):
        vel = ('with' if self.velocities else 'without') + ' velocities'
        str = ('random' if self.start_random else 'fixed') + ' starts'
        r = '[%s] %s, %s' % (self.__class__.__name__, vel, str)
        return r
        
        
if __name__ == '__main__':
    t = PoleBalanceTask()
    
    x, dx = 0.0, 0.0
    theta = np.array([0.017, 0])
    dtheta = np.array([0.0, 0])
    
    while (np.abs(theta) < t.r).all():
        (x, dx, theta, dtheta) = t._step(0.5, (x, dx, theta, dtheta))
        print theta
        
        
