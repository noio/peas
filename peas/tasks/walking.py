#! /usr/bin/python

""" Walking gait task.
"""

### IMPORTS ###
import random

# Libraries
import pymunk
import numpy as np

# Local
from ..networks.rnn import NeuralNetwork

# Shortcuts
pi = np.pi


class WalkingTask(object):
    """ Walking gait task.
    """
    
    def __init__(self, maxsteps=6000, tracklength=1000, maxrate=2.0, torsoheight=80):
        self.maxsteps = maxsteps
        self.tracklength = tracklength
        self.maxrate = maxrate
        self.torsoheight = torsoheight
        
    def makeleg(self, parent, position=(0,20), l=40, w=10):
        # Upperleg
        upperleg = pymunk.Body(10.0, pymunk.moment_for_box(5, l, w))
        upperleg.position = pymunk.Vec2d(parent.position) + pymunk.Vec2d(position) + pymunk.Vec2d(0, l/2.0 - w/2.0)
        shape = pymunk.Poly.create_box(upperleg, (w,l))
        shape.group = 1
        shape.friction = 2.0
        hip = pymunk.PivotJoint(parent, upperleg, pymunk.Vec2d(parent.position) + pymunk.Vec2d(position))
        hipmotor = pymunk.SimpleMotor(parent, upperleg, 0)
        hiplimit = pymunk.RotaryLimitJoint(parent, upperleg, -0.9*pi, 0.1*pi)
        hipmotor.max_force = 1e7
        self.space.add(hip, hipmotor, hiplimit, upperleg, shape)
        # Lower leg
        l *= 1.2
        lowerleg = pymunk.Body(10.0, pymunk.moment_for_box(5, l, w))
        lowerleg.position = pymunk.Vec2d(upperleg.position) + pymunk.Vec2d(0, l - w/2.0)
        shape = pymunk.Poly.create_box(lowerleg, (w,l))
        shape.group = 1
        shape.friction = 2.0
        knee = pymunk.PivotJoint(upperleg, lowerleg, pymunk.Vec2d(upperleg.position) + pymunk.Vec2d(0, l/2.0 - w/2.0))
        kneemotor = pymunk.SimpleMotor(upperleg, lowerleg, 0)
        kneelimit = pymunk.RotaryLimitJoint(upperleg, lowerleg, -0.1*pi, 0.9*pi)
        kneemotor.max_force = 1e7
        self.space.add(knee, kneemotor, kneelimit, lowerleg, shape)
        return (hipmotor, kneemotor), (upperleg, lowerleg)

        
    def evaluate(self, network, draw=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        if draw:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((self.tracklength, 200))
            pygame.display.set_caption("Simulation")
            clock = pygame.time.Clock()
            running = True
            font = pygame.font.Font(pygame.font.get_default_font(), 8)
        
        # Initialize pymunk
        self.space = space = pymunk.Space()
        space.gravity = (0.0, 900.0)
        space.damping = 0.9
        self.touching_floor = False
        # Create objects
        # Floor

        floor = pymunk.Body()
        floor.position = pymunk.Vec2d(self.tracklength/2.0 , 200)
        sfloor = pymunk.Poly.create_box(floor, (self.tracklength, 5))
        sfloor.friction = 1.0
        sfloor.collision_type = 1
        space.add_static(sfloor)

        # Torso
        torso = pymunk.Body(10.0, pymunk.moment_for_box(10, 40, self.torsoheight))
        torso.position = pymunk.Vec2d(20.0, self.torsoheight)
        storso = pymunk.Poly.create_box(torso, (40, self.torsoheight))
        storso.group = 1
        storso.collision_type = 1
        storso.friction = 2.0
        # space.add_static(storso)
        space.add(torso, storso)

        # Legs
        (hip1, knee1), leg1shapes = self.makeleg(torso)
        (hip2, knee2), leg2shapes = self.makeleg(torso)
        
        # Collision callback
        def oncollide(space, arb):
            self.touching_floor = True
        space.add_collision_handler(1, 1, post_solve=oncollide)
        
        for step in xrange(self.maxsteps):
            
            # Query network
            torso_y = torso.position.y
            torso_a = torso.angle
            leg1_a = leg1shapes[0].angle
            leg2_a = leg2shapes[0].angle
            output = network.feed(np.array([torso_y, torso_a, leg1_a, leg2_a]))
            
            output = np.clip(output[-4:] * self.maxrate, -self.maxrate, self.maxrate)
            hip1.rate = output[0]
            hip2.rate = output[1]
            knee1.rate = output[2]
            knee2.rate = output[3]
            
            # Advance simulation
            space.step(1/50.0)
            # Check for success/failure
            if torso.position.x < 0:
                break
            if torso.position.x > self.tracklength - 50:
                break
            if self.touching_floor:
                break

            # Draw
            if draw:
                print output
                # Clear
                screen.fill((255, 255, 255))
                # Do all drawing
                txt = font.render('%d' % step, False, (0,0,0) )
                screen.blit(txt, (0,0))
                # Draw objects
                for o in space.shapes + space.static_shapes:
                    if isinstance(o, pymunk.Circle):
                        pygame.draw.circle(screen, (0,0,0), (int(o.body.position.x), int(o.body.position.y)), int(o.radius))
                    else:
                        pygame.draw.lines(screen, (0,0,0), True, [(int(p.x), int(p.y)) for p in o.get_points()])
                # Flip buffers
                pygame.display.flip()
                # clock.tick(50)
                
        if draw:
            pygame.quit()
        
        distance = torso.position.x
        # print "Travelled %.2f in %d steps." % (distance, step)
        return distance
        
    def solve(self, network):
        return False
        
    def visualize(self, network, filename=None):
        """ Visualize a solution strategy by the given individual. """
        self.evaluate(network, draw=True)
        
        
    