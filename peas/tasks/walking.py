#! /usr/bin/python

""" Walking gait task.
"""

### IMPORTS ###
import random

# Libraries
import pymunk
import numpy as np

# Local

class WalkingTask(object):
    """ Walking gait task.
    """
    
    def __init__(self, maxsteps=6000, tracklength=1000):
        self.maxsteps = maxsteps
        self.tracklength = tracklength
        
    def makeleg(self, parent, position=(0,20), l=40, w=10):
        # Upperleg
        upperleg = pymunk.Body(10.0, pymunk.moment_for_box(5, l, w))
        upperleg.position = pymunk.Vec2d(parent.position) + pymunk.Vec2d(position) + pymunk.Vec2d(0, l/2.0 - w/2.0)
        shape = pymunk.Poly.create_box(upperleg, (w,l))
        shape.group = 1
        hip = pymunk.PivotJoint(parent, upperleg, pymunk.Vec2d(parent.position) + pymunk.Vec2d(position))
        hipmotor = pymunk.SimpleMotor(parent, upperleg, -1)
        hipmotor.max_force = 1e7
        self.space.add(hip, hipmotor, upperleg, shape)
        # Lower leg
        l *= 1.2
        lowerleg = pymunk.Body(10.0, pymunk.moment_for_box(5, l, w))
        lowerleg.position = pymunk.Vec2d(upperleg.position) + pymunk.Vec2d(0, l - w/2.0)
        shape = pymunk.Poly.create_box(lowerleg, (w,l))
        shape.group = 1
        shape.friction = 2.0
        knee = pymunk.PivotJoint(upperleg, lowerleg, pymunk.Vec2d(upperleg.position) + pymunk.Vec2d(0, l/2.0 - w/2.0))
        kneemotor = pymunk.SimpleMotor(upperleg, lowerleg, -2)
        kneemotor.max_force = 1e7
        self.space.add(knee, kneemotor, lowerleg, shape)

        
    def evaluate(self, network, draw=False):
        if draw:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((self.tracklength, 600))
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
        floor.position = pymunk.Vec2d(self.tracklength/2.0 , 500.0)
        sfloor = pymunk.Poly.create_box(floor, (self.tracklength, 5))
        sfloor.friction = 1.0
        sfloor.collision_type = 1
        space.add_static(sfloor)

        # Torso
        torso = pymunk.Body(10.0, pymunk.moment_for_box(10, 40, 50))
        torso.position = pymunk.Vec2d(20.0,380.0)
        storso = pymunk.Poly.create_box(torso, (40, 50))
        storso.group = 1
        storso.collision_type = 1
        # space.add_static(storso)
        space.add(torso, storso)

        # Legs
        self.makeleg(torso)
        self.makeleg(torso)
        
        # Collision callback
        def oncollide(space, arb):
            self.touching_floor = True
        space.add_collision_handler(1, 1, post_solve=oncollide)
        
        for step in xrange(self.maxsteps):
            
            # Query network
            torso_x = torso.position.x
            torso_y = torso.position.y
            output = network.feed(np.array([torso_x, torso_y]))
            
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
        print "Travelled %.2f in %d steps." % (distance, step)
        
    def solve(self, network):
        return False
        
    def visualize(self, network, filename=None):
        """ Visualize a solution strategy by the given individual. """
        self.evaluate(network, draw=True)
        
        
    