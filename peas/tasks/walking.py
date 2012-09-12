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

### FUNCTIONS ###

def angle_fix(theta):
    """ Fixes an angle to a value between -pi and pi.
        
        >>> angle_fix(-2*pi)
        0.0
    """
    return ((theta + pi) % (2*pi)) - pi
    
### CLASSES ###

class Joint(object):
    """ Joint object, contains pivot, motor and limit"""
    
    def __init__(self, a, b, position, range=(-pi, pi), max_rate=2.0):
        self.pivot = pymunk.PivotJoint(a, b, position)
        self.motor = pymunk.SimpleMotor(a, b, 0)
        self.motor.max_force = 1e7
        self.limit = pymunk.RotaryLimitJoint(a, b, range[0], range[1])
        self.limit.max_force = 1e1
        self.limit.max_bias = 0.5
        self.max_rate = max_rate
        
    def angle(self):
        return angle_fix(self.pivot.a.angle - self.pivot.b.angle)
        
    def set_target(self, target):
        """ Target is between 0 and 1, representing min and max angle. """
        cur = self.angle()
        tgt = angle_fix(target * (self.limit.max - self.limit.min) + self.limit.min)
        if tgt > cur + 0.1:
            self.motor.rate = self.max_rate
        elif tgt < cur - 0.1:
            self.motor.rate = -self.max_rate
        else: 
            self.motor.rate = 0
    
class Leg(object):
    """ Leg object, contains joints and shapes """
    
    def __init__(self, parent, position, walking_task):
        self.walking_task = walking_task
        (w, l) = walking_task.leg_length / 5.0, walking_task.leg_length
        mass = w * l * 0.2
        # Upper leg
        upperleg = pymunk.Body(mass, pymunk.moment_for_box(mass, w, l))
        upperleg.position = pymunk.Vec2d(parent.position) + pymunk.Vec2d(position) + pymunk.Vec2d(0, l/2.0 - w/2.0)
        shape = pymunk.Poly.create_box(upperleg, (w,l))
        shape.group = 1
        shape.friction = 2.0
        # Joints
        pos = pymunk.Vec2d(parent.position) + pymunk.Vec2d(position)
        hip = Joint(parent, upperleg, pos, (-0.1*pi, 0.9*pi), self.walking_task.max_rate)
        walking_task.space.add(hip.pivot, hip.motor, hip.limit, upperleg, shape)

        # Lower leg
        lowerleg = pymunk.Body(mass, pymunk.moment_for_box(mass, w, l * 1.2))
        lowerleg.position = pymunk.Vec2d(upperleg.position) + pymunk.Vec2d(0, l - w/2.0)
        shape = pymunk.Poly.create_box(lowerleg, (w, l * 1.2))
        shape.group = 1
        shape.friction = 2.0
        # Joints
        pos =  pymunk.Vec2d(upperleg.position) + pymunk.Vec2d(0, l/2.0 - w/2.0)
        knee = Joint(upperleg, lowerleg, pos, (-0.9*pi, 0.1*pi), self.walking_task.max_rate)
        walking_task.space.add(knee.pivot, knee.motor, knee.limit, lowerleg, shape)
        
        self.upperleg = upperleg
        self.lowerleg = lowerleg
        self.hip = hip
        self.knee = knee
            

class WalkingTask(object):
    """ Walking gait task.
    """
    
    def __init__(self, max_steps=1000, 
                       track_length=1000, 
                       max_rate=2.0, 
                       torso_height=40, 
                       torso_density=0.2,
                       leg_spacing=30,
                       leg_length=30,
                       num_legs=4):
        # Settings
        self.max_steps = max_steps
        self.track_length = track_length
        self.max_rate = max_rate
        self.torso_height = torso_height
        self.torso_density = torso_density
        self.leg_spacing = leg_spacing
        self.leg_length = leg_length
        self.num_legs = num_legs
        
    def evaluate(self, network, draw=False):
        """ Evaluate the efficiency of the given network. Returns the
            distance that the walker ran in the given time (max_steps).
        """
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        if draw:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((self.track_length, 200))
            pygame.display.set_caption("Simulation")
            clock = pygame.time.Clock()
            running = True
            font = pygame.font.Font(pygame.font.get_default_font(), 8)
        
        # Initialize pymunk
        self.space = space = pymunk.Space()
        space.gravity = (0.0, 900.0)
        space.damping = 0.7
        self.touching_floor = False
        # Create objects
        # Floor

        floor = pymunk.Body()
        floor.position = pymunk.Vec2d(self.track_length/2.0 , 210)
        sfloor = pymunk.Poly.create_box(floor, (self.track_length, 40))
        sfloor.friction = 1.0
        sfloor.collision_type = 1
        space.add_static(sfloor)

        # Torso
        torsolength = 20 + (self.num_legs // 2 - 1) * self.leg_spacing
        mass = torsolength * self.torso_height * self.torso_density
        torso = pymunk.Body(mass, pymunk.moment_for_box(mass, torsolength, self.torso_height))
        torso.position = pymunk.Vec2d(200, 200 - self.leg_length * 2 - self.torso_height)
        storso = pymunk.Poly.create_box(torso, (torsolength, self.torso_height))
        storso.group = 1
        storso.collision_type = 1
        storso.friction = 2.0
        # space.add_static(storso)
        space.add(torso, storso)

        # Legs
        legs = []
        for i in range(self.num_legs // 2):
            x = 10 - torsolength / 2.0 + i * self.leg_spacing
            y = self.torso_height / 2.0 - 10
            legs.append( Leg(torso, (x,y), self) )
            legs.append( Leg(torso, (x,y), self) )
        
        # Collision callback
        def oncollide(space, arb):
            self.touching_floor = True
        space.add_collision_handler(1, 1, post_solve=oncollide)
        
        for step in xrange(self.max_steps):
            
            # Query network
            input_width = max(len(legs), 4)
            net_input = np.zeros((3, input_width))
            torso_y = torso.position.y
            torso_a = torso.angle
            sine = np.sin(step / 10.0)
            hip_angles = [leg.hip.angle() for leg in legs]
            knee_angles = [leg.knee.angle() for leg in legs]
            other = [torso_y, torso_a, sine, 1.0]
            # Build a 2d input grid, 
            # as in Clune 2009 Evolving Quadruped Gaits, p4
            net_input[0, :len(legs)] = hip_angles
            net_input[1, :len(legs)] = knee_angles
            net_input[2, :4] = other
            act = network.feed(net_input, add_bias=False)

            output = np.clip(act[-self.num_legs*2:] * self.max_rate, -1.0, 1.0) / 2.0 + 0.5

            for i, leg in enumerate(legs):
                leg.hip.set_target( output[i * 2] )
                leg.knee.set_target( output[i * 2 + 1] )
            
            # Advance simulation
            space.step(1/50.0)
            # Check for success/failure
            if torso.position.x < 0:
                break
            if torso.position.x > self.track_length - 50:
                break
            if self.touching_floor:
                break

            # Draw
            if draw:
                print act
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
                clock.tick(50)
                
        if draw:
            pygame.quit()
        
        distance = torso.position.x
        # print "Travelled %.2f in %d steps." % (distance, step)
        return {'fitness':distance}
        
    def solve(self, network):
        return False
        
    def visualize(self, network, filename=None):
        """ Visualize a solution strategy by the given individual. """
        self.evaluate(network, draw=True)
        
        
    