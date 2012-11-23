#! /usr/bin/python

""" Top-down line following task
"""

### IMPORTS ###
import random
import os

# Libraries
import pymunk
import numpy as np
from scipy.misc import imread

# Local
from ...networks.rnn import NeuralNetwork

# Shortcuts
pi = np.pi

### CONSTANTS ###
DATA_DIR = os.path.abspath(os.path.split(__file__)[0])

### FUNCTIONS ###
    
def path_length(path):
    l = 0
    for i in xrange(1, len(path)):
        (x0, y0) = path[i-1]
        (x1, y1) = path[i]
        l += np.sqrt((x1-x0) ** 2 + (y1-y0) ** 2)
    return l
    
### CLASSES ###

class Robot(object):
    """ Robot that performs this task. """
    
    def __init__(self, space, field, 
                       size=8, 
                       wheeltorque=10, 
                       start=(256,256)):
        self.field = field
        self.size = size
        self.wheeltorque = wheeltorque
        
        mass = size ** 2 * 0.2
        self.body = body = pymunk.Body(mass, pymunk.moment_for_box(mass, size, size))
        body.position = pymunk.Vec2d(start[0], start[1])
        self.shape = shape = pymunk.Poly.create_box(body, (size, size))
        shape.group = 1
        shape.collision_type = 1
        space.add(body, shape)
        
        self.sensors = [(r, theta) for r in np.linspace(1,4,3) * size * 0.75
                                   for theta in np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)]
        
    
    def sensor_locations(self):
        for (r, theta) in self.sensors:
            (x, y) = np.cos(theta) * r, np.sin(theta) * r
            yield self.body.local_to_world((x, y))
            
    def sensor_response(self):
        for point in self.sensor_locations():
            yield self.field_at(point)

    def field_at(self, (x,y), border=1.0):
        if (0 <= x < self.field.shape[1] and 
            0 <= y < self.field.shape[0]):
            return self.field[int(y), int(x)]
        return border
                
    def apply_friction(self):
        f = self.field_at(self.body.position)
        self.body.velocity.x = self.body.velocity.x * (1 - 0.5 * f)
        self.body.velocity.y = self.body.velocity.y * (1 - 0.5 * f)

    def drive(self, l, r):
        l *= self.wheeltorque
        r *= self.wheeltorque
        self.body.apply_impulse((l,l) * self.body.rotation_vector, (0, -self.size / 2))
        self.body.apply_impulse((r,r) * self.body.rotation_vector, (0, self.size / 2))
        
    def draw(self, screen):
        import pygame
        pygame.draw.lines(screen, (255,0,0), True, [(int(p.x), int(p.y)) for p in self.shape.get_points()])
        
        for ((x, y), response) in zip(self.sensor_locations(), self.sensor_response()):
            r = 255 * response
            pygame.draw.circle(screen, (255 - r, r, 0), (int(x), int(y)), 2)
    
    
class LineFollowingTask(object):
    """ Line following task.
    """
    
    def __init__(self, field='eight.png',
                       max_steps=5000):
        # Settings
        self.max_steps = max_steps
        self.fieldpath = os.path.join(DATA_DIR,field)
        print "Using %s" % (self.fieldpath,)
        self.field = imread(self.fieldpath)
        self.field = self.field[:,:,0].astype(np.float)/255
        
        
    def evaluate(self, network, draw=False):
        """ Evaluate the efficiency of the given network. Returns the
            distance that the bot ran
        """
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        h,w = self.field.shape
        
        if draw:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Simulation")
            clock = pygame.time.Clock()
            running = True
            font = pygame.font.Font(pygame.font.get_default_font(), 8)
            field_image = pygame.image.load(self.fieldpath)
        
        # Initialize pymunk
        self.space = space = pymunk.Space()
        space.gravity = (0,0)
        space.damping = 0.2

        # Create objects
        robot = Robot(space, self.field)
    
        path = [(robot.body.position.int_tuple)]
        
        if network.cm.shape[0] != (3*5 + 3*3 + 1):
            raise Exception("Network shape must be a 2 layer controller: 3x5 input + 3x3 hidden + 1 bias. Has %d." % network.cm.shape[0])
        
        for step in xrange(self.max_steps):
            
            net_input = np.array(list(robot.sensor_response()))
            # The nodes used for output are somewhere in the middle of the network
            # so we extract them using -4 and -6
            action = network.feed(net_input)[[-4,-6]]

            robot.drive(*action)
            robot.apply_friction()
            space.step(1/50.0)
            
            if step % 100 == 0:
                path.append((robot.body.position.int_tuple))

            if draw:
                screen.fill((255, 255, 255))
                screen.blit(field_image, (0,0))
                # Draw path
                if len(path) > 1:
                    pygame.draw.lines(screen, (0,0,255), False, path, 2)
                robot.draw(screen)
                
                
                pygame.display.flip()
                clock.tick(50)
                
        if draw:
            pygame.quit()

        return {'fitness':1+path_length(path)}
        
    def solve(self, network):
        return False
        
    def visualize(self, network, filename=None):
        """ Visualize a solution strategy by the given individual. """
        self.evaluate(network, draw=True)
        

if __name__ == '__main__':
    a = LineFollowingTask()
