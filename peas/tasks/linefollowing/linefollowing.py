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
    
    def __init__(self, space, field_friction, field_observation, initial_pos,
                       size=8, 
                       motor_torque=6,
                       friction_scale=0.2,
                       angular_damping=0.9,
                       force_global=False):
        self.field_friction = field_friction
        self.field_observation = field_observation
        self.size = size
        self.friction_scale = friction_scale
        self.motor_torque = motor_torque
        self.angular_damping = angular_damping
        self.force_global = force_global
        
        mass = size ** 2 * 0.2
        self.body = body = pymunk.Body(mass, pymunk.moment_for_box(mass, size, size))
        body.position = pymunk.Vec2d(initial_pos[0], initial_pos[1])
        body.angle = initial_pos[2]
        self.shape = shape = pymunk.Poly.create_box(body, (size, size))
        shape.group = 1
        shape.collision_type = 1
        space.add(body, shape)
        
        self.sensors = [(r, theta) for r in np.linspace(1,4,3) * size * 0.75
                                   for theta in np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)]

        self.l = self.r = 0
        
    
    def sensor_locations(self):
        for (r, theta) in self.sensors:
            (x, y) = np.cos(theta) * r, np.sin(theta) * r
            yield self.body.local_to_world((x, y))

    def wheel_locations(self, rel=True):
        lwheel = self.body.local_to_world((0, -self.size / 2))
        rwheel = self.body.local_to_world((0, self.size / 2))
        if rel:
            lwheel -= self.body.position
            rwheel -= self.body.position
        return (lwheel, rwheel)
            
    def sensor_response(self):
        for point in self.sensor_locations():
            yield self.field_at(point, observation=True)

    def field_at(self, (x,y), border=1.0, observation=False):
        if (0 <= x < self.field_friction.shape[1] and 
            0 <= y < self.field_friction.shape[0]):
            if observation:
                return self.field_observation[int(y), int(x)]
            else:
                return self.field_friction[int(y), int(x)]
        return border
                
    def apply_friction(self):
        f = self.field_at(self.body.position)
        f = 1 - self.friction_scale * f
        self.body.velocity.x = self.body.velocity.x * f
        self.body.velocity.y = self.body.velocity.y * f
        self.body.angular_velocity *= self.angular_damping
        # Zero out sideways motion. (i.e. 100% perpendicular friction)
        self.body.velocity = self.body.velocity.projection(self.body.rotation_vector)

    def drive(self, l, r):
        self.l, self.r = np.clip([l, r], -1, 1)
        self.l *= self.motor_torque
        self.r *= self.motor_torque
        if not self.force_global:
            lw, rw = self.wheel_locations()
        else:
            lw, rw = (0, -self.size / 2), (0, self.size / 2)
        self.l = float(self.l)
        self.r = float(self.r)
        self.body.apply_impulse( self.l * self.body.rotation_vector, lw)
        self.body.apply_impulse( self.r * self.body.rotation_vector, rw)
        
    def draw(self, screen):
        import pygame
        pygame.draw.lines(screen, (255,0,0), True, [(int(p.x), int(p.y)) for p in self.shape.get_points()])
        
        lw, rw = self.wheel_locations(rel=False)
        pygame.draw.line(screen, (255, 0, 255), lw, lw + self.body.rotation_vector * 50. * self.l / self.motor_torque, 2)
        pygame.draw.line(screen, (255, 0, 255), rw, rw + self.body.rotation_vector * 50. * self.r / self.motor_torque, 2) 

        for ((x, y), response) in zip(self.sensor_locations(), self.sensor_response()):
            r = 255 * response
            pygame.draw.circle(screen, (255 - r, r, 0), (int(x), int(y)), 2)
    
    
class LineFollowingTask(object):
    """ Line following task.
    """
    
    def __init__(self, field='eight', observation='eight_striped',
                       max_steps=1000, 
                       friction_scale=0.2, 
                       motor_torque=6,
                       damping=0.2,
                       initial_pos=None,
                       flush_each_step=False,
                       force_global=False,
                       path_resolution=100,
                       check_coverage=False,
                       coverage_memory=20):
        # Settings
        self.max_steps = max_steps
        self.flush_each_step = flush_each_step
        self.force_global = force_global
        self.fieldpath = os.path.join(DATA_DIR,field) + '.png'
        self.observationpath = os.path.join(DATA_DIR,observation) + '.png'
        print "Using %s" % (self.fieldpath,)
        field_friction = imread(self.fieldpath)
        field_observation = imread(self.observationpath)
        self.path_resolution = path_resolution
        self.check_coverage = check_coverage
        self.coverage_memory = coverage_memory
        
        self.field_friction = field_friction[:,:,0].astype(np.float)/255
        self.field_observation   = field_observation[:,:,0].astype(np.float)/255
        
        self.friction_scale = friction_scale
        self.motor_torque   = motor_torque
        self.damping        = damping
        self.initial_pos    = initial_pos
        if self.initial_pos is None:
            self.initial_pos = self.field_friction.shape[0] / 2, self.field_friction.shape[1]/2 , 0
        
        
    def evaluate(self, network, draw=False, drawname='Simulation'):
        """ Evaluate the efficiency of the given network. Returns the
            distance that the bot ran
        """
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        h,w = self.field_friction.shape
        
        if draw:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption(drawname)
            clock = pygame.time.Clock()
            running = True
            font = pygame.font.Font(pygame.font.get_default_font(), 12)
            field_image = pygame.image.load(self.observationpath)
        
        # Initialize pymunk
        self.space = space = pymunk.Space()
        space.gravity = (0,0)
        space.damping = self.damping

        # Create objects
        robot = Robot(space, self.field_friction, self.field_observation, self.initial_pos,
                             friction_scale=self.friction_scale, motor_torque=self.motor_torque,
                             force_global=self.force_global)
    
        path = [(robot.body.position.int_tuple)]
        cells = []
        
        if network.cm.shape[0] != (3*5 + 3*3 + 1):
            raise Exception("Network shape must be a 2 layer controller: 3x5 input + 3x3 hidden + 1 bias. Has %d." % network.cm.shape[0])
        
        for step in xrange(self.max_steps):
            
            net_input = np.array(list(robot.sensor_response()))
            # The nodes used for output are somewhere in the middle of the network
            # so we extract them using -4 and -6
            action = network.feed(net_input)[[-4,-6]]
            if self.flush_each_step:
                network.flush()

            robot.drive(*action)
            robot.apply_friction()
            space.step(1/50.0)

            new_cell_covered = False
            current_cell = int(robot.body.position.x // 32), int(robot.body.position.y // 32)
            if current_cell not in cells:
                cells.append(current_cell)
                new_cell_covered = True
                if len(cells) > self.coverage_memory:
                    cells.pop(0)
            elif cells[-1] == current_cell:
                new_cell_covered = True
            
            if step % self.path_resolution == 0 and (not self.check_coverage or new_cell_covered):
                path.append((robot.body.position.int_tuple))


            if draw:
                screen.fill((255, 255, 255))
                screen.blit(field_image, (0,0))
                txt = font.render('%d - %.0f' % (step, path_length(path)), False, (0,0,0) )
                screen.blit(txt, (0,0))
                # Draw path
                if len(path) > 1:
                    pygame.draw.lines(screen, (0,0,255), False, path, 3)
                for cell in cells:
                    pygame.draw.rect(screen, (200,200,0), (cell[0]*32, cell[1]*32, 32, 32), 2)
                robot.draw(screen)
                
                if pygame.event.get(pygame.QUIT):
                    break

                for k in pygame.event.get(pygame.KEYDOWN):
                    if k.key == pygame.K_SPACE:
                        break
                
                pygame.display.flip()
                # clock.tick(50)
                
        if draw:
            pygame.quit()
            
        self.last_path = path
        dist = path_length(path)
        speed = dist / self.max_steps
        return {'fitness':1 + dist**2, 'dist':dist, 'speed':speed}
        
    def solve(self, network):
        stats = self.evaluate(network)
        return stats['speed'] > 0.20
        
    def visualize(self, network, filename=None):
        """ Visualize a solution strategy by the given individual. """
        import matplotlib.pyplot as plt
        self.evaluate(network, draw=False, drawname=filename)
        print "Saving as " + os.path.join(os.getcwd(), filename)
        plt.figure()
        plt.imshow(self.field_observation * 0.2, cmap='Greys', vmin=0, vmax=1)
        for i in range(len(self.last_path)-1):
            plt.plot(*zip(*self.last_path[i:i+2]), lw=4, alpha=0.3, color=(0.3,0,0.8))
        plt.ylim(0,512)
        plt.xlim(0,512)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


        

if __name__ == '__main__':
    a = LineFollowingTask()
