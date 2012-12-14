""" Implementation of the Shape Discrimination task described in
    "Coleman - Evolving Neural Networks for Visual Processing" 
"""

import random

import numpy as np

### HELPER FUNCTION

def line(im, x0, y0, x1, y1):
    """ Bresenham """
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0  
        x1, y1 = y1, x1
        
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        
    if y0 < y1: 
        ystep = 1
    else:
        ystep = -1
            
    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = 0
    y = y0
        
    for x in range(x0, x1 + 1): # We add 1 to x1 so that the range includes x1
        if steep:
            im[y, x] = 1
        else:
            im[x, y] = 1
            
        error = error + deltay
        if (error << 1) >= deltax:
            y = y + ystep
            error = error - deltax


class ShapeDiscriminationTask(object):
    
    def __init__(self, targetshape=None, 
                       distractorshapes=None, size=15, trials=75, fitnessmeasure='dist'):
        """ Constructor 
            If target shape and distractor shape isn't passed,
            the setup from the visual field experiment (big-box-little-box)
            is used. Otherwise, use makeshape() to initalize the target and
            distractor shapes.
        """
        self.target         = targetshape
        self.distractors    = distractorshapes
        self.size           = size
        self.trials         = trials
        self.fitnessmeasure = fitnessmeasure
             
        if self.target is None:
            self.target = self.makeshape('box', size//3)
        if self.distractors is None:
            self.distractors = [self.makeshape('box', 1)]
            
        print ":::: Target Shape ::::"
        print self.target
        print ":::: Distractor Shapes ::::"
        for d in self.distractors:
            print d
        
    @classmethod
    def makeshape(cls, shape, size=5):
        """ Create an image of the given shape.
        """
        im = np.zeros((size, size))
        xx, yy = np.mgrid[-1:1:size*1j, -1:1:size*1j]
        
        # Box used for big-box-little-box.
        if shape == 'box':
            im[:] = 1
            
        # Outlined square
        elif shape == 'square':
            im[:,0] = 1;
            im[0,:] = 1;
            im[:,-1] = 1;
            im[-1,:] = 1;
            
        # (roughly) a circle.
        elif shape == 'circle':                
            d = np.sqrt(xx * xx + yy * yy)
            im[ np.logical_and(0.65 <= d, d <= 1.01) ] = 1
            
        # An single-pixel lined X
        elif shape == 'x':
            line(im, 0, 0, size-1, size-1)
            line(im, 0, size-1, size-1, 0)
         
        else:
            raise Exception("Shape Unknown.")  
        
        return im
        
    def evaluate(self, network):
        if not network.sandwich:
            raise Exception("Object Discrimination task should be performed by a sandwich net.")
        
        dist = 0.0
        correct = 0.0
        wsose = 0.0
        pattern = np.zeros((self.size, self.size))
        for _ in xrange(self.trials):
            pattern *= 0.0
            targetsize = self.target.shape[0]
            distractor = random.choice(self.distractors)
            distsize = distractor.shape[0]
            x, y = np.random.randint(self.size - targetsize, size=2)
        
            pattern[x:x+targetsize, y:y+targetsize] = self.target
            cx, cy = x + targetsize // 2, y + targetsize // 2
        
            for i in xrange(100):
                x, y = np.random.randint(self.size - distsize, size=2)
                if not np.any(pattern[x:x+distsize, y:y+distsize]):
                    pattern[x:x+distsize, y:y+distsize] = distractor
                    break
                if i == 99:
                    raise Exception("No position found")

            network.flush()
            output = network.feed(pattern, add_bias=False)
            mx = output.argmax()
            (x_, y_) = mx // self.size, mx % self.size
            dist += np.sqrt(((x_ - cx) ** 2) + ((y_ - cy) ** 2))
            if dist == 0:
                correct += 1
            wsose += 0.5 * (1 - output.flat[mx]) + 0.5 * output.mean()
            
        correct /= self.trials
        dist /= self.trials
        wsose /= self.trials
        
        
        if self.fitnessmeasure == 'dist':
            fitness = 1. / (1. + dist)
        elif self.fitnessmeasure == 'wsose':
            fitness = 0.5 * correct + 0.5 * (1 - wsose)
        return {'fitness':fitness, 'correct':correct, 'dist':dist, 'wsose':wsose}
        
    def solve(self, network):
        return self.evaluate(network)['dist'] < 0.5
    
