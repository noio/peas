""" Implements different development modules to convert
    from genotype to phenotype (in)directly.
"""

### IMPORTS ###

# Libs
import numpy as np
import scipy.ndimage.filters

# Local
from ..methods.neat import NEATGenotype
from ..networks import NeuralNetwork

np.seterr(divide='raise')


### CLASSES ###

class ReactionDiffusionGenotype(object):
    
    def __init__(self, num_chemicals=3):
        pass

class ReactionDeveloper(object):
    """ Developer that converts a genotype into 
        a network using a HyperNEAT-like indirect
        encoding.
    """
    def __init__(self, substrate_shape=(10,), cm_range=(-30., 30.), reaction_steps=5, sandwich=False, 
                    diffusion=0.0, recursion=0.0):
        # Instance vars
        self.substrate_shape = substrate_shape
        self.cm_range        = cm_range
        self.reaction_steps  = reaction_steps
        self.diffusion       = diffusion
        self.sandwich        = sandwich
        self.recursion       = recursion
    
    def convert(self, network):
        """ Generates an n-dimensional connectivity matrix. """
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        os = np.atleast_1d(self.substrate_shape)
        # Unpack the genotype
        w, f = network.cm.copy(), network.node_types[:]
        
        # Create substrate
        if len(os) == 1:
            cm = np.mgrid[-1:1:os[0]*1j,-1:1:os[0]*1j].transpose((1,2,0))
        elif len(os) == 2:
            cm = np.mgrid[-1:1:os[0]*1j,-1:1:os[1]*1j,-1:1:os[0]*1j,-1:1:os[1]*1j].transpose(1,2,3,4,0)
        else:
            raise NotImplementedError("3+D substrates not supported yet.")
        # Insert a bias
        cm = np.insert(cm, 0, 1.0, -1)
        # Check if the genotype has enough weights
        if w.shape[0] < cm.shape[-1]:
            raise Exception("Genotype weight matrix is too small (%s)" % (w.shape,) )
        # Append zeros
        n_elems = len(f)
        nvals = np.zeros(cm.shape[:-1] + (n_elems - cm.shape[-1],))
        cm = np.concatenate((cm, nvals), -1)
        shape = cm.shape
        
        # Fix the input elements
        frozen = len(os) * 2 + 1
        w[:frozen] = 0.0
        w[np.diag_indices(frozen, 2)] = 1.0
        f[:frozen] = [lambda x: x] * frozen
        w[np.diag_indices(n_elems)] = (1 - self.recursion) * w[np.diag_indices(n_elems)] + self.recursion
        
        # Compute the reaction
        self._steps = []
        laplacian = np.empty_like(cm[..., frozen:])
        kernel = self.diffusion * np.array([1.,2.,1.])
        for _ in range(self.reaction_steps):
            cm = np.dot(w, cm.reshape((-1, n_elems)).T)
            cm = np.clip(cm, self.cm_range[0], self.cm_range[1])
            for el in xrange(cm.shape[0]):
                cm[el,:] = f[el](cm[el,:])
            cm = cm.T.reshape(shape)            
            # apply diffusion
            laplacian[:] = 0.0
            for ax in xrange(cm.ndim - 1):
                laplacian += scipy.ndimage.filters.convolve1d(cm[..., frozen:], kernel, axis=ax, mode='constant')
            cm[..., frozen:] += laplacian
            self._steps.append(cm[...,-1])
            
        # Return the values of the last element (indicating connectivity strength)
        output = cm[..., -1]
        # Build a network object
        net = NeuralNetwork().from_matrix(output)
        if self.sandwich:
            net.make_sandwich()
        return net
        
    def visualize(self, genotype, filename):
        self.convert(genotype)
        visualization.image_grid(map(visualization.conmat_to_im, self._steps)).save(filename)
        
