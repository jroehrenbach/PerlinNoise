# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 02:05:33 2019

@author: jroeh

module for generating k-dimensional perlin noise arrays
"""

import numpy as np


def interpolate(nodes, buffer, axis):
    """
    Interpolates node array over one axis using the cosine function
    
    Parameters
    ----------
    nodes : ndarray
        Node array to be interpolated
    buffer : int
        Buffer between nodes to be interpolated
    axis : int
        Axis over which node array should be interpolated
    
    Returns
    -------
    Interpolated nd-array
    
    Note
    ----
    Last elements of chosen axis won't be returned
    """
    # length axis over which nodes will be interpolated
    nnodes = nodes.shape[axis]
    
    # normalized position between nodes
    x = np.tile(np.arange(buffer),nnodes)[:-buffer] / buffer
    x_shape = np.ones(len(nodes.shape), int)
    x_shape[axis] = x.size
    x = x.reshape(*x_shape)
    
    # reference heights for each position
    n = np.repeat(np.arange(nnodes-1),buffer)
    h0 = nodes.take(n, axis)
    h1 = nodes.take(n+1, axis)
    
    # interpolating nodes using cosine function
    nodes = (1 - np.cos(x * np.pi)) / 2 * (h1 - h0) + h0
    return nodes


def random_nodes(shape, buffer):
    """
    Creates random array of nodes
    
    Paramters
    ---------
    shape : tuple
        Shape of complete noise array
    buffer : int
        Space between nodes on all axes
    
    Returns
    -------
    np.ndarray
    """
    # check arguments and calculate shape of random nodes
    shape = np.array(shape)
    if (shape % buffer != 0).any():
        raise IOError("shape and tilesize don't match!")
    node_shape = (shape / buffer).astype(int) + 1
    
    # random nodes which will be interpolated to noise
    return np.random.rand(*node_shape)


def generate_noise(nodes, buffer):
    """
    Turns node array into noise array
    
    Parameters
    ----------
    nodes : np.ndarray
        Array of nodes
    buffer : int
        Space between nodes to be interpolated
    """
    # interpolating over each axis
    for axis in range(len(nodes.shape)):
        nodes = interpolate(nodes, buffer, axis)
        
    return nodes


def random_noise(shape, buffer):
    """
    Generates noise array of any dimension
    
    Parameters
    ----------
    shape : tuple
        Can have any size but each axis has to match the buffer
    buffer : int
        must be smaller or equal to any element of shape
    
    Returns
    -------
    Array of generated noise with n dimensions
    """
    
    print(shape, buffer)
    nodes = random_nodes(shape, buffer)
    return generate_noise(nodes, buffer)


# implement Nodes class

class Perlin(object):
    """Generates k-dimensional perlin noise array"""
    
    def __init__(self, shape, base, lacunarity, depth):
        self.generate(shape, base, lacunarity, depth)
    
    def generate(self, shape, base, lacunarity, depth):
        """
        Generates noise for each layer
        
        Parameters
        ----------
        shape : tuple
            Can have any number of dimensions
        base : int
            Shape of bottom node layer
        lacunarity : int
            Exponential increase of nodes with each layer
            Typically 2
        depth : int
            Number of layers
            Note: the depth must match lacunarity and shape
        """
        # check arguments
        if ((np.array(shape) % (lacunarity ** depth)) != 0).any():
            raise IOError("shape, lacunarity and depth don't match!")
        
        # generate each noise layer and append to layers
        for d in range(depth):
            noise = random_noise(shape, base * lacunarity ** (d + 1))[None]
            if hasattr(self, "layers"):
                self.layers = np.append(noise, self.layers, 0)
            else:
                self.layers = noise
    
    def get_noise(self, persistance, normalize=True):
        """
        Updates noise with new persistance value
        
        Parameters
        ----------
        persistance : float
            Exponential decrease of height with each layer
            Typically between 0 and 1
        normalize : bool (optional)
            If true, noise array will be normalized
        
        Returns
        -------
        nd-array of perlin noise
        """
        p = persistance ** np.arange(self.layers.shape[0])
        p_shape = [p.size] + [1] * (len(self.layers.shape) - 1)
        noise = np.array(self.layers * p.reshape(*p_shape)).sum(0)
        if normalize:
            noise /= noise.max()
        return noise
    
    def add_nodes(self, axis, side, size=0, nodes=None):
        # function for adding nodes to any of the layers.
        
        # if no nodes are passed, it will be randomly generate with thinckness
        # of size.
        if nodes == None:
            # create random nodes
            # ...
            pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    noise = Perlin((256,512), 4, 2, 4).get_noise(0.44)
    plt.imshow(noise%.2)