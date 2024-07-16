
import random

import numpy as np


def skining_weights_to_color(skinning_weights, alpha):
    """ Given a skinning weight matrix NvxNj, return a color matrix of shape Nv*3. For each joint Ji i in [0, Nj] , 
    the color is colors[i]"""
    
    joints_ids = np.arange(0, skinning_weights.shape[1])
    colors = vertex_colors_from_weights(joints_ids, scale_to_range_1=True, alpha=alpha, shuffle=True, seed = 1)
    
    weights_color = np.matmul(skinning_weights, colors)
    return weights_color
        

def vertex_colors_from_weights(weights, scale_to_range_1=True, alpha=None, shuffle=False, seed=0):
    """
    Given an array of values of size N, generate an array of colors (Nx3) forming a gradient. 
    :param weights: Input values (N)
    :param scale_to_range_1: If False, the color gradient will cover the values 0 to 1 and plateau beyond
    :param alpha: If not None, add an alpha channel of value alpha
    :param shuffle: If True, shuffle the colors
    :param seed: Seed for the random number generator to shuffle
    :return: An array of rgb colors (N, 3).
    """
    if scale_to_range_1:
        weights = weights - np.min(weights)
        weights = weights / np.max(weights)

    from matplotlib import cm
    if alpha is None:
        vertex_colors = np.ones((len(weights), 3))
    else:
        vertex_colors = alpha * np.ones((len(weights), 4))
    vertex_colors[:, :3] = cm.jet(weights)[:, :3]

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(vertex_colors)

    return vertex_colors


def color_gradient(N, scale=1, shuffle=False, darken=False, pastel=False, alpha=None):

    """Return a Nx3 or Nx4 array of colors forming a gradient

    :param N: Number of colors to generate
    :param scale: Scale of the gradient
    :param shuffle: Shuffle the colors
    :param darken: Darken the colors
    :param pastel: Make the colors pastel
    :param alpha: Add an alpha channel of value alpha to the colors
    """
    import colorsys
    if darken:
        V = 0.75
    else:
        V = 1

    if pastel:
        S = 0.5
    else:
        S = 1

    HSV_tuples = [((N-x) * 1.0 / (scale*N), S, V) for x in range(N)] # blue - grean - yellow - red gradient
    RGB_list = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), HSV_tuples))

    if shuffle:
        random.Random(0).shuffle(RGB_list) #Seeding the random this way always produces the same shuffle
    
    RGB_array = np.array(RGB_list)
    
    if alpha is not None:
        RGB_array = np.concatenate([RGB_array, alpha * np.ones((N, 1))], axis=1)
    
    return RGB_array
