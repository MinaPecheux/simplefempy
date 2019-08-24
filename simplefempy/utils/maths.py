# Copyright 2019 - M. Pecheux
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# Utils: Subpackage with various util tools
# ------------------------------------------------------------------------------
# maths.py - Various util mathematical classes and methods
# ==============================================================================
import numpy as np
from .wrappers import typechecker
from .logger import Logger
from .misc import is_number

class FunctionalFunction(object):
    """Util class to get a custom callable object that can be summed or
    composed with others.
    
    From: https://stackoverflow.com/questions/4101244/how-to-add-functions
    """
    
    def __init__(self, func):
        if not callable(func):
            Logger.serror('You need to provide a callable object!')
        self.func = func
        
    def __call__(self, *args, **kwargs):
        """Returns the function computation."""
        return self.func(*args, **kwargs)
        
    def __add__(self, other):
        """Adds the two functions."""
        def summed(*args, **kwargs):
            return self(*args, **kwargs) + other(*args, **kwargs)
        return summed
        
    def __mul__(self, other):
        """Composes the two functions."""
        def composed(*args, **kwargs):
            if callable(other): return self(other(*args, **kwargs))
            else:               return other*self(*args, **kwargs)
        return composed
    def __rmul__(self, other):
        """Composes the two functions."""
        def composed(*args, **kwargs):
            if callable(other): return self(other(*args, **kwargs))
            else:               return other*self(*args, **kwargs)
        return composed

@typechecker(([int,float,np.float64],[int,float,np.float64],), None)
def dirac(x, y, tol=2e-2, bounds=None):
    """Creates a
    
    Parameters
    ----------
    x : int, float
        Horizontal position of the Dirac.
    y : int, float
        Vertical position of the Dirac.
    tol : float, optional
        Small offset around the position to avoid completely null Dirac if no
        vertex falls on the precise position.
    bounds : list[int or float, int or float, int or float, int or float] or None, optional
        Clamping bounds for the Dirac, if necessary.
    """
    xmin, xmax = x - tol, x + tol
    ymin, ymax = y - tol, y + tol
    if bounds is not None:
        xmin = max(xmin, bounds[0])
        xmax = min(xmax, bounds[1])
        ymin = max(ymin, bounds[2])
        ymax = min(ymax, bounds[3])
    return lambda x,y: 1. if (xmin <= x <= xmax and ymin <= y <= ymax) else 0.

def get_path(waypoints, steps, closed='none'):
    """Generates a path through the given waypoints (it can be open, closed or
    almost-closed).
    
    Parameters
    ----------
    waypoints : list(tuple(int or float,int or float))
        Coordinates of the points to go through.
    steps : int
        Number of intermediate points on each segments.
    closed : str, optional
        Type of closure for the path (among: "none", "closed", "almost-closed").
    """
    if len(waypoints) < 2:
        Logger.serror('Cannot generate a path with less than points!')
    if closed not in ['none', 'closed', 'almost-closed']:
        Logger.serror('Incorrect path closing type. Choose among: "none", '
                      '"closed" and "almost-closed".')
    steps += 1
    xpts  = [waypoints[0][0]]
    ypts  = [waypoints[0][1]]
    if closed != 'none': waypoints.append(waypoints[0])
    for i in range(1, len(waypoints)):
        w1, w2 = waypoints[i-1], waypoints[i]
        xlin = np.linspace(w1[0], w2[0], steps)
        ylin = np.linspace(w1[1], w2[1], steps)
        xpts.extend(list(xlin[1:]))
        ypts.extend(list(ylin[1:]))
    if closed == 'almost-closed':
        xpts = xpts[:-1]
        ypts = ypts[:-1]
    return list(zip(xpts, ypts))
