# Copyright 2019 - M. Pecheux
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# Utils: Subpackage with various util tools
# ------------------------------------------------------------------------------
# misc.py - Miscellaneous util classes and methods
# ==============================================================================
import os
import re
import sys
if sys.version_info[0] < 3:
    from inspect import getargspec
    PY_V = 2
else:
    from inspect import signature
    PY_V = 3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .wrappers import typechecker
from .logger import Logger

COLORS = [[1,0,0,1], [0,0,1,1], [0,1,0,1], [1,0,1,1], [0,1,1,1]]
PLOT_COLORS = ['r', 'b', 'g', 'm', 'y']
LATEX_REGEX = r'(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|kappa|' \
               'lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|tau|upsilon|phi|' \
               'varphi|chi|psi|omega|mathbb)[_|^]?'

class MultivalueDict(dict):
    """Dictionary that automatically stacks all values for the same key in a
    list.
    
    Adapted from:
    https://stackoverflow.com/questions/2390827/how-to-properly-subclass-dict-and-override-getitem-setitem
    """

    def __init__(self, *args, **kwargs) :
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        if key in self.keys():
            new_val = self[key]
            if not isinstance(new_val, list): new_val = [new_val]
            new_val.append(val)
        else:
            new_val = val
        dict.__setitem__(self, key, new_val)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

class InterpolationError(object):
    """Util holder for a series of (precision step, error) matches."""
    
    def __init__(self, norm):
        self.data = MultivalueDict()
        self.norm = norm
        
    def __str__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data.__getitem__(key)
        
    def __setitem__(self, key, val):
        self.data[key] = val
        
    @property
    def h(self): return np.array(list(self.data.keys()))
    @property
    def err(self): return np.array(list(self.data.values()))
    @property
    def info(self):
        err = self.err
        return self.h, err, err.shape[1]
    
    def output(self, compare_order=None):
        """
        Parameters
        ----------
        compare_order : int, list(int) or None, optional
            If it is not None, this parameter specifies a reference line to
            compare the errors to of the form: y = h^compare_order. If a list is
            passed in, a reference line is plotted for each order in the list.
        """
        s  = '\nInterpolation error ({} norm):\n'.format(self.norm)
        s += '=' * (28 + len(self.norm)) + '\n\n'
        x, y, N = self.info
        headers  = ['h'] + ['Error #{}'.format(i) for i in range(N)]
        row_data = [['{:.3f}'.format(x[i])] + ['{:.6f}'.format(j) for j in y[i,:]] \
                    for i in range(len(x))]
        if compare_order is not None:
            if not isinstance(compare_order, list):
                compare_order = [compare_order]
            for o in compare_order:
                if not isinstance(o, int):
                    Logger.serror('You can only compare to functions of the '
                                  'form: y = h^compare_order. Provide integer '
                                  'values for compare orders.')
                headers.append('Ref: y = h' + ('' if o == 1 else '^{}'.format(o)))
                for i in range(len(x)):
                    row_data[i].append('{:.6f}'.format(x[i]**o))
        rows = ['\t'.join(r) for r in row_data]
        head_row = '\t'.join(headers)
        s += head_row + '\n'
        s += '-' * (len(head_row) + 7*(len(headers) - 1)) + '\n'
        s += '\n'.join(rows)
        s += '\n'
        print(s)
        
    def plot(self, compare_order=None, reverse=True):
        """
        Parameters
        ----------
        compare_order : int, list(int) or None, optional
            If it is not None, this parameter specifies a reference line to
            compare the errors to of the form: y = h^compare_order. If a list is
            passed in, a reference line is plotted for each order in the list.
        reverse : bool, optional
            If true (by default), x values are inverted to have the error in
            descending order (the curve tends to zero on the right).
        """
        x, y, N = self.info
        if len(x) < 2:
            Logger.slog('Cannot plot a 0- or 1-point data. Aborting.',
                        level='warning')
        if reverse: x = 1./x
        x /= np.max(x)
        y /= np.max(y, axis=0)
        for i in range(N):
            plt.loglog(x, y[:,i], '{}.-'.format(PLOT_COLORS[i]))
        
        ref_leg = []
        if compare_order is not None:
            if not isinstance(compare_order, list):
                compare_order = [compare_order]
            for o in compare_order:
                if not isinstance(o, int):
                    Logger.serror('You can only compare to functions of the '
                                  'form: y = h^compare_order. Provide integer '
                                  'values for compare orders.')
                if reverse:
                    y_ref = 1./(x**o)
                    y_ref /= np.max(y_ref, axis=0)
                    plt.loglog(x, y_ref, '{}--'.format(PLOT_COLORS[o-1]))
                else:
                    y_ref = x**o
                    y_ref /= np.max(y_ref, axis=0)
                    plt.loglog(x, y_ref, '{}--'.format(PLOT_COLORS[o-1]))
                if o == 1: ref_leg.append('y = h'.format(o))
                else: ref_leg.append('y = h^{}'.format(o))
                
        plt.title('Interpolation Error ({} norm)'.format(self.norm))
        plt.xlabel('1/h' if reverse else 'h')
        plt.ylabel('Relative error e (||u - uh||/h)')
        plt.legend(['Error ({})'.format(i) for i in range(N)] + ref_leg)
        plt.show()

def is_number(v):
    """Checks if a given variable is a number (int, float or complex).
    
    Parameters
    ----------
    v : int, float, complex
        Variable to check.
    """
    return (isinstance(v,int) or isinstance(v,float) or isinstance(v,complex))

def nb_params(f):
    """Returns the number of input parameters of a given function.
    
    Parameters
    ----------
    f : func
        Function to analyze.
    """
    if PY_V == 2:
        argspec = getargspec(f)
        vargs_count = int(argspec.varargs is not None)
        vkwargs_count = int(argspec.keywords == 'kwargs')
        return len(argspec.args) + vargs_count + vkwargs_count
    elif PY_V == 3:
        return len(signature(f).parameters)

def to_func(v):
    """Transforms a numerical value to a callable Python object that returns
    this value (a constant function). If the provided object is already callable,
    it is returned as is.
    
    Parameters
    ----------
    v : int, float, complex
        Number to transform to a constant function.
    """
    if callable(v): return v
    if not is_number(v):
        Logger.serror('Only numerical values (int, real or complex) can be '
                      'converted to a function.')
    return lambda x,y: v

def to_3args(f):
    """Lambdaes a function to make it callable with 3 parameters.
    
    Parameters
    ----------
    f : func
        Function to transform.
    """
    n_params = nb_params(f)
    if n_params == 1:   return lambda x,y,z: f(x)
    elif n_params == 2: return lambda x,y,z: f(x,y)
    else:               return f
    
@typechecker((str,),(str,))
def to_latex(s):
    """Tries to transform a string to a Latex formula (if the string contains
    known special characters like Greek letters).
    
    Parameters
    ----------
    s : str
        String to convert.
    """
    if re.search(LATEX_REGEX, s, re.I) is None: return s
    return '$\\{}$'.format(s)
    
@typechecker((np.ndarray,str,), None)
def check_complex_solution(solution, complex):
    """Checks if a solution is complex and transforms it into a real solution
    according to the given rule.
    
    Parameters
    ----------
    solution : numpy.ndarray
        Solution to cast.
    complex : str
        Type of cast to perform: 'abs' takes the absolute value (default mode),
        'real' truncates to keep only the real part, 'imag' truncates to keep
        only the imaginary part.
    """
    if complex == 'none':  return solution
    if not solution.any(): return np.real(solution)
    if solution.dtype == np.complex128:
        if complex == 'abs':    solution = np.real(np.abs(solution))
        elif complex == 'real': solution = np.real(solution)
        elif complex == 'imag': solution = np.imag(solution)
        else:
            Logger.slog('Unknown type of cast for complex data exporting. '
                        'Defaulting to "abs" mode.', level='warning')
            solution = np.abs(solution)
    return solution

def list_in_list(a, l):
    """Checks if a list is in a list and returns its index if it is (otherwise
    returns -1).
    
    Parameters
    ----------
    a : list()
        List to search for.
    l : list()
        List to search through.
    """
    return next((i for i, elem in enumerate(l) if elem == a), -1)
    
def plot(domain, solution=None, solution_exact=None, **kwargs):
    """Plots a solution (and optionally a reference solution) onto the given
    DiscreteDomain instance.
    
    **kwargs are the same as the ones in the DiscreteDomain's visualize() method.
    
    Parameters
    ----------
    domain : DiscreteDomain
        Discrete domain instance to plot the solution onto. The solution must have
        as many values as the number of vertices in the domain.
    solution : numpy.ndarray, optional
        Values to plot at each vertex of the domain. If None, then the domain
        is plot as is with the z-component of each vertex as value.
    solution_exact : numpy.ndarray, optional
        Value references to plot at each vertex of the domain. If None, only the
        graph of the computed solution is shown; else two graphs are created to
        show the computed and exact solutions side by side.
    """
    if solution is None: return domain.visualize(**kwargs)
    else: return domain.visualize(z=solution, z_ex=solution_exact, **kwargs)

def make_animation(to_file, domains, solutions, titles=None, fps=50, **kwargs):
    """Creates an animation from a list of solutions (with their associated
    domains, one for each solution). The output can be a MP4 or GIF file.
    
    **kwargs are the same as the ones in the DiscreteDomain's visualize() method.
    
    Parameters
    ----------
    to_file : str
        Path to the file the animation should be saved to. The extension must be
        included and will determine the format of the file (among: "mp4", "gif").
    domains : list(DiscreteDomain)
        Instances associated with each solution to plot.
    solutions : list(numpy.ndarray)
        Solutions to plot.
    titles : list(str) or None, optional
        Titles for each plot.
    fps : int, optional
        Number of frames per second (sets the speed of the animation). The
        higher the FPS, the faster the video.
    """
    if len(domains) != len(solutions):
        Logger.serror('Cannot create an animation with a different number of '
                      'domains and solutions!')
    if titles is not None and len(titles) != len(solutions):
        Logger.serror('Cannot create an animation with a different number of '
                      'titles and solutions!')
    # precompute images and write them to a temporary file
    if titles is None:
        for i in range(len(solutions)):
            plot(domains[i], solution=solutions[i], no_plot=True,
                 to_file='/tmp/simplefempy-fig{}.jpg'.format(i), **kwargs)
            plt.close()
    else:
        for i in range(len(solutions)):
            plot(domains[i], solution=solutions[i], no_plot=True, title=titles[i],
                 to_file='/tmp/simplefempy-fig{}.jpg'.format(i), **kwargs)
            plt.close()
    # get type of animation (from file extension)
    ext = to_file.split('.')[-1].lower()
    # create final animation
    Logger.slog('Exporting to animation (with ffmpeg)...')
    if ext == 'mp4':
        os.system('ffmpeg -framerate {} -i /tmp/simplefempy-fig%01d.jpg -vcodec '
                  'mpeg4 -y {}'.format(fps, to_file))
    elif ext == 'gif':
        os.system('ffmpeg -framerate {} -i /tmp/simplefempy-fig%01d.jpg -y '
                  '-filter:v "setpts=30.0*PTS" {}'.format(fps, to_file))
    # clear tmp directory
    os.system('rm /tmp/simplefempy*')
