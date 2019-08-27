# Copyright 2019 Mina Pêcheux (mina.pecheux@gmail.com)
# ---------------------------
# Distributed under the MIT License:
# ==================================
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# Utils: Subpackage with various util tools
# ------------------------------------------------------------------------------
# wrappers.py - Various wrapping functions
# ==============================================================================
import inspect

from .logger import Logger

def typechecker(in_=(), out_=(type(None),), msg_=None):
    """
    Util function wrapper to check the type of the parameters passed to a
    function.
    
    Parameters
    ----------
    in_ : tuple(tuple(type))
        Reference of ingoing parameters' types.
    out_ : tuple(tuple(type))
        Reference of outgoing parameters' types.
    msg_ : str
        Custom message to display in case of invalid parameter (if none is
        provided, the wrapper uses a generic message instead). This message can
        use 4 predefined variables that will be instantiated with the current
        context:
        - $FUNCNAME: name of the wrapped function
        - $PARAMIDX: id of the parameter in the list of the function's parameters
        - $PARAMTYPE: expected type for the parameter
        - $IN?: is parameter in- or out-going?
    """
    def _typechecker(func):
        def _get_msg(state, idx, ok_type):
            if not type(ok_type) in (tuple, list):
                t = str(ok_type)
            else:
                t = ' or '.join(["{}".format(okt) for okt in ok_type])
            if msg_ is not None:
                m = msg_.replace('$FUNCNAME', func.__name__)
                m = m.replace('$PARAMIDX', str(idx))
                m = m.replace('$PARAMTYPE', t)
                m = m.replace('$IN?', '{}going'.format(state))
                return m
            else:
                return '{}(): Incorrect type for {}going parameter #{}: should ' \
                       'be of type {}.'.format(func.__name__, state, idx, t)
        
        def _check_types(elements, types, state):
            """Subfunction to check the arguments types."""
            if len(elements) != len(types):
                Logger.serror('{}(): Incorrect number of {}going arguments'
                              '.'.format(func.__name__, state), stackoffset=3)
            typed = enumerate(zip(elements, types))
            for idx, couple in typed:
                arg, ok_type = couple
                if not type(ok_type) in (tuple, list):
                    if isinstance(arg, ok_type): continue
                elif type(arg) in ok_type: continue
                Logger.serror(_get_msg(state, idx, ok_type), stackoffset=3)
        # wrapped function
        def run(*args, **kwargs):
            # check ingoing parameter types
            _check_types(args, in_, 'in')
            res = func(*args, **kwargs)
            # if necessary, check outgoing result types
            if out_ is not None:
                if not type(res) in (tuple, list): checkable_res = (res,)
                else: checkable_res = res
                _check_types(checkable_res, out_, 'out')
            return res
        return run
    return _typechecker

def funcunpacker(func):
    """Util function wrapper to unpack parameters in a function call to match
    the required number of arguments (flattens the parameters into one list)."""
    # wrapped function
    def run(*args, **kwargs):
        new_args = [item for sublist in args for item in sublist]
        return func(*new_args, **kwargs)
    return run

def multievaluate(func):
    """Util function wrapper to evaluate the same function multiple times by
    iterating over a set of values for one or more parameters."""
    # wrapped function
    def run(**kwargs):
        ref_args = inspect.getargspec(func).args
        for kw in kwargs:
            if kw not in ref_args:
                Logger.serror('Incorrect parameter: the function {}() does not '
                              'have a parameter: "{}".'.format(func.__name__, kw))
        new_kwargs = {}
        nb_tests = 0
        for k, v in kwargs.items():
            for i, val in enumerate(v):
                new_kwargs[k + '_' + str(i)] = val
                nb_tests += 1
        Logger.slog('Multi-evaluation of the problem: "{}".'.format(func.__name__))
        res = []
        idx = 0
        for k, v in new_kwargs.items():
            Logger.slog('[{:3d}%] {} = {}'.format(int(100.*idx/nb_tests), k, v))
            res.append(func(**{ '_'.join(k.split('_')[:-1]): v }))
            idx += 1
        return res
    return run
