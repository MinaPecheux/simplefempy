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
# solver.py - Classes to represent a weak formulation and its terms
# ==============================================================================
from enum import Enum
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from .geometrizor import interpolate
from .converter import formulation_parser
from .utils.logger import Logger
from .utils.wrappers import typechecker, funcunpacker
from .utils.misc import to_3args, to_func, is_number

class VariationalTerm(object):
    
    DPHI = {
        'P1': { 0: np.matrix([[-1], [-1]]),
             1: np.matrix([[1], [0]]),
             2: np.matrix([[0], [1]]) },
        'P2': { 0: np.matrix([[-1], [-1]]),
             1: np.matrix([[1], [0]]),
             2: np.matrix([[0], [1]]) }
    }
    
    def __init__(self, type, domain, func=None):
        self.type   = type
        # check for correct variational term type, or exit with error
        if not self.type in ['MASS', 'RIGIDITY', 'MIXED', 'RHS']:
            Logger.serror('Unknown Variational Term type: "{}"'.format(self.type))
            
        if self.type == 'MIXED':
            self.func = np.array([1.,1.]) if func is None else func
        else:
            if self.type == 'RHS': f = to_func(0. if func is None else func)
            else: f = to_func(1. if func is None else func)
            # check for number of input arguments: rematch to 3d space
            # (general case)
            self.func = to_3args(f)
        self.domain = domain
        
    def __str__(self):
        return 'Variational Term [domain: "{}"]\tType: "{}"\n'.format(self.domain.name,
                self.type)

    def assemble(self):
        """Assembles the matrix or vector corresponding to the Variational Term."""
        Nv = self.domain.Nv # domain size (number of vertices)
        f  = funcunpacker(self.func)
        
        def _mass1d(seg):
            row = list(); col = list(); A = list()
            n1 = self.domain.nodes[seg[0]]
            n2 = self.domain.nodes[seg[1]]
            length = np.linalg.norm(n2 - n1)
            Me = length/6.*np.matrix([[2.,1.], [1.,2.]])
            for i in range(2):
                I = seg[i]
                for j in range(2):
                    J = seg[j]
                    row.append((I))
                    col.append((J))
                    A.append((f(self.domain.nodes[J]) * Me[i,j]))
            return row, col, A
        def _mass2d(p, det):
            row = list(); col = list(); A = list()
            Me  = self.domain.mats_mass[p]
            for i in range(self.domain.tricount):
                I = self.domain.loc2glob(p,i)
                for j in range(self.domain.tricount):
                    J = self.domain.loc2glob(p,j)
                    row.append((I))
                    col.append((J))
                    A.append((f(self.domain.nodes[J])*Me[i,j]))
            return row, col, A
                
        def _rigidity1d(seg):
            row = list(); col = list(); A = list()
            n1 = self.domain.nodes[seg[0]]
            n2 = self.domain.nodes[seg[1]]
            length = np.linalg.norm(n2 - n1)
            De = length*np.matrix([[1.,-1.], [-1.,1.]])
            for i in range(2):
                I = seg[i]
                for j in range(2):
                    J = seg[j]
                    row.append((I))
                    col.append((J))
                    A.append((f(self.domain.nodes[J]) * De[i,j]))
            return row, col, A
        def _rigidity2d(p, det):
            row = list(); col = list(); A = list()
            De  = self.domain.mats_rigidity[p]
            for i in range(self.domain.tricount):
                I = self.domain.loc2glob(p,i)
                for j in range(self.domain.tricount):
                    J = self.domain.loc2glob(p,j)
                    row.append((I))
                    col.append((J))
                    A.append((f(self.domain.nodes[J]) * De[i,j]))
            return row, col, A
        
        def _mixed2d(p, det):
            row = list(); col = list(); A = list()
            T = self.domain.transfer_vector(p)
            w = self.func
            for i in range(3):
                I = self.domain.loc2glob(p,i)
                for j in range(3):
                    J = self.domain.loc2glob(p,j)
                    row.append((I))
                    col.append((J))
                    A.append((float(w.dot(T[i]))/6.))
            return row, col, A
            
        def _rhs1d(seg): return interpolate('edge', self.domain, f, seg)
        def _rhs2d(p):   return interpolate('surface', self.domain, f, p)
        
        if self.type == 'RHS':
            B = np.zeros((Nv,1))
            if self.domain.is_border:
                for seg in self.domain.edges: B += _rhs1d(seg)
            else:
                for p in range(self.domain.Nt): B += _rhs2d(p)
            return B
        else:
            row = list(); col = list(); A = list()
            if self.domain.is_border:
                for seg in self.domain.edges:
                    if self.type == 'MASS':
                        r, c, coeff = _mass1d(seg)
                    elif self.type == 'RIGIDITY':
                        r, c, coeff = _rigidity1d(seg)
                    else:
                        r, c, coeff = [], [], []
                    row.extend(r)
                    col.extend(c)
                    A.extend(coeff)
            else:
                # get all triangles determinants
                det_tri = self.domain.det_triangles
                for p in range(self.domain.Nt):
                    if self.type == 'MASS':
                        r, c, coeff = _mass2d(p, det_tri[p])
                    elif self.type == 'RIGIDITY':
                        r, c, coeff = _rigidity2d(p, det_tri[p])
                    else:
                        r, c, coeff = _mixed2d(p, det_tri[p])
                    row.extend(r)
                    col.extend(c)
                    A.extend(coeff)
            return sparse.coo_matrix((A, (row, col)), shape=(Nv,Nv)).tocsr()

class VariationalFormulation(object):
    
    """Object that represents a Variational Formulation, i.e. a 'nice'
    equivalent of a PDE with average quantities rather than point to point
    conditions."""
    
    def __init__(self, domain, a, l, dirichlet=None):
        for term in a:
            if term[0] == 'RHS':
                Logger.serror('You cannot define a right-hand side in the '
                              'left part of your variational formulation!')
        self.domain = domain
        self.a      = [VariationalTerm(*term) for term in a]
        self.l      = []
        for term in l:  # detailed unpacking for Python 2.x compatibility
            l_terms = ['RHS']
            l_terms = l_terms + [t for t in term]
            self.l.append(VariationalTerm(*l_terms))
                        
        self.dirichlet = []
        if dirichlet is not None:
            if not isinstance(dirichlet, list): dirichlet = [dirichlet]
            for d in dirichlet:
                if len(d) < 2 or not (callable(d[-1]) or is_number(d[-1])):
                    Logger.serror('Invalid Dirichlet condition: {}.'.format(d))
                t = to_func(d[-1])
                for s in d[:-1]:
                    if self.domain(s) == self.domain or not self.domain(s).is_border:
                        Logger.serror('Invalid Dirichlet condition: "{}" is not '
                                      'a border of your domain!'.format(s))
                    self.dirichlet.append((s,t))
                    
        self._precompute_domain()
        
    def __str__(self):
        """Overrides the print representation of this instance to provide
        useful information about it."""
        s = 'Variational Formulation:\n' + '=' * 24 + '\n'
        s += 'Left-hand side:\n' + '-' * 15 + '\n'
        for a in self.a: s += str(a)
        s += '\nRight-hand side:\n' + '-' * 16 + '\n'
        for l in self.l: s += str(l)
        if len(self.dirichlet) > 0:
            s += '\nDirichlet conditions on border(s): '
            s += ', '.join([ds for ds, _ in self.dirichlet])
            s += '\n'
        return s
        
    def _precompute_domain(self):
        """Precomputes mass and/or rigidity matrices of the associated
        DiscreteDomain instance, depending on the ones necessary for the problem
        to solve."""
        if not self.domain.mass_mtr_prepared:
            if len([term for term in self.a if term.type == 'MASS']) > 0:
                self.domain.precompute_mtr_mass()
        if not self.domain.rig_mtr_prepared:
            if len([term for term in self.a if term.type == 'RIGIDITY']) > 0:
                self.domain.precompute_mtr_rigidity()
            
        Logger.slog('Prepared Discrete Domain instance for problem to solve.',
                    stackoffset=2)
        
    @classmethod
    @typechecker((type,str,dict), None)
    def from_str(cls, s, loc, **kwargs):
        """Constructor to build a VariationalFormulation instance from a
        formatted string (by parsing).
        
        Parameters
        ----------
        s : str
            String to parse and deconstruct.
        loc : dict()
            Available local variables.
        """
        domain, a, l, dirichlet = formulation_parser(s, loc)
        return cls(domain, a, l, dirichlet=dirichlet, **kwargs)
        
    def split_dirichlet(self):
        """Returns the vertices indexes foreach Dirichlet condition of the
        problem instance."""
        masks = []
        for border, value in self.dirichlet:
            subdomain = self.domain(border)
            mask = np.zeros(subdomain.Nv, dtype=bool)
            edges = []
            for n0, n1 in subdomain.edges: edges.extend([n0,n1])
            mask[np.unique(edges)] = True
            masks.append((mask, value))
        return masks
        
    def mass_matrix(self, splitted=False):
        """Computes the mass matrix of the Variational Formulation."""
        M = sum([term.assemble() for term in self.a if term.type == 'MASS'])
        if not splitted: return M
        else:
            if len(self.dirichlet) > 1:
                Logger.serror('Mass matrix can only be splitted for a single '
                              'Dirichlet condition, for now!')
            # split 'interior' and 'Dirichlet' nodes
            mask = self.get_dirichlet_idx()[0][0]
            M_int = M[np.ix_(~mask,~mask)]
            M_dir = M[np.ix_(~mask, mask)]
            return M_int, M_dir, mask
        
    def rigidity_matrix(self, splitted=False):
        """Computes the rigidity matrix of the Variational Formulation."""
        D = sum([term.assemble() for term in self.a if term.type == 'RIGIDITY'])
        if not splitted: return D
        else:
            if len(self.dirichlet) > 1:
                Logger.serror('Rigidity matrix can only be splitted for a single '
                              'Dirichlet condition, for now!')
            # split 'interior' and 'Dirichlet' nodes
            mask = self.get_dirichlet_idx()[0][0]
            D_int = D[np.ix_(~mask,~mask)]
            D_dir = D[np.ix_(~mask, mask)]
            return D_int, D_dir, mask
        
    def mixed_matrix(self):
        """Computes the mixed matrix of the Variational Formulation."""
        return sum([term.assemble() for term in self.a if term.type == 'MIXED'])
        
    def rhs(self):
        """Computes the right-hand side (vector) of the Variational Formulation."""
        return sum([term.assemble() for term in self.l if term.type == 'RHS'])
        
    def apply_dirichlet(self, A, B=None):
        d = A.diagonal()
        v = d.sum() / A.shape[0]
        if B is not None:
            # .. check for complex equations: cast to complex if necessary
            A_is_complex = np.any(np.iscomplex(A.toarray()))
            B_is_complex = np.any(np.iscomplex(B))
            if A_is_complex or B_is_complex:
                v *= 1j
                if A_is_complex and not B_is_complex: B = B.astype(np.complex)
            for border, value in self.dirichlet:
                subdomain = self.domain(border)
                if subdomain.fe_type == 'P1':
                    for I, J in subdomain.edges:
                        d[I] = v; A[I,:] = 0.
                        d[J] = v; A[J,:] = 0.
                        s1x, s1y, _ = subdomain.nodes[I]
                        s2x, s2y, _ = subdomain.nodes[J]
                        B[I] = value(s1x,s1y)*v
                        B[J] = value(s2x,s2y)*v
                elif subdomain.fe_type == 'P2':
                    for I, J, K in subdomain.edges:
                        d[I] = v; A[I,:] = 0.
                        d[J] = v; A[J,:] = 0.
                        d[K] = v; A[K,:] = 0.
                        s1x, s1y, _     = subdomain.nodes[I]
                        s2x, s2y, _     = subdomain.nodes[J]
                        sMidx, sMidy, _ = subdomain.nodes[K]
                        B[I] = value(s1x,s1y)*v
                        B[J] = value(s2x,s2y)*v
                        B[K] = value(sMidx,sMidy)*v
                elif subdomain.fe_type == 'P3':
                    for I, J, K, L in subdomain.edges:
                        d[I] = v; A[I,:] = 0.
                        d[J] = v; A[J,:] = 0.
                        d[K] = v; A[K,:] = 0.
                        d[L] = v; A[L,:] = 0.
                        s1x, s1y, _ = subdomain.nodes[I]
                        s2x, s2y, _ = subdomain.nodes[J]
                        s3x, s3y, _ = subdomain.nodes[K]
                        s4x, s4y, _ = subdomain.nodes[L]
                        B[I] = value(s1x,s1y)*v
                        B[J] = value(s2x,s2y)*v
                        B[K] = value(s3x,s3y)*v
                        B[L] = value(s4x,s4y)*v
        else:
            # .. check for complex equations: cast to complex if necessary
            A_is_complex = np.any(np.iscomplex(A.toarray()))
            if A_is_complex: v *= 1j
            for border, value in self.dirichlet:
                subdomain = self.domain(border)
                if subdomain.fe_type == 'P1':
                    for I, J in subdomain.edges:
                        d[I] = v; A[I,:] = 0.
                        d[J] = v; A[J,:] = 0.
                elif subdomain.fe_type == 'P2':
                    for I, J, K in subdomain.edges:
                        d[I] = v; A[I,:] = 0.
                        d[J] = v; A[J,:] = 0.
                        d[K] = v; A[K,:] = 0.
                elif subdomain.fe_type == 'P3':
                    for I, J, K, L in subdomain.edges:
                        d[I] = v; A[I,:] = 0.
                        d[J] = v; A[J,:] = 0.
                        d[K] = v; A[K,:] = 0.
                        d[L] = v; A[L,:] = 0.
        A.setdiag(d)
        if B is not None: return A, B
        else: return A
        
    def assemble_matrices(self):
        """Assembles the matrix and right-hand side of the linear system for
        this Variational Formulation instance."""
        # compute system matrices and right-hand side
        A = (self.mass_matrix() + self.rigidity_matrix() + self.mixed_matrix()).tolil()
        B = self.rhs()
        # if necessary: set Dirichlet border conditions
        if len(self.dirichlet) > 0:
            A, B = self.apply_dirichlet(A, B)
        
        return A.tocsr(), B

    def solve(self, info=True):
        """Solves the linear system associated with the problem from its matrix
        and right-hand side.
        
        Parameters
        ----------
        info : bool, optional
            If true, some information is logged at the end of the computation.
        """
        A, B = self.assemble_matrices()
        u    = sparse.linalg.spsolve(A, B)
        if u.dtype == np.complex128:
            uabs = np.abs(u)
            min, max = np.min(uabs), np.max(uabs)
        else:
            min, max = np.min(u), np.max(u)
        if info:
            Logger.slog('Solved equation: min = {}\tmax = {}'.format(min, max))
        return u

class FiniteDifferenceScheme(object):
    
    """
    Abstract top class to implement a finite difference scheme.
    Warning: it does not implement any real computation!
    --------
    """
    
    def __init__(self, initial_sol, dt, **kwargs):
        self.init_u = initial_sol
        self.dt = dt
                
    def solve(self, nb_iters, all_solutions=False):
        """Computes a given number of iterations with a finite difference scheme.
        
        Parameters
        ----------
        nb_iters : int
            Number of iterations to compute.
        all_solutions : bool, optional
            If true, all intermediate results are recorded and returned. Else,
            only the last solution is returned.
        """
        u = np.copy(self.init_u)
        solutions = [u]
        next_u = np.zeros_like(u)
        for i in range(nb_iters):            
            # compute new solution vector
            next_u = self.iter(u, i)
            # record solution
            u = np.copy(next_u)
            solutions.append(u)

        if u.dtype == np.complex128:
            uabs = np.abs(u)
            min, max = np.min(uabs), np.max(uabs)
        else:
            min, max = np.min(u), np.max(u)
        Logger.slog('Solved {} iterations: min = {}\tmax = {}'.format(nb_iters, min, max))

        if all_solutions: return solutions
        else: return solutions[-1]
        
    def iter(self, u, i):
        """Computes a new solution vector from the previous one.
        
        Parameters
        ----------
        u : numpy.ndarray
            Current solution vector.
        i : int
            Index of the current timestep.
        """
        raise NotImplementedError('Define a specific iteration function!')

class ThetaScheme(FiniteDifferenceScheme):
    
    """Parent class to implement a theta-scheme."""
    
    def __init__(self, initial_sol, domain, dt, f, dirichlet, theta, **kwargs):
        super().__init__(initial_sol, dt, **kwargs)
        self.domain = domain
        self.f      = f

        system = VariationalFormulation(domain,
            [('RIGIDITY', domain), ('MASS', domain)], [(domain, f)],
            dirichlet=dirichlet)
        
        self.border_idx = system.split_dirichlet()
        M, D = system.mass_matrix(), system.rigidity_matrix()

        self.mtr_left  = sparse.linalg.inv((M + theta*dt*D).tocsc())
        self.mtr_right = M - (1-theta)*dt*D
        self.coeff_f1  = theta*dt
        self.coeff_f2  = (1-theta)*dt
        
    def iter(self, u, i):
        # compute right-hand side for current time
        cur_t, next_t = i * self.dt, (i+1) * self.dt
        f_cur  = lambda x, y: self.f(x, y, cur_t)
        F_cur  = f_cur(*self.domain.points).reshape((self.domain.Nv, 1))
        f_next = lambda x, y: self.f(x, y, next_t)
        F_next = f_next(*self.domain.points).reshape((self.domain.Nv, 1))
        
        next_u = np.empty_like(u)
        next_u = (self.mtr_left*self.mtr_right).dot(u) \
                 + self.coeff_f1 * F_next + self.coeff_f2 * F_cur
        
        for dir_idx, dir_val in self.border_idx:
            next_u[dir_idx] = dir_val(0.,0.) # the value is constant...
            
        return next_u

class EulerExplicit(ThetaScheme):
    
    """Class to implement the forward/explicit Euler scheme (theta-scheme,
    with theta = 0)."""
    
    def __init__(self, init_sol, dom, dt, f, dirichlet=None, **kwargs):
        super().__init__(init_sol, dom, dt, f, dirichlet, theta=0, **kwargs)

class EulerImplicit(ThetaScheme):
    
    """Class to implement the backward/implicit Euler scheme (theta-scheme,
    with theta = 1)."""
    
    def __init__(self, init_sol, dom, dt, f, dirichlet=None, **kwargs):
        super().__init__(init_sol, dom, dt, f, dirichlet, theta=1, **kwargs)

class CrankNicholson(ThetaScheme):
    
    """Class to implement the Crank-Nicholson scheme (theta-scheme, with
    theta = 0.5)."""
    
    def __init__(self, init_sol, dom, dt, f, dirichlet=None, **kwargs):
        super().__init__(init_sol, dom, dt, f, dirichlet, theta=0.5, **kwargs)
