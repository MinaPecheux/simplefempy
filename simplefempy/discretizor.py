# Copyright 2019 - M. Pecheux
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# discretizor.py - Class to represent a discretized domain (of finite dimension)
# ==============================================================================
import os
import numpy as np

from .settings import LIB_SETTINGS
if LIB_SETTINGS['using_tk']:
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.tri as mtri
import matplotlib.patches as mp
import matplotlib.collections as mc

from .geometrizor import make_rectangle, make_line, make_circle, make_ring
from .solver import VariationalTerm
from .converter import load_gmsh
from .utils.matrices import mass as mtr_mass, rigidity as mtr_rigidity
from .utils.logger import Logger
from .utils.wrappers import typechecker
from .utils.misc import COLORS, to_latex, check_complex_solution, list_in_list

class DiscreteDomain(object):
    
    """Core tool of the FEM to represent a discretized domain of finite dimension
    (an approximation of the continuous domain) defined by a set of nodes and
    their connectivity."""
    
    def __init__(self, filename=None, nodes=None, elements=None, fe_type='P1',
                 labels=None, name=None, tag=None, parent=None, scale=1.):
        self.name       = 'Domain' if name is None else name
        self.tag        = 0 if tag is None else tag
        self.parent     = parent
        self.fe_type = fe_type
        self.set_poly   = False
        self.scale      = scale if parent is None else parent.scale
        
        if filename is None:
            if nodes is None or elements is None:
                Logger.serror('A DiscreteDomain instance can only be created '
                              'from a file or from direct data. Provide either:\n'
                              '- the name of a mesh file (GMSH v2.2 format)\n'
                              '- or two NumPy arrays with the nodes and the '
                              'elements (connectivity) of your mesh')
            self.nodes    = nodes
            self.elements = elements
        else:
            self.nodes, self.elements = load_gmsh(filename)
            if parent is None: self.nodes *= scale
        
        # if all elements are edges, the domain is a border
        self.is_border  = np.all(self.elements[:,0] == 1)

        # prepare properties and element lists
        if self.fe_type == 'P1': self.tricount = 3
        elif self.fe_type == 'P2': self.tricount = 6
        elif self.fe_type == 'P3': self.tricount = 10
        else: Logger.serror('Finite Elements of type "{}" are not yet '
                            'implemented!'.format(self.fe_type), stackoffset=100)
        if parent is None:
            self.edges     = self._elements('edge')[:,-3:-1]
            self.triangles = self._elements('triangle')[:,-3:]
        else:
            l = -self.tricount + 1
            if self.fe_type == 'P1':   l += 1
            elif self.fe_type == 'P2': l += 2
            elif self.fe_type == 'P3': l += 3
            self.edges     = self._elements('edge')[:,-self.tricount:l]
            self.triangles = self._elements('triangle')[:,-self.tricount:]
        
        # precompute coordinate changes, transfer matrices...
        self._precompute(parent is None)
        self.points = [self.nodes[:,0], self.nodes[:,1]]
        self.z = self.nodes[:,2]
        
        # get bounds and cut down subdomains
        self.subdomains = {}
        if parent is None:
            self.is_subdomain = False
            self.bounds = [np.min(self.nodes[:,0]), np.max(self.nodes[:,0]),
                           np.min(self.nodes[:,1]), np.max(self.nodes[:,1])]

            subdomain_tags = np.unique(self.elements[:,1])
            for tag in subdomain_tags:
                elts_idx = np.where(self.elements[:,1] == tag)
                self.subdomains[tag] = DiscreteDomain.subdomain(
                    elements=self.elements[elts_idx], parent=self, tag=tag)
        else:
            self.is_subdomain = True
            self.bounds = parent.bounds
            
        self.nb_subdomains = len(self.subdomains)
            
        # if necessary, associate labels with subdomains
        if labels is not None:
            for lbl, lblTag in labels.items():
                if lblTag not in self.subdomains:
                    Logger.serror('There are no elements associated with this '
                                  'Physical label: "{}"'.format(lblTag))
                s = self.subdomains[lblTag]
                self.subdomains[lbl] = s
                s.name = lbl
        
        # log parent domain creation
        if parent is None:
            Logger.slog('Created a DiscreteDomain: {} vertices, {} triangles ({} '
                        'subdomain(s))'.format(self.Nv, self.Nt, self.nb_subdomains),
                        level='info', stackoffset=3)
    
    def __call__(self, *args):
        """Overrides the call method to select a subdomain in the instance. If
        no argument is passed, the whole domain is returned. Otherwise, the
        domain tries to access the requested label (int or string): if it is
        found, the subdomain is returned; else it is ignored. If multiple labels
        are requested, a merged domain combining the different correct subdomains
        is returned."""
        if len(args) == 0: return self
        subdomains = []
        for arg in args:
            if not (isinstance(arg, str) or isinstance(arg, int)):
                Logger.slog('To select a subdomain, provide an int or a string'
                      ' label (this label will be ignored)!', level='warning')
                continue
            if arg not in self.subdomains:
                Logger.slog('No subdomain with label "{}" is defined in your '
                     'discrete domain. Sub-selection is ignored.'.format(arg),
                     level='warning')
                continue
            subdomains.append(self.subdomains[arg])
        if len(subdomains) == 0: return self
        if len(subdomains) == 1: return subdomains[0]
        else:                    return (DiscreteDomain.merge(subdomains))
        
    def __str__(self):
        sub = '(Sub)' if self.parent != None else ''
        s  = 'Discrete {}domain "{}":\n'.format(sub, self.name)
        s += '-' * (19 + len(sub) + len(self.name)) + '\n'
        s += 'Interpolation type: "{}"\n'.format(self.fe_type)
        if self.nb_subdomains > 0: s += '({} subdomain(s))\n'.format(self.nb_subdomains)
        s += 'Nt = {} triangles'.format(self.Nt)
        if self.is_border: s += ' (Border)'
        s += '\n'
        s += 'Nv = {} vertices'.format(self.Nv) + '\n'
        s += '(Bounding box: [{},{}] x [{},{}])'.format(*self.bounds) + '\n'
        s += '\n'
        return s
    
    def _elements(self, type):
        """Util function to quickly get 'edge' or 'triangle' (full) elements.
        
        Parameters
        ----------
        type : str
            Type of elements to extract (among: "edge", "triangle").
        """
        if type == 'edge': return self.elements[self.elements[:,0] == 1]
        if type == 'triangle': return self.elements[self.elements[:,0] == 2]
    
    def _elements_idx(self, type):
        """Util function to quickly get 'edge' or 'triangle' (full) element
        indices (i.e.: position in full elements array).
        
        Parameters
        ----------
        type : str
            Type of elements to extract (among: "edge", "triangle").
        """
        if type == 'edge': return np.where(self.elements[:,0] == 1)[0]
        if type == 'triangle': return np.where(self.elements[:,0] == 2)[0]
                        
    def _precompute(self, set_poly=True):
        """Precomputes util variables for the domain (small overhead to speed
        up solving), but not the matrices."""
        def _add_points(t, l):
            """Util function to store the points to add depending on the type
            of Finite Elements.
            
            Parameters
            ----------
            t : str
                Type of Finite Elements.
            l : list(numpy.ndarray)
                List to append the new points to.
            """
            if t == 'P2':
                l.append(np.round(0.5*(l[0] + l[1]), decimals=6))
                l.append(np.round(0.5*(l[0] + l[2]), decimals=6))
                l.append(np.round(0.5*(l[1] + l[2]), decimals=6))
            elif t == 'P3':
                l.append(np.round(l[0] + (l[1] - l[0])/3., decimals=6))
                l.append(np.round(l[0] + 2.*(l[1] - l[0])/3., decimals=6))
                l.append(np.round(l[0] + (l[2] - l[0])/3., decimals=6))
                l.append(np.round(l[0] + 2.*(l[2] - l[0])/3., decimals=6))
                l.append(np.round(l[0] + 2.*(l[1] - l[0])/3. + (l[2] - l[0])/3., decimals=6))
                l.append(np.round(l[0] + (l[1] - l[0])/3. + 2.*(l[2] - l[0])/3., decimals=6))
                l.append(np.round(l[0] + (l[1] - l[0])/3. + (l[2] - l[0])/3., decimals=6))
        def _set_edges(t, lbls, tri, points, new_edges):
            """Util function to associate new points to the right edge depending
            on the type of Finite Elements.
            
            Parameters
            ----------
            t : str
                Type of Finite Elements.
            lbls : list(int)
                Integer label of each initial edge (in the order: "0-1", "0-2",
                "1-2").
            tri : numpy.ndarray
                Reference of the initial triangle points indices (real vertices).
            points : list(numpy.ndarray)
                New points that were added for this type of Finite Elements.
            new_edges : list(numpy.ndarray)
                List to append the new edges to.
            """
            if t == 'P2':
                new_edges.append([1, lbls[0], tri[0], tri[1], points[-3]])
                new_edges.append([1, lbls[1], tri[0], tri[2], points[-2]])
                new_edges.append([1, lbls[2], tri[1], tri[2], points[-1]])
            elif t == 'P3':
                new_edges.append([1, lbls[0], tri[0], tri[1], points[-6]])
                new_edges.append([1, lbls[0], tri[0], tri[1], points[-5]])
                new_edges.append([1, lbls[1], tri[0], tri[2], points[-4]])
                new_edges.append([1, lbls[1], tri[0], tri[2], points[-3]])
                new_edges.append([1, lbls[2], tri[1], tri[2], points[-2]])
                new_edges.append([1, lbls[2], tri[1], tri[2], points[-1]])
        
        no_parent = (self.parent is None)
        # prepare coordinate changes
        self.ref2tri_dict = {}
        # store new nodes for P2 polynomials
        new_nodes   = []
        new_edgeref = []
        new_triref  = []
        new_idx     = len(self.nodes)
        if self.fe_type == 'P1':
            for p in range(len(self.triangles)):
                tri = self.triangles[p]
                self.ref2tri_dict[p] = [self.nodes[tri[i]] for i in range(3)]
        else:
            edges_idx = self._elements_idx('edge')
            tris_idx  = self._elements_idx('triangle')
            for p in range(len(self.triangles)):
                tri = self.triangles[p]
                surf_ref = self.elements[tris_idx[p],1]
                l = [self.nodes[tri[i]] for i in range(3)]
                _add_points(self.fe_type, l)
                self.ref2tri_dict[p] = l
                if no_parent:
                    # new tri ref
                    new_l = [list(n) for n in l[-(self.tricount-3):]]
                    refs  = [2, surf_ref] + list(tri)
                    for new_node in new_l:
                        nidx = list_in_list(new_node, new_nodes)
                        if nidx == -1:
                            new_nodes.append(new_node)
                            refs.append(new_idx)
                            new_idx += 1
                        else: refs.append(len(self.nodes) + nidx)
                    new_triref.append(refs)

                    # new edge ref
                    # get back each edge label
                    lbls = []
                    # .. edge 0-1
                    ref_idx_1 = np.where((self.edges[:,0] == tri[0]) & (self.edges[:,1] == tri[1]))[0]
                    ref_idx_2 = np.where((self.edges[:,0] == tri[1]) & (self.edges[:,1] == tri[0]))[0]
                    if len(ref_idx_1) == 1: lbl = self.elements[edges_idx[ref_idx_1], 1]
                    elif len(ref_idx_2) == 1: lbl = self.elements[edges_idx[ref_idx_2], 1]
                    else: lbl = surf_ref
                    lbls.append(lbl)
                    # .. edge 0-2
                    ref_idx_1 = np.where((self.edges[:,0] == tri[0]) & (self.edges[:,1] == tri[2]))[0]
                    ref_idx_2 = np.where((self.edges[:,0] == tri[2]) & (self.edges[:,1] == tri[0]))[0]
                    if len(ref_idx_1) == 1: lbl = self.elements[edges_idx[ref_idx_1], 1]
                    elif len(ref_idx_2) == 1: lbl = self.elements[edges_idx[ref_idx_2], 1]
                    else: lbl = surf_ref
                    lbls.append(lbl)
                    # .. edge 1-2
                    ref_idx_1 = np.where((self.edges[:,0] == tri[1]) & (self.edges[:,1] == tri[2]))[0]
                    ref_idx_2 = np.where((self.edges[:,0] == tri[2]) & (self.edges[:,1] == tri[1]))[0]
                    if len(ref_idx_1) == 1: lbl = self.elements[edges_idx[ref_idx_1], 1]
                    elif len(ref_idx_2) == 1: lbl = self.elements[edges_idx[ref_idx_2], 1]
                    else: lbl = surf_ref
                    lbls.append(lbl)
                    # apply label on new points
                    _set_edges(self.fe_type, lbls, tri, refs, new_edgeref)
            if no_parent and len(new_nodes) > 0:
                # get useful indices
                tris_idx      = self._elements_idx('triangle')
                # make new triangles
                new_tris_arr  = np.array(new_triref)
                self.nodes    = np.concatenate((self.nodes, np.array(new_nodes)))
                # make new edges
                new_edges_arr = np.array(new_edgeref)
                if len(new_edges_arr) == 0:
                    new_edges_arr = self._elements('edge')
                Ne = len(new_edges_arr)
                d = new_tris_arr.shape[1] - new_edges_arr.shape[1]
                new_edges_arr = np.hstack((new_edges_arr, np.int_(-np.ones((Ne, d)))))
                # make new elements
                self.elements = np.concatenate((new_edges_arr, new_tris_arr))
                # recompute element arrays
                self.edges     = (self.elements[:Ne])[:,-self.tricount:-3]
                self.triangles = (self.elements[Ne:])[:,-self.tricount:]
                
        self.surf_refs = list(np.unique(self._elements('triangle')[:,1]))
        # sort edges array
        self.edges = self.edges[self.edges[:,0].argsort()]
        # get useful counts
        self.Nv = len(self.nodes)
        self.Ne = len(self.edges)
        self.Nt = len(self.triangles)
        
        # prepare determinants and triangle areas
        if no_parent:
            self.det_triangles = {}
            self.areas         = {}
            self.transf_vecs  = {}
            dphi = { 0: np.matrix([[-1], [-1]]),
                     1: np.matrix([[1], [0]]),
                     2: np.matrix([[0], [1]]) }
            for p in range(self.Nt):
                x1, y1, _ = self.ref2tri(p,0)
                x2, y2, _ = self.ref2tri(p,1)
                x3, y3, _ = self.ref2tri(p,2)
                d = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
                self.det_triangles[p] = lambda a, b, d=d: d
                self.areas[p]         = d * 0.5
                B = np.array([[y3-y1,y1-y2],[x1-x3,x2-x1]])
                self.transf_vecs[p] = np.array([B.dot(dphi[0]),
                                                B.dot(dphi[1]),
                                                B.dot(dphi[2])])
        else:
            self.det_triangles = self.parent.det_triangles
            self.areas         = self.parent.areas
            self.transf_vecs   = self.parent.transf_vecs

        # prepare precomputing flags
        self._prepared_mtr_mass     = False
        self._prepared_mtr_rigidity = False

    def precompute_mtr_mass(self):
        """Precomputes the mass matrix of the domain on each triangle (small
        overhead to speed up solving)."""
        self.mats_mass = {}
        o = self.fe_type
        if self.parent is None:
            for p in range(self.Nt):
                x1, y1, _ = self.ref2tri(p,0)
                x2, y2, _ = self.ref2tri(p,1)
                x3, y3, _ = self.ref2tri(p,2)
                self.mats_mass[p] = mtr_mass(o, x1, y1, x2, y2, x3, y3)
        else:
            if not self.parent.mass_mtr_prepared:
                self.parent.precompute_mtr_mass()
            self.mats_mass = self.parent.mats_mass
            
        # transfer to subdomains
        for s in self.subdomains:
            if isinstance(s, np.int64):
                self.subdomains[s].mats_mass = self.mats_mass
        self._prepared_mtr_mass = True

    def precompute_mtr_rigidity(self):
        """Precomputes the rigidity matrix of the domain on each triangle (small
        overhead to speed up solving)."""
        self.mats_rigidity = {}
        o = self.fe_type
        if self.parent is None:
            for p in range(self.Nt):
                x1, y1, _ = self.ref2tri(p,0)
                x2, y2, _ = self.ref2tri(p,1)
                x3, y3, _ = self.ref2tri(p,2)
                self.mats_rigidity[p] = mtr_rigidity(o, x1, y1, x2, y2, x3, y3)
        else:
            if not self.parent.rig_mtr_prepared:
                self.parent.precompute_mtr_rigidity()
            self.mats_rigidity = self.parent.mats_rigidity

        # transfer to subdomains
        for s in self.subdomains:
            if isinstance(s, np.int64):
                self.subdomains[s].mats_rigidity = self.mats_rigidity
        self._prepared_mtr_rigidity = True
        
    @property
    def mass_mtr_prepared(self):
        """Checks if the mass matrix was already precomputed for this domain."""
        return self._prepared_mtr_mass
    @property
    def rig_mtr_prepared(self):
        """Checks if the rigidity matrix was already precomputed for this domain."""
        return self._prepared_mtr_rigidity
        
    @classmethod
    @typechecker((type,str,), None)
    def from_file(cls, filename, **kwargs):
        """Constructor to build a ``DiscreteDomain`` instance from a source
        mesh file.
        
        Parameters
        ----------
        filename : str
            Path of the mesh file.
        """
        if not os.path.isfile(filename):
            Logger.serror('Could not load mesh file: "{}"'.format(filename))
        return cls(filename=filename, **kwargs)
    @classmethod
    @typechecker((type,np.ndarray,np.ndarray,), None)
    def from_data(cls, nodes, elements, **kwargs):
        """Constructor to build a ``DiscreteDomain`` instance directly from a
        set of nodes and a defined connectivity.
        
        Parameters
        ----------
        nodes : numpy.ndarray
            Array of nodes' coordinates in 3D cartesian coordinates (the array
            is of size [nb_vertices x 3]).
        elements : numpy.ndarray
            Connectivity for the nodes: edges and triangles definition (the array
            is of size [nb_elements x 5]). Each element has a type (1: edge,
            2: triangle), a Physical Label to set its subdomain and the indices
            of its nodes (2 for an edge, 3 for a triangle).
            For an edge, the last slot should be -1 (since the index is unused).
        """
        Logger.slog('This initialization method for DiscreteDomain is not fully '
                    'implemented...\nData formatting for "nodes" and "elements" '
                    'arrays can be painful and will not be checked automatically,'
                    ' so use it at your own risk!', level='info')
        if not isinstance(nodes, np.ndarray) or not isinstance(elements, np.ndarray):
            Logger.serror('Incorrect data type for direct DiscreteDomain '
                          'creation.\nYou must provide two NumPy arrays with the '
                          'nodes and the elements (connectivity) of your mesh.')
        return cls(nodes=nodes, elements=elements, **kwargs)
    @classmethod
    def subdomain(cls, elements, parent, tag):
        """Constructor to build a subdomain ``DiscreteDomain`` instance from its
        parent domain. The new domain nodes are assumed to be the same as its
        parent's.
        
        Parameters
        ----------
        elements : numpy.ndarray
            Connectivity for the nodes: edges and triangles definition (the array
            is of size [nb_elements x 5]). Each element has a type (1: edge,
            2: triangle), a Physical Label to set its subdomain and the indices
            of its nodes (2 for an edge, 3 for a triangle).
            For an edge, the last slot should be -1 (since the index is unused).
        parent : `.DiscreteDomain`
            Parent domain.
        tag : int
            Physical label specific to this subdomain.
        """
        return cls(nodes=parent.nodes, elements=elements, name=str(tag), tag=tag,
                   parent=parent, fe_type=parent.fe_type)
    @classmethod
    @typechecker((type,[int,float],[int,float]), None,
    msg_='Cannot initialize "$FUNCNAME" primitive: $IN? param #$PARAMIDX should ' \
         'be of type "$PARAMTYPE".')
    def rectangle(cls, width, height, step=10, nb_borders=1, **kwargs):
        """Constructor to build a DiscreteDomain instance on-the-go with
        a rectangular 2D geometry.
        
        The parameters are the same as in the Geometrizor's make_rectangle()
        function."""
        nodes, elements = make_rectangle(width, height, step+1, nb_borders)
        return cls(nodes=nodes, elements=elements, **kwargs)
    @classmethod
    @typechecker((type,[int,float]), None,
    msg_='Cannot initialize "$FUNCNAME" primitive: $IN? param #$PARAMIDX should ' \
         'be of type "$PARAMTYPE".')
    def line(cls, length, step=10, nb_borders=1, **kwargs):
        """Constructor to build a DiscreteDomain instance on-the-go with
        a rectangular 2D geometry.
        
        The parameters are the same as in the Geometrizor's make_line()
        function."""
        nodes, elements = make_line(length, step+1, nb_borders)
        return cls(nodes=nodes, elements=elements, **kwargs)
    @classmethod
    @typechecker((type,[int,float]), None,
    msg_='Cannot initialize "$FUNCNAME" primitive: $IN? param #$PARAMIDX should ' \
         'be of type "$PARAMTYPE".')
    def circle(cls, radius, astep=20, rstep=3, nb_borders=1, **kwargs):
        """Constructor to build a DiscreteDomain instance on-the-go with
        a circular 2D geometry.
        
        The parameters are the same as in the Geometrizor's make_circle()
        function."""
        nodes, elements = make_circle(radius, astep+1, rstep+1, nb_borders)
        return cls(nodes=nodes, elements=elements, **kwargs)
    @classmethod
    @typechecker((type,[int,float],[int,float]), None,
    msg_='Cannot initialize "$FUNCNAME" primitive: $IN? param #$PARAMIDX should ' \
         'be of type "$PARAMTYPE".')
    def ring(cls, rint, rext, astep=20, rstep=3, nb_borders=1, **kwargs):
        """Constructor to build a DiscreteDomain instance on-the-go with
        an annular 2D geometry.
        
        The parameters are the same as in the Geometrizor's make_ring()
        function."""
        if rint >= rext:
            Logger.serror('Cannot initialize "ring" primitive: "rint" is greater '
                          'than or equal to "rext"! ({} >= {})'.format(rint, rext))
        nodes, elements = make_ring(rint, rext, astep+1, rstep+1, nb_borders)
        return cls(nodes=nodes, elements=elements, **kwargs)
    
    @staticmethod
    def merge(domains):
        """Merges multiple `:class:DiscreteDomain` instances into one. The
        instances must share the same parent domain and the same set of nodes.
        
        Parameters
        ----------
        domains : list(DiscreteDomain)
            List of instances to merge.
        """
        parent   = domains[0].parent
        elements = []
        for d in domains: elements.extend(list(d.elements))
        elements = np.unique(elements, axis=0)
        names    = [d.name for d in domains]
        return DiscreteDomain.subdomain(elements, parent,
                                        'MERGED:' + ','.join(names))
    
    @property                    
    def base_functions(self):
        """Returns the base functions of this instance (i.e.: the identity
        matrix of the vertices)."""
        return np.eye(self.Nv, self.Nv)
    
    def loc2glob(self, p, i):
        """Returns the global index in the mesh corresponding to a local vertex
        in a triangle.
        
        Parameters
        ----------
        p : int
            Index of the triangle.
        i : int
            Local index of the vertex (inside the 'p' triangle).
        """
        return self.triangles[p,i]
        
    def ref2tri(self, p, i=None):
        """Returns the real coordinates of one or all vertices of a triangle of
        the instance.
        
        Parameters
        ----------
        p : int
            Index of the triangle.
        i : int or None, optional
            If None, all the vertices are checked; else only the vertex with
            local index 'i' in the triangle 'p' is checked.
        """
        if i is None: return self.ref2tri_dict[p]
        else:         return self.ref2tri_dict[p][i]
            
    def tridet(self, p):
        """Returns the determinant of a triangle of the instance, equal to
        twice its area.
        
        Parameters
        ----------
        p : int
            Index of the triangle.
        """
        return self.det_triangles[p]

    def transfer_vector(self, p):
        """Returns the transfer vector for a triangle of the instance.
        
        Parameters
        ----------
        p : int
            Index of the triangle.
        """
        return self.transf_vecs[p]
        
    def rigidity_transfer_matrix(self, p):
        """Returns the rigidty transfer matrix for a triangle of the instance
        (computed as the multiplication of the transfer matrix's transpose by
        itself).
        
        Parameters
        ----------
        p : int
            Index of the triangle.
        """
        return self.rig_transf_vecs[p]
        
    def error(self, u, u_exact, h, norm='L2'):
        """Computes the error between the computed and the exact solution on the
        discrete domain instance.
        
        See: https://sites.math.washington.edu/~greenbau/Math_595/fem1d.m
        
        Parameters
        ----------
        u : numpy.ndarray
            Computed (approximate) solution.
        u_exact : numpy.ndarray
            Exact (reference) solution.
        h : float
            Domain refinement step (i.e.: distance between two discretization
            points).
        norm : str, optional
            Type of error to compute, among: "L2" (default).
        """
        d = np.abs(u_exact - u)
        if norm == 'L2':
            err = np.sqrt(h*np.sum(np.square(d)))
            return err / h
        else:
            Logger.serror('This norm is invalid or not yet implemented!. For '
                          'now, you can choose among: "L2".')
    
    def visualize(self, z=None, z_ex=None, dim=2, value=True, show_labels=False,
                  show_triangulation=False, complex='abs', cmap='coolwarm',
                  levels=20, figscale=(1,1), use_subtriangulation=False,
                  to_file=None, title=None, no_plot=False):
        """Plots values on this domain instance: either the z-component of each
        vertex or a specific solution if one is provided. If two sets of values
        are given, the two plots are shown side by side.
        
        Parameters
        ----------
        z : numpy.ndarray, optional
            Values to plot (on first graph). The array must have as many values
            as the domain has vertices.
        z_ex : numpy.ndarray, optional
            Values to plot (on second graph). The array must have as many values
            as the domain has vertices.
        dim : int, optional
            Specifies if the plot is in 2D or 3D (can only be '2' or '3').
        value : bool, optional
            If true (default), a colorbar is added to show the range of plotted
            values (if there are 2 graphs, the colorbar is the same for both and
            takes on the range of values from the second solution that is
            assumed to be the reference).
        show_labels : bool, optional
            If true, the instance's subdomains' labels are displayed next to
            the subdomain.
        show_triangulation : bool, optional
            If true, the instance's nodes and elements are displayed to show how
            the mesh is discretized.
        complex : str, optional
            Specifies how complex sets of values should be handled: 'abs' takes
            the absolute value (default mode), 'real' truncates to keep only the
            real part, 'imag' truncates to keep only the imaginary part.
        cmap : str, optional
            Matplotlib color map to use. Must be one of the predefined color map
            names (see: https://matplotlib.org/users/colormaps.html).
        levels : int, optional
            Number of bins (i.e.: isolevel intervals) to consider for the set(s)
            of values.
        figscale : tuple(int,int)
            Scale factor for the complete figure.
        use_subtriangulation : bool, optional
            If true, recalculates a subtriangulation to use all points of a
            triangle (if FE elements are P2, P3... there are more than 3 points
            per triangle!).
        to_file : str or None, optional
            If it is not None, this parameter is the path to the file the plot(s)
            should be saved to. The extension must be included and will determine
            the format of the file (like it does in Matplotlib, see:
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html).
        title : str or None, optional
            If it is not None, this parameter is the title of the plot (only for
            a single plot).
        no_plot : bool, optional
            If true, the plot is not shown (but it can be saved to a file and/or
            returned by the function).
        """
        if dim != 2 and dim != 3:
            Logger.slog('Cannot plot solution for {}D space. Aborting.'.format(d),
                        level='warning')
            return
        if dim == 3 and show_triangulation:
            Logger.slog('Triangulation visualization not yet implemented in 3D '
                        'projection (parameter will be ignored).', level='info')

        xmin, xmax, ymin, ymax = self.bounds
        margin = max(self.scale*0.1, xmax / 20.)
        
        x = self.nodes[:,0]
        y = self.nodes[:,1]
        Z = z if z is not None else self.z
        Z = check_complex_solution(Z, complex)
        if x.shape != Z.shape: Z = Z.reshape((len(x),))
        if z_ex is None: lvls = np.linspace(np.min(Z), np.max(Z), levels)
        else:            lvls = np.linspace(np.min(z_ex), np.max(z_ex), levels)
        if lvls[0] >= lvls[1]: lvls = None
        sc_x, sc_y = figscale
        
        tri = None; tri_ex = None
        if z_ex is None:
            if dim == 2:
                if value: fig = plt.figure(figsize=(sc_x*6,sc_y*5), dpi=120)
                else: fig = plt.figure(figsize=(sc_x*5,sc_y*5), dpi=120)
                ax = plt.subplot(111)
                ax.axis('off')
            elif dim == 3:
                if value: fig = plt.figure(figsize=(sc_x*8,sc_y*5), dpi=120)
                else: fig = plt.figure(figsize=(sc_x*7,sc_y*5), dpi=120)
                ax = plt.subplot(111, projection='3d')

            ax.set_xlim((xmin-margin, xmax+margin))
            ax.set_ylim((ymin-margin, ymax+margin))            

            # plot triangulation
            if len(self.triangles) > 0:
                if dim == 2:
                    if use_subtriangulation:
                        triangulation = mtri.Triangulation(x, y)
                    else:
                        triangulation = mtri.Triangulation(x, y, self.triangles[:,:3])
                    try:
                        tri = ax.tricontourf(triangulation, Z, levels=lvls, cmap=cmap)
                        # for c in tri.collections:
                        #     print(c.get_offsets())
                    except ValueError:
                        Z = Z.T.tolist()[0]
                        tri = ax.tricontourf(triangulation, Z, levels=lvls, cmap=cmap)
                    if show_triangulation:
                        ax.triplot(mtri.Triangulation(x, y, self.triangles[:,:3]), 'k.:')
                else:
                    tri = ax.plot_trisurf(x, y, Z, triangles=self.triangles[:,:3],
                                          cmap=cmap)
                    ax.set_zlim(np.min(Z)-margin, np.max(Z)+margin)
            else:
                ax.add_patch(mp.Rectangle(
                    (xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False
                ))
            # plot border edges
            if dim == 2:
                if self.parent is None and self.nb_subdomains > 0:
                    lines = []
                    cols  = []
                    for lbl, s in self.subdomains.items():
                        if isinstance(lbl, np.int64) and s.is_border:
                            for edge in s.edges:
                                x1, x2 = edge[0], edge[1]
                                lines.append([s.nodes[x1,:2], s.nodes[x2,:2]])
                                cols.append(COLORS[(lbl-1) % len(COLORS)])
                    lc = mc.LineCollection(lines, colors=cols, linewidths=2)
                    ax.add_collection(lc)
                elif len(self.edges) > 0:
                    lines = []
                    cols  = []
                    for _, lbl, x1, x2, _ in self._elements('edge'):
                        lines.append([self.nodes[x1,:2], self.nodes[x2,:2]])
                        cols.append(COLORS[(lbl-1) % len(COLORS)])
                    lc = mc.LineCollection(lines, colors=cols, linewidths=2)
                    ax.add_collection(lc)
                
            if title is not None: ax.set_title(title)
        else:
            fig = plt.figure(figsize=(sc_x*12,sc_y*5), dpi=120)
            if dim == 2:
                ax    = plt.subplot(121)
                ax_ex = plt.subplot(122)
                ax.axis('off')
                ax_ex.axis('off')
            elif dim == 3:
                ax    = plt.subplot(121, projection='3d')
                ax_ex = plt.subplot(122, projection='3d')

            ax.set_xlim((xmin-margin, xmax+margin))
            ax.set_ylim((ymin-margin, ymax+margin))
            ax.set_title('Approximate solution')
            ax_ex.set_xlim((xmin-margin, xmax+margin))
            ax_ex.set_ylim((ymin-margin, ymax+margin))
            ax_ex.set_title('Exact solution')

            # plot triangulation
            if len(self.triangles) > 0:
                if dim == 2:
                    if use_subtriangulation:
                        triangulation = mtri.Triangulation(x, y)
                    else:
                        triangulation = mtri.Triangulation(x, y, self.triangles[:,:3])

                    try: tri = ax.tricontourf(triangulation, Z, levels=lvls, cmap=cmap)
                    except ValueError:
                        Z = Z.T.tolist()[0]
                        tri = ax.tricontourf(triangulation, Z, levels=lvls, cmap=cmap)
                    try: tri_ex = ax_ex.tricontourf(triangulation, z_ex, levels=lvls, cmap=cmap)
                    except ValueError:
                        z_ex = z_ex.T.tolist()[0]
                        tri_ex = ax_ex.tricontourf(triangulation, z_ex, levels=lvls, cmap=cmap)
                    if show_triangulation:
                        t = mtri.Triangulation(x, y, self.triangles[:,:3])
                        ax.triplot(t, 'k.:')
                        ax_ex.triplot(t, 'k.:')
                else:
                    tri = ax.plot_trisurf(x, y, Z, triangles=self.triangles[:,:3],
                                          cmap=cmap)
                    ax.set_zlim(np.min(Z)-margin, np.max(Z)+margin)
                    tri_ex = ax_ex.plot_trisurf(x, y, z_ex,
                                                triangles=self.triangles[:,:3],
                                                cmap=cmap)
                    ax_ex.set_zlim(np.min(z_ex)-margin, np.max(z_ex)+margin)
            else:
                ax.add_patch(mp.Rectangle(
                    (xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False
                ))
                ax_ex.add_patch(mp.Rectangle(
                    (xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False
                ))
            # plot border edges
            if dim == 2:
                if self.parent is None and self.nb_subdomains > 0:
                    lines = []
                    cols  = []
                    for lbl, s in self.subdomains.items():
                        if isinstance(lbl, np.int64) and s.is_border:
                            for edge in s.edges:
                                x1, x2 = edge[0], edge[1]
                                lines.append([s.nodes[x1,:2], s.nodes[x2,:2]])
                                cols.append(COLORS[(lbl-1) % len(COLORS)])
                    lc = mc.LineCollection(lines, colors=cols, linewidths=2)
                    ax.add_collection(lc)
                    lc_ex = mc.LineCollection(lines, colors=cols, linewidths=2)
                    ax_ex.add_collection(lc_ex)
                elif len(self.edges) > 0:
                    lines = []
                    cols  = []
                    for _, lbl, x1, x2, _ in self._elements('edge'):
                        lines.append([self.nodes[x1,:2], self.nodes[x2,:2]])
                        cols.append(COLORS[(lbl-1) % len(COLORS)])
                    lc = mc.LineCollection(lines, colors=cols, linewidths=2)
                    ax.add_collection(lc)
                    lc_ex = mc.LineCollection(lines, colors=cols, linewidths=2)
                    ax_ex.add_collection(lc_ex)

            ax.autoscale()
            ax_ex.autoscale()
            plt.tight_layout()
            
        if value:
            if tri is not None:    plt.colorbar(tri, ax=ax, ticks=lvls)
            if tri_ex is not None: plt.colorbar(tri_ex, ax=ax_ex, ticks=lvls)
        elif dim == 3:
            if tri is not None:    ax.set_axis_off()
            if tri_ex is not None: ax_ex.set_axis_off()
            
        if show_labels:
            if dim == 3:
                Logger.slog('Labels visualization not yet implemented in 3D '
                            'projection (parameter will be ignored).', level='info')
            else:
                subd = {}
                if self.nb_subdomains == 0:
                    tags = np.unique(self.elements[:,1])
                    subd = {k:d for k, d in self.parent.subdomains.items() \
                            if d.tag in tags}
                else:
                    subd = self.subdomains
                border_idx = 0
                for l, s in subd.items():
                    if isinstance(l, str): continue
                    if s.is_border:
                        c         = COLORS[border_idx]
                        n1, n2    = s.edges[len(s.edges)//2]
                        t1x, t1y, _ = s.nodes[n1]
                        t2x, t2y, _ = s.nodes[n2]
                        tx, ty = t1x + (t2x - t1x)/2., t1y + (t2y - t1y)/2.
                        border_idx += 1
                    else:
                        c = [0,0,0,1]
                        tx, ty = xmin + (xmax-xmin)/2., ymin + (ymax-ymin)/2.
                    if str(s.tag) != s.name:
                        lbl = to_latex(s.name) + ' [{}]'.format(s.tag)
                    else:
                        lbl = s.name
                    ax.text(tx, ty, lbl, ha='center', va='center', fontsize=13,
                            fontweight='medium', color=c, backgroundcolor=[1,1,1,0.8])
        
        if isinstance(to_file, str): plt.savefig(to_file)
        if not no_plot: plt.show()
        return fig
