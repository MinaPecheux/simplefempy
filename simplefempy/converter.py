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
# converter.py - I/O functions for mesh import/export and solution export
# ==============================================================================
import os
import re
import base64, binascii
from itertools import dropwhile
import numpy as np

from .utils.logger import Logger
from .utils.misc import to_3args, check_complex_solution, nb_params, is_number
from .utils.maths import FunctionalFunction, dirac

PARSER_REGEX = {
    'surface': r'(.?)int2d\(([^\)]+)\)\(([^\)]+)\)',
    'edge': r'(.?)int1d\(([^\)]+)\)\(([^\)]+)\)',
    'dirichlet': r'(on)\(([^\)]+)\)',
    'grad': r'grad{([^\}]*)}',
    'domain': r'([^\(]*){([^\}]*)}',
    'vars': r'(?P<name>[^\s]*)\s*=\s*(?P<value>[^\n;]*);?'
}

def load_gmsh(data):
    """Loads a mesh from a GMSH file (version 2.2). If the data provided is a
    filename, then the file is opened in the function. Otherwise, it is the
    content of the file directly encoded in base64.
    
    Parameters
    ----------
    data : str
        Data to analyze.
    """
    try:
        data = base64.b64decode(data).decode()
        lines = data.split('\n')
    except binascii.Error:
        with open(data, 'r') as f:
            lines = f.readlines()
        
    nb_nodes = 0
    i = 0   # go by index for Python 2.x compatibility
    # (.. drop top info)
    while not lines[i].split()[0].isdigit(): i += 1
        
    # read nodes
    nb_nodes = int(lines[i].split()[0])
    i += 1
    nodes = np.zeros((nb_nodes, 3))
    for node in lines[i:]:
        if node[0] == '$': break
        info = node.strip().split()
        node_id = int(info[0])-1
        nodes[node_id, 0] = info[1]
        nodes[node_id, 1] = info[2]
        nodes[node_id, 2] = info[3]
        i += 1

    i += 2 # skip middle lines
    
    # read elements
    line = lines[i]
    nb_elems = int(line.split()[0])
    i += 1
    # (type, physical number, elem1 id, elem2 id, elem3 id)
    elements = np.zeros((nb_elems, 5), dtype=np.int)
    for elem in lines[i:]:
        if elem[0] == '$': break
        info    = elem.strip().split()
        elem_id = int(info[0])-1
        elements[elem_id, 0] = info[1]
        elements[elem_id, 1] = info[3]
        elements[elem_id, 2] = int(info[5])-1
        elements[elem_id, 3] = int(info[6])-1
        try:               elements[elem_id, 4] = int(info[7])-1
        except IndexError: elements[elem_id, 4] = 0
        i += 1
    return nodes, elements
        
def save_gmsh(filename, domain):
    """Saves an instance of DiscreteDomain (i.e.: a mesh) to a GMSH file
    (version 2.2).
    
    Parameters
    ----------
    filename : str
        Path to the mesh file.
    domain : DiscreteDomain
        Instance to save.
    """
    with open(filename, 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write('$Nodes\n{}\n'.format(domain.Nv))
        for i, n in enumerate(domain.nodes):
            f.write(' '.join([str(i)] + [str(j) for j in n]) + '\n')
        f.write('$EndNodes\n')
        f.write('$Elements\n{}\n'.format(domain.Ne + domain.Nt))
        for i, e in enumerate(domain.elements):
            elt = [str(i), str(e[0]), '1'] + [str(j) for j in e[1:] if j != -1]
            f.write(' '.join(elt) + '\n')
        f.write('$EndElements\n')

def save_mesh_to_vtk(filename, domain):
    """Saves an instance of DiscreteDomain (i.e.: a mesh) to a VTK file.
    
    Parameters
    ----------
    filename : str
        Path to the mesh file.
    domain : DiscreteDomain
        Instance to save.
    """
    with open(filename, 'w') as f:
        f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write('<UnstructuredGrid>\n')
        f.write('<Piece NumberOfPoints="{}" NumberOfCells="{}">\n'.format(domain.Nv, domain.Nt))
        f.write('<Points>\n')
        f.write('<DataArray NumberOfComponents="3" type="Float64">\n')
        for n in domain.nodes: f.write(' '.join([str(t) for t in n]) + '\n')
        f.write('</DataArray>\n')
        f.write('</Points>\n')

        f.write('<Cells>\n')
        f.write('<DataArray type="Int32" Name="connectivity">\n')
        offsets = []
        offset  = 0
        pts_per_triangle = domain.tricount
        for n in domain.edges:
            f.write(' '.join([str(t) for t in n]) + '\n')
            offset += 2
            offsets.append(str(offset))
        for n in domain.triangles:
            f.write(' '.join([str(t) for t in n]) + '\n')
            offset += pts_per_triangle
            offsets.append(str(offset))
        f.write('</DataArray>\n')
        f.write('<DataArray type="Int32" Name="offsets">\n')
        f.write('\n'.join(offsets) + '\n')
        f.write('</DataArray>\n')
        f.write('<DataArray type="UInt8" Name="types">\n')
        f.write('\n'.join(['3'] * domain.Ne + ['5'] * domain.Nt) + '\n')
        f.write('</DataArray>\n')
        f.write('</Cells>\n')
        f.write('</Piece>\n')
        f.write('</UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

def save_to_csv(path, domain, solution, complex='abs', multiexport=False):
    """Saves a solution to a CSV file.
    
    Parameters
    ----------
    path : str
        Path to the folder/file to save.
    domain : :class:`DiscreteDomain`
        Domain instance where the solution was computed.
    solution : numpy.ndarray
        Solution to save.
    complex : str, optional
        Type of cast to apply to solution (if it is complex).
    multiexport : bool, optional
        If true, then data is considered to be a time series recording and will
        be saved to multiple csv files in a folder named after the "path"
        parameter. "Domain" and "solution" must be lists of solutions with their
        associated domains
    """
    if not multiexport:
        if not '.' in path:
            Logger.slog('You are exporting a .csv file but did not specify the '
                        'extension: did you mean to export a time series in a '
                        'folder?', level='warning')
        solution = check_complex_solution(solution, complex)
        with open(path, 'w') as f:
            f.write('"x","y","z","solution"\n')
            for i, p in enumerate(domain.nodes):
                f.write(','.join([str(c) for c in p]))
                f.write(',{}\n'.format(solution[i]))
        Logger.slog('Exported solution to .csv file: "{}"'.format(path))
    else:
        if not (isinstance(solution, list) or isinstance(solution, tuple)) or \
            not (isinstance(domain, list) or isinstance(domain, tuple)):
            Logger.serror('For "multiexport" mode, input domain and solution '
                          'must be lists.')
        if not os.path.exists(path): os.mkdir(path)
        for t, sol in enumerate(solution):
            sol = check_complex_solution(sol, complex)
            with open('{}/{}.csv.{}'.format(path, path, t), 'w') as f:
                f.write('"x","y","z","solution"\n')
                for i, p in enumerate(domain[t].nodes):
                    f.write(','.join([str(c) for c in p]))
                    f.write(',{}\n'.format(sol[i]))
        Logger.slog('Exported solution to folder: "{}/"'.format(path))

def save_to_vtk(file, domain, solution):
    """Saves a solution to a VTK format unstructured grid file.
    
    Parameters
    ----------
    file : str
        Path to the file to save.
    domain : :class:`DiscreteDomain`
        Domain instance where the solution was computed.
    solution : numpy.ndarray
        Solution to save.
    """
    solution = check_complex_solution(solution, 'none')

    with open(file, 'w') as f:
        f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write('<UnstructuredGrid>\n')
        f.write('<Piece NumberOfPoints="{}" NumberOfCells="{}">\n'.format(domain.Nv, domain.Nt))
        f.write('<Points>\n')
        f.write('<DataArray NumberOfComponents="3" type="Float64">\n')
        for n in domain.nodes: f.write(' '.join([str(t) for t in n]) + '\n')
        f.write('</DataArray>\n')
        f.write('</Points>\n')

        f.write('<Cells>\n')
        f.write('<DataArray type="Int32" Name="connectivity">\n')
        offsets = []
        offset  = 0
        pts_per_triangle = domain.tricount
        for n in domain.edges:
            f.write(' '.join([str(t) for t in n]) + '\n')
            offset += 2
            offsets.append(str(offset))
        for n in domain.triangles:
            f.write(' '.join([str(t) for t in n]) + '\n')
            offset += pts_per_triangle
            offsets.append(str(offset))
        f.write('</DataArray>\n')
        f.write('<DataArray type="Int32" Name="offsets">\n')
        f.write('\n'.join(offsets) + '\n')
        f.write('</DataArray>\n')
        f.write('<DataArray type="UInt8" Name="types">\n')
        f.write('\n'.join(['3'] * domain.Ne + ['5'] * domain.Nt) + '\n')
        f.write('</DataArray>\n')
        f.write('</Cells>\n')

        f.write('<PointData Scalars="solution">\n')
        f.write('<DataArray type="Float64" Name="real" format="ascii">\n')
        f.write('\n'.join([str(t) for t in np.real(solution)]) + '\n')
        f.write('</DataArray>\n')
        f.write('<DataArray type="Float64" Name="imag" format="ascii">\n')
        f.write('\n'.join([str(t) for t in np.imag(solution)]) + '\n')
        f.write('</DataArray>\n')
        f.write('<DataArray type="Float64" Name="abs" format="ascii">\n')
        f.write('\n'.join([str(t) for t in np.abs(solution)]) + '\n')
        f.write('</DataArray>\n')
        f.write('</PointData>\n')
        
        f.write('</Piece>\n')
        f.write('</UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

    Logger.slog('Exported solution to .vtu file: "{}"'.format(file), level='info')

def variable_parser(vars):
    """Parses a string with variables into the corresponding variables (as
    Python objects). The string must be of the form:
    "name1 = value1; name2 = value2; ...".
    
    Parameters
    ----------
    vars : str
        Input string to parse into variables.
    """
    matches = re.finditer(PARSER_REGEX['vars'], vars)
    if matches is not None:
        return { m.group('name'): eval(m.group('value')) for m in matches }
    return None

def formulation_parser(s, loc):
    """Parses a variational formulation-formatted string.
    
    Parameters
    ----------
    s : str
        String to parse into a VariationalFormulation object's properties.
    loc : dict()
        Available variables.
    """
    def _treat_terms(sign, expr, a, l, domain, vars):
        if sign == '' or sign == '+': coef = 1.
        elif sign == '-': coef = -1.
        else: Logger.serror('Invalid operator before variational term: '
                            '"{}"'.format(sign), stackoffset=4)
        terms    = expr.split('*')
        rhs      = None
        nb_grads = 0
        if vars[0] in expr: is_rhs = False
        else: is_rhs = True
        fcoef = None
        for term in terms:
            if term[-1] == 'j' and term[:-1].isdigit():
                coef *= int(term[:-1])*1j
            elif term.isdigit():
                coef *= int(term)
            else:
                m = re.match(PARSER_REGEX['grad'], term)
                if m is not None:
                    nb_grads += 1
                    term = m.group(1)
                if term not in vars:
                    if is_rhs: rhs = term
                    elif term in loc:
                        if callable(loc[term]): fcoef = to_3args(loc[term])
                        elif isinstance(loc[term], np.ndarray): fcoef = loc[term]
                        elif is_number(loc[term]): coef *= loc[term]
                    else:
                        Logger.serror('Unknown variable used in Variational '
                                      'Formulation: "{}".'.format(term),
                                      stackoffset=4)
        if rhs is None:
            if nb_grads == 0: mattype = 'MASS'
            elif nb_grads == 1: mattype = 'MIXED'
            else: mattype = 'RIGIDITY'
            if fcoef is not None:
                if isinstance(fcoef, np.ndarray): rescoef = fcoef
                else: rescoef = lambda x,y,z: coef*fcoef(x,y,z)
            else: rescoef = coef
            a.append((mattype, domain, rescoef))
        else:
            coef *= -1.
            if rhs not in loc:
                Logger.serror('Unknown variable used in Variational '
                              'Formulation: "{}".'.format(rhs),
                              stackoffset=4)
            old_func = loc[rhs]
            new_func = to_3args(old_func)
            l.append((domain, new_func))
    
    s = s.replace('\n', '').replace(' ', '')
    # get variable names
    if ':' not in s:
        Logger.serror('You must define the name of the variables in your '
                      'VariationalFormulation.\nE.g.: "u,v:int2d(Vh)(u*v)"')
    vars, s = s.split(':')
    vars    = vars.split(',')
    ref_domain = None
    for l in loc.values():
        if 'DiscreteDomain' in str(type(l)):
            ref_domain = l
            break
    if ref_domain is None:
        Logger.serror('There is currently no defined DiscreteDomain!')

    a = []  # left-hand side (depends on the solution)
    l = []  # right-hand side (does not depend on the solution)
    # get surface integrals
    matches = re.findall(PARSER_REGEX['surface'], s)
    for sign, domain, expr in matches:
        if domain not in loc:
            Logger.serror('Unknown DiscreteDomain used in Variational '
                          'Formulation: "{}".'.format(domain))
        if sign == '' or '+': coef = 1.
        elif sign == '-': coef = -1.
        else: Logger.serror('Invalid operator before variational term: "{}"'.format(sign))
        _treat_terms(sign, expr, a, l, ref_domain, vars)
    # get edge integrals
    matches = re.findall(PARSER_REGEX['edge'], s)
    for sign, domain, expr in matches:
        m = re.match(PARSER_REGEX['domain'], domain)
        subdomains = None
        if m is not None:
            subdomains = [int(t) if t.isdigit() else t for t in m.group(2).split(',')]
            parent = m.group(1)
        if subdomains is None:
            Logger.serror('Invalid border integral: you must specify one or more '
                          'borders of the DiscreteDomain.')
        if parent not in loc:
            Logger.serror('Unknown DiscreteDomain used in Variational '
                          'Formulation: "{}".'.format(parent))
        dom = ref_domain(*subdomains)
        _treat_terms(sign, expr, a, l, dom, vars)
    # get dirichlet conditions
    dirichlet = None
    matches = re.findall(PARSER_REGEX['dirichlet'], s)
    for _, match in matches:
        # detailed unpacking for Python 2.x compatibility
        spl = match.split(',')
        borders, func = spl[:-1], spl[-1]
        bords = [int(b) if b.isdigit() else b for b in borders]
        if func.isdigit(): func = int(func)
        else:
            try:
                func = float(func)
            except ValueError:
                if func in loc: func = loc[func]
                else: Logger.serror('Unknown variable used in Variational '
                                    'Formulation: "{}".'.format(func))
        if dirichlet is None: dirichlet = [tuple(bords + [func])]
        else: dirichlet.append(tuple(bords + [func]))
    
    return ref_domain, a, l, dirichlet
