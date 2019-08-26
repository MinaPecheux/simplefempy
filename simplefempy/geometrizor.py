# Copyright 2019 - M. Pecheux
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# geometrizor.py - Methods to create geomtric meshes and compute interpolations
# ==============================================================================
import numpy as np
import matplotlib.tri as mtri

from .utils.logger import Logger

# For reference tables, see:
# "SEVERAL NEW QUADRATURE FORMULAS FOR POLYNOMIAL INTEGRATION IN THE TRIANGLE",
# Mark A. Taylor, Beth A. Wingate and Len P. Bos (https://arxiv.org/abs/math/0501496v2)
# (pp. 10-14)
QUADRATURE = {
    'P1': {
        'phi': { 0: 1./6.*np.array([4,1,1]),
                 1: 1./6.*np.array([1,4,1]),
                 2: 1./6.*np.array([1,1,4]) },
        'weights': [1./3.,1./3.,1./3.]
    },
    'P2': {
        'phi': { 0: np.array([0.517632341987540, -0.0748038077482150, -0.0748038077482150, 0.299215230992860, 0.299215230992860, 0.0335448115231699]),
                 1: np.array([-0.0748038077481519, 0.517632341987767, -0.0748038077482150, 0.299215230992570, 0.0335448115231333, 0.299215230992897]),
                 2: np.array([-0.0748038077481519, -0.0748038077482150, 0.517632341987767, 0.0335448115231333, 0.299215230992570, 0.299215230992897]),
                 3: np.array([-0.0482083778155631, -0.0482083778154845, -0.0847304930939949, 0.795480226200853, 0.192833511262073, 0.192833511262116]),
                 4: np.array([-0.0482083778155631, -0.0847304930939949, -0.0482083778154845, 0.192833511262073, 0.795480226200853, 0.192833511262116]),
                 5: np.array([-0.0847304930939383, -0.0482083778154845, -0.0482083778154845, 0.192833511261938, 0.192833511261938, 0.795480226201031]) },
        'weights': [0.109951743655300,0.109951743655300,0.109951743655300,0.223381589678000,0.223381589678000,0.223381589678000]
    },
    'P3': {
        'phi': { 0: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                 2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                 3: np.array([-0.00143942836624332, 0.0308010619701353, 0.0474141224901398, 0.802868847425779, -0.154718847160122, 0.194939009239235, -0.157372800421027, -0.0152122530960442, -0.0637273482140034, 0.316447636132151]),
                 4: np.array([0.0334167324217187, 0.00509548877998407, 0.0471879691764868, -0.166979337920594, 0.808084636094602, -0.0161410914801349, -0.0619725733440553, 0.199164984832044, -0.158010338624507, 0.310153530064455]),
                 5: np.array([-0.00143941644282831, 0.0474130708335991, 0.0308000702283539, 0.194932338224586, -0.157368725960874, 0.802875536875356, -0.154714736599828, -0.0637262246052705, -0.0152113272664974, 0.316439414713403]),
                 6: np.array([0.0334174369462199, 0.0471872442318416, 0.00509878872742389, -0.0161409854308452, -0.0619710814096880, -0.166982971095874, 0.808090996386503, -0.158008251073522, 0.199163073944377, 0.310145748773563]),
                 7: np.array([0.0446974552862808, 0.00632218377882510, 0.0317082416128502, -0.148677905115648, 0.184561391796212, -0.0590738078993562, -0.0142574930503484, 0.824321295433865, -0.160269284585083, 0.290667922742403]),
                 8: np.array([0.0446970158901514, 0.0317089064030374, 0.00632468507826378, -0.0590728102798606, -0.0142575328118485, -0.148676755796022, 0.184560671155364, -0.160272604310969, 0.824325303502588, 0.290663121169296]),
                 9: np.array([0.00533809362112361, -0.00266957309919977, -0.00267264018933432, -0.0157653920652016, 0.00787816806380850, -0.0157656778338600, 0.00788736696779776, 0.00826967186911804, 0.00827917783950167, 0.999220804826246]) },
        'weights': [0.0137081973800000,0.0131356049752000,0.0131358306034000,0.125933026682900,0.124015246126050,0.125930230276450,0.124012589655700,0.117420611913400,0.117419193291150,0.225289469095700]
    },
}

def make_line(length, n_points, n_borders):
    """Creates a linear 2D geometry, i.e. a very thin rectangle (nodes and
    triangles, with a Delaunay triangulation).
    
    Parameters
    ----------
    length : int, float
        Total size of the line (in horizontal direction).
    n_points : int, float
        Number of discretization points along the horizontal axis.
    n_borders : int
        Number of distinct borders to create (with independent labels). Can be
        0 (only a surface), 1 (two borders each with one connex component, at
        each endpoint of the line) or 2 (two borders each with one connex
        component which is one of the line's endpoints).
    """
    if n_points < 1:
        Logger.serror('You must give a "n_points" of at least 1.', stackoffset=3)
    if isinstance(n_points, float):
        Logger.slog('You passed in a float "n_points" parameter: the value will be '
                    'truncated to an integer.', level='warning', stackoffset=3)

    # create nodes
    xx, yy = np.linspace(0, length, n_points), np.linspace(0, length * 5e-3, 2)
    x, y   = np.meshgrid(xx, yy)
    x, y   = x.flatten(), y.flatten()
    z      = np.zeros((len(x), 1))
    nodes  = np.column_stack((x,y,z))
    
    # create connectivity
    elts  = []
    # .. create triangles (surface)
    tri   = mtri.Triangulation(nodes[:,0], nodes[:,1])
    for elt in tri.triangles: elts.append([2,10,elt[0],elt[1],elt[2]])
    # .. create edges (border)
    if n_borders == 0: pass
    elif n_borders == 1:
        elts.append([1,1,0,len(xx),-1])
        elts.append([1,1,len(xx)-1,len(x)-1,-1])
    elif n_borders == 2:
        elts.append([1,1,0,len(xx),-1])
        elts.append([1,2,len(xx)-1,len(x)-1,-1])
    else:
        Logger.slog('A "line" primitive can only be created with 0, 1 or 2 '
                    'distinct borders.\nInvalid value: {} (the domain '
                    'will be created with 1 border).'.format(n_borders),
                    level='warning')
        for nx in range(len(xx)-1):
            elts.append([1,1,nx,nx+1,-1])
            elts.append([1,1,len(x)-len(xx)+nx,len(x)-len(xx)+nx+1,-1])
        for ny in range(len(yy)-1):
            elts.append([1,1,len(xx)*ny,len(xx)*(ny+1),-1])
            elts.append([1,1,len(xx)*ny+len(yy)-1,len(xx)*(ny+1)+len(yy)-1,-1])
    elements = np.array(elts)
    
    return nodes, elements

def make_rectangle(width, height, n_points, n_borders):
    """Creates a rectangle 2D geometry (nodes and triangles, with a Delaunay
    triangulation).
    
    Parameters
    ----------
    width : int, float
        Horizontal size of the rectangle (x direction).
    height : int, float
        Vertical size of the rectangle (y direction).
    n_points : int, float
        Number of discretization points along a side (identical in horizontal
        and vertical directions).
    n_borders : int
        Number of distinct borders to create (with independent labels). Can be
        0 (only a surface), 1 (one border with one connex component around the
        surface, 4 edges), 2 (two borders each with two connex components, 2
        horizontal edges and 2 vertical edges) or 4 (four borders each with
        one connex component which is one of the rectangle's sides).
    """
    if n_points < 1:
        Logger.serror('You must give a "n_points" of at least 1.', stackoffset=3)
    if isinstance(n_points, float):
        Logger.slog('You passed in a float "n_points" parameter: the value will be '
                    'truncated to an integer.', level='warning', stackoffset=3)
    
    # create nodes
    xx, yy = np.linspace(0, width, n_points), np.linspace(0, height, n_points)
    x, y   = np.meshgrid(xx, yy)
    x, y   = x.flatten(), y.flatten()
    z      = np.zeros((len(x), 1))
    nodes  = np.column_stack((x,y,z))
    
    # create connectivity
    elts  = []
    # .. create triangles (surface)
    tri   = mtri.Triangulation(nodes[:,0], nodes[:,1])
    for elt in tri.triangles: elts.append([2,10,elt[0],elt[1],elt[2]])
    # .. create edges (border)
    if n_borders == 0: pass
    elif n_borders == 1:
        for nx in range(len(xx)-1):
            elts.append([1,1,nx,nx+1,-1])
            elts.append([1,1,len(x)-len(xx)+nx,len(x)-len(xx)+nx+1,-1])
        for ny in range(len(yy)-1):
            elts.append([1,1,len(xx)*ny,len(xx)*(ny+1),-1])
            elts.append([1,1,len(xx)*ny+len(yy)-1,len(xx)*(ny+1)+len(yy)-1,-1])
    elif n_borders == 2:
        for nx in range(len(xx)-1):
            elts.append([1,1,nx,nx+1,-1])
            elts.append([1,1,len(x)-len(xx)+nx,len(x)-len(xx)+nx+1,-1])
        for ny in range(len(yy)-1):
            elts.append([1,2,len(xx)*ny,len(xx)*(ny+1),-1])
            elts.append([1,2,len(xx)*ny+len(yy)-1,len(xx)*(ny+1)+len(yy)-1,-1])
    elif n_borders == 4:
        for nx in range(len(xx)-1):
            elts.append([1,1,nx,nx+1,-1])
            elts.append([1,3,len(x)-len(xx)+nx,len(x)-len(xx)+nx+1,-1])
        for ny in range(len(yy)-1):
            elts.append([1,4,len(xx)*ny,len(xx)*(ny+1),-1])
            elts.append([1,2,len(xx)*ny+len(yy)-1,len(xx)*(ny+1)+len(yy)-1,-1])
    else:
        Logger.slog('A "square" primitive can only be created with 0, 1, 2 '
                    'or 4 distinct borders.\nInvalid value: {} (the domain '
                    'will be created with 1 border).'.format(n_borders),
                    level='warning')
        for nx in range(len(xx)-1):
            elts.append([1,1,nx,nx+1,-1])
            elts.append([1,1,len(x)-len(xx)+nx,len(x)-len(xx)+nx+1,-1])
        for ny in range(len(yy)-1):
            elts.append([1,1,len(xx)*ny,len(xx)*(ny+1),-1])
            elts.append([1,1,len(xx)*ny+len(yy)-1,len(xx)*(ny+1)+len(yy)-1,-1])
    elements = np.array(elts)
    
    return nodes, elements
    
def make_circle(radius, an_points, rn_points, n_borders):
    """Creates a circle 2D geometry (nodes and triangles, with a Delaunay
    triangulation).
    
    Parameters
    ----------
    radius : int, float
        Radius of the circle.
    an_points : int, float
        Number of discretization points angular-wise.
    rn_points : int, float
        Number of discretization points radius-wise.
    n_borders : int
        Number of distinct borders to create (with independent labels). Can be
        0 (only a surface) or 1 (one border with one connex component).
    """
    if an_points < 1:
        Logger.serror('You must give a "an_points" of at least 1.', stackoffset=3)
    if rn_points < 1:
        Logger.serror('You must give a "rn_points" of at least 1.', stackoffset=3)
    if isinstance(an_points, float):
        Logger.slog('You passed in a float "an_points" parameter: the value will be '
                    'truncated to an integer.', level='warning', stackoffset=3)
    if isinstance(rn_points, float):
        Logger.slog('You passed in a float "rn_points" parameter: the value will be '
                    'truncated to an integer.', level='warning', stackoffset=3)

    # create nodes
    radiuses = np.linspace(0, radius, rn_points+1)
    lx, ly   = [0.], [0.]
    m        = np.linspace(0,2*np.pi,an_points,endpoint=False)
    for r in radiuses[1:]:
        lx.extend(list(r*np.cos(m)))
        ly.extend(list(r*np.sin(m)))
    x, y  = np.array(lx), np.array(ly)
    z     = np.zeros((len(x), 1))
    nodes = np.column_stack((x,y,z))

    # create connectivity
    elts  = []
    # .. create triangles (surface)
    tri   = mtri.Triangulation(nodes[:,0], nodes[:,1])
    for elt in tri.triangles: elts.append([2,10,elt[0],elt[1],elt[2]])
    # .. create edges (border)
    if n_borders == 0: pass
    elif n_borders == 1:
        for n in range(len(m)-1):
            elts.append([1,1,len(x)-len(m)+n,len(x)-len(m)+n+1,-1])
        elts.append([1,1,len(x)-1,len(x)-len(m),-1])
    else:
        Logger.slog('A "circle" primitive can only be created with 0 or 1 '
                    'distinct borders.\nInvalid value: {} (the domain will '
                    'be created with 1 border).'.format(n_borders),
                    level='warning')
        for n in range(len(m)-1):
            elts.append([1,1,len(x)-len(m)+n,len(x)-len(m)+n+1,-1])
        elts.append([1,1,len(x)-1,len(x)-len(m),-1])
    elements = np.array(elts)
    
    return nodes, elements

def make_ring(rint, rext, an_points, rn_points, n_borders):
    """Creates a ring 2D geometry (nodes and triangles, with a Delaunay
    triangulation).
    
    Parameters
    ----------
    rint : int, float
        Internal radius of the ring.
    rext : int, float
        External radius of the ring.
    an_points : int, float
        Number of discretization points angular-wise.
    rn_points : int, float
        Number of discretization points radius-wise.
    n_borders : int
        Number of distinct borders to create (with independent labels). Can be
        0 (only a surface), 1 (one border with two connex components around the
        surface) or 2 (two borders each with one connex component).
    """
    if an_points < 1:
        Logger.serror('You must give a "an_points" of at least 1.', stackoffset=3)
    if rn_points < 1:
        Logger.serror('You must give a "rn_points" of at least 1.', stackoffset=3)
    if isinstance(an_points, float):
        Logger.slog('You passed in a float "an_points" parameter: the value will be '
                    'truncated to an integer.', level='warning', stackoffset=3)
    if isinstance(rn_points, float):
        Logger.slog('You passed in a float "rn_points" parameter: the value will be '
                    'truncated to an integer.', level='warning', stackoffset=3)

    # create nodes
    radiuses = np.linspace(rint, rext, rn_points+1)
    lx, ly   = [], []
    m        = np.linspace(0,2*np.pi,an_points,endpoint=False)
    for r in radiuses:
        lx.extend(list(r*np.cos(m)))
        ly.extend(list(r*np.sin(m)))
    x, y  = np.array(lx), np.array(ly)
    z     = np.zeros((len(x), 1))
    nodes = np.column_stack((x,y,z))

    # create connectivity
    elts  = []
    # .. create triangles (surface)
    tri   = mtri.Triangulation(nodes[:,0], nodes[:,1])
    for elt in tri.triangles:
        if np.all(elt < len(m)): continue # ignore inner triangles (in hole)
        elts.append([2,10,elt[0],elt[1],elt[2]])
    # .. create edges (border)
    if n_borders == 0: pass
    elif n_borders == 1:
        for n in range(len(m)-1):
            elts.append([1,1,n,n+1,-1])
        elts.append([1,1,len(m)-1,0,-1])
        for n in range(len(m)-1):
            elts.append([1,1,len(x)-len(m)+n,len(x)-len(m)+n+1,-1])
        elts.append([1,1,len(x)-1,len(x)-len(m),-1])
    elif n_borders == 2:
        for n in range(len(m)-1):
            elts.append([1,1,n,n+1,-1])
        elts.append([1,1,len(m)-1,0,-1])
        for n in range(len(m)-1):
            elts.append([1,2,len(x)-len(m)+n,len(x)-len(m)+n+1,-1])
        elts.append([1,2,len(x)-1,len(x)-len(m),-1])
    else:
        Logger.slog('A "ring" primitive can only be created with 0, 1 or 2 '
                    'distinct borders.\nInvalid value: {} (the domain will '
                    'be created with 2 borders).'.format(n_borders),
                    level='warning')
        for n in range(len(m)-1):
            elts.append([1,1,n,n+1,-1])
        elts.append([1,1,len(m)-1,0,-1])
        for n in range(len(m)-1):
            elts.append([1,2,len(x)-len(m)+n,len(x)-len(m)+n+1,-1])
        elts.append([1,2,len(x)-1,len(x)-len(m),-1])
    elements = np.array(elts)
    return nodes, elements

def interpolate(type, domain, f, element):
    """Interpolates a function on a DiscreteDomain instance. It can perform an
    edge-wise or surface-wise computation.
    
    For edges, the 1/3-Simpson rule is used.
    For surfaces, the usual Hammer's rules are used.
    The geometrical interpolation base functions are the classical hat functions
    for each type of finite elements.
    
    Parameters
    ----------
    type : str
        Specifies the computation type, among: "edge" or "surface".
    domain : DiscreteDomain
        Discrete domain to perform the computation on.
    f : func
        Function to interpolate (it will be multiplied by the base functions).
    element : array(int) or int
        Element to integrate (edge or surface).
    """
    fe_type = domain.fe_type
    if fe_type not in QUADRATURE:
        Logger.serror('Finite Elements of type "{}" are not yet implemented'
                      '!'.format(self.fe_type), stackoffset=100)
                      
    B = np.zeros((domain.Nv,1))
    if type == 'edge':
        s1 = domain.nodes[element[0]]
        s2 = domain.nodes[element[1]]
        length = np.linalg.norm(s2-s1)
        smid   = (s1+s2)/2.
        v1 = length/6. * (f(s1) + 2.*f(smid))
        v2 = length/6. * (2.*f(smid) + f(s2))
        B[element[0]] += v1
        B[element[1]] += v2
    elif type == 'surface':
        quad = QUADRATURE[fe_type]
        pts = domain.ref2tri(element)
        weighted_pts = []
        phi_ = quad['phi']
        for k in range(len(pts)):
            phiK = phi_[k]
            weighted_pts.append(sum([p*(phiK[idx]) for idx,p in enumerate(pts)]))
        vals = np.array(list(map(f, weighted_pts)))
        w    = quad['weights']
        for i in range(domain.tricount):
            I = domain.loc2glob(element,i)
            for m in range(domain.tricount):
                B[I] += w[m] * vals[m] * phi_[m][i]
        B *= domain.areas[element]
    else:
        Logger.serror('Unknown interpolation type: "{}".\nCannot perform '
                      'operation.'.format(type))
    return B
