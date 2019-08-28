# Copyright 2019 - M. Pecheux
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# demo.py - A set of examples to showcase most of the features of SimpleFEMPy
# ==============================================================================
import sys
if sys.version_info[0] < 3: PY_V = 2
else: PY_V = 3

import numpy as np

from simplefempy.discretizor import DiscreteDomain
from simplefempy.converter import save_to_csv, save_to_vtk
from simplefempy.solver import VariationalFormulation, EulerExplicit, \
    EulerImplicit, CrankNicholson
from simplefempy.utils.misc import plot, make_animation, InterpolationError
from simplefempy.utils.maths import FunctionalFunction, dirac
from simplefempy.utils.wrappers import multievaluate

## UTIL functions (reused by other functions)
## --------------
def _acoustic_diffraction(Vh):
    k     = 2*np.pi; alpha = 0.
    f     = lambda x,y: 0.
    uinc  = lambda x,y: np.exp(1j*k*(x*np.cos(alpha) + y*np.sin(alpha)))
    d     = lambda x,y: -uinc(x,y)

    # prepare variational formulation
    Problem = VariationalFormulation.from_str(
        'u,v:-int2d(Vh)(grad{u}*grad{v}) + int2d(Vh)(k*k*u*v)' \
        '- int1d(Vh{GammaExt})(1j*k*u*v) - int2d(Vh)(f*v)' \
        '+ on(GammaInt, d)', locals())
    # solve equation
    u = Problem.solve()
    # plot result
    sol = u+uinc(*Vh.points)
    plot(Vh, sol, cmap='jet', figscale=(1.1, 1))
    
vars_1d = {}
def _cable_1d(f_pt=1.):
    Vh = vars_1d['domain']
    f = lambda x: np.abs(x - f_pt) - 1.
    P = VariationalFormulation(Vh,
        [('RIGIDITY', Vh, vars_1d['p']), ('MASS', Vh, vars_1d['q'])],
        [(Vh, f)],
        dirichlet=(1, 0.))
    u = P.solve()
    return (Vh, u)

def _time_dependent(scheme):
    u0 = lambda x, y: np.sin(np.pi*x) * np.sin(np.pi*y)
    f = lambda x, y, t: (2*np.pi*np.pi - 1)*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)
    
    Vh = DiscreteDomain.rectangle(1,1,n_points=8)
    Tmax = 1; dt = 5e-4; n_iter = int(Tmax/dt)+1
    init_u = u0(*Vh.points).reshape((Vh.Nv,1))

    if scheme == 'euler-exp':
        Scheme = EulerExplicit(init_u, Vh, dt=dt, f=f, dirichlet=(1, 0.))
        all_u = Scheme.solve(n_iter, all_solutions=True)
        iter_u = all_u[::20]
        make_animation('heat-euler-exp.mp4', [Vh]*len(iter_u), iter_u, dim=3)
    elif scheme == 'euler-imp':
        Scheme = EulerImplicit(init_u, Vh, dt=dt, f=f, dirichlet=(1, 0.))
        all_u = Scheme.solve(n_iter, all_solutions=True)
        iter_u = all_u[::20]
        make_animation('heat-euler-imp.mp4', [Vh]*len(iter_u), iter_u, dim=3)
    elif scheme == 'cn':
        Scheme = CrankNicholson(init_u, Vh, dt=dt, f=f, dirichlet=(1, 0.))
        all_u = Scheme.solve(n_iter, all_solutions=True)
        iter_u = all_u[::20]
        make_animation('heat-cn.mp4', [Vh]*len(iter_u), iter_u, dim=3)

## DEMO functions (called from the menu)
## --------------
def acoustic_diffraction_simple():
    # create discrete domain
    Vh = DiscreteDomain.ring(0.5, 1.0, an_points=40, rn_points=20, n_borders=2,
        labels={ 'Omega': 10, 'GammaInt': 1, 'GammaExt': 2 })
    _acoustic_diffraction(Vh)

def acoustic_diffraction_file():
    # get input from user: name of the mesh file
    if PY_V == 2: filename = raw_input('Enter a .msh file name: ')
    elif PY_V == 3: filename = input('Enter a .msh file name: ')
    while not filename.endswith('.msh'):
        print('Incorrect file name. The mesh file must be a .msh file (GMSH'
              'v. 2.2 format)!')
        if PY_V == 2: filename = raw_input('Enter a .msh file name: ')
        elif PY_V == 3: filename = input('Enter a .msh file name: ')
    # create discrete domain
    Vh = DiscreteDomain.from_file(filename,
        labels={ 'Omega': 10, 'GammaInt': 1, 'GammaExt': 2 })
    _acoustic_diffraction(Vh)

def wave_interference():
    k = 1.5*np.pi
    f = FunctionalFunction(dirac(1.,2.,tol=0.1)) + FunctionalFunction(dirac(5.,2.,tol=0.1))

    # discretize domain
    Vh = DiscreteDomain.rectangle(6,4,n_points=50)
    # prepare variational formulation
    Problem = VariationalFormulation.from_str(
        'u,v:-int2d(Vh)(grad{u}*grad{v}) + int2d(Vh)(k*k*u*v)' \
        '+ int1d(Vh{1})(1j*k*u*v) + int2d(Vh)(f*v)', locals())
    # solve equation
    u = Problem.solve()
    # plot result
    plot(Vh, u, cmap='jet', value=False, figscale=(3,1))

def chemicals_propagation():
        f = lambda x,y: 0.
        w = np.array([-0.01, 0.])
        # discretize domain
        Vh = DiscreteDomain.rectangle(200, 5, n_points=60, n_borders=4,
            labels={'Omega': 10, 'Gamma_U': 2, 'Gamma_C': 4})
        # prepare variational formulation
        Problem = VariationalFormulation.from_str(
            'u,v : int2d(Vh)(grad{u}*grad{v}) + int2d(Vh)(w*u*grad{v})' \
            '- int2d(Vh)(f*v) + on(2, 10)', locals()
        )
        # solve equation
        u = Problem.solve()
        # plot result
        plot(Vh(10,2,4), u, cmap='viridis', figscale=(2,1), show_labels=True,
             title='Chemicals propagation in a river')

def deformation_1d_anim():
    # prepare variables common to all executions
    vars_1d['domain'] = DiscreteDomain.line(1, n_points=30)
    vars_1d['p'] = 1
    vars_1d['q'] = 1
    # prepare list of force application points
    f_pt_list = np.concatenate((
        np.linspace(0.05, 0.95, 30), np.linspace(0.9, 0.05, 28, endpoint=False)
    ))
    # run a multievaluation
    multi_solutions = multievaluate(_cable_1d)(f_pt=f_pt_list)
    # make an animation with all results
    Vh, u = zip(*multi_solutions)
    make_animation('1d-cable.gif', Vh, u, cmap='copper', value=False, fps=200,
                    dim=3)

def deformation_1d_csv():
    # prepare variables common to all executions
    vars_1d['domain'] = DiscreteDomain.line(1, n_points=30)
    vars_1d['p'] = 1
    vars_1d['q'] = 1
    # prepare list of force application points
    f_pt_list = np.concatenate((
        np.linspace(0.05, 0.95, 30), np.linspace(0.9, 0.05, 28, endpoint=False)
    ))
    # run a multievaluation
    multi_solutions = multievaluate(_cable_1d)(f_pt=f_pt_list)
    # make an animation with all results
    Vh, u = zip(*multi_solutions)
    save_to_csv('1d-cable', Vh, u, multiexport=True)

def poisson():
    f = lambda x, y: x*y
    for fe_type in ['P1', 'P2', 'P3']:
        # discretize domain
        Vh = DiscreteDomain.circle(1, an_points=40, rn_points=20, fe_type=fe_type)
        # prepare variational formulation
        Problem = VariationalFormulation(Vh,
            [('RIGIDITY', Vh)], [(Vh, f)], dirichlet=(1,0.))
        # solve equation
        u = Problem.solve()
        # saves directly (no plot display)
        plot(Vh, u, title='Poisson equation ({} polynomials)'.format(fe_type),
             value=False, no_plot=True, to_file='poisson-{}.jpg'.format(fe_type))

def poisson_robin():
    f = lambda x, y: (2*np.pi*np.pi + 1)*np.sin(np.pi*x)*np.sin(np.pi*y)
    # discretize domain
    Vh = DiscreteDomain.rectangle(1, 1, n_points=20, n_borders=4)
    # prepare variational formulation
    Problem = VariationalFormulation(Vh,
        [('RIGIDITY', Vh), ('MASS', Vh(1), -10.)],
        [(Vh, f), (Vh(1), lambda x,y: np.cos(y))],
        dirichlet=(2,3,4, 0.)
    )
    # solve equation
    u = Problem.solve()
    # plot result
    plot(Vh, u, cmap='CMRmap', dim=3)

def heat_diffusion():
    D = 0.54 # heat diffusion coefficient
    f = dirac(1.,1.)
    # discretize domain
    Vh1 = DiscreteDomain.from_file('resources/piece-1border.msh',
        labels={'Omega': 10, 'Gamma': 1})
    Vh2 = DiscreteDomain.from_file('resources/piece-2borders.msh',
        labels={'Omega': 10, 'Gamma_N': 1, 'Gamma_D': 2})
    # prepare variational formulations
    Problem1 = VariationalFormulation.from_str(
        'u,v:int2d(Vh1)(D*grad{u}*grad{v}) - int2d(Vh1)(f*v)' \
        '+ on(1, 10)', locals())
    Problem2 = VariationalFormulation.from_str(
        'u,v:int2d(Vh2)(D*grad{u}*grad{v}) - int2d(Vh2)(f*v)' \
        '+ on(2, 10)', locals())
    # solve equations
    u1 = Problem1.solve()
    u2 = Problem2.solve()
    plot(Vh1, u1, show_labels=True, to_file='heat-diffusion1.jpg', value=False)
    plot(Vh2, u2, show_labels=True, to_file='heat-diffusion2.jpg', value=False)

def wifi_diffusion():
    f = dirac(0.1, 0.1, bounds=[0.05,2.95,0.05,1.95])
    frequency = 2.4e9
    k = 2*np.pi*frequency / 3e8
    n = 2.4+0.01j

    # discretize domain
    Vh = DiscreteDomain.from_file('resources/room.msh',
        labels={ 'OmegaAir': 10, 'OmegaWall': 11, 'Gamma': 1 })
    # prepare variational formulation
    Problem = VariationalFormulation(Vh,
        [('RIGIDITY', Vh, -1.),
         ('MASS', Vh('OmegaAir'), k*k),
         ('MASS', Vh('OmegaWall'), n*n*k*k),
         ('MASS', Vh('Gamma'), 1j*k*n)],
        [(Vh, f)])
    # solve equation
    u = Problem.solve()
    # plot result
    save_to_vtk('wifi-solution.vtu', Vh, u)
    plot(Vh, u, cmap='gnuplot',value=False,figscale=(2,1), to_file='wifi.jpg')

def time_dependent_euler_exp():
    _time_dependent('euler-exp')

def time_dependent_euler_imp():
    _time_dependent('euler-imp')

def time_dependent_cn():
    _time_dependent('cn')

def basefunc():
    for fe_type in ['P1', 'P2', 'P3']:
        Vh = DiscreteDomain.from_file('resources/square-simple.msh',
                                      fe_type=fe_type)
        bf = Vh.base_functions
        t = ['Phi_' + str(i) for i in range(Vh.Nv)] # plot titles
        make_animation('base_functions-{}.mp4'.format(fe_type),
            [Vh]*len(bf), bf, t, fps=10, use_subtriangulation=True,
            show_triangulation=True)

def error_analysis():
    f      = lambda x,y: (2*np.pi*np.pi + 1)*np.sin(np.pi*x)*np.sin(np.pi*y)
    u_ex   = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
    norm   = 'L2'
    errors = InterpolationError(norm) # create the object to store the errors
    size   = 1
    # study the error for several finite elements and several discretization
    # n_pointss
    for n_points in range(5, 35, 5):
        # discretize domain with 2 types of finite elements
        Vh1 = DiscreteDomain.rectangle(size,size,n_points=n_points,fe_type='P1')
        Vh2 = DiscreteDomain.rectangle(size,size,n_points=n_points,fe_type='P2')
        # prepare 1st variational formulation
        Problem1 = VariationalFormulation(Vh1,
            [('RIGIDITY',Vh1), ('MASS',Vh1)], [(Vh1, f)], dirichlet=(1,0.))
        # prepare 2nd variational formulation
        Problem2 = VariationalFormulation(Vh2,
            [('RIGIDITY',Vh2), ('MASS',Vh2)], [(Vh2, f)], dirichlet=(1,0.))
        # solve equation
        u1 = Problem1.solve()
        u2 = Problem2.solve()
        # get exact solution reference
        u_exact1 = u_ex(*Vh1.points)
        u_exact2 = u_ex(*Vh2.points)
        # compute L2 error
        h = size/n_points
        errors[h] = Vh1.error(u1, u_exact1, h, norm=norm)
        errors[h] = Vh2.error(u2, u_exact2, h, norm=norm)
    # output to table and graph
    errors.output(compare_order=[1,2])
    errors.plot(compare_order=[1,2])

## MAIN functions and variables
## ----------------------------
MENU = [
    ('Acoustic Diffraction (on a ring)', acoustic_diffraction_simple,
     'This example shows the computation of the diffraction of an acoustic '
     'wave on a simple geometry: a ring.'),
    ('Acoustic Diffraction (with a .msh file)', acoustic_diffraction_file,
     'This example shows the computation of the diffraction of an acoustic '
     'wave on a geometry loaded from a .msh file.'),
    ('Wave interference (on a rectangle)', wave_interference,
     'This example illustrates the wave interference phenomenon by computing '
     'the interaction between two identical waves (with the Helmholtz equation).'),
    ('Chemicals propagation in a river', chemicals_propagation,
     'This example solves an advection-diffusion equation to study the '
     'propagation of chemicals in a river.'),
    ('Lateral deformation of a 1D cable (exported to an animation)', deformation_1d_anim,
     'This example shows how a 1D cable is deformed by the application of a '
     'lateral force (depending on where the force is applied) and makes a 3D '
     'animation of the result.'),
    ('Lateral deformation of a 1D cable (exported to multiple csv)', deformation_1d_csv,
     'This example shows how a 1D cable is deformed by the application of a '
     'lateral force (depending on where the force is applied) and exports the '
     'results to multiple CSV files (to rearrange in Paraview).'),
    ('Poisson equation for P1, P2 or P3-Lagrange finite elements (exported to images)', poisson,
     'This example compares the solutions for the Poisson equation depending on '
     'the type of Lagrange polynomials used (it exports the results to .jpg files).'),
    ('Poisson equation with Fourier-Robin conditions', poisson_robin,
     'This example computes the solution of the Poisson equation for a problem '
     'with Fourier-Robin conditions.'),
    ('Heat diffusion in a mechanical piece of hardware (with image export)', heat_diffusion,
     'This examples solves the usual heat equation on a relatively complex 2D '
     'geometry with 2 different sets of borders. It then exports the result to '
     'two .jpg files.'),
    ('Wi-Fi wave diffusion in a small appartment', wifi_diffusion,
     'This examples shows a Wi-Fi wave propagates in a small appartment (with '
     'plaster walls).'),
    ('Time-dependent heat equation (Euler explicit scheme)', time_dependent_euler_exp,
     'This example solves the time-dependent heat equation with a Euler explicit '
     'scheme (and exports the result to a .mp4 movie).'),
    ('Time-dependent heat equation (Euler implicit scheme)', time_dependent_euler_imp,
     'This example solves the time-dependent heat equation with a Euler implicit '
     'scheme (and exports the result to a .mp4 movie).'),
    ('Time-dependent heat equation (Crank-Nicholson scheme)', time_dependent_cn,
     'This example solves the time-dependent heat equation with a Crank-Nicholson '
     'scheme (and exports the result to a .mp4 movie).'),
    ('Base functions (for P1, P2, P3-Lagrange finite elements)', basefunc,
     'This example computes the base functions for the P1, P2 and P3-Lagrange '
     'polynomials and exports the result to a .mp4 movie.'),
    ('Interpolation error analysis (for P1, P2-Lagrange finite elements)', error_analysis,
     'This example shows the evolution of the interpolation error depending on '
     'the type of finite elements and the discretization n_points.'),
]

def _show_menu():
    print('SimpleFEMPy (v. 1.0) - DEMO PROGRAM')
    print('===================================')
    print('Choose an example from the menu (enter its id from 0 to {}):'.format(len(MENU)))
    for i, (title, _, _) in enumerate(MENU):
        print('{:2d} - {}'.format(i, title))
        
    print('\n<c> - Cancel and exit')

def _process_menu():
    # get input from user: example id
    if PY_V == 2: inp = raw_input('Example id: ')
    elif PY_V == 3: inp = input('Example id: ')
    while inp.lower() != 'c' and not (0 <= int(inp) < len(MENU)):
        print('Incorrect id. Choose an id from 0 to {}!'.format(len(MENU)))
        if PY_V == 2: inp = raw_input('Example id: ')
        elif PY_V == 3: inp = input('Example id: ')
    # check for cancel
    if inp.lower() == 'c': return
    # else convert to example id (integer) and print info on the example
    inp = int(inp)
    print('')
    print(MENU[inp][0])
    print('-' * len(MENU[inp][0]))
    print(MENU[inp][2])
    print('')
    # run function
    MENU[inp][1]()

if __name__ == '__main__':
    _show_menu()
    _process_menu()
