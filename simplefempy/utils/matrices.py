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
# matrices.py - Hardcoded matrices for FEM common element types
# ==============================================================================
import numpy as np

from .logger import Logger

def mass(order, x1, y1, x2, y2, x3, y3, **kwargs):
    if order == 'P1':
        M_11 = -(x1 - x3)*(y1 - y2) + (x1 - x2)*(y1 - y3)
        M_12 = -1/2*(x1 - x3)*(y1 - y2) + 1/2*(x1 - x2)*(y1 - y3)
        M_13 = -1/2*(x1 - x3)*(y1 - y2) + 1/2*(x1 - x2)*(y1 - y3)
        M_22 = -(x1 - x3)*(y1 - y2) + (x1 - x2)*(y1 - y3)
        M_23 = -1/2*(x1 - x3)*(y1 - y2) + 1/2*(x1 - x2)*(y1 - y3)
        M_33 = -(x1 - x3)*(y1 - y2) + (x1 - x2)*(y1 - y3)
        M    = 1./12. * np.matrix([
            [M_11, M_12, M_13],
            [M_12, M_22, M_23],
            [M_13, M_23, M_33]
        ])
    elif order == 'P2':
        M_11 = -1/4*(x1 - x3)*(y1 - y2) + 1/4*(x1 - x2)*(y1 - y3)
        M_12 = 1/24*(x1 - x3)*(y1 - y2) - 1/24*(x1 - x2)*(y1 - y3)
        M_13 = 1/24*(x1 - x3)*(y1 - y2) - 1/24*(x1 - x2)*(y1 - y3)
        M_14 = 0
        M_15 = 0
        M_16 = 1/6*(x1 - x3)*(y1 - y2) - 1/6*(x1 - x2)*(y1 - y3)
        M_22 = -1/4*(x1 - x3)*(y1 - y2) + 1/4*(x1 - x2)*(y1 - y3)
        M_23 = 1/24*(x1 - x3)*(y1 - y2) - 1/24*(x1 - x2)*(y1 - y3)
        M_24 = 0
        M_25 = 1/6*(x1 - x3)*(y1 - y2) - 1/6*(x1 - x2)*(y1 - y3)
        M_26 = 0
        M_33 = -1/4*(x1 - x3)*(y1 - y2) + 1/4*(x1 - x2)*(y1 - y3)
        M_34 = 1/6*(x1 - x3)*(y1 - y2) - 1/6*(x1 - x2)*(y1 - y3)
        M_35 = 0
        M_36 = 0
        M_44 = -4/3*(x1 - x3)*(y1 - y2) + 4/3*(x1 - x2)*(y1 - y3)
        M_45 = -2/3*(x1 - x3)*(y1 - y2) + 2/3*(x1 - x2)*(y1 - y3)
        M_46 = -2/3*(x1 - x3)*(y1 - y2) + 2/3*(x1 - x2)*(y1 - y3)
        M_55 = -4/3*(x1 - x3)*(y1 - y2) + 4/3*(x1 - x2)*(y1 - y3)
        M_56 = -2/3*(x1 - x3)*(y1 - y2) + 2/3*(x1 - x2)*(y1 - y3)
        M_66 = -4/3*(x1 - x3)*(y1 - y2) + 4/3*(x1 - x2)*(y1 - y3)
        M    = 1./15. * np.matrix([
            [M_11, M_12, M_13, M_14, M_15, M_16],
            [M_12, M_22, M_23, M_24, M_25, M_26],
            [M_13, M_23, M_33, M_34, M_35, M_36],
            [M_14, M_24, M_34, M_44, M_45, M_46],
            [M_15, M_25, M_35, M_45, M_55, M_56],
            [M_16, M_26, M_36, M_46, M_56, M_66]
        ])
    elif order == 'P3':
        M_11 = -19/15*(x1 - x3)*(y1 - y2) + 19/15*(x1 - x2)*(y1 - y3)
        M_12 = -11/60*(x1 - x3)*(y1 - y2) + 11/60*(x1 - x2)*(y1 - y3)
        M_13 = -11/60*(x1 - x3)*(y1 - y2) + 11/60*(x1 - x2)*(y1 - y3)
        M_14 = -3/10*(x1 - x3)*(y1 - y2) + 3/10*(x1 - x2)*(y1 - y3)
        M_15 = 0
        M_16 = -3/10*(x1 - x3)*(y1 - y2) + 3/10*(x1 - x2)*(y1 - y3)
        M_17 = 0
        M_18 = -9/20*(x1 - x3)*(y1 - y2) + 9/20*(x1 - x2)*(y1 - y3)
        M_19 = -9/20*(x1 - x3)*(y1 - y2) + 9/20*(x1 - x2)*(y1 - y3)
        M_110 = -3/5*(x1 - x3)*(y1 - y2) + 3/5*(x1 - x2)*(y1 - y3)
        M_22 = -19/15*(x1 - x3)*(y1 - y2) + 19/15*(x1 - x2)*(y1 - y3)
        M_23 = -11/60*(x1 - x3)*(y1 - y2) + 11/60*(x1 - x2)*(y1 - y3)
        M_24 = 0
        M_25 = -3/10*(x1 - x3)*(y1 - y2) + 3/10*(x1 - x2)*(y1 - y3)
        M_26 = -9/20*(x1 - x3)*(y1 - y2) + 9/20*(x1 - x2)*(y1 - y3)
        M_27 = -9/20*(x1 - x3)*(y1 - y2) + 9/20*(x1 - x2)*(y1 - y3)
        M_28 = -3/10*(x1 - x3)*(y1 - y2) + 3/10*(x1 - x2)*(y1 - y3)
        M_29 = 0
        M_210 = -3/5*(x1 - x3)*(y1 - y2) + 3/5*(x1 - x2)*(y1 - y3)
        M_33 = -19/15*(x1 - x3)*(y1 - y2) + 19/15*(x1 - x2)*(y1 - y3)
        M_34 = -9/20*(x1 - x3)*(y1 - y2) + 9/20*(x1 - x2)*(y1 - y3)
        M_35 = -9/20*(x1 - x3)*(y1 - y2) + 9/20*(x1 - x2)*(y1 - y3)
        M_36 = 0
        M_37 = -3/10*(x1 - x3)*(y1 - y2) + 3/10*(x1 - x2)*(y1 - y3)
        M_38 = 0
        M_39 = -3/10*(x1 - x3)*(y1 - y2) + 3/10*(x1 - x2)*(y1 - y3)
        M_310 = -3/5*(x1 - x3)*(y1 - y2) + 3/5*(x1 - x2)*(y1 - y3)
        M_44 = -9*(x1 - x3)*(y1 - y2) + 9*(x1 - x2)*(y1 - y3)
        M_45 = 63/20*(x1 - x3)*(y1 - y2) - 63/20*(x1 - x2)*(y1 - y3)
        M_46 = -9/2*(x1 - x3)*(y1 - y2) + 9/2*(x1 - x2)*(y1 - y3)
        M_47 = 9/4*(x1 - x3)*(y1 - y2) - 9/4*(x1 - x2)*(y1 - y3)
        M_48 = 9/4*(x1 - x3)*(y1 - y2) - 9/4*(x1 - x2)*(y1 - y3)
        M_49 = 9/10*(x1 - x3)*(y1 - y2) - 9/10*(x1 - x2)*(y1 - y3)
        M_410 = -27/10*(x1 - x3)*(y1 - y2) + 27/10*(x1 - x2)*(y1 - y3)
        M_55 = -9*(x1 - x3)*(y1 - y2) + 9*(x1 - x2)*(y1 - y3)
        M_56 = 9/4*(x1 - x3)*(y1 - y2) - 9/4*(x1 - x2)*(y1 - y3)
        M_57 = 9/10*(x1 - x3)*(y1 - y2) - 9/10*(x1 - x2)*(y1 - y3)
        M_58 = -9/2*(x1 - x3)*(y1 - y2) + 9/2*(x1 - x2)*(y1 - y3)
        M_59 = 9/4*(x1 - x3)*(y1 - y2) - 9/4*(x1 - x2)*(y1 - y3)
        M_510 = -27/10*(x1 - x3)*(y1 - y2) + 27/10*(x1 - x2)*(y1 - y3)
        M_66 = -9*(x1 - x3)*(y1 - y2) + 9*(x1 - x2)*(y1 - y3)
        M_67 = 63/20*(x1 - x3)*(y1 - y2) - 63/20*(x1 - x2)*(y1 - y3)
        M_68 = 9/10*(x1 - x3)*(y1 - y2) - 9/10*(x1 - x2)*(y1 - y3)
        M_69 = 9/4*(x1 - x3)*(y1 - y2) - 9/4*(x1 - x2)*(y1 - y3)
        M_610 = -27/10*(x1 - x3)*(y1 - y2) + 27/10*(x1 - x2)*(y1 - y3)
        M_77 = -9*(x1 - x3)*(y1 - y2) + 9*(x1 - x2)*(y1 - y3)
        M_78 = 9/4*(x1 - x3)*(y1 - y2) - 9/4*(x1 - x2)*(y1 - y3)
        M_79 = -9/2*(x1 - x3)*(y1 - y2) + 9/2*(x1 - x2)*(y1 - y3)
        M_710 = -27/10*(x1 - x3)*(y1 - y2) + 27/10*(x1 - x2)*(y1 - y3)
        M_88 = -9*(x1 - x3)*(y1 - y2) + 9*(x1 - x2)*(y1 - y3)
        M_89 = 63/20*(x1 - x3)*(y1 - y2) - 63/20*(x1 - x2)*(y1 - y3)
        M_810 = -27/10*(x1 - x3)*(y1 - y2) + 27/10*(x1 - x2)*(y1 - y3)
        M_99 = -9*(x1 - x3)*(y1 - y2) + 9*(x1 - x2)*(y1 - y3)
        M_910 = -27/10*(x1 - x3)*(y1 - y2) + 27/10*(x1 - x2)*(y1 - y3)
        M_1010 = -162/5*(x1 - x3)*(y1 - y2) + 162/5*(x1 - x2)*(y1 - y3)
        M = 1./224. * np.matrix([
            [M_11,   M_12,  M_13,  M_14,  M_15,  M_16,  M_17,  M_18,  M_19,  M_110],
            [M_12,   M_22,  M_23,  M_24,  M_25,  M_26,  M_27,  M_28,  M_29,  M_210],
            [M_13,   M_23,  M_33,  M_34,  M_35,  M_36,  M_37,  M_38,  M_39,  M_310],
            [M_14,   M_24,  M_34,  M_44,  M_45,  M_46,  M_47,  M_48,  M_49,  M_410],
            [M_15,   M_25,  M_35,  M_45,  M_55,  M_56,  M_57,  M_58,  M_59,  M_510],
            [M_16,   M_26,  M_36,  M_46,  M_56,  M_66,  M_67,  M_68,  M_69,  M_610],
            [M_17,   M_27,  M_37,  M_47,  M_57,  M_67,  M_77,  M_78,  M_79,  M_710],
            [M_18,   M_28,  M_38,  M_48,  M_58,  M_68,  M_78,  M_88,  M_89,  M_810],
            [M_19,   M_29,  M_39,  M_49,  M_59,  M_69,  M_79,  M_89,  M_99,  M_910],
            [M_110, M_210, M_310, M_410, M_510, M_610, M_710, M_810, M_910, M_1010]
        ])
    return M

def rigidity(order, x1, y1, x2, y2, x3, y3, **kwargs):
    if order == 'P1':
        D_11 = -(x1 - x2)*((x1 - x2)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (x1 - x3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))) + (x1 - x3)*((x1 - x2)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (x1 - x3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))) - (y1 - y2)*((y1 - y2)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (y1 - y3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))) + (y1 - y3)*((y1 - y2)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (y1 - y3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)))
        D_12 = -(x1 - x2)*(x1 - x3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) + (x1 - x3)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (y1 - y2)*(y1 - y3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) + (y1 - y3)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))
        D_13 = (x1 - x2)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (x1 - x2)*(x1 - x3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) + (y1 - y2)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (y1 - y2)*(y1 - y3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))
        D_22 = -(x1 - x3)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (y1 - y3)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))
        D_23 = (x1 - x2)*(x1 - x3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) + (y1 - y2)*(y1 - y3)/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))
        D_33 = -(x1 - x2)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3)) - (y1 - y2)**2/((x1 - x3)*(y1 - y2) - (x1 - x2)*(y1 - y3))
        D    = 1./2. * np.matrix([
            [D_11, D_12, D_13],
            [D_12, D_22, D_23],
            [D_13, D_23, D_33]
        ])
    elif order == 'P2':
        D_11 = -(x2**2 - 2*x2*x3 + x3**2 + y2**2 - 2*y2*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_12 = -1/3*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_13 = 1/3*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_14 = 4/3*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_15 = -4/3*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_16 = 0
        D_22 = -(x1**2 - 2*x1*x3 + x3**2 + y1**2 - 2*y1*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_23 = -1/3*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_24 = 4/3*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_25 = 0
        D_26 = 4/3*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_33 = -(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_34 = 0
        D_35 = -4/3*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_36 = 4/3*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_44 = -8/3*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_45 = 8/3*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_46 = -8/3*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_55 = -8/3*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_56 = 8/3*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_66 = -8/3*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D    = 1./2. * np.matrix([
            [D_11, D_12, D_13, D_14, D_15, D_16],
            [D_12, D_22, D_23, D_24, D_25, D_26],
            [D_13, D_23, D_33, D_34, D_35, D_36],
            [D_14, D_24, D_34, D_44, D_45, D_46],
            [D_15, D_25, D_35, D_45, D_55, D_56],
            [D_16, D_26, D_36, D_46, D_56, D_66]
        ])
    elif order == 'P3':
        D_11 = -17/2*(x2**2 - 2*x2*x3 + x3**2 + y2**2 - 2*y2*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_12 = 7/4*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_13 = -7/4*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_14 = 3/4*(19*x1*x2 - x2**2 - (19*x1 + 17*x2)*x3 + 18*x3**2 + 19*y1*y2 - y2**2 - (19*y1 + 17*y2)*y3 + 18*y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_15 = -3/4*(8*x1*x2 + x2**2 - 2*(4*x1 + 5*x2)*x3 + 9*x3**2 + 8*y1*y2 + y2**2 - 2*(4*y1 + 5*y2)*y3 + 9*y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_16 = -3/4*(19*x1*x2 - 18*x2**2 - (19*x1 - 17*x2)*x3 + x3**2 + 19*y1*y2 - 18*y2**2 - (19*y1 - 17*y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_17 = 3/4*(8*x1*x2 - 9*x2**2 - 2*(4*x1 - 5*x2)*x3 - x3**2 + 8*y1*y2 - 9*y2**2 - 2*(4*y1 - 5*y2)*y3 - y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_18 = 3/4*(x2**2 - 2*x2*x3 + x3**2 + y2**2 - 2*y2*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_19 = 3/4*(x2**2 - 2*x2*x3 + x3**2 + y2**2 - 2*y2*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_110 = 0
        D_22 = -17/2*(x1**2 - 2*x1*x3 + x3**2 + y1**2 - 2*y1*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_23 = 7/4*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_24 = -3/4*(x1**2 + 8*x1*x2 - 2*(5*x1 + 4*x2)*x3 + 9*x3**2 + y1**2 + 8*y1*y2 - 2*(5*y1 + 4*y2)*y3 + 9*y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_25 = -3/4*(x1**2 - 19*x1*x2 + (17*x1 + 19*x2)*x3 - 18*x3**2 + y1**2 - 19*y1*y2 + (17*y1 + 19*y2)*y3 - 18*y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_26 = 3/4*(x1**2 - 2*x1*x3 + x3**2 + y1**2 - 2*y1*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_27 = 3/4*(x1**2 - 2*x1*x3 + x3**2 + y1**2 - 2*y1*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_28 = 3/4*(18*x1**2 - 19*x1*x2 - (17*x1 - 19*x2)*x3 - x3**2 + 18*y1**2 - 19*y1*y2 - (17*y1 - 19*y2)*y3 - y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_29 = -3/4*(9*x1**2 - 8*x1*x2 - 2*(5*x1 - 4*x2)*x3 + x3**2 + 9*y1**2 - 8*y1*y2 - 2*(5*y1 - 4*y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_210 = 0
        D_33 = -17/2*(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_34 = 3/4*(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_35 = 3/4*(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_36 = -3/4*(x1**2 - 10*x1*x2 + 9*x2**2 + 8*(x1 - x2)*x3 + y1**2 - 10*y1*y2 + 9*y2**2 + 8*(y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_37 = -3/4*(x1**2 + 17*x1*x2 - 18*x2**2 - 19*(x1 - x2)*x3 + y1**2 + 17*y1*y2 - 18*y2**2 - 19*(y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_38 = -3/4*(9*x1**2 - 10*x1*x2 + x2**2 - 8*(x1 - x2)*x3 + 9*y1**2 - 10*y1*y2 + y2**2 - 8*(y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_39 = 3/4*(18*x1**2 - 17*x1*x2 - x2**2 - 19*(x1 - x2)*x3 + 18*y1**2 - 17*y1*y2 - y2**2 - 19*(y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_310 = 0
        D_44 = -135/4*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_45 = 27/4*(x1**2 + 2*x1*x2 + x2**2 - 4*(x1 + x2)*x3 + 4*x3**2 + y1**2 + 2*y1*y2 + y2**2 - 4*(y1 + y2)*y3 + 4*y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_46 = 135/4*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_47 = -27/4*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_48 = 27/4*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_49 = 27/4*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_410 = -81/2*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_55 = -135/4*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_56 = -27/4*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_57 = -27/4*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_58 = -135/4*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_59 = 27/4*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_510 = 81/2*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_66 = -135/4*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_67 = 27/4*(x1**2 - 4*x1*x2 + 4*x2**2 + 2*(x1 - 2*x2)*x3 + x3**2 + y1**2 - 4*y1*y2 + 4*y2**2 + 2*(y1 - 2*y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_68 = -27/4*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_69 = -27/4*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_610 = 81/2*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_77 = -135/4*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_78 = -27/4*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_79 = 135/4*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_710 = 81/2*(x1**2 - x1*x2 - (x1 - x2)*x3 + y1**2 - y1*y2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_88 = -135/4*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_89 = 27/4*(4*x1**2 - 4*x1*x2 + x2**2 - 2*(2*x1 - x2)*x3 + x3**2 + 4*y1**2 - 4*y1*y2 + y2**2 - 2*(2*y1 - y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_810 = 81/2*(x1*x2 - (x1 + x2)*x3 + x3**2 + y1*y2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_99 = -135/4*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_910 = -81/2*(x1*x2 - x2**2 - (x1 - x2)*x3 + y1*y2 - y2**2 - (y1 - y2)*y3)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D_1010 = -81*(x1**2 - x1*x2 + x2**2 - (x1 + x2)*x3 + x3**2 + y1**2 - y1*y2 + y2**2 - (y1 + y2)*y3 + y3**2)/((x2 - x3)*y1 - (x1 - x3)*y2 + (x1 - x2)*y3)
        D = 1./20. * np.matrix([
            [D_11,   D_12,  D_13,  D_14,  D_15,  D_16,  D_17,  D_18,  D_19,  D_110],
            [D_12,   D_22,  D_23,  D_24,  D_25,  D_26,  D_27,  D_28,  D_29,  D_210],
            [D_13,   D_23,  D_33,  D_34,  D_35,  D_36,  D_37,  D_38,  D_39,  D_310],
            [D_14,   D_24,  D_34,  D_44,  D_45,  D_46,  D_47,  D_48,  D_49,  D_410],
            [D_15,   D_25,  D_35,  D_45,  D_55,  D_56,  D_57,  D_58,  D_59,  D_510],
            [D_16,   D_26,  D_36,  D_46,  D_56,  D_66,  D_67,  D_68,  D_69,  D_610],
            [D_17,   D_27,  D_37,  D_47,  D_57,  D_67,  D_77,  D_78,  D_79,  D_710],
            [D_18,   D_28,  D_38,  D_48,  D_58,  D_68,  D_78,  D_88,  D_89,  D_810],
            [D_19,   D_29,  D_39,  D_49,  D_59,  D_69,  D_79,  D_89,  D_99,  D_910],
            [D_110, D_210, D_310, D_410, D_510, D_610, D_710, D_810, D_910, D_1010]
        ])
    return D
