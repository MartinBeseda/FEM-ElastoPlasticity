"""
This module contains the Python3 implementation of matlabfem library (https://github.com/matlabfem/)
intended to solve elasticity and plasticity problems.
"""

# TODO consider object-oriented architecture!

#################
# Import modules
#################
import enum
from typing import Tuple, Union, Any

import numpy as np
import numpy.matlib as npm
import scipy as sp
import scipy.sparse as ssp
import scipy.sparse.linalg as sspl
import typing as tp
from numba import njit
import logging as log
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.cm as cm
import os

##############
# Qt settings
##############
from numpy import matrix, ndarray
from scipy.sparse import csr_matrix

os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

####################
# Set logging level
####################
log.basicConfig(level=log.INFO)


# os.environ["NUMBA_DISABLE_JIT"] = "1"


# @njit
def flatten_row(v: np.array) -> np.array:
    return np.reshape(v, (1, -1), order='F')


# @njit
def flatten_col(v: np.array) -> np.array:
    return np.reshape(v, (np.size(v), 1), order='F')


class LagrangeElementType(enum.Enum):
    """ Enumeration type allowing users to select the type of Lagrange finite elements."""
    P1 = 1
    P2 = 2
    Q1 = 3
    Q2 = 4
    P4 = 5


# @njit
def get_quadrature_volume(el_type: LagrangeElementType) -> tp.Tuple[np.array, np.array]:
    """ This function specifies a numerical quadrature for volume integration,
    depending on a chosen finite element. The quadratures suggested
    below can be simply replaced by another ones.

    :param el_type: the type of Lagrange finite elements
    :type el_type: LagrangeElementType

    :return: (Xi, WF): Xi - local coordinates of quadrature points, size(Xi)=(2,n_q),
                       WF - weight factors, size(WF)=(1,n_q)
    :rtype: tuple
    """

    pt = 1 / np.sqrt(3)

    """
    Initialization values of Xi and WF (first and second element of the tuples, respectively)
    
    P1
    - the reference triangle with coordinates: [0,0], [1,0], [0,1]
    - 1-point quadrature rule, i.e., n_q=1
    
    P2
    - the reference triangle with coordinates: [0,0], [1,0], [0,1]
    - 7-point quadrature rule, i.e., n_q=7
    
    Q1
    - the reference cube with coordinates: [-1,-1], [1,-1], [1,1], [-1,1]
    - (2x2)-point quadrature rule, i.e., n_q=4
    
    Q2
    - the reference cube with coordinates: [-1,-1], [1,-1], [1,1], [-1,1]
    - (3x3)-point quadrature rule, i.e., n_q=9    
    """
    init_dict = {LagrangeElementType.P1: (np.array([[1 / 3], [1 / 3]]), np.array([[0.5]])),
                 LagrangeElementType.P2: (np.array([[0.1012865073235, 0.7974269853531, 0.1012865073235,
                                                     0.4701420641051, 0.4701420641051, 0.0597158717898, 1 / 3],
                                                    [0.1012865073235, 0.1012865073235, 0.7974269853531,
                                                     0.0597158717898, 0.4701420641051, 0.4701420641051, 1 / 3]]),
                                          0.5 * np.array([[0.1259391805448, 0.1259391805448, 0.1259391805448,
                                                           0.1323941527885, 0.1323941527885, 0.1323941527885, 0.225]])),
                 LagrangeElementType.Q1: (np.array([[-pt, -pt, pt, pt], [-pt, pt, -pt, pt]]),
                                          np.array([[1, 1, 1, 1]])),
                 LagrangeElementType.Q2: (np.array([[-pt, pt, pt, -pt, 0, pt, 0, -pt, 0],
                                                    [-pt, -pt, pt, pt, -pt, 0, pt, 0, 0]]),
                                          np.array([[25 / 81, 25 / 81, 25 / 81, 25 / 81, 40 / 81, 40 / 81, 40 / 81,
                                                     40 / 81, 64 / 81]])),
                 LagrangeElementType.P4: (np.array([[0.063089014491502, 0.06308901449102, 0.873821971016996,
                                                     0.249286745170910, 0.249286745170910, 0.501426509658179,
                                                     0.310352451033785, 0.310352451033785, 0.053145049844816,
                                                     0.053145049844816, 0.636502499121399, 0.636502499121399],
                                                    [0.063089014491502, 0.873821971016996, 0.063089014491502,
                                                     0.249286745170910, 0.501426509658179, 0.249286745170910,
                                                     0.053145049844816, 0.636502499121399, 0.310352451033785,
                                                     0.636502499121399, 0.310352451033785, 0.053145049844816]]),
                                          np.array([[0.050844906370207, 0.050844906370207, 0.050844906370207,
                                                     0.116786275726379, 0.116786275726379, 0.116786275726379,
                                                     0.082851075618374, 0.082851075618374, 0.082851075618374,
                                                     0.082851075618374, 0.082851075618374, 0.082851075618374]])/2)
                 }

    return init_dict[el_type]


def get_quadrature_surface(el_type: LagrangeElementType) -> tp.Tuple[np.array, np.array]:
    """ This function specifies a numerical quadrature for volume integration,
    depending on a chosen finite element. The quadratures suggested
    below can be simply replaced by another ones.

    :param el_type: the type of Lagrange finite elements
    :type el_type: LagrangeElementType

    :return: (Xi, WF): Xi - local coordinates of quadrature points, size(Xi)=(2,n_q),
                       WF - weight factors, size(WF)=(1,n_q)
    :rtype: tuple
    """

    pt = 1 / np.sqrt(3)

    init_dict = {LagrangeElementType.P1: (np.array([0]), np.array([2])),
                 LagrangeElementType.Q1: (np.array([0]), np.array([2])),
                 LagrangeElementType.P2: (np.array([-pt, pt]), np.array([1, 1])),
                 LagrangeElementType.Q2: (np.array([-pt, pt]), np.array([1, 1]))}

    return init_dict[el_type]


# @njit
def get_local_basis_volume(el_type: LagrangeElementType, xi: np.array) -> tuple:
    """ This function evaluates local basis functions and their derivatives at
    prescribed quadrature points depending on a chosen finite elements.

    :param el_type: the type of Lagrange finite elements
    :type el_type: LagrangeElementType

    :param xi: coordinates of the quadrature points, size(Xi)=(2,n_q)
    :type xi: np.array

    :return: (HatP, DHatP1, DHatP2): HatP   - values of basis functions at the quadrature points, size(HatP)=(n_p,n_q)
                                     DHatP1 - derivatives of basis functions at the quadrature points
                                              in the direction xi_1, size(DHatP1)=(n_p,n_q)
                                     DHatP2 - derivatives of basis functions at the quadrature points
                                              in the direction xi_2, size(DHatP2)=(n_p,n_q)
                                     Note: n_p - no. of basis functions, n_q - no. of integration points / 1 element
    :rtype: tuple
    """

    xi_1 = xi[0, ]
    xi_2 = xi[1, ]

    # For P2 case
    xi_0 = 1 - xi_1 - xi_2
    n_q = np.size(xi_1, 0)

    init_dict = {LagrangeElementType.P1: (np.array([1 - xi_1 - xi_2, xi_1, xi_2]),
                                          np.array([[-1], [1], [0]]),
                                          np.array([[-1], [0], [1]])),
                 LagrangeElementType.P2: (np.array([xi_0 * (2 * xi_0 - 1), xi_1 * (2 * xi_1 - 1), xi_2 * (2 * xi_2 - 1),
                                                    4 * xi_1 * xi_2, 4 * xi_0 * xi_2, 4 * xi_0 * xi_1]),
                                          np.array([-4 * xi_0 + 1, 4 * xi_1 - 1, np.zeros(n_q), 4 * xi_2, -4 * xi_2,
                                                    4 * (xi_0 - xi_1)]),
                                          np.array(
                                              [-4 * xi_0 + 1, np.zeros(n_q), 4 * xi_2 - 1, 4 * xi_1, 4 * (xi_0 - xi_2),
                                               -4 * xi_1])),
                 LagrangeElementType.Q2: (np.array([(1 - xi_1) * (1 - xi_2) * (-1 - xi_1 - xi_2) / 4,
                                                    (1 + xi_1) * (1 - xi_2) * (-1 + xi_1 - xi_2) / 4,
                                                    (1 + xi_1) * (1 + xi_2) * (-1 + xi_1 + xi_2) / 4,
                                                    (1 - xi_1) * (1 + xi_2) * (-1 - xi_1 + xi_2) / 4,
                                                    (1 - pow(xi_1, 2)) * (1 - xi_2) / 2,
                                                    (1 + xi_1) * (1 - pow(xi_2, 2)) / 2,
                                                    (1 - pow(xi_1, 2)) * (1 + xi_2) / 2,
                                                    (1 - xi_1) * (1 - pow(xi_2, 2)) / 2]),
                                          np.array([(1 - xi_2) * (2 * xi_1 + xi_2) / 4,
                                                    (1 - xi_2) * (2 * xi_1 - xi_2) / 4,
                                                    (1 + xi_2) * (2 * xi_1 + xi_2) / 4,
                                                    (1 + xi_2) * (2 * xi_1 - xi_2) / 4,
                                                    -xi_1 * (1 - xi_2),
                                                    (1 - pow(xi_2, 2)) / 2,
                                                    -xi_1 * (1 + xi_2),
                                                    -(1 - pow(xi_2, 2)) / 2]),
                                          np.array([(1 - xi_1) * (xi_1 + 2 * xi_2) / 4,
                                                    (1 + xi_1) * (-xi_1 + 2 * xi_2) / 4,
                                                    (1 + xi_1) * (xi_1 + 2 * xi_2) / 4,
                                                    (1 - xi_1) * (-xi_1 + 2 * xi_2) / 4,
                                                    -(1 - pow(xi_1, 2)) / 2,
                                                    -(1 + xi_1) * xi_2,
                                                    (1 - pow(xi_1, 2)) / 2,
                                                    -(1 - xi_1) * xi_2])),
                 LagrangeElementType.Q1: (np.array([(1 - xi_1) * (1 - xi_2) / 4,
                                                    (1 + xi_1) * (1 - xi_2) / 4,
                                                    (1 + xi_1) * (1 + xi_2) / 4,
                                                    (1 - xi_1) * (1 + xi_2) / 4]),
                                          np.array([-(1 - xi_2) / 4,
                                                    (1 - xi_2) / 4,
                                                    (1 + xi_2) / 4,
                                                    -(1 + xi_2) / 4]),
                                          np.array([-(1 - xi_1) / 4,
                                                    -(1 + xi_1) / 4,
                                                    (1 + xi_1) / 4,
                                                    (1 - xi_1) / 4])),
                 LagrangeElementType.P4: (np.array([ xi_0*(4*xi_0-1)*(4*xi_0-2)*(4*xi_0-3)/6,
                                                     xi_1*(4*xi_1-1)*(4*xi_1-2)*(4*xi_1-3)/6,
                                                     xi_2*(4*xi_2-1)*(4*xi_2-2)*(4*xi_2-3)/6,
                                                     4*xi_0*xi_1*(4*xi_0-1)*(4*xi_1-1),
                                                     4*xi_1*xi_2*(4*xi_1-1)*(4*xi_2-1),
                                                     4*xi_0*xi_2*(4*xi_0-1)*(4*xi_2-1),
                                                     8*xi_0*xi_1*(4*xi_0-1)*(4*xi_0-2)/3,
                                                     8*xi_0*xi_1*(4*xi_1-1)*(4*xi_1-2)/3,
                                                     8*xi_1*xi_2*(4*xi_1-1)*(4*xi_1-2)/3,
                                                     8*xi_1*xi_2*(4*xi_2-1)*(4*xi_2-2)/3,
                                                     8*xi_0*xi_2*(4*xi_2-1)*(4*xi_2-2)/3,
                                                     8*xi_0*xi_2*(4*xi_0-1)*(4*xi_0-2)/3,
                                                     32*xi_0*xi_1*xi_2*(4*xi_0-1),
                                                     32*xi_0*xi_1*xi_2*(4*xi_1-1),
                                                     32*xi_0*xi_1*xi_2*(4*xi_2-1)]),
                                          np.array([-((4*xi_0-1)*(4*xi_0-2)*(4*xi_0-3)+4*xi_0*(4*xi_0-2)*(4*xi_0-3)+4*xi_0*(4*xi_0-1)*(4*xi_0-3)+4*xi_0*(4*xi_0-1)*(4*xi_0-2))/6,
                                                    ((4*xi_1-1)*(4*xi_1-2)*(4*xi_1-3)+4*xi_1*(4*xi_1-2)*(4*xi_1-3)+4*xi_1*(4*xi_1-1)*(4*xi_1-3)+4*xi_1*(4*xi_1-1)*(4*xi_1-2))/6,
                                                    np.zeros(n_q),
                                                    4*(-xi_1*(4*xi_0-1)*(4*xi_1-1)+xi_0*(4*xi_0-1)*(4*xi_1-1)-4*xi_0*xi_1*(4*xi_1-1)+4*xi_0*xi_1*(4*xi_0-1)),
                                                    4*(xi_2*(4*xi_1-1)*(4*xi_2-1)+4*xi_1*xi_2*(4*xi_2-1)),
                                                    4*(-xi_2*(4*xi_0-1)*(4*xi_2-1)-4*xi_0*xi_2*(4*xi_2-1)),
                                                    8*(-xi_1*(4*xi_0-1)*(4*xi_0-2)+xi_0*(4*xi_0-1)*(4*xi_0-2)-4*xi_0*xi_1*(4*xi_0-2)-4*xi_0*xi_1*(4*xi_0-1))/3,
                                                    8*(-xi_1*(4*xi_1-1)*(4*xi_1-2)+xi_0*(4*xi_1-1)*(4*xi_1-2)+4*xi_0*xi_1*(4*xi_1-2)+4*xi_0*xi_1*(4*xi_1-1))/3,
                                                    8*(xi_2*(4*xi_1-1)*(4*xi_1-2)+4*xi_1*xi_2*(4*xi_1-2)+4*xi_1*xi_2*(4*xi_1-1))/3,
                                                    8*xi_2*(4*xi_2-1)*(4*xi_2-2)/3,
                                                   -8*xi_2*(4*xi_2-1)*(4*xi_2-2)/3,
                                                    8*(-xi_2*(4*xi_0-1)*(4*xi_0-2)-4*xi_0*xi_2*(4*xi_0-2)-4*xi_0*xi_2*(4*xi_0-1))/3,
                                                    32*(-xi_1*xi_2*(4*xi_0-1)+xi_0*xi_2*(4*xi_0-1)-4*xi_0*xi_1*xi_2),
                                                    32*(-xi_1*xi_2*(4*xi_1-1)+xi_0*xi_2*(4*xi_1-1)+4*xi_0*xi_1*xi_2),
                                                    32*(-xi_1*xi_2*(4*xi_2-1)+xi_0*xi_2*(4*xi_2-1))]),
                                          np.array([-((4*xi_0-1)*(4*xi_0-2)*(4*xi_0-3)+4*xi_0*(4*xi_0-2)*(4*xi_0-3)+4*xi_0*(4*xi_0-1)*(4*xi_0-3)+4*xi_0*(4*xi_0-1)*(4*xi_0-2))/6, 
                                                    np.zeros(n_q),
                                                    ((4*xi_2-1)*(4*xi_2-2)*(4*xi_2-3)+4*xi_2*(4*xi_2-2)*(4*xi_2-3)+4*xi_2*(4*xi_2-1)*(4*xi_2-3)+4*xi_2*(4*xi_2-1)*(4*xi_2-2))/6,
                                                    4*(-xi_1*(4*xi_0-1)*(4*xi_1-1)-4*xi_0*xi_1*(4*xi_1-1)),
                                                    4*(xi_1*(4*xi_1-1)*(4*xi_2-1)+4*xi_1*xi_2*(4*xi_1-1)),
                                                    4*(-xi_2*(4*xi_0-1)*(4*xi_2-1)+xi_0*(4*xi_0-1)*(4*xi_2-1)-4*xi_0*xi_2*(4*xi_2-1)+4*xi_0*xi_2*(4*xi_0-1)),
                                                    8*(-xi_1*(4*xi_0-1)*(4*xi_0-2)-4*xi_0*xi_1*(4*xi_0-2)-4*xi_0*xi_1*(4*xi_0-1))/3,
                                                   -8*xi_1*(4*xi_1-1)*(4*xi_1-2)/3,
                                                    8*xi_1*(4*xi_1-1)*(4*xi_1-2)/3,
                                                    8*(xi_1*(4*xi_2-1)*(4*xi_2-2)+4*xi_1*xi_2*(4*xi_2-2)+4*xi_1*xi_2*(4*xi_2-1))/3,
                                                    8*(-xi_2*(4*xi_2-1)*(4*xi_2-2)+xi_0*(4*xi_2-1)*(4*xi_2-2)+4*xi_0*xi_2*(4*xi_2-2)+4*xi_0*xi_2*(4*xi_2-1))/3,
                                                    8*(-xi_2*(4*xi_0-1)*(4*xi_0-2)+xi_0*(4*xi_0-1)*(4*xi_0-2)-4*xi_0*xi_2*(4*xi_0-2)-4*xi_0*xi_2*(4*xi_0-1))/3,
                                                    32*(-xi_1*xi_2*(4*xi_0-1)+xi_0*xi_1*(4*xi_0-1)-4*xi_0*xi_1*xi_2),
                                                    32*(-xi_1*xi_2*(4*xi_1-1)+xi_0*xi_1*(4*xi_1-1)),
                                                    32*(-xi_1*xi_2*(4*xi_2-1)+xi_0*xi_1*(4*xi_2-1)+4*xi_0*xi_1*xi_2)]))
                 }

    return init_dict[el_type]


def get_local_basis_surface(el_type: LagrangeElementType, xi_s: np.array) -> tuple:
    """ This function evaluates local basis functions and their derivatives at
    prescribed quadrature points depending on a chosen finite elements.

    :param el_type: the type of Lagrange finite elements
    :type el_type: LagrangeElementType

    :param xi_s: coordinates of the quadrature points, size(Xi)=(2,n_q)
    :type xi_s: np.array

    :return: (HatP, DHatP1, DHatP2): HatP   - values of basis functions at the quadrature points, size(HatP)=(n_p,n_q)
                                     DHatP1 - derivatives of basis functions at the quadrature points
                                              in the direction xi_1, size(DHatP1)=(n_p,n_q)
                                     Note: n_p - no. of basis functions, n_q - no. of integration points / 1 element
    :rtype: tuple
    """

    # xi = xi_s[0, ]
    xi = xi_s

    init_1 = (0.5 * np.array([1 - xi, 1 + xi]), np.array([[-0.5], [0.5]]))
    init_2 = (np.array([np.multiply(xi, (xi - 1) / 2),
                        np.multiply(xi, (xi + 1) / 2),
                        np.multiply(xi + 1, 1 - xi)]),
              np.array([xi - 0.5, xi + 0.5, -2 * xi]))

    init_dict = {LagrangeElementType.P1: init_1,
                 LagrangeElementType.Q1: init_1,
                 LagrangeElementType.P2: init_2,
                 LagrangeElementType.Q2: init_2}

    return init_dict[el_type]


def get_vector_volume(elements: np.array,
                      coordinates: np.array,
                      f_V_int: np.array,
                      hatp: np.array,
                      weight: np.array) -> sp.sparse.csc_matrix:
    """
    Assembling of the vector of volume forces

    :param elements: to indicate nodes belonging to each element
                        elements.shape=(n_p,n_e) where n_e is a number of elements and n_p is a number of the nodes
                        within one element
    :type elements: np.array

    :param coordinates: coordinates of the nodes, coordinates.shape=(2,n_n)
    :type coordinates: np.array

    :param f_V_int: values of volume forces at integration points f_V_int.shape=(2,n_int), where n_int=n_e*n_q
                    is a number of integration points, n_q is a number of quadrature points
    :type f_V_int: np.array

    :param hatp: values of the basis functions at quadrature points hatp.shape=(n_p,n_q)
    :type hatp: np.array

    :param weight: weight coefficients, weight.shape=(1,n_int)
    :type weight: np.array

    :return: f_V - vector of volume forces, size(f_V)=(2,n_n) where n_n is the number of nodes
    :rtype: np.array
    """

    n_n = coordinates.shape[1]
    n_e = elements.shape[1]
    n_p = elements.shape[0]
    n_q = hatp.shape[1]
    n_int = n_e * n_q

    hatphi = np.tile(hatp, (1, n_e))
    vf1 = np.multiply(hatphi, np.ones((n_p, 1)) * np.multiply(weight, f_V_int[0,])).flatten(order='F')
    vf2 = np.multiply(hatphi, np.ones((n_p, 1)) * np.multiply(weight, f_V_int[1,])).flatten(order='F')

    i_f = np.zeros((n_p, n_int)).flatten(order='F')
    j_f = np.kron(elements, np.ones((1, n_q))).flatten(order='F')

    f_V = sp.sparse.vstack((sp.sparse.csc_matrix((vf1, (i_f, j_f)), shape=(1, n_n)),
                            sp.sparse.csc_matrix((vf2, (i_f, j_f)), shape=(1, n_n))))

    return f_V


def get_vector_traction(elements_s: np.array,
                        coordinates: np.array,
                        f_t_int: np.array,
                        hatp_s: np.array,
                        dhatp1_s: np.array,
                        wf_s: np.array) -> sp.sparse.csc_matrix:
    """
    Assembling of the vector of traction forces acting on the upper side of the 2D body

    :param elements_s: to indicate nodes belonging to each surface element elements_s.shape=(n_p_s,n_e_s)
                       where n_e_s is a number of surface elements n_p_s is a number of the nodes within one
                       surface element
    :type elements_s: np.array

    :param coordinates: coordinates of the nodes, coordinates.shape=(2,n_n)
    :type coordinates: np.array

    :param f_t_int: values of traction forces at integration points f_t_int.shape=(2,n_int), where n_int=n_e*n_q
                    is a number of integration points, n_q is a number of quadrature points
    :type f_t_int: np.array

    :param hatp_s: values of the basis functions at quadrature points
    :type hatp_s: np.array

    :param dhatp1_s: xi_1-derivatives of the surface basis functions at q. p. hatps.shape=dhatp1_s.shape=(n_p_s,n_q_s)
    :type dhatp1_s: np.array

    :param wf_s: weight factors at surface quadrature points, wf_s.shape=(1,n_q_s)
    :type wf_s: np.array

    :return: f_t - vector of traction forces, f_t.shape=(2,n_n), where n_n is the number of nodes
    :rtype: np.array
    """

    n_n = coordinates.shape[1]
    n_p_s, n_e_s = elements_s.shape
    n_q_s = wf_s.shape[0]
    n_int_s = n_e_s * n_q_s

    #################################################################
    # Jacobians and their determinants at surface integration points
    #################################################################
    dhatphi1_s = np.tile(dhatp1_s, (1, n_e_s))
    hatphi_s = np.tile(hatp_s, (1, n_e_s))

    coords1 = np.reshape(coordinates[0, elements_s.flatten(order='F').astype(int)], (n_p_s, n_e_s), order='F')
    coord_int1 = np.kron(coords1, np.ones((1, n_q_s)))

    # Components of Jacobians
    j11 = sum(np.multiply(coord_int1, dhatphi1_s))

    # Determinant of the Jacobian
    det_s = abs(j11)

    # Weight coefficients
    weight_s = np.multiply(det_s, np.tile(wf_s, (1, n_e_s)))

    ##############################################
    # Assembling of the vector of traction forces
    ##############################################

    v_f1 = np.multiply(hatphi_s, np.dot(np.ones((n_p_s, 1)), np.multiply(weight_s, f_t_int[0, -1]))).flatten(order='F')
    v_f2 = np.multiply(hatphi_s, np.dot(np.ones((n_p_s, 1)), np.multiply(weight_s, f_t_int[1, -1]))).flatten(order='F')
    i_f = np.zeros((n_p_s, n_int_s)).flatten(order='F')
    j_f = np.kron(elements_s, np.ones((1, n_q_s))).flatten(order='F')

    f_t = sp.sparse.vstack((sp.sparse.csc_matrix((v_f1, (i_f, j_f)), shape=(1, n_n)),
                            sp.sparse.csc_matrix((v_f2, (i_f, j_f)), shape=(1, n_n))))

    return f_t


def get_elastic_stiffness_matrix(elements: np.array,
                                 coordinates: np.array,
                                 shear: np.array,
                                 bulk: np.array,
                                 dhatp1: np.array,
                                 dhatp2: np.array,
                                 wf: np.array):
    # TODO rewrite as class method to prevent code duplicating!

    n_n = np.size(coordinates, 1)  # number of nodes including midpoints
    n_e = np.size(elements, 1)  # number of elements
    n_p = np.size(elements, 0)  # number of vertices per element
    n_q = np.size(wf)  # number of quadrature points
    n_int = n_e * n_q  # total number of integrations points

    # Jacobian

    # extension of the input arrays DHatP1,DHatP2 by replication
    # size(dhat_phi1)=size(dhat_phi2)=(n_p,n_int)
    dhat_phi1 = np.matlib.repmat(dhatp1, 1, n_e)
    dhat_phi2 = np.matlib.repmat(dhatp2, 1, n_e)

    # Shift the elements' indices - TODO remove after 3rd party modules are rewritten to Python!
    # elements -= 1

    # coordinates of nodes defining each element
    coord_e1 = np.reshape(coordinates[0, ][[int(e) for e in np.reshape(elements, np.size(elements), order='F')]],
                          (n_p, n_e),
                          order='F')

    coord_e2 = np.reshape(coordinates[1, ][[int(e) for e in np.reshape(elements, np.size(elements), order='F')]],
                          (n_p, n_e),
                          order='F')

    # coordinates of nodes around each integration point
    coord_int1 = np.kron(coord_e1, np.ones(n_q))
    coord_int2 = np.kron(coord_e2, np.ones(n_q))

    # components of the Jacobians: size=(1,n_int)
    j_11 = sum(coord_int1 * dhat_phi1)
    j_12 = sum(coord_int2 * dhat_phi1)
    j_21 = sum(coord_int1 * dhat_phi2)
    j_22 = sum(coord_int2 * dhat_phi2)

    # determinant of the Jacobian: size=(1,n_int)
    det = j_11 * j_22 - j_12 * j_21

    # components of the inverse to the Jacobian: size=(1,n_int)
    j_inv11 = j_22 / det
    j_inv12 = -j_12 / det
    j_inv21 = -j_21 / det
    j_inv22 = j_11 / det

    # derivatives of local basis functions w.r.t the coordinates x_1,x_2
    dphi_1 = np.matlib.repmat(j_inv11, n_p, 1) * dhat_phi1 + np.matlib.repmat(j_inv12, n_p, 1) * dhat_phi2
    dphi_2 = np.matlib.repmat(j_inv21, n_p, 1) * dhat_phi1 + np.matlib.repmat(j_inv22, n_p, 1) * dhat_phi2

    # assembling of the strain-displacement matrix B
    n_b = 6 * n_p
    vB = np.zeros((n_b, n_int))
    vB[0:n_b - 5:6] = dphi_1
    vB[5:n_b:6] = dphi_1
    vB[4:n_b - 1:6] = dphi_2
    vB[2:n_b - 3:6] = dphi_2

    # i-th and j-th indices of B
    aux = np.array(range(3 * n_int)).reshape((3, n_int), order='F') + 1
    iB = np.matlib.repmat(aux, 2 * n_p, 1)

    aux_1 = np.array([[1], [1]]) * np.array(range(n_p))
    aux_2 = np.array([[1], [0]]) * np.ones(n_p)
    aux_3 = 2 * (elements[np.reshape(aux_1, np.size(aux_1), order='F')] + 1) - np.kron(np.ones(n_e),
                                                                                       np.reshape(aux_2,
                                                                                                  (np.size(aux_2), 1),
                                                                                                  order='F'))

    jB = np.kron(aux_3, np.ones((3, n_q)))

    # the sparse strain-displacement matrix B
    B = ssp.csr_matrix((flatten_row(vB)[0], (flatten_row(iB)[0] - 1, flatten_row(jB)[0] - 1)),
                       shape=(3 * n_int, 2 * n_n))  # .toarray()
    # print(f'B-sparse shape: {B.shape}')
    # print(f'B-dense shape: {B.todense().shape}')
    # print(f'B-array shape: {B.toarray().shape}')

    # assembling of the elastic stress-strain matrix

    # elastic tensor at each integration point:
    iota = np.array([[1], [1], [0]])
    vol = iota * iota.transpose()
    dev = np.diag([1, 1, 0.5]) - vol / 3
    elast = 2 * flatten_col(dev) * shear + flatten_col(vol) * bulk

    # weight coefficients
    weight = np.abs(det) * np.matlib.repmat(wf, 1, n_e)

    # assemblinng of the sparse matrix D
    # id = np.matlib.repmat(aux, 3, 1)
    id = np.tile(aux, (3, 1))
    jd = np.kron(aux, flatten_col(np.array([1] * 3)))
    vd = elast * (flatten_col(np.array([1] * 9)) * weight)
    D = ssp.csr_matrix((flatten_row(vd)[0], (flatten_row(id)[0] - 1, flatten_row(jd)[0] - 1)))  # .toarray()

    # elastic stiffness matrix
    K = B.transpose() @ D @ B
    # print(f'rank B: {np.linalg.matrix_rank(B.todense())}, {np.linalg.matrix_rank(B.toarray())}')
    # print(f'rank D: {np.linalg.matrix_rank(D.todense())}, {np.linalg.matrix_rank(D.toarray())}')
    log.info(f'K: nonzero elms. / no. elms. = {K.nnz}/{K.shape[0] * K.shape[1]}, '
             f'Density = {K.nnz / (K.shape[0] * K.shape[1])}')

    return K, B, weight, id, jd, D


# @njit
def get_nodes_1(level: int,
                element_type: LagrangeElementType,
                size_xy: float,
                size_hole: float) -> tp.Dict[str, np.array]:
    # Numbers of segments, nodes etc.
    N_x = size_xy * 2 ** level
    N_y = N_x
    N1_x = size_hole * 2 ** level
    N2_x = N_x - N1_x
    N1_y = size_hole * 2 ** level
    N2_y = N_y - N1_y

    n_n = (N_x + 1) * (N_y + 1) - N1_x * N1_y
    n_cell_xy = N_x * N_y - N1_x * N1_y
    n_e = n_cell_xy * 2

    # C - auxiliary array for the further mesh construction
    C = np.zeros((N_x + 1, N_y + 1))
    C1 = np.array(range(1, (N2_x + 1) * N1_y + 1)).reshape((N2_x + 1, N1_y), order='F')
    C2 = np.array(range(((N2_x + 1) * N1_y + 1), n_n + 1)).reshape((N_x + 1, N2_y + 1), order='F')
    C[N1_x:N_x + 1, 0:N1_y] = C1
    C[0:N_x + 1, N1_y:N_y + 1] = C2

    coord_x = np.linspace(0, size_xy, N_x + 1)
    coord_y = np.linspace(0, size_xy, N_y + 1)

    c_x = np.concatenate((np.tile(coord_x[N1_x:N_x + 1], (1, N1_y)),
                          np.tile(coord_x, (1, N2_y + 1))),
                         axis=1).flatten(order='F')
    c_y = np.concatenate((np.tile(np.kron(coord_y[0:N1_y], np.ones((1, N2_x + 1))), 1),
                          np.tile(np.kron(coord_y[N1_y:N_y + 1], np.ones((1, N_x + 1))), 1)),
                         axis=1).flatten(order='F')
    coord = np.array([c_x, c_y])

    V1 = np.zeros((N_x + 1, N_y + 1), dtype=bool)
    V1[N1_x:N_x, 0:N1_y] = 1
    V1[0:N_x, N1_y:N_y] = 1
    V1 = V1.transpose()

    V2 = np.zeros((N_x + 1, N_y + 1), dtype=bool)
    V2[N1_x + 1:N_x + 1, 0:N1_y] = 1
    V2[1:N_x + 1, N1_y:N_y] = 1
    V2 = V2.transpose()

    V3 = np.zeros((N_x + 1, N_y + 1), dtype=bool)
    V3[N1_x + 1:N_x + 1, 1:N1_y + 1] = 1
    V3[1:N_x + 1, N1_y + 1:N_y + 1] = 1
    V3 = V3.transpose()

    V4 = np.zeros((N_x + 1, N_y + 1), dtype=bool)
    V4[N1_x:N_x, 1:N1_y + 1] = 1
    V4[0:N_x, N1_y + 1:N_y + 1] = 1
    V4 = V4.transpose()

    # TODO check if col-wise indexing is possible natively!
    Ct = C.transpose()

    elem = None
    if element_type == LagrangeElementType.P1:
        aux_elem = np.array([Ct[V1],
                             Ct[V2],
                             Ct[V4],
                             Ct[V2],
                             Ct[V3],
                             Ct[V4]])

        elem = aux_elem.reshape((3, n_e), order='F')

    elif element_type == LagrangeElementType.Q1:
        elem = np.array([Ct[V1],
                         Ct[V2],
                         Ct[V3],
                         Ct[V4]])

    elem = elem.astype(int)

    ###############
    # Body surface
    ###############

    # Face 1 (y=0)
    C_s = C[:, 0].copy().reshape((C[:, 0].shape[0], -1))
    v1_s = np.zeros((N_x + 1, 1), dtype=bool)
    v2_s = v1_s.copy()  # np.zeros((N_x, 1), dtype=bool)
    v1_s[N1_x:N_x, 0] = 1
    v2_s[N1_x + 1:N_x + 1, 0] = 1

    surf1 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf1 = surf1.reshape((2, N2_x))

    # Face 2 (x=size_xy)
    C_s = C[-1, :].copy().reshape((C[-1, :].shape[0], -1))
    v1_s = np.zeros((N_y + 1, 1), dtype=bool)
    v2_s = v1_s.copy()  # np.zeros((N_y+1, 1), dtype=bool)
    v1_s[0:N_y, 0] = 1
    v2_s[1:N_y + 1, 0] = 1

    surf2 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf2 = surf2.reshape((2, N_y))

    # Face 3 (y=size_xy)
    C_s = C[:, -1].copy().reshape((C[:, -1].shape[0], -1))
    v1_s = np.zeros((N_x + 1, 1), dtype=bool)
    v2_s = v1_s.copy()  # np.zeros((N_x+1, 1), dtype=bool)
    v1_s[0:N_x, 0] = 1
    v2_s[1:N_x + 1, 0] = 1

    surf3 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf3 = surf3.reshape((2, N_x))

    # Face 4 (x=0)
    C_s = C[0, :].copy().reshape((C[0, :].shape[0], -1))
    v1_s = np.zeros((N_y + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v1_s[N1_y:N_y, 0] = 1
    v2_s[N1_y + 1:N_y + 1, 0] = 1

    surf4 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf4 = surf4.reshape((2, N2_y))

    # Face 5 (y=size_hole)
    C_s = C[N1_x, :].copy().reshape((C[N1_x, :].shape[0], -1))
    v1_s = np.zeros((N_x + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v1_s[0:N1_x, 0] = 1
    v2_s[1:N1_x + 1, 0] = 1

    surf5 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf5 = surf5.reshape((2, N1_x))

    # Face 6 (x=size_hole)
    C_s = C[N1_x + 1, :].copy().reshape((C[N1_x + 1, :].shape[0], -1))
    v1_s = np.zeros((N_y + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v1_s[0:N1_y, 0] = 1
    v2_s[1:N1_y + 1, 0] = 1

    surf6 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf6 = surf6.reshape((2, N1_y))

    ###################
    # Complete surface
    ###################
    surf = np.concatenate((surf1, surf2, surf3, surf4, surf5, surf6), axis=1)

    ######################
    # Boundary conditions
    ######################

    # Non-homogeneous Neumann b. c. on Face 3
    neumann = surf3

    # Nodes with non-homogeneous Dirichlet b. c.
    dirichlet = np.zeros(coord.shape)
    dirichlet[0, coord[1, :] == 0] = 1

    # Nodes with Dirichlet b. c.
    Q = coord > 0
    Q[0, coord[1, :] == 0] = 0

    return {'coordinates': coord, 'elements': elem, 'surface': surf,
            'neumann_nodes': neumann - 1, 'dirichlet_nodes': dirichlet, 'Q': Q}


def get_nodes_2(level: int,
                element_type: LagrangeElementType,
                size_xy: float,
                size_hole: float) -> tp.Dict[str, np.array]:
    # Numbers of segments, nodes etc.
    N_x = size_xy * 2**level
    N_y = N_x
    N1_x = size_hole * 2**level
    N2_x = N_x - N1_x
    N1_y = size_hole * 2**level
    N2_y = N_y - N1_y

    n_n = (2 * N_x + 1) * (2 * N_y + 1) - 4 * N1_x * N1_y
    n_cell_xy = N_x * N_y - N1_x * N1_y
    n_e = n_cell_xy * 2

    # C - auxiliary array for the further mesh construction
    C = np.zeros((2 * N_x + 1, 2 * N_y + 1))

    C1 = None
    C2 = None
    if element_type == LagrangeElementType.P2:
        C1 = np.array(range(1, (2 * N2_x + 1) * 2 * N1_y + 1)).reshape((2 * N2_x + 1, 2 * N1_y), order='F').transpose()
        C2 = np.array(range(((2 * N2_x + 1) * 2 * N1_y + 1), n_n + 1)).reshape((2 * N_x + 1, 2 * N2_y + 1), order='F').transpose()

    elif element_type == LagrangeElementType.Q2:
        C1 = np.zeros((2 * N2_x + 1, 2 * N1_y))
        Q1 = np.ones((2 * N2_x + 1, 2 * N1_y), dtype=bool)
        Q1[1::2, 1::2] = 0
        C1.transpose()[Q1.transpose()] = np.array(range(1, len(C1[Q1]) + 1))
        C2 = np.zeros((2 * N_x + 1, 2 * N2_y + 1))
        Q2 = np.ones((2 * N_x + 1, 2 * N2_y + 1), dtype=bool)
        Q2[1::2, 1::2] = 0
        C2.transpose()[Q2.transpose()] = len(C1[Q1]) + np.array(range(1, len(C2[Q2]) + 1))

    C[2 * N1_x:2 * N_x + 1, 0:2 * N1_y] = C1
    C[0:2 * N_x + 1, 2 * N1_y:2 * N_y + 1] = C2

    coord_x = np.linspace(0, size_xy, 2 * N_x + 1)
    coord_y = np.linspace(0, size_xy, 2 * N_y + 1)

    c_x = None
    c_y = None
    coord = None
    if element_type == LagrangeElementType.P2:
        c_x = np.concatenate((np.tile(coord_x[2 * N1_x:2 * N_x + 1], (1, 2 * N1_y)),
                              np.tile(coord_x, (1, 2 * N2_y + 1))),
                             axis=1).flatten(order='F')
        c_y = np.concatenate((np.tile(np.kron(coord_y[0:2 * N1_y], np.ones((1, 2 * N2_x + 1))), 1),
                              np.tile(np.kron(coord_y[2 * N1_y:2 * N_y + 1], np.ones((1, 2 * N_x + 1))), 1)),
                             axis=1).flatten(order='F')
        coord = np.array([c_x, c_y])

    elif element_type == LagrangeElementType.Q2:
        c1_x = np.tile(np.reshape(coord_x[2*N1_x: 2*N_x + 1], (-1, 1), order='F'), (1, 2*N1_y))
        c2_x = np.tile(np.reshape(coord_x, (-1, 1)), (1, 2*N2_y + 1))
        c1_y = np.tile(np.reshape(coord_y[0:2*N1_y], (1, -1)), (2*N2_x + 1, 1))
        c2_y = np.tile(np.reshape(coord_y[2*N1_y:2*N_y + 1], (1, -1)), (2*N_x + 1, 1))

        coord = np.concatenate([np.concatenate((np.reshape(c1_x.transpose()[Q1.transpose()], (1, -1), order='F'),
                                          np.reshape(c2_x.transpose()[Q2.transpose()], (1, -1), order='F')), axis=1),
                          np.concatenate((np.reshape(c1_y.transpose()[Q1.transpose()], (1, -1), order='F'),
                                          np.reshape(c2_y.transpose()[Q2.transpose()], (1, -1), order='F')), axis=1)],
                               axis=0)
    else:
        raise NotImplementedError()

    V1 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V1[2 * N1_x:2 * N_x - 1:2, 0:2 * N1_y - 1:2] = 1
    V1[0:2 * N_x - 1:2, 2 * N1_y:2 * N_y - 1:2] = 1
    V1 = V1.transpose()

    V2 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V2[2 * N1_x + 2:2 * N_x + 1:2, 0:2 * N1_y - 1:2] = 1
    V2[2:2 * N_x + 1:2, 2 * N1_y:2 * N_y - 1:2] = 1
    V2 = V2.transpose()

    V3 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V3[2 * N1_x + 2:2 * N_x + 1:2, 2:2 * N1_y + 1:2] = 1
    V3[2:2 * N_x + 1:2, 2 * N1_y + 2:2 * N_y + 1:2] = 1
    V3 = V3.transpose()

    V4 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V4[2 * N1_x:2 * N_x - 1:2, 2:2 * N1_y + 1:2] = 1
    V4[0:2 * N_x - 1:2, 2 * N1_y + 2:2 * N_y + 1:2] = 1
    V4 = V4.transpose()

    V12 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V12[2 * N1_x + 1:2 * N_x:2, 0:2 * N1_y - 1:2] = 1
    V12[1:2 * N_x:2, 2 * N1_y:2 * N_y - 1:2] = 1
    V12 = V12.transpose()

    V14 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V14[2 * N1_x:2 * N_x - 1:2, 1:2 * N1_y:2] = 1
    V14[0:2 * N_x - 1:2, 2 * N1_y + 1:2 * N_y:2] = 1
    V14 = V14.transpose()

    V23 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V23[2 * N1_x + 2:2 * N_x + 1:2, 1:2 * N1_y:2] = 1
    V23[2:2 * N_x + 1:2, 2 * N1_y + 1:2 * N_y:2] = 1
    V23 = V23.transpose()

    V24 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V24[2 * N1_x + 1:2 * N_x:2, 1:2 * N1_y:2] = 1
    V24[1:2 * N_x:2, 2 * N1_y + 1:2 * N_y:2] = 1
    V24 = V24.transpose()

    V34 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V34[2 * N1_x + 1:2 * N_x:2, 2:2 * N1_y + 1:2] = 1
    V34[1:2 * N_x:2, 2 * N1_y + 2:2 * N_y + 1:2] = 1
    V34 = V34.transpose()

    # TODO check if col-wise indexing is possible natively!
    Ct = C.transpose()

    elem = None
    if element_type == LagrangeElementType.P2:
        aux_elem = np.array([Ct[V1],
                             Ct[V2],
                             Ct[V4],
                             Ct[V24],
                             Ct[V14],
                             Ct[V12],
                             Ct[V2],
                             Ct[V3],
                             Ct[V4],
                             Ct[V34],
                             Ct[V24],
                             Ct[V23]])

        elem = aux_elem.reshape((6, n_e), order='F')

    elif element_type == LagrangeElementType.Q2:
        elem = np.array([Ct[V1],
                         Ct[V2],
                         Ct[V3],
                         Ct[V4],
                         Ct[V12],
                         Ct[V23],
                         Ct[V34],
                         Ct[V14]])

    elem = elem.astype(int)

    ###############
    # Body surface
    ###############

    # Face 1 (y=0)
    C_s = C[:, 0].copy().reshape((C[:, 0].shape[0], -1))
    v1_s = np.zeros((2 * N_x + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v12_s = v1_s.copy()
    v1_s[2 * N1_x:2 * N_x - 1:2, 0] = 1
    v2_s[2 * N1_x + 1:2 * N_x + 1:2, 0] = 1
    v12_s[2 * N1_x + 1:2 * N_x:2, 0] = 1
    surf1 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F'),
                      C_s[v12_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P2:
        surf1 = surf1.reshape((3, N2_x))

    # Face 2 (x=size_xy)
    C_s = C[-1, :].copy().reshape((C[-1, :].shape[0], -1))
    v1_s = np.zeros((2 * N_y + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v12_s = v1_s.copy()
    v1_s[0:2 * N_y - 1:2, 0] = 1
    v2_s[2:2 * N_y + 1:2, 0] = 1
    v12_s[1:2 * N_y:2, 0] = 1
    surf2 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F'),
                      C_s[v12_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P2:
        surf2 = surf2.reshape((3, N_y))

    # Face 3 (y=size_xy)
    C_s = C[:, -1].copy().reshape((C[:, -1].shape[0], -1))
    v1_s = np.zeros((2 * N_x + 1, 1), dtype=bool)
    v2_s = v1_s.copy()  # np.zeros((N_x+1, 1), dtype=bool)
    v12_s = v1_s.copy()
    v1_s[0:2 * N_x - 1:2, 0] = 1
    v2_s[2:2 * N_x + 1:2, 0] = 1
    v12_s[1:2 * N_x:2, 0] = 1
    surf3 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F'),
                      C_s[v12_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P2:
        surf3 = surf3.reshape((3, N_x))

    # Face 4 (x=0)
    C_s = C[0, :].copy().reshape((C[0, :].shape[0], -1))
    v1_s = np.zeros((2 * N_y + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v12_s = v1_s.copy()
    v1_s[2 * N1_y:2 * N_y - 1:2, 0] = 1
    v2_s[2 * N1_y + 2:2 * N_y + 1:2, 0] = 1
    v12_s[2 * N1_y + 1:2 * N_y:2, 0] = 1
    surf4 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F'),
                      C_s[v12_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P2:
        surf4 = surf4.reshape((3, N2_y))

    # Face 5 (y=size_hole)
    C_s = C[N1_x, :].copy().reshape((C[N1_x, :].shape[0], -1))
    v1_s = np.zeros((2 * N_x + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v12_s = v1_s.copy()
    v1_s[0:2 * N1_x - 1:2, 0] = 1
    v2_s[2:2 * N1_x + 1:2, 0] = 1
    v12_s[1:2 * N1_x:2, 0] = 1
    surf5 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F'),
                      C_s[v12_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P2:
        surf5 = surf5.reshape((3, N1_x))

    # Face 6 (x=size_hole)
    C_s = C[N1_x + 1, :].copy().reshape((C[N1_x + 1, :].shape[0], -1))
    v1_s = np.zeros((2 * N_y + 1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v12_s = v1_s.copy()
    v1_s[0:2 * N1_y - 1:2, 0] = 1
    v2_s[2:2 * N1_y + 1:2, 0] = 1
    v12_s[1:2 * N1_y:2, 0] = 1
    surf6 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F'),
                      C_s[v12_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P2:
        surf6 = surf6.reshape((3, N1_y))

    ###################
    # Complete surface
    ###################
    surf = np.concatenate((surf1, surf2, surf3, surf4, surf5, surf6), axis=1)

    ######################
    # Boundary conditions
    ######################

    # Non-homogeneous Neumann b. c. on Face 3
    neumann = surf3

    # Nodes with non-homogeneous Dirichlet b. c.
    dirichlet = np.zeros(coord.shape)
    dirichlet[0, coord[1, :] == 0] = 1

    # Nodes with Dirichlet b. c.
    Q = coord > 0
    Q[0, coord[1, :] == 0] = 0

    return {'coordinates': coord, 'elements': elem, 'surface': surf,
            'neumann_nodes': neumann - 1, 'dirichlet_nodes': dirichlet, 'Q': Q}


def construct_constitutive_problem(e: np.array, e0: np.array, ep_prev: np.array, shear: np.array, bulk: np.array, eta: np.array,
                                   c: np.array, apply_plastic_strain: bool = False) -> tp.Dict[str, np.array]:
    """
    The aim of this function is to construct constitutive and consistent
    tangent operators at integration points 1,2,...,n_int. These operators
    are related to elastic-perfectly plastic model containing the
    Drucker-Prager yield criterion.

    :param e: current strain tensor, size(E)=(3,n_int)
    :type e: numpy.array

    :param e0: initial stain tensor, size(E0)=(4,n_int)
    :type e0: numpy.array

    :param ep_prev: plastic strain tensor from the previous time step, size(Ep_prev)=(4,n_int)
    :type ep_prev: numpy.array

    :param shear: shear moduli at integration points, size(shear)=(1,n_int)
    :type shear: numpy.array

    :param bulk: bulk moduli at integration points, size(shear)=(1,n_int)
    :type bulk: numpy.array

    :param eta: inelastic parameter at integration points, size(eta)=size(c)=(1,n_int)
    :type eta: numpy.array

    :param c: inelastic parameters at integration points, size(eta)=size(c)=(1,n_int)
    :type c: numpy.array

    :param apply_plastic_strain: flag to signify if the plastic strain is applied or not
    :type apply_plastic_strain: bool

    :return Dictionary with keys:
                s      - stress tensors at integration points, size(s)=(4,n_int)
                ds     - consistent tangent operators at integr. points,
                         size(ds)=(9,n_plast)
                ind_p  - logical array indicating integration points with plastic response,
                         size(ind_p)=(1,n_int),
                         n_plast=number of the points with plastic response
                lambda - plastic multipliers, size(lambda)=(1,n_int)
                ep     - plastic strains, size(ep)=(4,n_int)

    :rtype: dict
    """

    # Number of integration points
    n_int = len(shear)

    # Elastic tensor at integration points, size(Elast)=(9,n_int).
    # Deviatoric and volumetric 4x4 matrices
    iota = np.array([1, 1, 0, 1])
    vol = np.outer(iota, iota)
    dev = np.diag([1, 1, 1/2, 1]) - vol / 3
    Dev = dev[0:3, 0:3]
    Vol = vol[0:3, 0:3]

    # Trial variables:
    #   E_tr   - trial strain tensors, size(E_tr)=(4,n_int)
    #   S_tr   - trial stress tensors, size(S_tr)=(4,n_int)
    #   dev_E  - deviatoric part of the trial strain, size(dev_E)=(4,n_int)
    #   norm_E - norm of dev_E, size(norm_E)=(1,n_int)

    E4 = np.concatenate([e, np.zeros((1, n_int))], axis=0) + e0

    # Trial strain
    E_tr = E4
    if ep_prev is not None:
        E_tr -= ep_prev

    S_tr = 2 * np.tile(shear, (4, 1)) * (dev @ E_tr) + np.tile(bulk, (4, 1)) * (vol @ E_tr)

    # deviatoric part of E_tr
    dev_E = dev @ E_tr

    # norm of the deviatoric strain
    norm_E = np.sqrt([e if e > 0 else 0 for e in sum(E_tr * dev_E)])

    # rho^{tr}
    rho_tr = 2 * (shear * norm_E)

    # trial volumetric stress
    p_tr = bulk * (iota.T @ E_tr)

    # return criteria and specification of integration points with plastic
    # return to the sooth portion and to the apex of the yield surface

    denom_a = bulk * (eta**2)
    denom_s = shear + denom_a
    CRIT1 = rho_tr/np.sqrt(2) + eta * p_tr - c
    CRIT2 = eta * p_tr - denom_a * rho_tr / (shear * np.sqrt(2)) - c

    # logical array indicating plastic integration points
    IND_p = CRIT1 > 0
    IND_p_sum = np.sum(IND_p)

    # logical array indicating int. p. with the return to the smooth portion
    IND_s = np.logical_and((CRIT1 > 0), (CRIT2 <= 0))
    IND_s_sum = np.sum(IND_s)

    # logical array indicating integr. points with the return to the apex
    IND_a = np.logical_and((CRIT1 > 0), (CRIT2 > 0))
    IND_a_sum = np.sum(IND_a)

    # The elastic prediction of unknowns
    S = S_tr
    DS= 2 * Dev.reshape(-1, 1) * shear + Vol.reshape(-1, 1) * bulk

    # plastic multiplier
    lambda_final = np.zeros((1, n_int))

    # plastic strain
    ep = np.zeros((4, n_int))

    if not (IND_s_sum == 0 and IND_a_sum == 0):

        ############################################################
        # The plastic correction at the selected integration points
        ############################################################

        # plastic multipliers for smooth portion of the yield surface
        lambda_s = CRIT1[IND_s] / denom_s[IND_s.flatten(order='F')]
        n_smooth = len(lambda_s)

        # plastic multipliers for apex of the yield surface
        lambda_a = (np.outer(eta[IND_a], p_tr[IND_a])-c[IND_a] / denom_a[IND_a])
        n_apex = len(lambda_a)

        # correction of the stress tensors
        N_hat = dev_E[:, IND_s] / np.tile(norm_E[IND_s], (4, 1))
        M_hat = (np.tile(np.sqrt(2) * shear[IND_s], (4, 1)) * N_hat) + np.outer(iota, (bulk[IND_s] * eta[IND_s]))
        S[:, IND_s] = S[:, IND_s] - np.tile(lambda_s, (4,1)) * M_hat
        S[:, IND_a] = np.outer(iota, (c[IND_a] / eta[IND_a]))

        # correction of the consistent tangent operator
        ID = np.outer(Dev.flatten(), np.ones(n_smooth))
        NN_hat = np.tile(N_hat[0:3, :], (3, 1)) * np.kron(N_hat[0:3, :], np.ones((3, 1)))
        MM_hat = np.tile(M_hat[0:3, :], (3, 1)) * np.kron(M_hat[0:3, :], np.ones((3, 1)))
        DS[:, IND_s] = DS[:, IND_s] - np.tile(2 * np.sqrt(2)*(shear[IND_s]**2) * lambda_s / rho_tr[IND_s], (9, 1)) * (ID-NN_hat) - MM_hat / np.tile(denom_s[IND_s], (9, 1))
        DS[:, IND_a] = np.zeros((9, n_apex))

        log.log(log.INFO, f'plastic integration points: smooth portion = {n_smooth}, apex = {n_apex} (of {n_int})')

        #################################################################
        # Update of the plastic multiplier and plastic strain (optional)
        #################################################################

        # # plastic multiplier
        # lambda_final = np.zeros((1, n_int))

        # correction on the smooth portion
        lambda_final[IND_s.reshape((1, -1), order='F')] = lambda_s

        # correction at the apex
        try:
            lambda_final[IND_a.reshape((1, -1), order='F')] = lambda_a
        except TypeError:
            lambda_final = None

        # # plastic strain
        # ep = np.zeros((4, n_int))
        if apply_plastic_strain:
            ep = ep_prev
            ep[:, IND_s] += np.outer(np.array([1, 1, 2, 1]), lambda_s) * (N_hat/np.sqrt(2) + np.outer(iota, (eta[IND_s]/3)))

            if sum(IND_a) > 0:
                ep[:, IND_a] = E4[:, IND_a] - np.outer(iota, (c[IND_a] / (3 * bulk[IND_a] * eta[IND_a])))

    return {'s': S, 'ds': DS, 'ind_p': IND_p, 'lambda_final': lambda_final, 'ep': ep}


def transform(q_int: np.array, elements: np.array, weight: np.array) -> np.array:
    """
    Transformation of function values at integration points to function
    values at nodes of the finite element mesh.

    :param q_int: values of a function Q at integration points, size(Q_int)=(1,n_int)
    :type q_int: numpy.array

    :param elements: to indicate nodes belonging to each element size(elem)=(n_p,n_e) where n_e is a number of elements
                     and n_p is a number of the nodes within one element
    :type elements: numpy.array

    :param weight: weight factors at integration points, size(weight)=(1,n_int)
    :type weight: numpy.array

    :return Q_node: values of a function Q at nodes of the FE mesh, size(Q_node)=(1,n_n),
                    where n_n is the number of nodes
    :rtype: numpy.array
    """

    #####################
    # Auxiliary notation
    #####################

    # Number of elements
    n_e = elements.shape[1]

    # Number of vertices / 1 element
    n_p = elements.shape[0]

    # Total number of integration points
    n_int = np.size(weight)

    # Number of quadrature points / 1 element
    n_q = int(n_int/n_e)

    # F1 - 1*n_n array, to each node we compute numerically the integral of Q
    #      through a vicinity of the node
    # F2 - 1*n_n array, to each node we compute the area of the vicinity

    # values at integration points, size(vF1)=size(vF2)=(n_p,n_int)
    vF1 = np.ones((n_p, 1)) * (weight * q_int)
    vF2 = np.ones((n_p, 1)) * weight

    # row and column indices, size(iF)=size(jF)=(n_p,n_int)
    iF = np.zeros((n_p, n_int), dtype=int)
    jF = np.kron(elements, np.ones((1, n_q), dtype=int)).astype(int)

    # the asssembling by using the sparse command - values v for duplicate
    # doubles i,j are automatically added together
    F1 = ssp.csr_matrix((flatten_row(vF1)[0], (flatten_row(iF)[0], flatten_row(jF)[0])))
    F2 = ssp.csr_matrix((flatten_row(vF2)[0], (flatten_row(iF)[0], flatten_row(jF)[0])))

    # Approximated values of the function Q at nodes of the FE mesh
    q_node = F1/F2

    return np.array(q_node)


def draw_mesh(coordinates: np.array, elements: np.array, elem_type: LagrangeElementType):
    """
    This function draws mesh and nodal point on the surface of the body

    :param coordinates: coordinates of the nodes, size(coord)=(2,n_n) where n_n is a number of nodes
    :type coordinates: numpy.array

    :param elements: array containing numbers of nodes defining each element,
                     elem.shape = (n_p, n_e), n_e = number of elements
    :type elements: numpy.array

    :param elem_type: type of Lagrange finite element
    :type elem_type: LagrangeElementType
    """

    # coord_aux = [coord;
    # zeros(1, size(coord, 2))];
    # patch('Faces', elem(1: 3,:)','
    # Vertices
    # ',coord_aux', 'FaceVertexCData', ...
    # 0 * ones(size(coord, 2), 1), 'FaceColor', 'white', 'EdgeColor', 'blue');
    # plot(coord(1,:), coord(2,:), 'r.', 'MarkerSize', 10);
    coord_aux = list(zip(*coordinates))

    fig = plt.figure()

    ax = fig.add_subplot(111, aspect='equal')
    # triangle = patch.Polygon(((0.05, 0.1), (0.396, 0.1), (0.223, 0.38)))
    # pythonFEM.plt.scatter([8], [4])
    # ax.autoscale()

    plt.plot(coordinates[0,], coordinates[1,], 'b.')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')

    # Draw polygons
    polygons = None
    if elem_type in (LagrangeElementType.P1, LagrangeElementType.P2, LagrangeElementType.P4):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:3, ].transpose()]
    elif elem_type in (LagrangeElementType.Q1, LagrangeElementType.Q2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:4, ].transpose()]

    for poly in polygons:
        test = patch.Polygon(poly, fc='w', ec='b')
        ax.add_artist(test)

    plt.show()


def draw_displacement(coordinates: np.array, elements: np.array, u: np.array, u_disp: np.array,
                      elem_type: LagrangeElementType, size_xy: int, size_hole: int):
    """
    This function depicts prescribed displacements, deformed and undeformed shape of the body

    :param coordinates: coordinates of the nodes, size(coord)=(2,n_n) where n_n is a number of nodes
    :type coordinates: numpy.array

    :param elements: array containing numbers of nodes defining each element, elem.shape=(n_p,n_e),
                     n_e = number of elements
    :type elements: numpy.array

    :param u: nodal displacements, u.shape=(2,n_n)
    :type u: numpy.array

    :param u_disp: prescribed displacements (e.g. total displacement or displacements in x directions, etc.),
                   u_disp.shape=(1,n_n)
    :type u_disp: numpy.array

    :param elem_type: the type of finite elements
    :type elem_type: LagrangeElementType

    :param size_xy: size of the body in x and y direction
    :type size_xy: int

    :param size_hole: size of the hole in the body, size_hole < size_xy
    :type size_hole: int
    """

    fig = plt.figure()
    coord_aux = list(zip(*coordinates + u))
    ax = fig.add_subplot(111, aspect='equal')

    # Draw polygons
    polygons = None
    colors = None

    # TODO implement color gradient
    cmap = cm.get_cmap('gist_rainbow')
    if elem_type in (LagrangeElementType.P1, LagrangeElementType.P2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:3, ].transpose()]
        colors = [list(map(np.mean, list(zip(*[cmap(u_disp[idx]) for i, idx in enumerate(idx_list)]))))
                  for idx_list in elements[0:3, ].transpose()]
    elif elem_type in (LagrangeElementType.Q1, LagrangeElementType.Q2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:4, ].transpose()]
        colors = [list(map(np.mean, list(zip(*[cmap(u_disp[idx]) for i, idx in enumerate(idx_list)]))))
                  for idx_list in elements[0:4, ].transpose()]

    for i, poly in enumerate(polygons):
        test = patch.Polygon(poly, color=colors[i], alpha=1)
        ax.add_artist(test)

    plt.plot((size_hole, size_xy), (0, 0))
    plt.plot([0, size_xy], [size_xy, size_xy])
    plt.plot([0, size_hole], [size_hole, size_hole])
    plt.plot([0, 0], [size_hole, size_xy])
    plt.plot([size_hole, size_hole], [0, size_hole])
    plt.plot([size_xy, size_xy], [0, size_xy])
    plt.show()


def draw_quantity(coordinates, elements, u, q_node):
    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    coord_aux = list(zip(*coordinates + u))
    ax = fig.add_subplot(111, aspect='equal')

    # if element_type in (LagrangeElementType.P1, LagrangeElementType.P2):
    polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:3, ].T]
    colors = [list(map(np.mean, list(zip(*[cmap(q_node[idx]) for i, idx in enumerate(idx_list)]))))
              for idx_list in elements[0:3, ].T]

    for i, poly in enumerate(polygons):
        test = patch.Polygon(poly, color=colors[i], alpha=1)
        ax.add_artist(test)

    plt.plot([-50, 50], [-50, -50])
    plt.plot([-50, 50], [50, 50])
    plt.plot([-50, -50], [-50, 50])
    plt.plot([50, 50], [-50, 50])

    t = np.arange(0, 2*np.pi, 0.01 * np.pi)
    plt.plot(2.1875 * np.cos(t), 1.75 * np.sin(t))
    plt.show()


def create_midpoints_P4(coord, elem):
    # numbers of elements and vertices
    n_e = elem.shape[1]
    n_n = coord.shape[1]

    # predefinition of unknown arrays
    coord_mid = np.zeros((2, 10*n_e))
    elem_mid = np.zeros((12, n_e))
    surf = np.zeros((5, n_e))

    # for cyclus over elements
    ind = -1
    ind_s = -1
    for i in range(n_e):
        # vertices defining the i-th element
        V1 = elem[0, i]
        V2 = elem[1, i]
        V3 = elem[2, i]

        # creation of the midpoints which do not belong on edges
        coord_mid[:, ind+1] = coord[:, V1]/2 + coord[:, V2]/4 + coord[:, V3]/4
        elem_mid[9, i] = n_n + ind + 1

        coord_mid[:, ind+2] = coord[:, V1]/4 + coord[:, V2]/2 + coord[:, V3]/4
        elem_mid[10, i] = n_n + ind + 2

        coord_mid[:, ind+3] = coord[:, V1]/4 + coord[:, V2]/4 + coord[:, V3]/2
        elem_mid[11, i] = n_n + ind + 3

        ind = ind + 3

        # analysis of the edge V1-V2
        if elem_mid[0, i] == 0:
            # creation of new midpoints lying on the edge V1-V2
            coord_mid[:, ind+1] = (coord[:, V1] + coord[:, V2])/2
            elem_mid[0, i] = n_n+ind+1

            coord_mid[:, ind+2] = 3*coord[:,V1]/4+coord[:, V2]/4
            elem_mid[3, i] = n_n+ind+2

            coord_mid[:, ind+3] = coord[:, V1]/4+3*coord[:, V2]/4
            elem_mid[4, i] = n_n+ind+3

            # finding the adjacent element j to i which contains the edge V1-V2
            row1, col1 = np.where(elem==V1)
            row2, col2 = np.where(elem==V2)
            j = np.setdiff1d(np.intersect1d(col1,col2), i)
            if len(j):
                # This case means that the edge V1-V2 is the intersection of the
                # elements i and j.
                v = elem[:, j]
                if V2 == v[0]:
                    elem_mid[0, j] = n_n+ind+1
                    elem_mid[3, j] = n_n+ind+3
                    elem_mid[4, j] = n_n+ind+2
                elif V2 == v[1]:
                    elem_mid[1, j] = n_n+ind+1
                    elem_mid[5, j] = n_n+ind+3
                    elem_mid[6, j] = n_n+ind+2
                else:
                    elem_mid[2, j] = n_n+ind+1
                    elem_mid[7, j] = n_n+ind+3
                    elem_mid[8, j] = n_n+ind+2
            else:
                ind_s += 1
                surf[:, ind_s] = np.array([[V2], [V1], [n_n+ind+1], [n_n+ind+2], [n_n+ind+3]])

            ind += 3

        # analysis of the edge V2-V3
        if elem_mid[1, i] == 0:
            # creation of new midpoints lying on the edge V2-V3
            coord_mid[:, ind+1] = (coord[:, V2] + coord[:, V3])/2
            elem_mid[1, i] = n_n+ind+1

            coord_mid[:, ind+2] = 3*coord[:, V2]/4 + coord[:, V3]/4
            elem_mid[5, i] = n_n+ind+2

            coord_mid[:, ind+3] = coord[:, V2]/4 + 3*coord[:, V3]/4
            elem_mid[6, i] = n_n+ind+3

            # finding the adjacent element j to i which contains the edge V2-V3
            row1, col1 = np.where(elem == V2)
            row2, col2 = np.where(elem == V3)
            j = np.setdiff1d(np.intersect1d(col1, col2), i)

            if len(j):
                # This case means that the edge V2-V3 is the intersection of the
                # elements i and j.
                v = elem[:, j]
                if V3 == v[0]:
                    elem_mid[0, j] = n_n+ind+1
                    elem_mid[3, j] = n_n+ind+3
                    elem_mid[4, j] = n_n+ind+2
                elif V3 == v[1]:
                    elem_mid[1, j] = n_n+ind+1
                    elem_mid[5, j] = n_n+ind+3
                    elem_mid[6, j] = n_n+ind+2
                else:
                    elem_mid[2, j] = n_n+ind+1
                    elem_mid[7, j] = n_n+ind+3
                    elem_mid[8, j] = n_n+ind+2
            else:
                ind_s += 1
                surf[:, ind_s] = np.array([[V3], [V2], [n_n+ind+1], [n_n+ind+2], [n_n+ind+3]])

            ind += 3

        # analysis of the edge V3-V1
        if elem_mid[2, i] == 0:
            # creation of new midpoints lying on the edge V3-V1
            coord_mid[:, ind+1] = (coord[:, V3] + coord[:, V1])/2
            elem_mid[2, i] = n_n+ind+1

            coord_mid[:, ind+2] = 3*coord[:, V3]/4+coord[:, V1]/4
            elem_mid[7, i] = n_n+ind+2

            coord_mid[:, ind+3] = coord[:, V3]/4+3*coord[:, V1]/4
            elem_mid[8, i] = n_n+ind+3

            # finding the adjacent element j to i which contains the edge V3-V1
            row1, col1 = np.where(elem == V3)
            row2, col2 = np.where(elem == V1)
            j = np.setdiff1d(np.intersect1d(col1, col2), i)
            if len(j):
                # This case means that the edge V3-V1 is the intersection of the
                # elements i and j.
                v = elem[:, j]
                if V1 == v[0]:
                    elem_mid[0, j] = n_n+ind+1
                    elem_mid[3, j] = n_n+ind+3
                    elem_mid[4, j] = n_n+ind+2
                elif V1 == v[1]:
                    elem_mid[1, j] = n_n+ind+1
                    elem_mid[5, j] = n_n+ind+3
                    elem_mid[6, j] = n_n+ind+2
                else:
                    elem_mid[2, j] = n_n+ind+1
                    elem_mid[7, j] = n_n+ind+3
                    elem_mid[8, j] = n_n+ind+2
            else:
                ind_s += 1
                surf[:, ind_s] = np.array([[V1], [V3], [n_n+ind+1], [n_n+ind+2], [n_n+ind+3]]).flatten()

            ind += 3

    coord_mid = coord_mid[:, 0:ind+1]
    surf = surf[:, 0:ind_s+1]
    coord_ext = np.concatenate([coord, coord_mid], axis=1)
    elem_ext = np.array(np.concatenate([elem, elem_mid], axis=0), dtype=int)

    return {'coord_mid': coord_mid, 'surf': surf, 'coord_ext': coord_ext, 'elem_ext': elem_ext}


def create_midpoints_P2(coord, elem):
    # numbers of elements and vertices
    n_e = elem.shape[1]
    n_n = coord.shape[1]

    # predefinition of unknown arrays
    coord_mid = np.zeros((2, 2*n_e))
    elem_mid = np.zeros((3, n_e))
    elem_ed = np.zeros((3, n_e))
    edge_el = np.zeros((2, 2*n_e))
    surf = np.zeros((3, n_e))

    # for cyclus over elements
    ind = 0   # enlarging index specifying midpoints
    ind_s = 0  # enlarging index specifying surf
    for i in range(n_e):
        # vertices defining the i-th element
        V1 = elem[0, i]
        V2 = elem[1, i]
        V3 = elem[2, i]

        # analysis of the edge V2-V3
        if elem_mid[0, i] == 0:
            # creation of a new midpoint

            coord_mid[:, ind] = (coord[:, V2] + coord[:, V3])/2  # its coordinates
            elem_mid[0, i] = n_n + ind
            elem_ed[0, i] = ind
            edge_el[0, ind] = i
            # finding the adjacent element j to i which contains the edge V2-V3
            row1, col1 = np.where(elem == V2)
            row2, col2 = np.where(elem == V3)
            j = np.setdiff1d(np.intersect1d(col1, col2), i)
            if len(j):
                # This case means that the edge V2-V3 is the intersection of the
                # elements i and j.
                edge_el[1, ind] = j
                v = elem[:, j]
                if V3 == v[0]:
                    elem_mid[2, j] = n_n + ind
                    elem_ed[2, j] = ind
                elif V3 == v[1]:
                    elem_mid[0, j] = n_n + ind
                    elem_ed[0, j] = ind
                else:
                    elem_mid[1, j] = n_n + ind
                    elem_ed[1, j] = ind
            else:
                surf[:, ind_s] = np.array([[V3], [V2], [n_n+ind]])
                ind_s += 1
            ind += 1  # number of a new midpoint

        # analysis of the edge V3-V1
        if elem_mid[1, i] == 0:
            # creation of a new midpoint
            coord_mid[:, ind] = (coord[:, V3] + coord[:, V1])/2 # its coordinates
            elem_mid[1, i] = n_n + ind
            elem_ed[1, i] = ind
            edge_el[0, ind] = i
            # finding the adjacent element j to i which contains the edge V3-V1
            row1, col1 = np.where(elem == V3)
            row2, col2 = np.where(elem == V1)
            j = np.setdiff1d(np.intersect1d(col1, col2), i)
            if len(j):
                # This case means that the edge V2-V3 is the intersection of the
                # elements i and j.
                edge_el[1, ind] = j
                v=elem[:, j]
                if V1 == v[0]:
                    elem_mid[2,j] = n_n + ind
                    elem_ed[2,j] = ind
                elif V1 == v[1]:
                    elem_mid[0,j] = n_n + ind
                    elem_ed[0,j] = ind
                else:
                    elem_mid[1,j] = n_n + ind
                    elem_ed[1, j] = ind
            else:
                surf[:, ind_s] = np.array([V1, V3, n_n+ind])
                ind_s += 1
            ind += 1

        # analysis of the edge V1-V2
        if elem_mid[2, i] == 0:
            # creation of a new midpoint
            coord_mid[:, ind] = (coord[:, V1] + coord[:, V2])/2 # its coordinates
            elem_mid[2, i] = n_n+ind
            elem_ed[2, i] = ind
            edge_el[0, ind] = i
            # finding the adjacent element j to i which contains the edge V1-V2
            row1, col1 = np.where(elem==V1)
            row2, col2 = np.where(elem==V2)
            j = np.setdiff1d(np.intersect1d(col1,col2) , i)
            if len(j):
                # This case means that the edge V2-V3 is the intersection of the
                # elements i and j.
                edge_el[1, ind] = j
                v = elem[:, j]
                if V2 == v[0]:
                    elem_mid[2, j] = n_n+ind
                    elem_ed[2, j] = ind
                elif V2 == v[1]:
                  elem_mid[0, j] = n_n+ind
                  elem_ed[0, j] = ind
                else:
                  elem_mid[1, j] = n_n+ind
                  elem_ed[1, j] = ind
            else:
                surf[:, ind_s] = np.array([V2, V1, n_n+ind])
                ind_s += 1
            ind += 1

    coord_mid = coord_mid[:, 0:ind]
    surf = surf[:, 0:ind_s]
    coord_ext = np.concatenate([coord, coord_mid], axis=1)
    elem_ext = np.array(np.concatenate([elem, elem_mid], axis=0), dtype=int)

    return {'coord_mid': coord_mid, 'surf': surf, 'coord_ext': coord_ext, 'elem_ext': elem_ext,
            'elem_ed': elem_ed, 'edge_el': edge_el}


def create_midpoints(elem_type, coord, elem):
    if elem_type == LagrangeElementType.P2:
        return create_midpoints_P2(coord, elem)
    elif elem_type == LagrangeElementType.P4:
        return create_midpoints_P4(coord, elem)


# @njit
def elasticity_fem(element_type: LagrangeElementType = LagrangeElementType.P1,
                   level: int = 1,
                   draw: bool = True) -> tp.Dict[str, int]:
    """
    This function solves a plane strain elastic-perfectly plastic problem. It
    is considered a tunnel geometry inspired by the TSX in-situ experiment
    in Canada. We use the Drucker-Prager yield criterion and the associated
    plastic flow rule, for the sake of simplicity. The mesh is provided by
    the team from TU Liberec. P1 and P2 elements are considered. The tunnel
    excavation is simulated such that the initial stress tensor is
    parametrized by a scalar factor varying from 0 to 1.

    :param element_type: the type of finite elements
    :type element_type: LagrangeElementType

    :param level: an integer defining mesh density

    :param draw: a logical value enbling visualization of the results

    :return: Dictionary containing two keys (elements): 'time' - assembly time for the elastic stiffness matrix (in seconds),
    'rows' - number of rows in the stiffness matrix
    :rtype: Dictionary[str: int]
    """

    # Elastic material parameters
    # TODO maybe set from function parameters?
    young_mod = 60000
    poisson_ratio = 0.2
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    # Inelastic parameters
    cohesion = 18.7
    friction_ang = 49*np.pi / 180
    plane_strain1 = 3*np.tan(friction_ang)/np.sqrt(9+12*(np.tan(friction_ang))**2)
    plane_strain2 = 3*cohesion/np.sqrt(9+12*(np.tan(friction_ang))**2)

    # Initial stress and strain tensors
    init_stress = np.array([-45, -11, 0, -60]).reshape((-1, 1), order='F')
    trace_init_stress = init_stress[0] + init_stress[1] + init_stress[3]
    init_strain = np.array([-poisson_ratio*trace_init_stress+(1+poisson_ratio)*init_stress[0],
                            -poisson_ratio*trace_init_stress+(1+poisson_ratio)*init_stress[1],
                            0,
                            -poisson_ratio*trace_init_stress+(1+poisson_ratio)*init_stress[3]], dtype=float).reshape((-1, 1),
                                                                                                        order='F') / young_mod

    ################
    # Generate mesh
    ################

    coords = np.genfromtxt('coord.csv', delimiter=',')
    elem = np.genfromtxt('elem.csv', delimiter=',', dtype=int) - 1

    midp_dict = create_midpoints(element_type, coords, elem)
    coords = midp_dict['coord_ext']
    elem = midp_dict['elem_ext']

    # Nodes with Dirichlet boundary condition
    Q = np.ones(np.shape(coords), dtype=bool)
    Q[0, coords[0, :] < -49.99] = 0
    Q[0, coords[0, :] > 49.99] = 0
    Q[1, coords[1, :] < -49.99] = 0
    Q[1, coords[1, :] > 49.99] = 0

    ##############################
    # Data from reference element
    ##############################
    xi, wf = get_quadrature_volume(element_type)
    # xi_s, wf_s = get_quadrature_surface(element_type)

    hatp, dhatp1, dhatp2 = get_local_basis_volume(element_type, xi)
    # hatp_s, dhatp1_s = get_local_basis_surface(element_type, xi_s)

    #####################################
    # Assembling of the stiffness matrix
    #####################################
    n_n = coords.shape[1]
    n_unknown = len(coords[Q])
    n_e = elem.shape[1]
    n_q = wf.size
    n_int = n_e * n_q

    shear_mod = shear_mod * np.ones(n_int)
    bulk_mod = bulk_mod * np.ones(n_int)

    K, B, weight, iD, jD, D_elast = get_elastic_stiffness_matrix(elem, coords, shear_mod, bulk_mod, dhatp1, dhatp2, wf)
    weight = weight.flatten(order='F')

    # Plastic material parameters
    eta = plane_strain1 * np.ones(n_int)
    c = plane_strain2 * np.ones(n_int)

    # Loading process
    d_zeta = 1/17
    d_zeta_min = d_zeta / 10
    d_zeta_old = d_zeta
    zeta = 0
    zeta_old = zeta
    zeta_max = 1

    F0 = (B.T @ np.reshape(np.tile(weight, (3, 1)) * init_stress[0:3, :], (3*n_int, 1), order='F')).reshape((2, -1), order='F')
    U_elast = np.zeros((2, n_n))

    # TODO min(min(reshape(K_elast(logical(Q(:) * Q(:)')), 908, 908) == K_elast(Q, Q))) !!!
    Q_flat = Q.flatten(order='F')
    Q_logic = np.outer(Q_flat, Q_flat)
    nonzero_dim = int(np.sqrt(np.count_nonzero(Q_logic)))
    K_masked = K.T[Q_logic].reshape((nonzero_dim, nonzero_dim), order='F')

    # K_masked_matlab = np.genfromtxt('kelast_qq.csv', dtype=float, delimiter=',')
    # F0q_matlab = np.genfromtxt('f0q.csv', dtype=float, delimiter=',')
    U_elast.T[Q.T] = np.linalg.solve(K_masked, -F0.T[Q.T])
    # U_elast.T[Q.T] = np.linalg.solve(K_masked_matlab, -F0q_matlab)

    U_it = d_zeta * U_elast
    dU = np.zeros((2, n_n))
    U = np.zeros((2, n_n))
    U_old = -U_it
    F = np.zeros((2, n_n))
    E = np.zeros((3, n_int))
    Ep_old = np.zeros((4, n_int))
    zeta_hist = np.zeros(100)
    displ_hist = np.zeros(100)

    step = 0
    # print(f'step: {step}')
    while True:
        zeta = zeta_old + d_zeta
        E0_zeta = zeta * init_strain
        # print(f'load factor = {zeta}')
        # print(f'load increment = {d_zeta}')

        n_it = 25
        for it in range(n_it):
            E = (B @ U_it.reshape((-1, 1), order='F')).reshape((3, -1), order='F')
            const_problem = construct_constitutive_problem(E, E0_zeta, Ep_old, shear_mod, bulk_mod, eta, c)
            vD = np.tile(weight, (9, 1)) * const_problem['ds']
            # print(f'vD: {vD[:10]}')
            D_p = ssp.csr_matrix((flatten_row(vD)[0], (flatten_row(iD)[0] - 1, flatten_row(jD)[0] - 1)),
                                 shape=(3 * n_int, 3 * n_int))
            K_tangent = K + B.T * (D_p-D_elast) * B
            F = (B.T @ (np.tile(weight, (3, 1)) * const_problem['s'][0:3, :]).reshape((3*n_int, 1), order='F')).reshape((2, n_n), order='F')
            K_tangent_masked = K_tangent.T[Q_logic].reshape((nonzero_dim, nonzero_dim), order='F')

            dU.T[Q.T] = np.linalg.solve(K_tangent_masked, -F.T[Q.T])
            dU_flat = dU.flatten(order='F')
            # print(f'inner iter: {it}')
            U_new = U_it + dU_flat.reshape((2, -1), order='F')
            U_it_flat = U_it.flatten(order='F')
            U_new_flat = U_new.flatten(order='F')

            q1 = np.sqrt(dU_flat @ K @ dU_flat)
            q2 = np.sqrt(U_it_flat @ K @ U_it_flat)
            q3 = np.sqrt(U_new_flat @ K @ U_new_flat)

            criterion = q1/(q2+q3)
            if np.isnan(criterion):
                it = n_it
                break

            # print(f'criterion: {criterion}')

            U_it = U_new

            if criterion < 1e-12:
                break

        if criterion < 1e-10:
            U_old = U
            U = U_it
            E = (B @ U.flatten(order='F')).reshape((3, -1), order='F')
            const_problem = construct_constitutive_problem(E, E0_zeta, Ep_old, shear_mod, bulk_mod, eta, c)
            Ep_old = const_problem['ep']
            zeta_old = zeta
            d_zeta_old = d_zeta
            zeta_hist[step] = zeta
            displ = U[0, 40]
            displ_hist[step] = displ
            step += 1
        else:
            print(f'The Newton solver does not converge.')
            d_zeta = d_zeta/2

        # Initialization for the next iteration
        U_it = d_zeta*(U-U_old)/d_zeta_old+U

        # Stopping criteria for the loading process
        if zeta_old >= zeta_max:
            print('Maximal load factor was achieved.')
            break

        if d_zeta < d_zeta_min:
            print('Too small load increments.')
            break

        # print(f'step = {step}')

    # TODO move drawing outside of the function
    # TODO return variables important for subsequent plotting
    if draw:
        draw_mesh(coords, elem, element_type)

        plt.plot(17*zeta_hist[0:step], displ_hist[0:step])
        plt.xlabel('day')
        plt.ylabel('displacement on the tunnel wall')
        plt.show()

        U_total = np.sqrt(U[0, :]**2 + U[1, :]**2)
        draw_quantity(coords, elem, 300*U, U_total)

        Ep_node = transform(np.sqrt(sum(const_problem['ep']*const_problem['ep'])), elem, weight).flatten(order='F')
        draw_quantity(coords, elem, 0*U, np.zeros(Ep_node.shape) + Ep_node)

        # Q_plast = sum((const_problem['lambda'].reshape((n_q, n_e)), 1) > 0)
        # draw_zones(coords, elem, Q_plast)


if __name__ == '__main__':
    # TODO add "an introduction hint" for new users
    pass
