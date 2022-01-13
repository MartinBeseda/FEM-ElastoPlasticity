"""
This module contains the Python3 implementation of matlabfem library (https://github.com/matlabfem/)
intended to solve elasticity and plasticity problems.

This program triggers an assembly test for a 2D elastoplastic body. It
is considered perfectly plastic model with the Drucker-Prager yield
criterion and the strip-footing benchmark. The main aim is to compare
the assembling time for the elastic and tangent stiffness matrices.
The tangent stiffness matrices are computed in each time step.
One can set optionally 4 types of finite elements,
levels of mesh density and many other parameters.
"""

#################
# Import modules
#################
import enum
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
import matplotlib.collections as col
import os


##############
# Qt settings
##############
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

####################
# Set logging level
####################
log.basicConfig(level=log.INFO)


def flatten_row(v: np.array) -> np.array:
    return np.reshape(v, (1, -1), order='F')


def flatten_col(v: np.array) -> np.array:
    return np.reshape(v, (np.size(v), 1), order='F')


class LagrangeElementType(enum.Enum):
    """ Enumeration type allowing users to select the type of Lagrange finite elements."""
    P1 = 1
    P2 = 2
    Q1 = 3
    Q2 = 4


def get_nodes_1(level: int,
                element_type: LagrangeElementType,
                size_xy: float) -> tp.Dict[str, np.array]:
    # Number of segments, nodes and elements
    N_x = size_xy * 2**level
    N_y = N_x
    n_e = 2 * N_x * N_y

    # C - 2D auxilliary array that contains node numbers and that is important
    # for the mesh construction.
    C = np.array(range((N_x+1)*(N_y+1))).reshape((N_x+1, N_y+1), order='F')
    Ct = C.T

    # Coordinates of nodes
    #
    # coordinates in directions x and y
    coord_x = np.linspace(0, size_xy, N_x+1)
    coord_y = np.linspace(0, size_xy, N_y+1)

    # long 1D arrays containing coordinates of all nodes in x,y directions
    c_x = np.tile(coord_x, N_y+1)
    c_y = np.kron(coord_y, np.ones(N_x+1))

    # the required array of coordinates, size(coord)=(2,n_n)
    coord = np.array([c_x, c_y])


    # construction of the array elem
    #
    # ordering of the nodes creating the unit cube:
    #  V1 -> [0 0], V2 -> [1 0], V3 -> [1 1], V4 -> [0 1]
    #  V1,...,V4 are logical 2D arrays which enable to select appropriate
    #  nodes from the array C.

    V1 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V1[0:N_x, 0:N_y] = 1
    V1 = V1.T

    V2 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V2[1:(N_x+1), 0:N_y] = 1
    V2 = V2.T

    V3 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V3[1:(N_x+1), 1:(N_y+1)] = 1
    V3 = V3.T

    V4 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V4[0:N_x, 1:(N_y+1)] = 1
    V4 = V4.T

    elem = None
    if element_type == LagrangeElementType.P1:
        # used division of a rectangle into 2 triangles:
        #   V1 V2 V4
        #   V2 V3 V4
        # size(aux_elem)=(2*3,n_e/2)
        aux_elem = np.array((Ct[V1], Ct[V2], Ct[V4], Ct[V2], Ct[V3], Ct[V4]))

        # the array elem, size(elem)=(3,n_e)
        elem = aux_elem.reshape((3, n_e), order='F')

    elif element_type == LagrangeElementType.Q1:
        # Unit cube
        elem = np.array([Ct[V1], Ct[V2], Ct[V3], Ct[V4]])

    # Surface of the body - the array "surf"

    # For each face of the body, we define the restriction C_s of the array C
    # and logical 2D arrays V1_s, V2_s which enable to select appropriate
    # nodes from the array C_s. We consider the following ordering of the
    # nodes within the unit line:
    #   V1_s -> [0 0], V2_s -> [1 0]

    # Edge 1: y=0 (the bottom of the body)
    C_s = C[:, 0].reshape(-1, 1)
    V1_s = np.zeros((N_x+1, 1), dtype=bool)
    V1_s[0:N_x, 0] = 1
    V2_s = np.zeros((N_x+1, 1), dtype=bool)
    V2_s[1:(N_x+1), 0] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s]))
    surf1 = aux_surf.reshape(2, N_x)

    # Edge 2: x=size_xy (the right hand side of the body)
    C_s = C[-1, :].reshape(-1, 1)
    V1_s = np.zeros((N_y+1, 1), dtype=bool)
    V1_s[0:N_y, 0] = 1
    V2_s = np.zeros((N_y+1, 1), dtype=bool)
    V2_s[1:(N_y+1), 0] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s]))
    surf2 = aux_surf.reshape(2, N_y)

    # Edge 3: y=size_xy (the top of the body)
    C_s = C[:, -1].reshape(-1, 1)
    V1_s = np.zeros((N_x+1, 1), dtype=bool)
    V1_s[0:N_x, 0] = 1
    V2_s = np.zeros((N_x+1, 1), dtype=bool)
    V2_s[1:(N_x+1), 0] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s]))
    surf3 = aux_surf.reshape(2,N_x)

    # Edge 4: x=0 (the left hand side of the body)
    C_s = C[0, :].reshape(-1, 1)
    V1_s = np.zeros((N_y+1, 1), dtype=bool)
    V1_s[0:N_y, 0] = 1
    V2_s = np.zeros((N_y+1, 1), dtype=bool)
    V2_s[1:(N_y+1), 0] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s]))
    surf4 = aux_surf.reshape(2, N_y)

    # the array "surf"
    surf = np.concatenate((surf1, surf2, surf3, surf4), axis=1)

    # Boundary conditions

    # array indicating the nodes with non-homogen. Dirichlet boundary cond.
    dirichlet = np.zeros(coord.shape)
    dirichlet[1, np.logical_and((coord[1, :] == size_xy), (coord[0, :] <= 1.0001))] = 1

    # logical array indicating the nodes with the Dirichlet boundary cond.
    Q = coord > 0
    Q[1, np.logical_and((coord[1, :] == size_xy), (coord[0, :] <= 1.0001))] = 0
    Q[0, (coord[0, :] == size_xy)] = 0

    return {'coordinates': coord, 'elements': elem, 'surface': surf, 'dirichlet_nodes': dirichlet, 'Q': Q}


def get_nodes_2(level: int,
                element_type: LagrangeElementType,
                size_xy: float) -> tp.Dict[str, np.array]:
    # Number of segments, nodes and elements
    N_x = size_xy * 2**level
    N_y = N_x
    n_e = 2 * N_x * N_y

    # C - 2D auxilliary array that contains node numbers and that is important
    # for the mesh construction.
    Ct = None
    Q_mid = None
    if element_type == LagrangeElementType.P2:
        C = np.array(range((2*N_x+1)*(2*N_y+1))).reshape((2*N_x+1, 2*N_y+1), order='F')
        Ct = C.T
    elif element_type == LagrangeElementType.Q2:
        C = np.zeros((2 * N_x + 1, 2 * N_y + 1))
        Ct = C.T
        Q_mid = np.ones((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
        Q_mid[1::2, 1::2] = 0
        Ct[Q_mid] = np.array(range(np.size(C[Q_mid])))

    # Coordinates of nodes
    #
    # coordinates in directions x and y
    coord_x = np.linspace(0, size_xy, 2*N_x+1)
    coord_y = np.linspace(0, size_xy, 2*N_y+1)

    # long 1D arrays containing coordinates of all nodes in x,y directions
    coord = None
    if element_type == LagrangeElementType.P2:
        c_x = np.tile(coord_x, 2 * N_y + 1)
        c_y = np.kron(coord_y, np.ones(2*N_x+1))

        # the required array of coordinates, size(coord)=(2,n_n)
        coord = np.array([c_x, c_y])

    elif element_type == LagrangeElementType.Q2:
        c_x = np.tile(coord_x.reshape((-1, 1), order='F'), (1, 2 * N_y + 1))
        c_y = np.tile(coord_y, (2*N_x + 1, 1))
        coord = np.array([c_x.T[Q_mid], c_y.T[Q_mid]])

    # construction of the array elem
    #
    # ordering of the nodes creating the unit cube:
    #  V1 -> [0 0], V2 -> [1 0], V3 -> [1 1], V4 -> [0 1]
    #  V1,...,V4 are logical 2D arrays which enable to select appropriate
    #  nodes from the array C
    V1 = np.zeros((2*N_x+1, 2*N_y+1), dtype=bool)
    V1[0:2*N_x-1:2, 0:2*N_y-1:2] = 1
    V1 = V1.T

    V2 = np.zeros((2*N_x+1, 2*N_y+1), dtype=bool)
    V2[2:(2*N_x+1):2, 0:2*N_y-1:2] = 1
    V2 = V2.T

    V3 = np.zeros((2*N_x+1, 2*N_y+1), dtype=bool)
    V3[2:(2*N_x+1):2, 2:(2*N_y+1):2] = 1
    V3 = V3.T

    V4 = np.zeros((2*N_x+1, 2*N_y+1), dtype=bool)
    V4[0:2*N_x-1:2, 2:(2*N_y+1):2] = 1
    V4 = V4.T

    # logical arrays for midpoints, e.g.V12 represents the midpoints between V1 and V2
    V12 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V12[1:(2 * N_x):2, 0:(2 * N_y - 1):2] = 1
    V12 = V12.T

    V14 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V14[0:(2 * N_x - 1):2, 1:(2 * N_y):2] = 1
    V14 = V14.T

    V23 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V23[2:(2 * N_x + 1):2, 1:(2 * N_y):2] = 1
    V23 = V23.T

    V24 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V24[1:(2 * N_x):2, 1:(2 * N_y):2] = 1
    V24 = V24.T

    V34 = np.zeros((2 * N_x + 1, 2 * N_y + 1), dtype=bool)
    V34[1:(2 * N_x):2, 2:(2 * N_y + 1):2] = 1
    V34 = V34.T

    elem = None
    if element_type == LagrangeElementType.P2:
        aux_elem = np.array((Ct[V1], Ct[V2], Ct[V4], Ct[V24], Ct[V14], Ct[V12],
                             Ct[V2], Ct[V3], Ct[V4], Ct[V34], Ct[V24], Ct[V23]))
        elem = aux_elem.reshape((6, n_e), order='F')

    elif element_type == LagrangeElementType.Q2:
        elem = np.array((Ct[V1], Ct[V2], Ct[V3], Ct[V4], Ct[V12], Ct[V23], Ct[V34], Ct[V14]))

    # Surface of the body - the array "surf"

    # Edge 1: y=0 (the bottom of the body)
    C_s = C[:, 0].reshape(-1, 1)
    V1_s = np.zeros((2*N_x+1, 1), dtype=bool)
    V1_s[0:2*N_x-1:2, 0] = 1
    V2_s = np.zeros((2*N_x+1, 1), dtype=bool)
    V2_s[2:(2*N_x+1):2, 0] = 1
    V12_s = np.zeros((2*N_x+1, 1), dtype=bool)
    V12_s[1:2*N_x:2, 0] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s], C_s[V12_s]))
    surf1 = aux_surf.reshape(3, N_x)

    # Edge 2: x=size_xy (the right hand side of the body)
    C_s = C[-1, :].reshape(-1, 1)
    V1_s = np.zeros((2*N_y+1, 1), dtype=bool)
    V1_s[0:2*N_y-1:2, 0] = 1
    V2_s = np.zeros((2*N_y+1, 1), dtype=bool)
    V2_s[2:(2*N_y+1):2, 0] = 1
    V12_s = np.zeros((2*N_y+1, 1), dtype=bool)
    V12_s[1:2*N_y:2] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s], C_s[V12_s]))
    surf2 = aux_surf.reshape(3, N_y)

    # Edge 3: y=size_xy (the top of the body)
    C_s = C[:, -1].reshape(-1, 1)
    V1_s = np.zeros((2*N_x+1, 1), dtype=bool)
    V1_s[0:2*N_x-1:2, 0] = 1
    V2_s = np.zeros((2*N_x+1, 1), dtype=bool)
    V2_s[2:(2*N_x+1):2, 0] = 1
    V12_s = np.zeros((2*N_x+1, 1), dtype=bool)
    V12_s[1:2*N_x:2] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s], C_s[V12_s]))
    surf3 = aux_surf.reshape(3, N_x)

    # Edge 4: x=0 (the left hand side of the body)
    C_s = C[0, :].reshape(-1, 1)
    V1_s = np.zeros((2*N_y+1, 1), dtype=bool)
    V1_s[0:2*N_y-1:2, 0] = 1
    V2_s = np.zeros((2*N_y+1, 1), dtype=bool)
    V2_s[2:(2*N_y+1):2, 0] = 1
    V12_s = np.zeros((2*N_y+1, 1), dtype=bool)
    V12_s[1:2*N_y:2, 0] = 1
    aux_surf = np.array((C_s[V1_s], C_s[V2_s], C_s[V12_s]))
    surf4 = aux_surf.reshape(3, N_y)

    # the array "surf"
    surf = np.concatenate((surf1, surf2, surf3, surf4), axis=1)

    # Boundary conditions

    # array indicating the nodes with non-homogen. Dirichlet boundary cond.
    dirichlet = np.zeros(coord.shape)
    dirichlet[1, np.logical_and((coord[1, :] == size_xy), (coord[0, :] <= 1.0001))] = 1

    # logical array indicating the nodes with the Dirichlet boundary cond.
    Q = coord > 0
    Q[1, np.logical_and((coord[1, :] == size_xy), (coord[0, :] <= 1.0001))] = 0
    Q[0, (coord[0, :] == size_xy)] = 0

    return {'coordinates': coord, 'elements': elem.astype(int), 'surface': surf, 'dirichlet_nodes': dirichlet, 'Q': Q}


def assemble_mesh_1(level: int, element_type: LagrangeElementType, size_xy: float):
    return get_nodes_1(level, element_type, size_xy)


def assemble_mesh_2(level: int, element_type: LagrangeElementType, size_xy: float):
    return get_nodes_2(level, element_type, size_xy)


def assemble_mesh(level: int, element_type: LagrangeElementType, size_xy: float):
    mesh = None
    if element_type in (LagrangeElementType.P1, LagrangeElementType.Q1):
        mesh = assemble_mesh_1(level, element_type, size_xy)
    elif element_type in (LagrangeElementType.P2, LagrangeElementType.Q2):
        mesh = assemble_mesh_2(level, element_type, size_xy)

    return mesh


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
                                                     40 / 81, 64 / 81]]))}

    return init_dict[el_type]


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

    xi_1 = xi[0,]
    xi_2 = xi[1,]

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
                                                    (1 - xi_1) / 4]))}

    return init_dict[el_type]


def get_elastic_stiffness_matrix(elements: np.array,
                                 coordinates: np.array,
                                 shear: np.array,
                                 bulk: np.array,
                                 dhatp1: np.array,
                                 dhatp2: np.array,
                                 wf: np.array) -> tp.Tuple[np.array, np.array]:
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


def construct_constitutive_problem(e: np.array, ep_prev: np.array, shear: np.array, bulk: np.array, eta: np.array,
                                   c: np.array, apply_plastic_strain: bool = False) -> tp.Dict[str, np.array]:
    """
    The aim of this function is to construct constitutive and consistent
    tangent operators at integration points 1,2,...,n_int. These operators
    are related to elastic-perfectly plastic model containing the
    Drucker-Prager yield criterion.

    :param e: current strain tensor, size(E)=(3,n_int)
    :type e: numpy.array

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

    E4 = np.concatenate([e, np.zeros((1, n_int))])

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

    # logical array indicating int. p. with the return to the smooth portion
    IND_s = np.logical_and((CRIT1 > 0), (CRIT2 <= 0))

    # logical array indicating integr. points with the return to the apex
    IND_a = np.logical_and((CRIT1 > 0), (CRIT2 > 0))

    # The elastic prediction of unknowns
    S = S_tr
    DS= 2 * Dev.reshape(-1, 1) * shear + Vol.reshape(-1, 1) * bulk

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

    # plastic multiplier
    lambda_final = np.zeros((1, n_int))

    # correction on the smooth portion
    lambda_final[IND_s.reshape((1, -1), order='F')] = lambda_s

    # correction at the apex
    try:
        lambda_final[IND_a.reshape((1, -1), order='F')] = lambda_a
    except TypeError:
        lambda_final = None

    # plastic strain
    ep = np.zeros((4, n_int))
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
    vF1 = np.ones((n_p, 1)) @ (weight * q_int)
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

    return q_node


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

    # coord_aux = np.vstack([coordinates,
    #                        np.zeros((1, coordinates.shape[1]))]).transpose()

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
    if elem_type in (LagrangeElementType.P1, LagrangeElementType.P2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:3, ].transpose()]
    elif elem_type in (LagrangeElementType.Q1, LagrangeElementType.Q2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements.astype(int)[0:4, ].transpose()]

    for poly in polygons:
        test = patch.Polygon(poly, fc='w', ec='b')
        ax.add_artist(test)

    plt.show()


def draw_quantity(coordinates, elements, u, q_node, element_type, size_xy):

    cmap = cm.get_cmap('gist_rainbow')
    fig = plt.figure()
    coord_aux = list(zip(*coordinates + u))
    ax = fig.add_subplot(111, aspect='equal')

    polygons = None
    colors = None
    if element_type in (LagrangeElementType.P1, LagrangeElementType.P2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements[0:3, ].transpose()]
        colors = [list(map(np.mean, list(zip(*[cmap(q_node[idx]) for i, idx in enumerate(idx_list)]))))
                  for idx_list in elements[0:3, ].transpose()]

    elif element_type in (LagrangeElementType.Q1, LagrangeElementType.Q2):
        polygons = [[coord_aux[idx] for idx in idx_list] for idx_list in elements.astype(int)[0:4, ].transpose()]
        colors = [list(map(np.mean, list(zip(*[cmap(q_node[idx]) for i, idx in enumerate(idx_list)]))))
                  for idx_list in elements[0:4, ].transpose()]

    for i, poly in enumerate(polygons):
        test = patch.Polygon(poly, color=colors[i], alpha=1)
        ax.add_artist(test)

    # p = col.PatchCollection(polygons, alpha=0.5)
    # p.set_array(cmap)
    # # ax.add_collection(p)
    # fig.colorbar(p)

    # undeformed shape of the body
    plt.plot([0, size_xy], [0, 0])
    plt.plot([0, size_xy], [size_xy, size_xy])
    plt.plot([0, 0], [0, size_xy])
    plt.plot([size_xy, size_xy], [0, size_xy])
    # plt.colorbar(cmap)
    plt.show()


def elasticity_fem(element_type: LagrangeElementType = LagrangeElementType.P1,
                   level: int = 1,
                   draw: bool = True) -> tp.Dict[str, int]:

    ##############################
    # Elastic material parameters
    ##############################

    # Young's modulus
    young = 1e7

    # Poisson's ration
    poisson = 0.48

    # Shear modulus
    shear = young / (2 * (1 + poisson))

    # Bulk modulus
    bulk = young / (3 * (1 - 2 * poisson))

    ########################################
    # Values of plastic material parameters
    ########################################

    # Cohesion
    c0 = 450

    # Frictional angle
    phi = np.pi / 9

    # Plane strain approximation
    eta = 3 * np.tan(phi) / np.sqrt(9 + 12 * (np.tan(phi))**2)
    c = 3 * c0 / np.sqrt(9 + 12 * (np.tan(phi))**2)

    #########################
    # Geometrical parameters
    #########################

    # Size of body in direction of x and y
    size_xy = 10

    ################
    # Generate mesh
    ################
    mesh = assemble_mesh(level, element_type, size_xy)
    q_nd = mesh['dirichlet_nodes'][1, :] > 0

    # Data from the reference element

    # quadrature points and weights for volume integration
    xi, wf = get_quadrature_volume(element_type)

    # local basis functions and their derivatives for volume
    hatp, dhatp1, dhatp2 = get_local_basis_volume(element_type, xi)

    # Number of nodes, unknowns, elements and integration points + print

    n_n = mesh['coordinates'].shape[1]
    n_unknown = len(mesh['coordinates'][mesh['Q']])
    n_e = mesh['elements'].shape[1]
    n_q = np.size(wf)
    n_int = n_e*n_q

    log.log(log.INFO, f'number of nodes = {n_n}')
    log.log(log.INFO, f'number of unknowns = {n_unknown}')
    log.log(log.INFO, f'number of elements = {n_e}')
    log.log(log.INFO, f'number of integration points = {n_int}')

    # Assembling of the elastic stiffness matrix

    # values of elastic material parameters at integration points
    shear = shear * np.ones(n_int)
    bulk = bulk * np.ones(n_int)

    # stiffness matrix assembly and the assembly time
    # TODO measure time of matrix assembling (OPTIONALLY!)
    K_elast, B, weight, iD, jD, D_elast = get_elastic_stiffness_matrix(mesh['elements'], mesh['coordinates'], shear,
                                                                       bulk, dhatp1, dhatp2, wf)

    # Plastic material parameters

    # values of plastic material parematers at integration points
    eta = eta * np.ones(n_int)
    c = c * np.ones(n_int)

    # Loading process

    # initial load increment and factor
    d_zeta = 1 / 1000
    d_zeta_min = d_zeta / 1300
    d_zeta_old = d_zeta
    zeta = 0
    zeta_old = zeta
    zeta_max = 1

    # elastic initial displacements
    Ud = -d_zeta * mesh['dirichlet_nodes']
    f = - K_elast * Ud.reshape((-1, 1), order='F')
    U_it = Ud
    Q_flat = mesh['Q'].reshape((-1, 1), order='F')
    K_bool_indices = (Q_flat @ Q_flat.transpose()).astype(bool)
    stiff_mat = K_elast[K_bool_indices]
    stiff_mat = stiff_mat.reshape((int(np.sqrt(stiff_mat.shape[1])), -1), order='F')
    U_it.transpose()[mesh['Q'].transpose()] = np.linalg.solve(stiff_mat, f[Q_flat].transpose()).flatten(order='F')

    # other initialization
    dU = np.zeros((2, n_n))
    U = np.zeros((2, n_n))
    U_old = -U_it
    F = np.zeros((2, n_n))
    E = np.zeros((3, n_int))
    Ep_old = np.zeros((4, n_int))
    zeta_hist = np.zeros(1000)
    pressure_old = 0
    pressure_hist = np.zeros(1000)

    # storage of assembly time in dependence on plastic integration points
    assembly = np.zeros((20000, 2))
    assembly_step = 0
    aux = np.array(range(3 * n_int)).reshape((3, n_int), order='F')

    ####################
    # Iterative loading
    ####################
    step = 1
    criterion = None
    # TODO rewrite loop with stopping condition
    while True:

        # Load factor update
        zeta = zeta_old + d_zeta
        log.log(log.INFO, f'load factor = {zeta}')
        log.log(log.INFO, f'load increment = {d_zeta}')
        log.log(log.INFO, f'pressure = {pressure_old}')

        ############################
        # Semi-smooth Newton method
        ############################
        max_it = 25
        for i in range(max_it):

            # consistent tangent stiffness matrix
            E = (B@U_it.reshape((-1, 1), order='F')).reshape((3, -1), order='F')
            constitutive_problem = construct_constitutive_problem(E, Ep_old, shear, bulk, eta, c,
                                                                  apply_plastic_strain=False)

            vD = np.tile(weight, (9, 1)) * constitutive_problem['ds']
            D_p = ssp.csr_matrix((flatten_row(vD)[0], (flatten_row(iD)[0] - 1, flatten_row(jD)[0] - 1)),
                                 shape=(3*n_int, 3*n_int))
            K_tangent = K_elast + B.T * (D_p-D_elast) * B

            # measuring assembly dependance on plastic integration points
            # n_plast = len(weight[constitutive_problem['ind_p'].reshape((1, 800))])

            # TODO measure time of matrix assembling!

            # vector of internal forces
            F = B.T * np.reshape(np.tile(weight, (3, 1)) * constitutive_problem['s'][0:3, :], (3*n_int, 1), order='F')

            # Newton's increment
            # TODO rewrite more efficiently!
            Q_flat = mesh['Q'].reshape((-1, 1), order='F')
            K_bool_indices = (Q_flat @ Q_flat.T).astype(bool)
            stiff_mat = K_tangent[K_bool_indices]
            stiff_mat = stiff_mat.reshape((int(np.sqrt(stiff_mat.shape[1])), -1), order='F')
            dU.T[mesh['Q'].T] = np.linalg.solve(stiff_mat, -F[Q_flat].T).flatten(order='F')

            # next iteration
            U_new = U_it + dU

            # stopping criterion
            q1 = np.sqrt(flatten_row(dU) @ K_elast @ flatten_col(dU))
            q2 = np.sqrt(flatten_row(U_it) @ K_elast @ flatten_col(U_it))
            q3 = np.sqrt(flatten_row(U_new) @ K_elast @ flatten_col(U_new))
            criterion = (q1/(q2+q3))[0][0]
            if np.isnan(criterion):
                it = max_it
                break

            log.log(log.INFO, f'stopping criterion: {criterion}')

            # update of unknown arrays
            U_it = U_new

            # test on the stopping criterion
            if criterion < 1e-12:
                break

        # TODO implement Newton solver AS A SEPARATE FUNCTION!
        constitutive_problem = None
        if criterion < 1e-10:
            # Successful convergence
            U_old = U
            U = U_it
            E = (B @ flatten_col(U)).reshape((3, -1), order='F')
            constitutive_problem = construct_constitutive_problem(E, Ep_old, shear, bulk, eta, c,
                                                                  apply_plastic_strain=True)
            Ep_old = constitutive_problem['ep']
            zeta_old = zeta
            d_zeta_old = d_zeta
            zeta_hist[step] = zeta
            step += 1

            # normalized mean pressure on the footing
            pressure_array = transform(constitutive_problem['s'][1, :], mesh['elements'], weight)
            pressure = -np.mean(pressure_array[q_nd.reshape((1, -1))])/c0
            pressure_hist[step] = pressure

            if pressure-pressure_old < 0.1 and criterion < 1e-12:
                d_zeta *= 2

            pressure_old = pressure
        else:
            log.log(log.WARNING, 'The Newton solver does not converge! Decreasing load increment...')

            # Decrease load increment
            d_zeta /= 2

        # initialization for the next iteration
        U_it = d_zeta * (U-U_old) / d_zeta_old + U

        # stopping criteria for the loading process
        if zeta_old >= zeta_max:
            log.log(log.WARNING, 'Maximal load factor was achieved.')
            break

        if d_zeta < d_zeta_min:
            log.log(log.WARNING, 'Too small load increments"')
            break

        print(f'step = {step}')

    if draw:
        # TODO move outside this function!

        ####################################
        # visualization of selected results
        ####################################

        # mesh visualization
        if level < 2:
            draw_mesh(mesh['coordinates'], mesh['elements'], element_type)

        # total displacements + deformed shape
        U_max = max(U[1, :])
        U_total = np.sqrt(U[0, :]**2 + U[1, :]**2)
        draw_quantity(mesh['coordinates'], mesh['elements'], U/U_max, U_total, element_type, size_xy)

        # total displacements less than 0.01
        draw_quantity(mesh['coordinates'], mesh['elements'], 0*U, np.array([e if e < 0.01 else 0.01 for e in U_total]),
                      element_type, size_xy)

        # plastic multipliers
        Ep_node = transform(np.sqrt(sum(constitutive_problem['ep']*constitutive_problem['ep'])), mesh['elements'],
                            weight)
        draw_quantity(mesh['coordinates'], mesh['elements'], 0*U, (np.zeros(Ep_node.shape) + np.array(Ep_node))[0],
                      element_type, size_xy)


if __name__ == '__main__':
    # TODO add usage "help" here
    pass
