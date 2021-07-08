"""
This module contains the Python3 implementation of matlabfem library (https://github.com/matlabfem/)
intended to solve elasticity and plasticity problems.
"""

# TODO consider object-oriented architecture!

import enum
import os

import numpy as np
import numpy.matlib as npm
import scipy as sp
import scipy.sparse as ssp
import typing as tp
from numba import njit

# os.environ["NUMBA_DISABLE_JIT"] = "1"


#@njit
def flatten_row(v: np.array) -> np.array:
    return np.reshape(v, (1, np.size(v)), order='F')


#@njit
def flatten_col(v: np.array) -> np.array:
    return np.reshape(v, (np.size(v), 1), order='F')


class LagrangeElementType(enum.Enum):
    """ Enumeration type allowing users to select the type of Lagrange finite elements."""
    P1 = 1
    P2 = 2
    Q1 = 3
    Q2 = 4


#@njit
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
    init_dict = {LagrangeElementType.P1: (np.array([[1/3], [1/3]]), np.array([[0.5]])),
                 LagrangeElementType.P2: (np.array([[0.1012865073235, 0.7974269853531, 0.1012865073235,
                                                     0.4701420641051, 0.4701420641051, 0.0597158717898, 1/3],
                                                    [0.1012865073235, 0.1012865073235, 0.7974269853531,
                                                     0.0597158717898, 0.4701420641051, 0.4701420641051, 1/3]]),
                                          0.5 * np.array([[0.1259391805448, 0.1259391805448, 0.1259391805448,
                                                           0.1323941527885, 0.1323941527885, 0.1323941527885, 0.225]])),
                 LagrangeElementType.Q1: (np.array([[-pt, -pt, pt, pt], [-pt, pt, -pt, pt]]),
                                          np.array([[1, 1, 1, 1]])),
                 LagrangeElementType.Q2: (np.array([[-pt,  pt, pt, -pt, 0, pt, 0, -pt, 0],
                                                    [-pt, -pt, pt,  pt, -pt,  0, pt, 0, 0]]),
                                          np.array([[25/81, 25/81, 25/81, 25/81, 40/81, 40/81, 40/81, 40/81, 64/81]]))}

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

    pt = 1/np.sqrt(3)

    init_dict = {LagrangeElementType.P1: (np.array([0]), np.array([2])),
                 LagrangeElementType.Q1: (np.array([0]), np.array([2])),
                 LagrangeElementType.P2: (np.array([-pt, pt]), np.array([1, 1])),
                 LagrangeElementType.Q2: (np.array([-pt, pt]), np.array([1, 1]))}

    return init_dict[el_type]


#@njit
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

    init_dict = {LagrangeElementType.P1: (np.array([1-xi_1-xi_2, xi_1, xi_2]),
                                          np.array([[-1], [1], [0]]),
                                          np.array([[-1], [0], [1]])),
                 LagrangeElementType.P2: (np.array([xi_0*(2*xi_0-1), xi_1*(2*xi_1-1), xi_2*(2*xi_2-1),
                                                    4*xi_1*xi_2, 4*xi_0*xi_2, 4*xi_0*xi_1]),
                                          np.array([-4*xi_0+1, 4*xi_1-1, np.zeros(n_q), 4*xi_2, -4*xi_2,
                                                    4*(xi_0-xi_1)]),
                                          np.array([-4*xi_0+1, np.zeros(n_q), 4*xi_2-1, 4*xi_1, 4*(xi_0-xi_2),
                                                    -4*xi_1])),
                 LagrangeElementType.Q2: (np.array([(1-xi_1)*(1-xi_2)*(-1-xi_1-xi_2)/4,
                                                    (1+xi_1)*(1-xi_2)*(-1+xi_1-xi_2)/4,
                                                    (1+xi_1)*(1+xi_2)*(-1+xi_1+xi_2)/4,
                                                    (1-xi_1)*(1+xi_2)*(-1-xi_1+xi_2)/4,
                                                    (1-pow(xi_1, 2))*(1-xi_2)/2,
                                                    (1+xi_1)*(1-pow(xi_2, 2))/2,
                                                    (1-pow(xi_1, 2))*(1+xi_2)/2,
                                                    (1-xi_1)*(1-pow(xi_2, 2))/2]),
                                          np.array([(1-xi_2)*(2*xi_1+xi_2)/4,
                                                    (1-xi_2)*(2*xi_1-xi_2)/4,
                                                    (1+xi_2)*(2*xi_1+xi_2)/4,
                                                    (1+xi_2)*(2*xi_1-xi_2)/4,
                                                    -xi_1*(1-xi_2),
                                                    (1-pow(xi_2, 2))/2,
                                                    -xi_1*(1+xi_2),
                                                    -(1-pow(xi_2, 2))/2]),
                                          np.array([(1-xi_1)*( xi_1+2*xi_2)/4,
                                                    (1+xi_1)*(-xi_1+2*xi_2)/4,
                                                    (1+xi_1)*( xi_1+2*xi_2)/4,
                                                    (1-xi_1)*(-xi_1+2*xi_2)/4,
                                                    -(1-pow(xi_1, 2))/2,
                                                    -(1+xi_1)*xi_2,
                                                    (1-pow(xi_1, 2))/2,
                                                    -(1-xi_1)*xi_2])),
                 LagrangeElementType.Q1: (np.array([(1-xi_1)*(1-xi_2)/4,
                                                    (1+xi_1)*(1-xi_2)/4,
                                                    (1+xi_1)*(1+xi_2)/4,
                                                    (1-xi_1)*(1+xi_2)/4]),
                                          np.array([-(1-xi_2)/4,
                                                    (1-xi_2)/4,
                                                    (1+xi_2)/4,
                                                    -(1+xi_2)/4]),
                                          np.array([-(1-xi_1)/4,
                                                    -(1+xi_1)/4,
                                                    (1+xi_1)/4,
                                                    (1-xi_1)/4]))}

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

    xi = xi_s[0, ]

    init_1 = (0.5 * np.array([[1-xi], [1+xi]]), np.array([[-0.5], [0.5]]))
    init_2 = (np.array([[np.dot(xi, (xi-1)/2)],
                        [np.dot(xi, (xi+1)/2)],
                        [np.dot(xi+1, 1-xi)]]),
              np.array([[xi-0.5], [xi+0.5], [-2*xi]]))

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
    vf1 = np.multiply(hatphi, np.ones((n_p, 1)) * np.multiply(weight, f_V_int[0, ])).flatten(order='F')
    vf2 = np.multiply(hatphi, np.ones((n_p, 1)) * np.multiply(weight, f_V_int[1, ])).flatten(order='F')

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


#@njit
def get_elastic_stiffness_matrix(elements: np.array,
                                 coordinates: np.array,
                                 shear: np.array,
                                 bulk: np.array,
                                 dhatp1: np.array,
                                 dhatp2: np.array,
                                 wf: np.array) -> tp.Tuple[np.array, float]:
    n_n = np.size(coordinates, 1)  # number of nodes including midpoints
    n_e = np.size(elements, 1)  # number of elements
    n_p = np.size(elements, 0)  # number of vertices per element
    n_q = np.size(wf, 0)  # number of quadrature points
    n_int = n_e * n_q  # total number of integrations points

    # Jacobian

    # extension of the input arrays DHatP1,DHatP2 by replication
    # size(dhat_phi1)=size(dhat_phi2)=(n_p,n_int)
    dhat_phi1 = np.matlib.repmat(dhatp1, 1, n_e)
    dhat_phi2 = np.matlib.repmat(dhatp2, 1, n_e)

    # Shift the elements' indices - TODO remove after 3rd party modules are rewritten to Python!
    elements -= 1

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
    aux_3 = 2 * (elements[np.reshape(aux_1, np.size(aux_1), order='F')]+1) - np.kron(np.ones(n_e),
                                                                                     np.reshape(aux_2,
                                                                                                (np.size(aux_2), 1),
                                                                                                order='F'))

    jB = np.kron(aux_3, np.ones((3, n_q)))

    # the sparse strain-displacement matrix B
    B = ssp.csr_matrix((flatten_row(vB)[0], (flatten_row(iB)[0] - 1, flatten_row(jB)[0] - 1)),
                       shape=(3 * n_int, 2 * n_n))

    # assembling of the elastic stress-strain matrix

    # elastic tensor at each integration point:
    iota = np.array([[1], [1], [0]])
    vol = iota * iota.transpose()
    dev = np.diag([1, 1, 0.5]) - vol/3
    elast = 2 * flatten_col(dev) * shear + flatten_col(vol) * bulk

    # weight coefficients
    weight = np.abs(det) * np.matlib.repmat(wf, 1, n_e)

    # assemblinng of the sparse matrix D
    id = np.matlib.repmat(aux, 3, 1)
    jd = np.kron(aux, flatten_col(np.array([1] * 3)))
    vd = elast * (flatten_col(np.array([1] * 9)) * weight)
    D = ssp.csr_matrix((flatten_row(vd)[0], (flatten_row(id)[0] - 1, flatten_row(jd)[0] - 1)))

    # elastic stiffness matrix
    K = B.transpose() * D * B

    return K.toarray(), weight


#@njit
def get_nodes_1(level: int,
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

    n_n = (N_x + 1) * (N_y + 1) - N1_x * N1_y
    n_cell_xy = N_x * N_y - N1_x * N1_y
    n_e = n_cell_xy * 2

    # C - auxiliary array for the further mesh construction
    C = np.zeros((N_x + 1, N_y + 1))
    C1 = np.array(range(1, (N2_x + 1) * N1_y + 1)).reshape((N2_x+1, N1_y), order='F')
    C2 = np.array(range(((N2_x+1)*N1_y+1), n_n + 1)).reshape((N_x + 1, N2_y + 1), order='F')
    C[N1_x:N_x+1, 0:N1_y] = C1
    C[0:N_x+1, N1_y:N_y+1] = C2

    coord_x = np.linspace(0, size_xy, N_x + 1)
    coord_y = np.linspace(0, size_xy, N_y + 1)

    c_x = np.concatenate((np.tile(coord_x[N1_x:N_x+1], (1, N1_y)),
                          np.tile(coord_x, (1, N2_y+1))),
                         axis=1).flatten(order='F')
    c_y = np.concatenate((np.tile(np.kron(coord_y[0:N1_y], np.ones((1, N2_x+1))), 1),
                          np.tile(np.kron(coord_y[N1_y:N_y+1], np.ones((1, N_x+1))), 1)),
                         axis=1).flatten(order='F')
    coord = np.array([c_x, c_y])

    V1 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V1[N1_x:N_x, 0:N1_y] = 1
    V1[0:N_x, N1_y:N_y] = 1
    V1 = V1.transpose()

    V2 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V2[N1_x+1:N_x+1, 0:N1_y] = 1
    V2[1:N_x+1, N1_y:N_y] = 1
    V2 = V2.transpose()

    V3 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V3[N1_x+1:N_x+1, 1:N1_y+1] = 1
    V3[1:N_x+1, N1_y+1:N_y+1] = 1
    V3 = V3.transpose()

    V4 = np.zeros((N_x+1, N_y+1), dtype=bool)
    V4[N1_x:N_x, 1:N1_y+1] = 1
    V4[0:N_x, N1_y+1:N_y+1] = 1
    V4 = V4.transpose()

    #TODO check if col-wise indexing is possible natively!
    Ct = C.transpose()

    elem = None
    if element_type == LagrangeElementType.P1:
        #
        # aux_elem = np.array([C[V1].transpose().flatten(order='F'),
        #                      C[V2].transpose().flatten(order='F'),
        #                      C[V4].transpose().flatten(order='F'),
        #                      C[V2].transpose().flatten(order='F'),
        #                      C[V3].transpose().flatten(order='F'),
        #                      C[V4].transpose().flatten(order='F')])
        #

        aux_elem = np.array([Ct[V1],
                             Ct[V2],
                             Ct[V4],
                             Ct[V2],
                             Ct[V3],
                             Ct[V4]])

        elem = aux_elem.reshape((3, n_e), order='F')

    elif element_type == LagrangeElementType.Q1:
        elem = np.array([C[V1].transpose().flatten(order='F'),
                         C[V2].transpose().flatten(order='F'),
                         C[V3].transpose().flatten(order='F'),
                         C[V4].transpose().flatten(order='F')])

    ###############
    # Body surface
    ###############

    # Face 1 (y=0)
    C_s = C[:, 0].copy().reshape((C[:, 0].shape[0], -1))
    v1_s = np.zeros((N_x+1, 1), dtype=bool)
    v2_s = v1_s.copy() # np.zeros((N_x, 1), dtype=bool)
    v1_s[N1_x:N_x, 0] = 1
    v2_s[N1_x+1:N_x+1, 0] = 1

    surf1 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf1 = surf1.reshape((2, N2_x))

    # Face 2 (x=size_xy)
    C_s = C[-1, :].copy().reshape((C[-1, :].shape[0], -1))
    v1_s = np.zeros((N_y+1, 1), dtype=bool)
    v2_s = v1_s.copy() # np.zeros((N_y+1, 1), dtype=bool)
    v1_s[0:N_y, 0] = 1
    v2_s[1:N_y+1, 0] = 1

    surf2 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf2 = surf2.reshape((2, N_y))

    # Face 3 (y=size_xy)
    C_s = C[:, -1].copy().reshape((C[:, -1].shape[0], -1))
    v1_s = np.zeros((N_x+1, 1), dtype=bool)
    v2_s = v1_s.copy() # np.zeros((N_x+1, 1), dtype=bool)
    v1_s[0:N_x, 0] = 1
    v2_s[1:N_x+1, 0] = 1

    surf3 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf3 = surf3.reshape((2, N_x))

    # Face 4 (x=0)
    C_s = C[0, :].copy().reshape((C[0, :].shape[0], -1))
    v1_s = np.zeros((N_y+1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v1_s[N1_y:N_y, 0] = 1
    v2_s[N1_y+1:N_y+1, 0] = 1

    surf4 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf4 = surf4.reshape((2, N2_y))

    # Face 5 (y=size_hole)
    C_s = C[N1_x, :].copy().reshape((C[N1_x, :].shape[0], -1))
    v1_s = np.zeros((N_x+1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v1_s[0:N1_y, 0] = 1
    v2_s[1:N1_x+1, 0] = 1

    surf5 = np.array([C_s[v1_s].transpose().flatten(order='F'),
                      C_s[v2_s].transpose().flatten(order='F')])

    if element_type == LagrangeElementType.P1:
        surf5 = surf5.reshape((2, N1_x))

    # Face 6 (x=size_hole)
    C_s = C[N1_x+1, :].copy().reshape((C[N1_x+1, :].shape[0], -1))
    v1_s = np.zeros((N_y+1, 1), dtype=bool)
    v2_s = v1_s.copy()
    v1_s[0:N1_y, 0] = 1
    v2_s[1:N1_y+1, 0] = 1

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
            'neumann_nodes': neumann-1, 'dirichlet_nodes': dirichlet, 'Q': Q}


def get_nodes_2(level: int,
                element_type: LagrangeElementType,
                size_xy: float,
                size_hole: float) -> tp.Dict[str, np.array]:
    # TODO implement
    raise NotImplementedError()


#@njit
def assemble_mesh_1(level: int, element_type: LagrangeElementType, size_xy: float, size_hole: float):
    return get_nodes_1(level, element_type, size_xy, size_hole)


def assemble_mesh_2(level: int, element_type: LagrangeElementType, size_xy: float, size_hole: float):
    return get_nodes_2(level, element_type, size_xy, size_hole)


#@njit
def assemble_mesh(level: int, element_type: LagrangeElementType, size_xy: float, size_hole: float):
    mesh = None
    if element_type in (LagrangeElementType.P1, LagrangeElementType.Q1):
        mesh = assemble_mesh_1(level, element_type, size_xy, size_hole)
    else:
        mesh = assemble_mesh_1(level, element_type, size_xy, size_hole)

    return mesh


def draw_mesh(coordinates: np.array, elements: np.array, elem_type: LagrangeElementType):
    # TODO implement
    raise NotImplementedError()


def draw_displacement(coordinates , elem , U, U_disp, elem_type, size_xy,size_hole):
    # TODO implement
    raise NotImplementedError()


#@njit
def elasticity_fem(element_type: LagrangeElementType = LagrangeElementType.P1,
                   level: int = 1,
                   draw: bool = True) -> tp.Dict[str, int]:
    """
    This function constructs the mesh, assembles the elastic stiffness
    matrix and the load vector, and computes displacements depending on
    an element type and mesh density. Optionally, it enables to visualize
    the results and the mesh.

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
    young_mod = 206900
    poisson_ratio = 0.29
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    # Volume forces
    volume_force = np.array([[0, -1]])

    # Traction in each direction
    traction_force = np.array([[0, 450]])

    ################
    # Generate mesh
    ################

    # Size of the body
    size_xy = 10

    # Size of the hole in the body
    size_hole = 5

    # Assemble mesh
    mesh = assemble_mesh(level, element_type, size_xy, size_hole)

    ##############################
    # Data from reference element
    ##############################
    xi, wf = get_quadrature_volume(element_type)
    xi_s, wf_s = get_quadrature_surface(element_type)

    hatp, dhatp1, dhatp2 = get_local_basis_volume(element_type, xi)
    hatp_s, dhatp1_s = get_local_basis_surface(element_type, xi_s)

    #####################################
    # Assembling of the stiffness matrix
    #####################################
    n_e = mesh['elements'].shape[1]
    n_q = wf.size
    n_int = n_e * n_q

    shear_mod = shear_mod * np.ones(n_int)
    bulk_mod = bulk_mod * np.ones(n_int)

    #TODO add matrix assembly time measurement
    K, weight = get_elastic_stiffness_matrix(mesh['elements'], mesh['coordinates'], shear_mod,
                                             bulk_mod, dhatp1, dhatp2, wf)

    # rows = K.shape[1]

    ############################################
    # Assembling of the vector of volume forces
    ############################################
    f_V_int = np.dot(volume_force.transpose(), np.ones((1, n_int)))
    f_V = get_vector_volume(mesh['elements'], mesh['coordinates'], f_V_int, hatp, weight).toarray().flatten(order='F')

    ##############################################
    # Assembling of the vector of traction forces
    ##############################################
    n_e_s = mesh['neumann_nodes'].shape[1]
    n_q_s = len(wf_s)
    n_int_s = n_e_s * n_q_s
    f_t_int = np.dot(traction_force.transpose(), np.ones((1, n_int_s)))
    f_t = get_vector_traction(mesh['neumann_nodes'],
                              mesh['coordinates'],
                              f_t_int, hatp_s, dhatp1_s, wf_s).toarray().flatten(order='F')

    ud = (0.5*mesh['dirichlet_nodes'])
    ud_flat = ud.flatten(order='F')

    #############
    # Processing
    #############
    f = (f_t + f_V - np.dot(K, ud_flat)).reshape((-1, 1), order='F')
    u = ud.copy()
    # Computation of displacements
    # TODO is possible in a simpler way?
    Q_flat = mesh['Q'].reshape((-1, 1), order='F')
    K_bool_indices = (Q_flat @ Q_flat.transpose()).astype(bool)
    stiff_mat = K[K_bool_indices]
    stiff_mat = stiff_mat.reshape((int(np.sqrt(stiff_mat.shape[0])), -1), order='F')
    u.transpose()[mesh['Q'].transpose()] = np.linalg.solve(stiff_mat, f[Q_flat])
    u_flat = u.flatten(order='F')

    # Stored energy
    e = 0.5*u_flat @ K @ u_flat - (f_t + f_V) @ u_flat
    print(f'Stored energy: {e}')

    # TODO implement drawing functions!


if __name__ == '__main__':
    # TODO add "an introduction hint" for new users
    pass
