"""
This module contains the Python3 implementation of matlabfem library (https://github.com/matlabfem/)
intended to solve elasticity and plasticity problems.
"""

# TODO consider object-oriented architecture!

import enum
import numpy as np
import numpy.matlib as npm
import scipy as sp
import scipy.sparse as ssp


def flatten_row(v: np.array) -> np.array:
    return np.reshape(v, (1, np.size(v)), order='F')


def flatten_col(v: np.array) -> np.array:
    return np.reshape(v, (np.size(v), 1), order='F')


class LagrangeElementType(enum.Enum):
    """ Enumeration type allowing users to select the type of Lagrange finite elements."""
    P1 = 1
    P2 = 2
    Q1 = 3
    Q2 = 4


def get_quadrature_volume(el_type: LagrangeElementType) -> tuple:
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


def get_elastic_stiffness_matrix(elements: np.array,
                                 coordinates: np.array,
                                 shear: np.array,
                                 bulk: np.array,
                                 dhatp1: np.array,
                                 dhatp2: np.array,
                                 wf: np.array) -> ssp.csr_matrix:
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

    return K


if __name__ == '__main__':
    # TODO add "an introduction hint" for new users
    pass
