import matlab.engine
import pythonFEM
import pprint as pp
import numpy as np
import time

def el_num(collection):
    return sum([len(e) for e in collection])


mat_eng = matlab.engine.start_matlab()
# print(mat_eng)

# Add path to the 3rd party MATLAB modules
mat_eng.addpath(
    mat_eng.genpath('/home/martin/matlab_fem_elastoplasticity/elasticity/3rd_party_elasticity_codes_for_testing'))

levels = 8  # maximum uniform refinement level

# homogeneous material parameters
young = 206900  # Young's modulus E
poisson = 0.29  # Poisson's ratio nu

lmbd = young * poisson / ((1 + poisson) * (1 - 2 * poisson))  # Lamme first parameter
mu = young / (2 * (1 + poisson))  # Lamme second parameter

bulkC = young / (3 * (1 - 2 * poisson))  # bulk modulus K
shearC = mu  # shear modulus G (equal to Lamme second parameter)

demo = 0
mat_eng.workspace['demo'] = demo
mat_eng.create_2D_mesh(nargout=0)
coordinates = mat_eng.workspace['coordinates']
elements = mat_eng.workspace['elements']
dirichlet = mat_eng.workspace['dirichlet']

# print(mat_eng.workspace)

# mat_eng.test(nargout=0)
# print(mat_eng.workspace['a'])

level_size_P1 = []
for level in range(levels + 1):
    # uniform refinement
    if level > 0:
        # coordinates, elements, dirichlet = mat_eng.refinement_uniform(coordinates, elements, dirichlet)
        coordinates, elements, dirichlet = mat_eng.refinement_uniform(coordinates, elements, dirichlet, nargout=3)
        # pp.pprint(coordinates)
        # pp.pprint(elements)
        # pp.pprint(dirichlet)

    mat_eng.workspace['level'] = mat_eng.double(level)
    mat_eng.workspace['coordinates'] = coordinates
    mat_eng.workspace['elements'] = elements
    mat_eng.workspace['dirichlet'] = dirichlet
    mat_eng.visualize_mesh(nargout=0)

    rows = el_num(coordinates)
    level_size_P1.append(rows)

    # stiffness matrix assembly - method 1
    # print('Technique of Cermak, Sysala and Valdman:')
    xi, wf = pythonFEM.get_quadrature_volume(pythonFEM.LagrangeElementType.P1)
    hatp, dhatp1, dhatp2 = pythonFEM.get_local_basis_volume(pythonFEM.LagrangeElementType.P1, xi)
    n_e = len(elements)  # TODO check later - may be insufficient after elements will not be matlab.double!
    n_q = len(wf)
    n_int = n_e * n_q
    shear = shearC * np.ones(n_int)
    bulk = bulkC * np.ones(n_int)

    # Cast matlab.engine-types to numpy-types
    # TODO remove when everything rewritten to Python
    elements_numpy = np.array(elements)
    coordinates_numpy = np.array(coordinates)

    assemble_start = time.process_time()
    K = pythonFEM.get_elastic_stiffness_matrix(elements_numpy.transpose(), coordinates_numpy.transpose(),
                                               shear, bulk, dhatp1, dhatp2, wf)
    assemble_end = time.process_time()

    # stiffness matrix assembly - method 2
    # print('Technique of Rahman and Valdman:')
    # K2, areas = mat_eng.stiffness_matrixP1_2D_elasticity(elements, coordinates, lmbd, mu, nargout=2)
    print('Level {}: Assemble time: {}, rows = {}'.format(level, assemble_end - assemble_start, rows))

# Quit matlab engine
mat_eng.quit()
