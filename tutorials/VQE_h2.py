from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# electronic-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(
    atom='H .0 .0 .0; H .0 .0 0.735',
    unit=DistanceUnit.ANGSTROM,
    basis='sto3g',
)
problem = driver.run()

# setup the qubit mapper
from qiskit_nature.second_q.mappers import ParityMapper

mapper = ParityMapper(num_particles=problem.num_particles)

# setup the classical optimizer for the VQE
from qiskit_algorithms.optimizers import L_BFGS_B

optimizer = L_BFGS_B()

# setup the estimator primitive for the VQE
from qiskit.primitives import Estimator

estimator = Estimator()

# setup the ansatz for VQE
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    ),
)

# set up our actual VQE instance
from qiskit_algorithms import VQE

vqe = VQE(estimator, ansatz, optimizer)
# ensure that the optimizer starts in the all-zero state which corresponds to
# the Hartree-Fock starting point
vqe.initial_point = [0] * ansatz.num_parameters

# prepare the ground-state solver and run it
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

algorithm = GroundStateEigensolver(mapper, vqe)

electronic_structure_result = algorithm.solve(problem)
electronic_structure_result.formatting_precision = 6
print(electronic_structure_result)
