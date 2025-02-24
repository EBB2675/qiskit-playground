from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

es_problem = driver.run()


# use a Jordan-Wigner mapping into the qubit space 
# it maps the occupation of one spin-orbital to the occupation of one qubit
# possible fermionic mappers in Qiskit Nature:
# Jordan-Wigner (Zeitschrift für Physik, 47, 631-651 (1928))
# Parity (The Journal of chemical physics, 137(22), 224109 (2012))
# Bravyi-Kitaev (Annals of Physics, 298(1), 210-226 (2002
from qiskit_nature.second_q.mappers import JordanWignerMapper

mapper = JordanWignerMapper()

# numpy exact solver
from qiskit_algorithms import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()

# VQE solver
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

ansatz = UCCSD(
    es_problem.num_spatial_orbitals,
    es_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
    ),
)

vqe_solver = VQE(Estimator(), ansatz, SLSQP())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

from qiskit_nature.second_q.algorithms import GroundStateEigensolver

calc = GroundStateEigensolver(mapper, vqe_solver)

res = calc.solve(es_problem)
print(res)

calc_exact = GroundStateEigensolver(mapper, numpy_solver)
res_exact = calc_exact.solve(es_problem)
print(res_exact)
