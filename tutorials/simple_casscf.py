from pyscf import gto, scf, mcscf

import numpy as np

from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import ParityMapper

from qiskit_nature_pyscf import QiskitSolver

mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")

h_f = scf.RHF(mol).run()

norb = 2
nalpha, nbeta = 1, 1
nelec = nalpha + nbeta

cas = mcscf.CASCI(h_f, norb, nelec)

mapper = ParityMapper(num_particles=(nalpha, nbeta))

ansatz = UCCSD(
    norb,
    (nalpha, nbeta),
    mapper,
    initial_state=HartreeFock(
        norb,
        (nalpha, nbeta),
        mapper,
    ),
)

vqe = VQE(Estimator(), ansatz, SLSQP())
vqe.initial_point = np.zeros(ansatz.num_parameters)

algorithm = GroundStateEigensolver(mapper, vqe)

cas.fcisolver = QiskitSolver(algorithm)

cas.run()
