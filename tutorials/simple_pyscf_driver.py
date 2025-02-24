from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver


# this driver will yield a simple electronic structure problem

driver = PySCFDriver(
    atom="H 0 0 0; H 0 0 0.735",
    basis="sto3g",
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
)

problem = driver.run()
print(problem)


# generate the second-quantized operator from the 1- and 2-body integrals 
# which the classical code has computed for us
hamiltonian = problem.hamiltonian

coefficients = hamiltonian.electronic_integrals
print(coefficients.alpha)

second_q_op = hamiltonian.second_q_op()
print(second_q_op)

# up until this point it is purely electronic!
# add the nuclear repulsion energy

hamiltonian.nuclear_repulsion_energy  # NOT included in the second_q_op above

# additional attributes of the ElectronicStructureProblem

print(problem.molecule)
print(problem.reference_energy)
print(problem.num_particles)
print(problem.num_spatial_orbitals)
print(problem.basis)
