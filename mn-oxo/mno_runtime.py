import numpy as np
import pandas as pd

# Qiskit Nature and chemistry-specific imports.
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver, MethodType
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper

# Qiskit Algorithms and optimizer.
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import SLSQP

# Import Qiskit Runtime components.
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

# Define the Mn(II)O molecule.
molecule = "Mn 0 0 0; O 0 0 1.6"
charge = 0
multiplicity = 6  # High-spin Mn(II): d5 state (S = 5/2)

def compute_energy(solver_type, basis="sto3g",
                   active_electrons_tuple=(6, 1), num_active_orbitals=8):
    """
    Computes the ground-state energy of Mn(II)O using ROHF and an active space reduction.

    Parameters:
        solver_type (str): "numpy" for exact diagonalization or "vqe" for the variational method.
        basis (str): Basis set (default "sto3g").
        active_electrons_tuple (tuple): Active electrons tuple (alpha, beta). For example, (6, 1).
        num_active_orbitals (int): Number of active spatial orbitals.

    Returns:
        energy (float): Ground-state energy in Hartree.
        symmetry (str): Extracted symmetry information, if available.
        es_problem: The electronic structure problem instance.
    """
    # Configure the PySCF driver using ROHF.
    driver = PySCFDriver(
        atom=molecule,
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,  # PySCF expects 2S = multiplicity - 1.
        unit=DistanceUnit.ANGSTROM,
        method=MethodType.ROHF
    )
    es_problem = driver.run()

    # Apply an active space transformer.
    transformer = ActiveSpaceTransformer(active_electrons_tuple, num_active_orbitals)
    es_problem = transformer.transform(es_problem)

    # Obtain the second-quantized Hamiltonian.
    hamiltonian = es_problem.hamiltonian.second_q_op()

    # Set up the fermionic-to-qubit mapping using the Jordan-Wigner mapper.
    mapper = JordanWignerMapper()

    # Apply symmetry (Z₂) tapering.
    tapered_mapper = es_problem.get_tapered_mapper(mapper)

    if solver_type.lower() == "numpy":
        # Use exact diagonalization.
        solver = NumPyMinimumEigensolver()
        gs_solver = GroundStateEigensolver(tapered_mapper, solver)
    elif solver_type.lower() == "vqe":
        # Set up a chemistry-specific ansatz.
        initial_state = HartreeFock(es_problem.num_spatial_orbitals,
                                    es_problem.num_particles, tapered_mapper)
        ansatz = UCCSD(es_problem.num_spatial_orbitals,
                       es_problem.num_particles, tapered_mapper,
                       initial_state=initial_state)
        optimizer = SLSQP(maxiter=200)

        # --- Qiskit Runtime configuration ---
        # Initialize the Qiskit Runtime service.
        service = QiskitRuntimeService(channel="ibm_quantum")
        # Select a backend; here we choose a simulator available on the cloud.
        backend = service.least_busy(operational=True, simulator=True)
        # Create the runtime estimator.
        estimator = EstimatorV2(service=service, backend=backend)
        # ------------------------------------

        # Create the VQE instance with the runtime estimator.
        solver = VQE(estimator, ansatz, optimizer=optimizer)
        solver.initial_point = [0.0] * ansatz.num_parameters
        gs_solver = GroundStateEigensolver(tapered_mapper, solver)
    else:
        raise ValueError("solver_type must be 'numpy' or 'vqe'.")

    # Solve for the ground state.
    result = gs_solver.solve(es_problem)
    energy = result.total_energies[0]

    # Attempt to extract symmetry information.
    try:
        symmetry = result.eigenstate.transformed_data.get("Z2Symmetries", "Not available")
    except Exception:
        symmetry = "Not available"

    return energy, symmetry, es_problem

# Run calculations for both solvers.
results = []
for solver_type in ["numpy", "vqe"]:
    energy, symmetry, _ = compute_energy(solver_type)
    results.append([solver_type, energy, symmetry])

df_results = pd.DataFrame(results, columns=["Solver", "Total Energy (Ha)", "Symmetry"])
print(df_results)
df_results.to_csv("mn_oxide_results_runtime.csv", index=False)
print("Results saved to 'mn_oxide_results_runtime.csv'")
