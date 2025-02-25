# Qiskit Playground

This repository contains example scripts demonstrating quantum chemistry calculations using Qiskit Nature and variational quantum algorithms (VQE). The examples focus on computing the ground-state energy of a Mn(II)O molecule using two different approaches:

- **Local Simulation:** Using Qiskit's local simulator with the `NumPyMinimumEigensolver` and VQE.
- **Cloud Execution with Qiskit Runtime:** Running VQE on IBM Quantum’s cloud infrastructure via the Qiskit Runtime service.

## Repository Contents

- **mno.py:**  
  A script that sets up the Mn(II)O quantum chemistry problem using Qiskit Nature, applies active space reduction and symmetry tapering, and computes the ground-state energy using both exact diagonalization and a VQE approach on a local simulator.

- **mno_runtime.py:**  
  An adapted version of the quantum chemistry workflow that integrates with Qiskit Runtime. This example demonstrates how to execute the VQE algorithm on IBM Quantum’s cloud using the `EstimatorV2` and selecting an appropriate cloud backend.

## Prerequisites

Ensure you have the following installed in your Python environment:

- [Qiskit](https://qiskit.org) (version 1.0 or later)
- [Qiskit Nature](https://qiskit.org/documentation/nature/)  
- [Qiskit IBM Runtime](https://pypi.org/project/qiskit-ibm-runtime/)  
- [Qiskit Aer](https://qiskit.org/documentation/aer/)  
- Other dependencies: `numpy`, `pandas`, `scipy`, `matplotlib` 

Install the required packages via pip:

```bash
pip install qiskit qiskit-nature qiskit-ibm-runtime qiskit-aer numpy pandas scipy matplotlib
