from pyscf import gto
mol = gto.M(
    atom="Mn 0 0 0; O 0 0 1.6",
    basis="sto3g",
    charge=0,
    spin=5,  
    symmetry=True
)
mol.build()
print("Symmetry enabled:", mol.symmetry)   
print("Point group label:", mol.irrep_name) 
