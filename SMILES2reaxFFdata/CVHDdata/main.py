#!/usr/bin/env python
"""
Usage example: Demonstrates how to use the extended SmilesToLammps class to generate LAMMPS input files.
"""
from smiles_to_lammps import SmilesToLammps

def main():
    # Example: ester + diphenylamine antioxidant + oxygen
    smiles_list = [
        "O=C(CCCCCC)OCC(COC(CCCC)=O)(COC(CCCCCC)=O)COC(CCCC)=O",  # Ester
        "CC(C)(C)CC(C)(C)C1=CC=C(NC2=CC=C(C(C)(C)CC(C)(C)C)C=C2)C=C1",  # Diphenylamine
        "O=O"  # Oxygen
    ]
    counts = [20, 20, 100] # [5, 5, 100]
    template_names = ["EST", "DPA", "OXY"]   # Custom template names
    target_density = 50.0  # kg/m³
    
    builder = SmilesToLammps(workspace_name="my_data", use_pdb_mode=True)
    success = builder.build_lammps_input(
        smiles_list=smiles_list,
        counts=counts,
        target_density_kg_per_m3=target_density,
        output_file="system.data",
        template_names=template_names
    )
    
    if success:
        print("Success! Generated files:")
        print("  - LAMMPS data: system.data")
        print("  - atoms.ndx and groups.lmp in tmp/my_data/")
    else:
        print("Failed.")

if __name__ == "__main__":
    main()