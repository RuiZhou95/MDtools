"""
Global configuration: default role recognition rules, atom type mappings, etc.
"""
from typing import Dict

# Default role recognition SMARTS patterns
# Key: role name, Value: SMARTS expression
# Note: Matching order is important, more specific patterns should be placed first
DEFAULT_ROLE_SMARTS: Dict[str, str] = {
    # Carbonyl carbon (C=O) - CVHD requirement
    "c_carbonyl": "[CX3]=[OX1]",
    # Carbonyl oxygen - CVHD requirement
    "o_carbonyl": "[OX1]=[CX3]",
    # Ester single-bond oxygen (C-O-C=O) - CVHD requirement (o_ester_single)
    "o_ester_single": "[OX2H0][CX3](=O)",
    # Aromatic carbon (six-membered ring) - CVHD requirement (c_ar)
    "c_ar": "[c;!$(c=O)]",
    # Amine nitrogen (sp3) - CVHD requirement
    "n_amine": "[NX3;H2,H1,H0;!$(N=O)]",
    # Peroxide/oxygen molecule oxygen (O=O) - CVHD requirement
    "o2_oxygen": "O=O",
    
    # New CVHD required roles
    # α-carbonyl carbon (aliphatic carbon attached to carbonyl carbon)
    "c_alpha_to_carbonyl": "[CX4;H2,H1,H0][CX3](=O)",
    # α-hydrogen (hydrogen attached to α-carbonyl carbon)
    "h_alpha": "[H][CX4][CX3](=O)",
    # N-H hydrogen (hydrogen attached to amine nitrogen)
    "h_nh": "[H][NX3]",
    
    # Compatibility roles (maintain backward compatibility)
    "c_aromatic": "[c]",                    # Compatible with old name
    "o_ester": "[OX2H0][CX3](=O)",          # Compatible with old name
    "n_aromatic": "[n]",                    # Aromatic nitrogen
    "c_aliphatic": "[CX4;!$(C(=O))]",       # Aliphatic carbon
    "h_aliphatic": "[H][CX4]",              # Aliphatic hydrogen
    "h_aromatic": "[H][c]",                 # Aromatic hydrogen
    "o_hydroxyl": "[OX2H]",                 # Hydroxyl oxygen
    
    # Default role for unmatched atoms
    "other": "*"
}

# Role to LAMMPS atom type mapping (customizable)
# Key: role name, Value: atom type ID (integer)
DEFAULT_ROLE_TO_ATOM_TYPE: Dict[str, int] = {
    # CVHD required roles
    "c_carbonyl": 1,
    "c_ar": 1,
    "o_carbonyl": 3,
    "o_ester_single": 3,
    "n_amine": 4,
    "o2_oxygen": 3,
    "c_alpha_to_carbonyl": 1,
    "h_alpha": 2,
    "h_nh": 2,
    
    # Compatibility roles
    "c_aromatic": 1,
    "o_ester": 3,
    "n_aromatic": 4,
    "c_aliphatic": 1,
    "h_aliphatic": 2,
    "h_aromatic": 2,
    "o_hydroxyl": 3,
    "other": 1,
}

# Default residue names (three-letter codes) for PDB files
DEFAULT_RESIDUE_NAMES = {
    "EST": "EST",   # Esters
    "DPA": "DPA",   # Diphenylamines
    "OXY": "OXY",   # Oxygen
}

# Packmol default parameters
PACKMOL_TOLERANCE = 6.0   # Angstroms
PACKMOL_TIMEOUT = 300     # Seconds