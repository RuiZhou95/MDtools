"""
Data model definitions: for template roles, molecule copies, packed systems, etc.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class TemplateRoleAssignment:
    """Role assignment for a single atom"""
    atom_index: int          # Index in the template molecule (0-based)
    role: str                # Chemical role name, e.g., 'c_carbonyl', 'o_carbonyl'
    element: str             # Element symbol
    is_aromatic: bool = False

@dataclass
class TemplateInfo:
    """Complete information for a single molecule template"""
    template_name: str               # User-specified name, e.g., 'EST', 'DPA', 'OXY'
    smiles: str                      # Original SMILES
    num_atoms: int                   # Total number of atoms (after adding hydrogens)
    pdb_file: str                    # Single-molecule template PDB file path (relative to workspace)
    role_to_indices: Dict[str, List[int]] = field(default_factory=dict)   # Role -> list of atom indices within template
    atom_index_to_role: Dict[int, str] = field(default_factory=dict)      # Atom index -> role
    assignments: List[TemplateRoleAssignment] = field(default_factory=list)

@dataclass
class PackedAtomRecord:
    """An atom in the packed system"""
    global_id: int           # Atom ID in final LAMMPS data (1-based)
    mol_id: int              # Molecule copy ID (1-based)
    template_name: str       # Template name it belongs to
    template_atom_index: int # Atom index in the template molecule (0-based)
    role: str                # Chemical role (inherited from template)
    element: str
    x: float
    y: float
    z: float
    charge: float = 0.0

@dataclass
class PackedMoleculeRecord:
    """A molecule copy in the packed system"""
    mol_id: int
    template_name: str
    atom_indices: List[int]  # List of global_id for all atoms in this molecule

@dataclass
class PackedSystem:
    """Complete packed system"""
    atoms: List[PackedAtomRecord]                    # All atoms
    molecules: List[PackedMoleculeRecord]            # 所有分子副本
    box_size: List[float]                            # [xlo, xhi, ylo, yhi, zlo, zhi]
    template_infos: Dict[str, TemplateInfo]          # 模板名称 -> 模板信息