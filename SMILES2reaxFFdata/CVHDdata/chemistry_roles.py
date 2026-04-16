"""
Identify atomic roles in template molecules using RDKit and SMARTS patterns.
(No 3D conformation involved, only atomic type and connectivity analysis)
"""
import logging
from typing import Dict, List, Optional
from rdkit import Chem

from models import TemplateInfo, TemplateRoleAssignment
from config import DEFAULT_ROLE_SMARTS

logger = logging.getLogger(__name__)

def classify_template_atoms(
    smiles: str,
    template_name: str,
    role_smarts: Optional[Dict[str, str]] = None,
    add_hs: bool = True
) -> TemplateInfo:
    """
    Classify atomic roles for a given SMILES molecule (based on SMARTS substructure matching).
    Note: This function does not generate 3D conformations, only topological analysis.
    
    Args:
        smiles: SMILES string
        template_name: Template name
        role_smarts: Custom role -> SMARTS mapping
        add_hs: Whether to add hydrogen atoms (strongly recommended True)
    
    Returns:
        TemplateInfo object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    if add_hs:
        mol = Chem.AddHs(mol)
    
    if role_smarts is None:
        role_smarts = DEFAULT_ROLE_SMARTS
    
    num_atoms = mol.GetNumAtoms()
    atom_roles: List[str] = ['other'] * num_atoms
    
    # CVHD required roles list (ensure these roles have priority)
    cvhd_roles = {
        "c_carbonyl", "o_carbonyl", "o_ester_single",
        "c_alpha_to_carbonyl", "h_alpha", "c_ar",
        "n_amine", "h_nh", "o2_oxygen"
    }
    
    # Custom sorting: CVHD roles first, then by SMARTS length descending
    def sort_key(item):
        role, smarts = item
        # CVHD roles have higher priority (smaller value)
        priority = 0 if role in cvhd_roles else 1
        # Then by length descending
        return (priority, -len(smarts))
    
    sorted_roles = sorted(role_smarts.items(), key=sort_key)
    
    for role, smarts_pattern in sorted_roles:
        pat = Chem.MolFromSmarts(smarts_pattern)
        if pat is None:
            logger.warning(f"Invalid SMARTS pattern for role {role}: {smarts_pattern}")
            continue
        matches = mol.GetSubstructMatches(pat)
        if matches and role in cvhd_roles:
            logger.debug(f"CVHD role {role} matched {len(matches)} positions")
        for match in matches:
            # For each match, only mark the first atom
            # This avoids marking the entire match group with the same role
            # e.g., [OX2H0][CX3](=O) should only mark the oxygen atom, not the carbon
            if match:  # Ensure match is not empty
                atom_idx = match[0]
                if atom_roles[atom_idx] == 'other':
                    atom_roles[atom_idx] = role
                    if role in cvhd_roles:
                        logger.debug(f"  Atom {atom_idx} marked as {role}")
                else:
                    # Atom already marked, skip
                    if role in cvhd_roles:
                        logger.debug(f"  Atom {atom_idx} already marked as {atom_roles[atom_idx]}, skipping {role}")
    
    role_to_indices: Dict[str, List[int]] = {}
    atom_index_to_role: Dict[int, str] = {}
    assignments: List[TemplateRoleAssignment] = []
    
    for idx, role in enumerate(atom_roles):
        atom = mol.GetAtomWithIdx(idx)
        element = atom.GetSymbol()
        is_aromatic = atom.GetIsAromatic()
        
        role_to_indices.setdefault(role, []).append(idx)
        atom_index_to_role[idx] = role
        assignments.append(TemplateRoleAssignment(
            atom_index=idx,
            role=role,
            element=element,
            is_aromatic=is_aromatic
        ))
    
    other_count = len(role_to_indices.get('other', []))
    if other_count > 0:
        logger.warning(f"Template {template_name}: {other_count} atoms have role 'other'")
    
    return TemplateInfo(
        template_name=template_name,
        smiles=smiles,
        num_atoms=num_atoms,
        pdb_file="",  # Will be filled later by pdb_utils
        role_to_indices=role_to_indices,
        atom_index_to_role=atom_index_to_role,
        assignments=assignments
    )