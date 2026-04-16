import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Enhanced Functional Group Capping Strategy Dictionary ---
CUSTOM_CAPPING_RULES = {
    # Capped Functional Groups (Cap with C)
    '[#0]-[O]-[#0]': (True, 2, 'COC'),  
    '[#0]-C(=O)-[#0]': (True, 2, 'C(C)=O'),
    '[#0]-[#16]-[#0]': (True, 2, 'CSC'),
    '[#0]-C(=O)[H]': (True, 1, 'C(C)=O'),
    '[#0]-C(=O)-[N]-[#0]': (True, 2, 'C(N)=O'),
    '[#0]-C(=O)-[O]-[#0]': (True, 2, 'C(OC)=O'),
    '[#0]-C(=O)[O;H1]': (True, 1, 'C(C)(O)=O'),
    
    # Uncapped Functional Groups
    '[#0]-[O;H1]': (False, 1, '[OH]'),
    '[#0]-[N;H2]': (False, 1, '[NH2]'),
    '[#0]-[N;H1]': (False, 1, '[NH]'),
    '[#0]-[N;H0]': (False, 1, 'N'),
    '[#0]-[S;H1]': (False, 1, '[SH]'),
    '[#0]-[F]': (False, 1, 'F'),
    '[#0]-[Cl]': (False, 1, 'Cl'),
    '[#0]-[Br]': (False, 1, 'Br'),
    '[#0]-[I]': (False, 1, 'I'),
    # Cyanide group - both carbon and nitrogen attachment points
    '[#0]-C#[N]': (False, 1, 'C#N'),
    '[#0]-N#C': (False, 1, 'C#N'),
    '[#0]-N(=O)=O': (False, 1, 'N(=O)=O'),
}

# --- Fragment Normalization Function ---
def normalize_fragment_smiles(smi):
    """
    Normalizes specific common fragment SMILES to a unified representation.
    """
    if not smi:
        return smi
    
    # Quick check for common fragments
    if smi in ['[OH]', 'C', 'C=C', 'C#C', 'C#N']:
        return smi
    
    # First, check if this is a defined capped functional group
    for smarts, (should_cap, num_points, target_smiles) in CUSTOM_CAPPING_RULES.items():
        if should_cap and target_smiles in smi:
            return smi
    
    # Try to parse SMILES, suppressing possible warnings
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            # Try sanitize but skip kekulization
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except:
        mol = None
    
    if mol is None:
        # If parsing fails, return original SMILES
        return smi
    
    # Special handling for cyanide group
    # Check for C#N pattern
    cyanide_pattern = Chem.MolFromSmarts('C#N')
    if cyanide_pattern and mol.HasSubstructMatch(cyanide_pattern):
        # Count atoms
        atoms = mol.GetAtoms()
        if len(atoms) == 2:
            # Check if it's a simple C#N fragment
            bond = mol.GetBondBetweenAtoms(0, 1)
            if bond and bond.GetBondType() == Chem.BondType.TRIPLE:
                # Check for connection points
                dummy_atoms = [atom for atom in atoms if atom.GetAtomicNum() == 0]
                if len(dummy_atoms) == 1:
                    return 'C#N'
    
    # Simplification: all alkyl carbons return as 'C'
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    if len(carbon_atoms) == 1:
        return 'C'
    
    # Check if it's a simple alkyl fragment
    non_carbon_hydrogen = [atom for atom in mol.GetAtoms() 
                          if atom.GetAtomicNum() not in [1, 6, 0]]
    if len(non_carbon_hydrogen) == 0 and len(carbon_atoms) <= 4:
        # Check for double or triple bonds
        has_double = any(bond.GetBondType() == Chem.BondType.DOUBLE 
                        for bond in mol.GetBonds())
        has_triple = any(bond.GetBondType() == Chem.BondType.TRIPLE 
                        for bond in mol.GetBonds())
        
        if has_double and not has_triple and len(carbon_atoms) == 2:
            return 'C=C'
        elif has_triple and not has_double and len(carbon_atoms) == 2:
            return 'C#C'
        
        return 'C'
    
    return smi

def should_fragment_be_capped(frag_mol):
    """
    Determine if a fragment should be capped with carbon.
    Returns: (should_cap, num_attachment_points, target_smiles)
    """
    if frag_mol is None:
        return (False, 0, None)
    
    if frag_mol.GetRingInfo().NumRings() > 0:
        return (False, 0, None)
    
    # Check all rules
    for smarts, (should_cap, num_points, target_smiles) in CUSTOM_CAPPING_RULES.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and frag_mol.HasSubstructMatch(pattern):
                return (should_cap, num_points, target_smiles)
        except Exception:
            continue
    
    return (False, 0, None)

def post_process_fragment(frag_mol):
    """
    Process fragments after splitting: apply capping and normalization.
    """
    if frag_mol is None:
        return None
    
    # Special handling for cyanide fragments
    # Check if it's a cyanide fragment
    cyanide_pattern = Chem.MolFromSmarts('C#N')
    if cyanide_pattern and frag_mol.HasSubstructMatch(cyanide_pattern):
        # Count atoms
        atoms = frag_mol.GetAtoms()
        if len(atoms) == 2 or len(atoms) == 3:  # C#N or C#N with dummy atom
            # Check for triple bond between C and N
            for bond in frag_mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.TRIPLE:
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    atomic_nums = sorted([a1.GetAtomicNum(), a2.GetAtomicNum()])
                    if atomic_nums == [6, 7]:  # C#N bond
                        # Check for connection points
                        dummy_atoms = [atom for atom in atoms if atom.GetAtomicNum() == 0]
                        if len(dummy_atoms) <= 1:
                            return 'C#N'
    
    # Get capping decision
    should_cap, num_points, target_smiles = should_fragment_be_capped(frag_mol)
    
    if not should_cap:
        # If target_smiles is provided, use it
        if target_smiles:
            return target_smiles
        
        # No C-capping needed
        rw_mol = Chem.RWMol(frag_mol)
        dummy_indices = sorted([at.GetIdx() for at in rw_mol.GetAtoms() 
                              if at.GetAtomicNum() == 0], reverse=True)
        
        for idx in dummy_indices:
            rw_mol.RemoveAtom(idx)
        
        if rw_mol.GetNumAtoms() == 0:
            return None

        try:
            mol = rw_mol.GetMol()
            # Remove stereochemistry tags
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            
            # Try sanitize but skip kekulization
            Chem.SanitizeMol(mol, 
                             sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            
            # Return canonical SMILES
            result = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            
            # Special handling for hydroxyl
            if result == 'O':
                return '[OH]'
            
            return result
        except Exception:
            # If sanitize fails, try to return without sanitization
            try:
                return Chem.MolToSmiles(rw_mol.GetMol(), canonical=True)
            except:
                return None

    else:
        # C-capping required - use target smiles if available
        if target_smiles:
            return target_smiles
        
        # Fallback: manual capping
        rw_mol = Chem.RWMol(frag_mol)
        dummy_indices = [at.GetIdx() for at in rw_mol.GetAtoms() 
                        if at.GetAtomicNum() == 0]
        
        # Replace '*' with Carbon
        for idx in dummy_indices:
            atom = rw_mol.GetAtomWithIdx(idx)
            atom.SetAtomicNum(6) 
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)

        try:
            Chem.SanitizeMol(rw_mol)
            capped_mol = rw_mol.GetMol()
            return Chem.MolToSmiles(capped_mol, canonical=True)
        except Exception:
            return None

# --- Atomic Element Extraction ---
def extract_atomic_elements(smiles):
    """
    Extract all unique atomic elements from a molecule.
    Charged atoms are treated as neutral elements.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return set()
    
    elements = set()
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num > 0:  # Exclude dummy atoms
            # Treat charged atoms as neutral
            element_symbol = atom.GetSymbol()
            elements.add(element_symbol)
    
    return elements

# --- Bond Protection Functions ---
def get_bond_connectivity_info(mol):
    """
    Get information about which bonds connect to rings.
    Returns a dictionary with bond indices as keys and a boolean value
    indicating if the bond connects to a ring atom.
    """
    bond_info = {}
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        # Check if either atom is in a ring
        connects_to_ring = begin_atom.IsInRing() or end_atom.IsInRing()
        bond_info[bond.GetIdx()] = connects_to_ring
    return bond_info

def get_protected_bonds_strategy_A(mol):
    """
    Strategy A: Protect all functional group bonds regardless of ring connection.
    This is the original strategy where functional groups are kept intact.
    For cyanide connected to ring: protect the bond between cyanide and ring.
    """
    protected_bonds = set()
    
    # 1. Protect all bonds in rings
    for bond in mol.GetBonds():
        if bond.IsInRing():
            protected_bonds.add(bond.GetIdx())
    
    # 2. Protect all non-single bonds (double, triple bonds)
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            protected_bonds.add(bond.GetIdx())
    
    # 3. Protect bonds within functional groups (all functional groups)
    functional_group_smarts = [
        # Amide - protect entire CON group
        "[CX3](=[OX1])-[NX3]",
        # Ester - protect entire COO group
        "[CX3](=[OX1])-[OX2;!R]",
        # Carboxylic acid
        "[CX3](=[OX1])-[OX2H1]",
        # Ketone - protect C=O and adjacent bonds
        "[#6]-[#6](=[#8])-[#6]",
        # Ether - protect C-O-C bonds
        "[#6]-[#8;!R]-[#6]",
        # Thioether
        "[#6]-[#16;!R]-[#6]",
        # Sulfonamide
        "[SX4](=[OX1])(=[OX1])-[NX3]",
        # Cyanide group - protect C#N triple bond
        "[#6]#[#7]",
    ]
    
    for smarts in functional_group_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # Protect all bonds between atoms in this match
                for i in range(len(match)):
                    for j in range(i+1, len(match)):
                        bond = mol.GetBondBetweenAtoms(match[i], match[j])
                        if bond:
                            protected_bonds.add(bond.GetIdx())
    
    # 4. For Strategy A, also protect bonds between functional groups and rings
    # Find bonds connecting functional group atoms to ring atoms
    for bond in mol.GetBonds():
        if bond.GetIdx() not in protected_bonds:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            
            # Check if this is a bond between a ring atom and a functional group atom
            ring_atom = a1 if a1.IsInRing() else (a2 if a2.IsInRing() else None)
            non_ring_atom = a2 if a1.IsInRing() else (a1 if a2.IsInRing() else None)
            
            if ring_atom and non_ring_atom:
                # Check if non-ring atom is part of a functional group
                for smarts in functional_group_smarts:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern:
                        # Check if non_ring_atom is part of this functional group
                        matches = mol.GetSubstructMatches(pattern)
                        for match in matches:
                            if non_ring_atom.GetIdx() in match:
                                # This bond connects a ring to a functional group
                                # Protect it in Strategy A
                                protected_bonds.add(bond.GetIdx())
                                break
    
    return protected_bonds

def get_protected_bonds_strategy_B(mol):
    """
    Strategy B: Protect functional group bonds only if not connected to rings.
    If a functional group is connected to a ring, its single bonds are not protected.
    Note: Triple bonds (like in cyanide) should always be protected.
    For cyanide connected to ring: allow cutting the bond between cyanide and ring.
    """
    protected_bonds = set()
    
    # 1. Protect all bonds in rings
    for bond in mol.GetBonds():
        if bond.IsInRing():
            protected_bonds.add(bond.GetIdx())
    
    # 2. Protect all non-single bonds (double, triple bonds)
    # This includes cyanide C#N bonds
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            protected_bonds.add(bond.GetIdx())
    
    # 3. Get bond connectivity information
    bond_info = get_bond_connectivity_info(mol)
    
    # 4. Protect bonds within functional groups, but only if the functional group
    #    is not connected to a ring. However, triple bonds are already protected above.
    functional_group_smarts = [
        # Amide - protect entire CON group
        "[CX3](=[OX1])-[NX3]",
        # Ester - protect entire COO group
        "[CX3](=[OX1])-[OX2;!R]",
        # Carboxylic acid
        "[CX3](=[OX1])-[OX2H1]",
        # Ketone - protect C=O and adjacent bonds
        "[#6]-[#6](=[#8])-[#6]",
        # Ether - protect C-O-C bonds
        "[#6]-[#8;!R]-[#6]",
        # Thioether
        "[#6]-[#16;!R]-[#6]",
        # Sulfonamide
        "[SX4](=[OX1])(=[OX1])-[NX3]",
        # Cyanide is already protected by triple bond protection
    ]
    
    for smarts in functional_group_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # Check if this functional group is connected to a ring
                group_connected_to_ring = False
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.IsInRing():
                        group_connected_to_ring = True
                        break
                
                # Only protect bonds if the functional group is not connected to a ring
                if not group_connected_to_ring:
                    for i in range(len(match)):
                        for j in range(i+1, len(match)):
                            bond = mol.GetBondBetweenAtoms(match[i], match[j])
                            if bond and bond.GetBondType() == Chem.BondType.SINGLE:
                                protected_bonds.add(bond.GetIdx())
    
    # 5. Special rule for Strategy B: always allow cutting bonds connecting cyanide to rings
    cyanide_ring_pattern = Chem.MolFromSmarts("[c,C;R]-C#[N]")
    if cyanide_ring_pattern:
        matches = mol.GetSubstructMatches(cyanide_ring_pattern)
        for match in matches:
            # Get bond between ring and cyanide carbon
            ring_atom_idx = match[0]
            cyanide_carbon_idx = match[1]
            bond = mol.GetBondBetweenAtoms(ring_atom_idx, cyanide_carbon_idx)
            if bond and bond.GetIdx() in protected_bonds:
                # Remove this bond from protection set, allow cutting
                protected_bonds.remove(bond.GetIdx())
    
    return protected_bonds

# --- Molecule Disassembly Functions ---
def disassemble_with_strategy(smiles, strategy_func):
    """
    Disassemble a molecule using a specific bond protection strategy.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return []
    
    # Get protected bonds using the specified strategy
    protected = strategy_func(mol)
    
    # Identify bonds to cut: all unprotected single bonds
    bonds_to_cut = []
    for bond in mol.GetBonds():
        if (bond.GetIdx() not in protected and 
            bond.GetBondType() == Chem.BondType.SINGLE):
            bonds_to_cut.append(bond.GetIdx())
    
    # If no bonds to cut, return the whole molecule
    if not bonds_to_cut:
        return [Chem.MolToSmiles(mol, canonical=True)]
    
    # Fragment the molecule
    fragmented = Chem.FragmentOnBonds(mol, bonds_to_cut, addDummies=True)
    frags_smiles = Chem.MolToSmiles(fragmented).split('.')
    
    # Process each fragment
    final_fragments = set()
    
    for fsmi in frags_smiles:
        # Quick processing for common fragments
        if fsmi == 'O':
            fsmi = '[OH]'
        
        # Try to parse fragment, suppress possible warnings
        fmol = None
        try:
            fmol = Chem.MolFromSmiles(fsmi, sanitize=False)
            if fmol is not None:
                # Try sanitize but skip kekulization
                Chem.SanitizeMol(fmol, 
                                 sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except:
            pass
        
        if fmol is None:
            # If parsing fails, skip this fragment
            continue
        
        processed_smi = post_process_fragment(fmol)
        
        if processed_smi:
            # Apply normalization
            normalized_smi = normalize_fragment_smiles(processed_smi)
            if normalized_smi:
                final_fragments.add(normalized_smi)
    
    return list(final_fragments)

def disassemble_molecule_all_strategies(smiles):
    """
    Disassemble a molecule using all strategies and return all unique fragments.
    """
    # Get atomic elements
    atomic_elements = extract_atomic_elements(smiles)
    
    # Get fragments from Strategy A (functional groups protected even when connected to rings)
    fragments_A = disassemble_with_strategy(smiles, get_protected_bonds_strategy_A)
    
    # Get fragments from Strategy B (functional groups not protected when connected to rings)
    fragments_B = disassemble_with_strategy(smiles, get_protected_bonds_strategy_B)
    
    # Combine all fragments
    all_fragments = set()
    
    # Add atomic elements
    for element in atomic_elements:
        all_fragments.add(element)
    
    # Add fragments from both strategies
    for frag in fragments_A:
        all_fragments.add(frag)
    
    for frag in fragments_B:
        all_fragments.add(frag)
    
    return list(all_fragments)

# --------------------------------------------------------
# Test Cases
# --------------------------------------------------------
if __name__ == "__main__":
    test_mols = {
        "Ether_Capped": "COC(C)C",                 
        "Hydroxyl_Uncapped": "CC(C)O",             
        "Ketone_Capped": "CC(=O)CC",               
        "Amino_Uncapped": "CCN(C)C",               
        "Phenol_Uncapped": "Oc1ccccc1",            
        "Amide_Capped": "CC(=O)Nc1ccccc1",         
        "Alkane_Test": "CCCC",  # Butane -> should give CH3 and CH2
        "Alkene_Test": "C=CC",  # Propene -> should give C=C and CH3
        "Alkyne_Test": "C#CC",  # Propyne -> should give C#C and CH3
        "Complex_Test": "CC(=O)Oc1ccccc1",  # Aspirin-like
        "ZINC1": "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
        "ZINC2": "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
        "ZINC3": "N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1",
        "ZINC4": "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1",
        "ZINC5": "N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#N)C12CCCCC2",
        "ZINC6": "CC[NH+](CC)[C@](C)(CC)[C@H](O)c1cscc1Br",
        "ZINC7": "COc1ccc(C(=O)N(C)[C@@H](C)C/C(N)=N/O)cc1O",
        "ZINC8": "O=C(Nc1nc[nH]n1)c1cccnc1Nc1cccc(F)c1",
        "ZINC9": "Cc1c(/C=N/c2cc(Br)ccn2)c(O)n2c(nc3ccccc32)c1C#N",
        "ZINC10": "C[C@@H]1CN(C(=O)c2cc(Br)cn2C)CC[C@H]1[NH3+]"
    }

    print("--- Running Tests with All Disassembly Strategies ---")
    for name, smi in test_mols.items():
        print(f"--- Test: {name} ({smi}) ---")
        
        # Get atomic elements
        atomic_elements = extract_atomic_elements(smi)
        print(f"  Atomic Elements: {sorted(atomic_elements)}")
        
        # Get fragments from Strategy A (functional groups protected even when connected to rings)
        fragments_A = disassemble_with_strategy(smi, get_protected_bonds_strategy_A)
        print(f"  Strategy A - Functional Groups Protected ({len(fragments_A)} fragments):")
        for f in sorted(fragments_A):
            print(f"    -> {f}")
        
        # Get fragments from Strategy B (functional groups not protected when connected to rings)
        fragments_B = disassemble_with_strategy(smi, get_protected_bonds_strategy_B)
        print(f"  Strategy B - Functional Groups Split When Connected to Rings ({len(fragments_B)} fragments):")
        for f in sorted(fragments_B):
            print(f"    -> {f}")
        
        # Get all fragments combined
        all_fragments = disassemble_molecule_all_strategies(smi)
        print(f"  All Unique Fragments ({len(all_fragments)}):")
        for f in sorted(all_fragments):
            print(f"    -> {f}")
        
        print()