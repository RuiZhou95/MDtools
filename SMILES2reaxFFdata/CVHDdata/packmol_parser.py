"""
Parse Packmol output packed.pdb file, reconstruct molecule copies based on atom order and template information.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import PackedAtomRecord, PackedMoleculeRecord, PackedSystem, TemplateInfo

logger = logging.getLogger(__name__)

def parse_packed_pdb(
    pdb_file: Path,
    template_infos: Dict[str, TemplateInfo],
    box_size: Optional[List[float]] = None,
    template_names_order: Optional[List[str]] = None,
    counts: Optional[List[int]] = None
) -> Optional[PackedSystem]:
    """
    Parse Packmol output PDB file.
    
    Strategy:
    1. First try grouping by residue number (standard PDB behavior).
    2. If that fails, split molecules based on given template_names_order and counts, following atom appearance order.
    
    Args:
        pdb_file: packed.pdb file path
        template_infos: Template name -> TemplateInfo mapping
        box_size: Optional box size
        template_names_order: List of molecule type order (corresponding to counts)
        counts: List of counts for each molecule type
    
    Returns:
        PackedSystem object, None if failed
    """
    try:
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        # Parse all atom lines, extract information
        atoms_data = []  # Each element: (global_id, element, x, y, z, resname, resid, atom_name)
        global_id_counter = 1
        
        for line in lines:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            
            # 原子序号（PDB 列 7-11）
            try:
                serial = int(line[6:11].strip())
                global_id = serial
            except ValueError:
                global_id = global_id_counter
            
            # 原子名（列 13-16）
            atom_name = line[12:16].strip()
            # 残基名（列 17-20）
            residue_name = line[17:20].strip()
            # 残基序号（列 22-26）
            try:
                residue_seq = int(line[22:26].strip())
            except ValueError:
                residue_seq = 0
            
            # 坐标（列 30-54）
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                logger.warning(f"Invalid coordinates in line: {line}")
                continue
            
            # 元素（列 76-78）
            element = line[76:78].strip()
            if not element:
                element = atom_name[0].upper()
            else:
                element = element.upper()
            
            atoms_data.append((global_id, element, x, y, z, residue_name, residue_seq, atom_name))
            global_id_counter += 1
        
        if not atoms_data:
            logger.error("No atoms found in PDB file")
            return None
        
        # 方法1：按残基序号分组
        from collections import defaultdict
        resid_groups = defaultdict(list)
        for atom in atoms_data:
            resid = atom[6]
            resid_groups[resid].append(atom)
        
        # 检查是否所有组的大小都能匹配某个模板
        def find_template_by_atom_count(num_atoms):
            for tinfo in template_infos.values():
                if tinfo.num_atoms == num_atoms:
                    return tinfo
            return None
        
        all_groups_match = True
        for resid, group in resid_groups.items():
            if find_template_by_atom_count(len(group)) is None:
                all_groups_match = False
                break
        
        if all_groups_match:
            logger.info("Using residue number grouping for molecule separation")
            all_atoms = []
            all_molecules = []
            for resid, group in resid_groups.items():
                num_atoms = len(group)
                tinfo = find_template_by_atom_count(num_atoms)
                if tinfo is None:
                    continue
                mol_global_ids = []
                for idx, (gid, elem, x, y, z, resname, resid_val, atname) in enumerate(group):
                    role = tinfo.atom_index_to_role.get(idx, "other")
                    record = PackedAtomRecord(
                        global_id=gid,
                        mol_id=resid,
                        template_name=tinfo.template_name,
                        template_atom_index=idx,
                        role=role,
                        element=elem,
                        x=x, y=y, z=z,
                        charge=0.0
                    )
                    all_atoms.append(record)
                    mol_global_ids.append(gid)
                mol_record = PackedMoleculeRecord(
                    mol_id=resid,
                    template_name=tinfo.template_name,
                    atom_indices=mol_global_ids
                )
                all_molecules.append(mol_record)
        else:
            # 方法2：按顺序切分（需要 template_names_order 和 counts）
            if template_names_order is None or counts is None:
                logger.error("Cannot separate molecules: residue grouping failed and no order/counts provided")
                return None
            
            # 构建分子序列：[tname1]*count1 + [tname2]*count2 + ...
            mol_sequence = []
            for tname, cnt in zip(template_names_order, counts):
                mol_sequence.extend([tname] * cnt)
            
            # 按顺序切分原子
            all_atoms = []
            all_molecules = []
            atom_idx = 0
            mol_id = 1
            for tname in mol_sequence:
                tinfo = template_infos.get(tname)
                if tinfo is None:
                    logger.error(f"Template {tname} not found")
                    return None
                num_atoms = tinfo.num_atoms
                if atom_idx + num_atoms > len(atoms_data):
                    logger.error(f"Not enough atoms for molecule {mol_id} (need {num_atoms}, have {len(atoms_data)-atom_idx})")
                    return None
                # 提取该分子的原子
                mol_global_ids = []
                for offset in range(num_atoms):
                    gid, elem, x, y, z, resname, resid, atname = atoms_data[atom_idx + offset]
                    role = tinfo.atom_index_to_role.get(offset, "other")
                    record = PackedAtomRecord(
                        global_id=gid,
                        mol_id=mol_id,
                        template_name=tname,
                        template_atom_index=offset,
                        role=role,
                        element=elem,
                        x=x, y=y, z=z,
                        charge=0.0
                    )
                    all_atoms.append(record)
                    mol_global_ids.append(gid)
                mol_record = PackedMoleculeRecord(
                    mol_id=mol_id,
                    template_name=tname,
                    atom_indices=mol_global_ids
                )
                all_molecules.append(mol_record)
                atom_idx += num_atoms
                mol_id += 1
            
            if atom_idx != len(atoms_data):
                logger.warning(f"Not all atoms used: {atom_idx} of {len(atoms_data)}")
        
        # 按全局 ID 排序
        all_atoms.sort(key=lambda a: a.global_id)
        
        # 盒子尺寸
        box = box_size if box_size else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for line in lines:
            if line.startswith("CRYST1"):
                parts = line.split()
                if len(parts) >= 4:
                    a = float(parts[1])
                    b = float(parts[2])
                    c = float(parts[3])
                    box = [0.0, a, 0.0, b, 0.0, c]
                break
        
        system = PackedSystem(
            atoms=all_atoms,
            molecules=all_molecules,
            box_size=box,
            template_infos=template_infos
        )
        logger.info(f"Parsed packed PDB: {len(system.atoms)} atoms, {len(system.molecules)} molecules")
        return system
    
    except Exception as e:
        logger.exception(f"Failed to parse packed PDB: {e}")
        return None