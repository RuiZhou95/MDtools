"""
Single-molecule template PDB file generation and reading.
Contains robust 3D conformation generation logic, supporting ETKDG embedding and MMFF/UFF optimization.
"""
import logging
from pathlib import Path
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem

from models import TemplateInfo

logger = logging.getLogger(__name__)

def _embed_and_optimize(mol: Chem.Mol, use_etkdg: bool = True, optimize_mmff: bool = True, max_attempts: int = 3) -> bool:
    """
    Generate and optimize 3D conformation for a molecule.
    
    Args:
        mol: RDKit Mol object (should already contain hydrogens)
        use_etkdg: Whether to use ETKDG algorithm (otherwise use random distance geometry)
        optimize_mmff: Whether to perform force field optimization (if True, try MMFF first, fallback to UFF)
        max_attempts: Number of embedding attempts
    
    Returns:
        True if successful, False otherwise
    """
    if mol.GetNumAtoms() == 0:
        return False
    
    # Embed conformation
    success = False
    for attempt in range(max_attempts):
        seed = 42 + attempt
        if use_etkdg:
            # ETKDG embedding, returns 0 for success
            if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) == 0:
                success = True
                break
        else:
            # Simple distance geometry
            if AllChem.EmbedMolecule(mol, randomSeed=seed) == 0:
                success = True
                break
    if not success:
        logger.error("Failed to embed molecule after multiple attempts")
        return False
    
    # Force field optimization
    if optimize_mmff:
        try:
            # 先尝试 MMFF
            if AllChem.MMFFOptimizeMolecule(mol, maxIters=200) == 0:
                logger.info("MMFF optimization succeeded")
            else:
                # MMFF 失败，尝试 UFF
                logger.warning("MMFF optimization did not converge, trying UFF")
                if AllChem.UFFOptimizeMolecule(mol, maxIters=200) == 0:
                    logger.info("UFF optimization succeeded")
                else:
                    logger.warning("UFF optimization did not converge, using raw coordinates")
        except Exception as e:
            logger.warning(f"Force field optimization failed: {e}, using raw coordinates")
    else:
        # 即使不优化，也尝试简单的 UFF 几次迭代改善几何
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=50)
        except:
            pass
    
    return True

def smiles_to_pdb(
    smiles: str,
    output_pdb_path: Path,
    template_info: Optional[TemplateInfo] = None,
    residue_name: str = "UNK",
    add_hs: bool = True,
    use_etkdg: bool = True,
    optimize_mmff: bool = True
) -> bool:
    """
    将 SMILES 转换为单分子 PDB 文件。
    
    Args:
        smiles: SMILES 字符串
        output_pdb_path: 输出 PDB 文件路径
        template_info: 若提供，将更新其 pdb_file 字段
        residue_name: 残基名称（三字母）
        add_hs: 是否添加氢原子
        use_etkdg: 是否使用 ETKDG 生成 3D 构象
        optimize_mmff: 是否进行力场优化
    
    Returns:
        成功返回 True
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {smiles}")
            return False
        
        if add_hs:
            mol = Chem.AddHs(mol)
        
        # 生成 3D 构象
        if not _embed_and_optimize(mol, use_etkdg=use_etkdg, optimize_mmff=optimize_mmff):
            logger.error("Failed to generate 3D conformation for PDB export")
            return False
        
        # 设置残基信息
        for atom in mol.GetAtoms():
            atom.SetIntProp('residueNumber', 1)
            atom.SetProp('residueName', residue_name)
        
        # 写入 PDB
        with open(output_pdb_path, 'w') as f:
            writer = Chem.PDBWriter(f)
            writer.write(mol)
            writer.close()
        
        logger.info(f"PDB template saved: {output_pdb_path}")
        
        if template_info is not None:
            template_info.pdb_file = str(output_pdb_path)
        
        return True
    
    except Exception as e:
        logger.exception(f"Failed to generate PDB for SMILES {smiles}: {e}")
        return False

def read_pdb_as_mol(pdb_path: Path) -> Optional[Chem.Mol]:
    """从 PDB 文件读取分子"""
    try:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is None:
            logger.error(f"Failed to read PDB file: {pdb_path}")
            return None
        return mol
    except Exception as e:
        logger.exception(f"Error reading PDB {pdb_path}: {e}")
        return None