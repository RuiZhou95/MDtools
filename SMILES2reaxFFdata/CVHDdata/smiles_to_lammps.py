"""
Extended SmilesToLammps class with new PDB template, role recognition, and index export features.
Keeps the original XYZ-based workflow as a fallback, defaults to the new PDB workflow.
"""
import os
import subprocess
import shutil
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors

# New module imports
from models import TemplateInfo, PackedSystem
from chemistry_roles import classify_template_atoms
from pdb_utils import smiles_to_pdb
from packmol_parser import parse_packed_pdb
from group_export import write_atoms_ndx, write_lammps_groups
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmilesToLammps:
    def __init__(self, workspace_name="lammps_build", use_pdb_mode=True):
        self.workshop_dir = None
        self.workspace_name = workspace_name
        self.use_pdb_mode = use_pdb_mode   # New: whether to use PDB mode
        
        # Define constants
        self.AVOGADRO_CONSTANT = 6.02214076e23  # mol^-1
        self.M3_TO_ANGSTROM3 = 1e30  # 1 m^3 = 1e30 Å^3
        
        # Store template information (template name -> TemplateInfo)
        self.template_infos: Dict[str, TemplateInfo] = {}
        # Store the final packed system
        self.packed_system: Optional[PackedSystem] = None
        
    def setup_workshop(self):
        """Create working directory in the tmp folder under current path"""
        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            print(f"Created tmp directory: {tmp_dir}")
        
        self.workshop_dir = os.path.join(tmp_dir, self.workspace_name)
        if os.path.exists(self.workshop_dir):
            shutil.rmtree(self.workshop_dir)
        os.makedirs(self.workshop_dir)
        
        print(f"Working directory: {self.workshop_dir}")
        return self.workshop_dir
    
    def calculate_box_size(self, smiles_list, counts, target_density_kg_per_m3):
        """Keep unchanged, same as original code"""
        if target_density_kg_per_m3 is None:
            raise ValueError("Target density target_density_kg_per_m3 must be specified")
        
        if target_density_kg_per_m3 <= 0:
            raise ValueError("Target density must be greater than 0")
        
        print(f"Using target density: {target_density_kg_per_m3:.2f} kg/m³")        
        
        molecular_weights = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Unable to parse SMILES: {smiles}")
            mol = Chem.AddHs(mol)
            molecular_weight = Descriptors.MolWt(mol)
            molecular_weights.append(molecular_weight)
            print(f"Molecular weight of {smiles}: {molecular_weight:.2f} g/mol")
        
        total_mass_g = 0.0
        for count, molecular_weight in zip(counts, molecular_weights):
            single_molecule_mass_g = molecular_weight / self.AVOGADRO_CONSTANT
            total_mass_g += count * single_molecule_mass_g
        
        print(f"Total number of molecules: {sum(counts)}")
        print(f"Total system mass: {total_mass_g:.10f} g")
        
        total_mass_kg = total_mass_g / 1000.0
        print(f"Total system mass: {total_mass_kg:.12f} kg")
        
        volume_m3 = total_mass_kg / target_density_kg_per_m3
        print(f"Calculated box volume: {volume_m3:.12e} m³")
        
        side_length_m = volume_m3 ** (1/3)
        side_length_angstrom = side_length_m * 1e10
        volume_angstrom3 = volume_m3 * 1e30
        
        print(f"Box volume: {volume_angstrom3:.2f} Å³")
        print(f"Calculated cube box side length: {side_length_angstrom:.2f} Å")
        
        calculated_density_kg_per_m3 = total_mass_kg / (volume_angstrom3 / 1e30)
        print(f"Verification - calculated density: {calculated_density_kg_per_m3:.2f} kg/m³")
        
        calculated_volume_angstrom3 = side_length_angstrom ** 3
        print(f"Verification - volume calculated from side length: {calculated_volume_angstrom3:.2f} Å³")
        
        box_size = [0.0, side_length_angstrom, 0.0, side_length_angstrom, 0.0, side_length_angstrom]
        return box_size, side_length_angstrom
    
    def cleanup(self, keep_files=True):
        """Clean up working directory"""
        if not keep_files and self.workshop_dir and os.path.exists(self.workshop_dir):
            shutil.rmtree(self.workshop_dir)
            print("Cleaned up working directory")
        elif keep_files:
            print(f"Intermediate files saved at: {self.workshop_dir}")
    
    # 保留原有的 smiles_to_xyz 方法
    def smiles_to_xyz(self, smiles, output_filename, charge_method='mmff94'):
        """Convert SMILES to XYZ using Open Babel (保持不变)"""
        try:
            output_path = os.path.join(self.workshop_dir, output_filename)
            cmd = f'obabel -:"{smiles}" -o xyz -O {output_path} --gen3D --partialcharge {charge_method}'
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                print(f"Successfully generated XYZ file: {output_path}")
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return True
                else:
                    print(f"XYZ file is empty or does not exist: {output_path}")
                    return False
            else:
                print(f"Open Babel error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("Open Babel execution timeout")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Open Babel execution failed: {e}")
            return False
    
    # 新增：PDB 模式下的模板生成
    def _prepare_templates_pdb(self, smiles_list, template_names):
        """
        使用 RDKit 生成单分子 PDB 模板，并执行原子角色分类。
        
        Args:
            smiles_list: SMILES 列表
            template_names: 对应的模板名称列表（与 smiles_list 顺序一致）
        
        Returns:
            bool: 成功返回 True
        """
        self.template_infos.clear()
        for i, (smiles, tname) in enumerate(zip(smiles_list, template_names)):
            # 分类角色
            tinfo = classify_template_atoms(smiles, tname, add_hs=True)
            # 生成 PDB 文件
            pdb_filename = f"template_{tname}.pdb"
            pdb_path = Path(self.workshop_dir) / pdb_filename
            success = smiles_to_pdb(
                smiles,
                pdb_path,
                template_info=tinfo,
                residue_name=config.DEFAULT_RESIDUE_NAMES.get(tname, tname[:3]),
                add_hs=True,
                use_etkdg=True,
                optimize_mmff=True
            )
            if not success:
                logger.error(f"Failed to generate PDB for template {tname}")
                return False
            self.template_infos[tname] = tinfo
            logger.info(f"Template {tname}: {tinfo.num_atoms} atoms, roles: {list(tinfo.role_to_indices.keys())}")
        return True
    
    # 新增：创建 Packmol 输入文件（PDB 模式）
    def create_packmol_input_pdb(self, template_names, counts, box_size, output_file="packed.pdb"):
        """
        创建使用 PDB 格式的 Packmol 输入文件。
        
        Args:
            template_names: 模板名称列表（顺序与 counts 对应）
            counts: 每种分子的数量
            box_size: 盒子边界 [xlo, xhi, ylo, yhi, zlo, zhi]
            output_file: Packmol 输出文件名
        
        Returns:
            输入文件名（相对于工作区）
        """
        input_filename = "packmol.inp"
        input_path = Path(self.workshop_dir) / input_filename
        
        # 构建 structure 块
        structures = []
        for tname, count in zip(template_names, counts):
            pdb_file = self.template_infos[tname].pdb_file
            # 使用相对路径（工作区内）
            pdb_rel = Path(pdb_file).name
            struct_block = f"""structure {pdb_rel}
  number {count}
  inside box {box_size[0]} {box_size[2]} {box_size[4]} {box_size[1]} {box_size[3]} {box_size[5]}
end structure
"""
            structures.append(struct_block)
        
        packmol_content = f"""tolerance {config.PACKMOL_TOLERANCE}
filetype pdb
output {output_file}

{''.join(structures)}
"""
        with open(input_path, 'w') as f:
            f.write(packmol_content)
        
        logger.info(f"Created Packmol input (PDB mode): {input_path}")
        return input_filename
    
    # 保留原有的 create_packmol_input 用于 XYZ 模式（兼容）
    def create_packmol_input(self, xyz_files, counts, box_size, output_file="packed.xyz"):
        """原始 XYZ 模式的 Packmol 输入创建（保持不变）"""
        xyz_files_rel = xyz_files
        output_file_rel = output_file
        
        packmol_input = f"""tolerance 6.0
filetype xyz
output {output_file_rel}

"""
        for i, (xyz_file, count) in enumerate(zip(xyz_files_rel, counts)):
            packmol_input += f"""structure {xyz_file}
  number {count}
  inside box {box_size[0]} {box_size[2]} {box_size[4]} {box_size[1]} {box_size[3]} {box_size[5]}
end structure

"""
        input_filename = "packmol.inp"
        input_path = os.path.join(self.workshop_dir, input_filename)
        with open(input_path, 'w') as f:
            f.write(packmol_input)
        
        print(f"Created Packmol input file: {input_path}")
        return input_filename
    
    def run_packmol(self, input_file):
        """运行 Packmol，与原始代码基本相同，但增加输出文件检查"""
        try:
            input_path = os.path.join(self.workshop_dir, input_file)
            if not os.path.exists(input_path):
                print(f"Packmol input file does not exist: {input_path}")
                return False
            
            # 检查依赖的模板文件是否存在（如果是 PDB 模式，检查 PDB 文件）
            with open(input_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('structure'):
                        struct_file = line.split()[1]
                        struct_path = os.path.join(self.workshop_dir, struct_file)
                        if not os.path.exists(struct_path):
                            print(f"Structure file does not exist: {struct_path}")
                            return False
            
            original_dir = os.getcwd()
            os.chdir(self.workshop_dir)
            cmd = f"packmol < {input_file}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=config.PACKMOL_TIMEOUT)
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("Packmol execution successful")
                # 根据输入文件中的 output 指令确定输出文件名
                output_file = None
                for line in content.split('\n'):
                    if line.startswith('output'):
                        output_file = line.split()[1]
                        break
                if output_file:
                    output_path = os.path.join(self.workshop_dir, output_file)
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"Successfully generated packed file: {output_path}")
                        return True
                print("Packmol output file not found or empty")
                return False
            else:
                print(f"Packmol error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("Packmol execution timeout")
            return False
        except subprocess.CalledProcessError as e:
            print(f"Packmol execution failed, return code: {e.returncode}")
            print(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            print(f"Packmol execution exception: {e}")
            return False
    
    # 新增：基于 PDB 解析结果生成 LAMMPS data 文件（正确设置 mol_id 和 atom_type）
    def _pdb_to_lammps_data(self, packed_system: PackedSystem, output_file: str) -> bool:
        """
        根据 PackedSystem 生成 LAMMPS data 文件，每个分子副本具有唯一的 mol_id。
        
        Args:
            packed_system: 解析后的体系
            output_file: 输出文件名
        
        Returns:
            成功返回 True
        """
        try:
            atoms = packed_system.atoms
            box = packed_system.box_size
            
            # 收集原子类型（基于元素）
            atom_types_set = set()
            for atom in atoms:
                # 根据元素分配原子类型
                if atom.element == "C":
                    atype = 1
                elif atom.element == "H":
                    atype = 2
                elif atom.element == "O":
                    atype = 3
                elif atom.element == "N":
                    atype = 4
                else:
                    atype = config.DEFAULT_ROLE_TO_ATOM_TYPE.get(atom.role, 1)
                atom_types_set.add(atype)
            num_atom_types = len(atom_types_set)
            
            # 写入文件
            output_path = output_file if os.path.isabs(output_file) else os.path.join(self.workshop_dir, output_file)
            with open(output_path, 'w') as f:
                f.write("# LAMMPS data file generated by SmilesToLammps (PDB mode)\n")
                f.write("\n")
                f.write(f"{len(atoms)} atoms\n")
                f.write(f"{num_atom_types} atom types\n")
                f.write(f"{box[0]:.6f} {box[1]:.6f} xlo xhi\n")
                f.write(f"{box[2]:.6f} {box[3]:.6f} ylo yhi\n")
                f.write(f"{box[4]:.6f} {box[5]:.6f} zlo zhi\n")
                f.write("\n")
                
                # Masses
                f.write("Masses\n\n")
                # 根据原子类型分配质量
                type_to_mass = {1: 12.01, 2: 1.008, 3: 16.00, 4: 14.01}
                type_to_element = {1: "C", 2: "H", 3: "O", 4: "N"}
                for atype in sorted(atom_types_set):
                    mass = type_to_mass.get(atype, 12.01)
                    elem = type_to_element.get(atype, "C")
                    f.write(f"{atype} {mass:.3f}  # {elem}\n")
                f.write("\n")
                
                # Atoms section (full format: atom-ID molecule-ID atom-type charge x y z)
                f.write("Atoms # full\n\n")
                for atom in atoms:
                    # 根据元素分配原子类型，而不是依赖角色（因为角色识别可能失败）
                    if atom.element == "C":
                        atype = 1
                    elif atom.element == "H":
                        atype = 2
                    elif atom.element == "O":
                        atype = 3
                    elif atom.element == "N":
                        atype = 4
                    else:
                        atype = config.DEFAULT_ROLE_TO_ATOM_TYPE.get(atom.role, 1)
                    f.write(f"{atom.global_id} {atom.mol_id} {atype} {atom.charge:.6f} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n")
            
            logger.info(f"Generated LAMMPS data file: {output_path}")
            return True
        except Exception as e:
            logger.exception(f"Failed to generate LAMMPS data: {e}")
            return False
    
    # 保留原始 xyz_to_lammps_data（修复 mol_id 问题，但旧模式无法区分分子，仍需改进）
    # 为了不破坏原有功能，我们保留但添加警告
    def xyz_to_lammps_data(self, xyz_file, box_size, output_file="lammps.data"):
        """
        原始 XYZ 转换（存在 mol_id 均为 1 的问题，保留仅为兼容性）
        建议使用 PDB 模式以获得正确的分子 ID。
        """
        if not self.use_pdb_mode:
            logger.warning("Using legacy XYZ mode: mol_id will be 1 for all atoms. Consider switching to PDB mode.")
        # 原始代码（保持不变）
        try:
            xyz_path = os.path.join(self.workshop_dir, xyz_file)
            with open(xyz_path, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                print("XYZ file format error: insufficient lines")
                return False
            try:
                num_atoms = int(lines[0].strip())
            except ValueError:
                print("XYZ file format error: first line is not atom count")
                return False
            atom_lines = lines[2:]
            atoms = []
            atom_types = set()
            mol_id = 1
            atom_counter = 0
            for i, line in enumerate(atom_lines):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        element = parts[0]
                        x, y, z = map(float, parts[1:4])
                        if element.upper() == 'C':
                            atom_type = 1
                        elif element.upper() == 'H':
                            atom_type = 2  
                        elif element.upper() == 'O':
                            atom_type = 3
                        elif element.upper() == 'N':
                            atom_type = 4
                        else:
                            atom_type = 1
                        atom_types.add(atom_type)
                        # 注意：这里 mol_id 始终为 1，无法修复因为 XYZ 没有分子边界信息
                        mol_id = 1
                        charge = 0.0
                        atoms.append({
                            'id': i + 1,
                            'mol_id': mol_id,
                            'atom_type': atom_type,
                            'charge': charge,
                            'x': x, 'y': y, 'z': z
                        })
                        atom_counter += 1
            output_path = output_file
            with open(output_path, 'w') as f:
                f.write("# ReaxFF data for LAMMPS\n")
                f.write("\n")
                f.write(f"{len(atoms)} atoms\n")
                f.write(f"{len(atom_types)} atom types\n")
                f.write(f"{box_size[0]:.6f} {box_size[1]:.6f} xlo xhi\n")
                f.write(f"{box_size[2]:.6f} {box_size[3]:.6f} ylo yhi\n")
                f.write(f"{box_size[4]:.6f} {box_size[5]:.6f} zlo zhi\n")
                f.write("\n")
                f.write("Masses\n\n")
                f.write("1 12.0  # C\n")
                f.write("2 1.0  # H\n")
                f.write("3 16.0  # O\n")
                f.write("\n")
                f.write("Atoms # full\n\n")
                for atom in atoms:
                    f.write(f"{atom['id']} {atom['mol_id']} {atom['atom_type']} {atom['charge']:.6f} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
            print(f"Successfully generated LAMMPS data file: {output_path}")
            print(f"Box dimensions: {box_size}")
            return True
        except Exception as e:
            print(f"Failed to convert LAMMPS data file: {e}")
            return False
    
    # 新增：导出索引和 group 文件
    def _export_groups_and_ndx(self, system: PackedSystem):
        """导出 atoms.ndx 和 groups.lmp"""
        ndx_path = Path(self.workshop_dir) / "atoms.ndx"
        groups_path = Path(self.workshop_dir) / "groups.lmp"
        write_atoms_ndx(system, ndx_path)
        write_lammps_groups(system, groups_path)
    
    # 主流程：重构 build_lammps_input 以支持 PDB 模式
    def build_lammps_input(self, smiles_list, counts, target_density_kg_per_m3, output_file="lammps.data", template_names=None):
        """
        主函数：从 SMILES 列表构建 LAMMPS 输入文件。
        
        Args:
            smiles_list: SMILES 字符串列表
            counts: 每种分子的数量列表
            target_density_kg_per_m3: 目标密度 (kg/m³)
            output_file: 输出 LAMMPS data 文件名
            template_names: 可选，为每种分子指定模板名称（默认使用 'MOL1', 'MOL2', ...）
        
        Returns:
            成功返回 True
        """
        try:
            # 设置工作目录
            self.setup_workshop()
            
            # 确定模板名称
            if template_names is None:
                template_names = [f"MOL{i+1}" for i in range(len(smiles_list))]
            if len(template_names) != len(smiles_list):
                raise ValueError("template_names length must match smiles_list")
            
            # 计算盒子尺寸
            print("=== Calculate Box Dimensions ===")
            box_size, side_length = self.calculate_box_size(smiles_list, counts, target_density_kg_per_m3)
            print(f"Box dimensions: {box_size}")
            
            if self.use_pdb_mode:
                # ========== PDB 模式（新流程）==========
                print("\n=== PDB Mode: Generating Templates with Role Recognition ===")
                if not self._prepare_templates_pdb(smiles_list, template_names):
                    raise Exception("Failed to generate PDB templates")
                
                # 创建 Packmol 输入文件（PDB 格式）
                print("\n=== Creating Packmol Input (PDB) ===")
                packmol_input = self.create_packmol_input_pdb(template_names, counts, box_size, output_file="packed.pdb")
                
                # 运行 Packmol
                print("\n=== Running Packmol ===")
                if not self.run_packmol(packmol_input):
                    raise Exception("Packmol packing failed")
                
                # 解析 packed.pdb
                print("\n=== Parsing Packmol Output ===")
                packed_pdb_path = Path(self.workshop_dir) / "packed.pdb"
                self.packed_system = parse_packed_pdb(
                    packed_pdb_path,
                    self.template_infos,
                    box_size,
                    template_names_order=template_names,
                    counts=counts
                )
                if self.packed_system is None:
                    raise Exception("Failed to parse packed PDB")
                
                # 生成 LAMMPS data 文件（正确 mol_id）
                print("\n=== Generating LAMMPS Data File ===")
                if not self._pdb_to_lammps_data(self.packed_system, output_file):
                    raise Exception("Failed to generate LAMMPS data file")
                
                # 导出 atoms.ndx 和 groups.lmp
                print("\n=== Exporting atoms.ndx and LAMMPS groups ===")
                self._export_groups_and_ndx(self.packed_system)
                
            else:
                # ========== 原有 XYZ 模式（保留兼容）==========
                print("\n=== Legacy XYZ Mode (mol_id will be 1 for all atoms) ===")
                xyz_files = []
                for i, smiles in enumerate(smiles_list):
                    xyz_file = f"mol_{i+1}.xyz"
                    if self.smiles_to_xyz(smiles, xyz_file):
                        xyz_path = os.path.join(self.workshop_dir, xyz_file)
                        if os.path.exists(xyz_path) and os.path.getsize(xyz_path) > 0:
                            xyz_files.append(xyz_file)
                        else:
                            raise Exception(f"Generated XYZ file is invalid: {xyz_path}")
                    else:
                        raise Exception(f"Unable to convert SMILES: {smiles}")
                
                print("\n=== Molecular Packing (XYZ) ===")
                packmol_input = self.create_packmol_input(xyz_files, counts, box_size)
                if not self.run_packmol(packmol_input):
                    raise Exception("Packmol packing failed")
                
                print("\n=== Format Conversion (XYZ to LAMMPS) ===")
                packed_xyz = "packed.xyz"
                if not self.xyz_to_lammps_data(packed_xyz, box_size, output_file):
                    raise Exception("LAMMPS format conversion failed")
            
            print("\n=== Completed ===")
            print("LAMMPS input file construction completed!")
            print(f"All intermediate files saved at: {self.workshop_dir}")
            for file in os.listdir(self.workshop_dir):
                file_path = os.path.join(self.workshop_dir, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
            
            return True
            
        except Exception as e:
            logger.exception(f"Build process failed: {e}")
            return False
        finally:
            self.cleanup(keep_files=True)