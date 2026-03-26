import os
import subprocess
import shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

class SmilesToLammps:
    def __init__(self, workspace_name="lammps_build"):
        self.workshop_dir = None
        self.workspace_name = workspace_name
        
        # Define constants
        self.AVOGADRO_CONSTANT = 6.02214076e23  # mol^-1
        self.M3_TO_ANGSTROM3 = 1e30  # 1 m^3 = 1e30 Å^3
        
    def setup_workshop(self):
        """Create working directory in the tmp folder under current path"""
        # Create tmp folder (if it doesn't exist)
        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            print(f"Created tmp directory: {tmp_dir}")
        
        # Create working directory
        self.workshop_dir = os.path.join(tmp_dir, self.workspace_name)
        if os.path.exists(self.workshop_dir):
            # If it exists, clear the directory
            shutil.rmtree(self.workshop_dir)
        os.makedirs(self.workshop_dir)
        
        print(f"Working directory: {self.workshop_dir}")
        return self.workshop_dir
    
    def calculate_box_size(self, smiles_list, counts, target_density_kg_per_m3):
        """
        Calculate the side length of a cubic box based on molecular counts and molecular weights
        
        Args:
            smiles_list: List of SMILES strings
            counts: List of molecular counts
            target_density_kg_per_m3: Target density (kg/m³), must be specified
            
        Returns:
            box_size: Box dimensions in [0.0, A, 0.0, A, 0.0, A] format
            side_length_angstrom: Box side length (Å)
        """
        if target_density_kg_per_m3 is None:
            raise ValueError("Target density target_density_kg_per_m3 must be specified")
        
        if target_density_kg_per_m3 <= 0:
            raise ValueError("Target density must be greater than 0")
        
        print(f"Using target density: {target_density_kg_per_m3:.2f} kg/m³")        
        
        # Calculate molecular weight for each molecule
        molecular_weights = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Unable to parse SMILES: {smiles}")
            
            # Add hydrogen atoms to get complete molecule
            mol = Chem.AddHs(mol)
            molecular_weight = Descriptors.MolWt(mol)
            molecular_weights.append(molecular_weight)
            print(f"Molecular weight of {smiles}: {molecular_weight:.2f} g/mol")
        
        # Calculate total mass (unit: grams)
        # Note: Molecular weight is in g/mol, so need to divide by Avogadro's constant to get mass of a single molecule
        total_mass_g = 0.0
        for count, molecular_weight in zip(counts, molecular_weights):
            # Mass of a single molecule = molecular weight (g/mol) / Avogadro's constant (mol^-1)
            single_molecule_mass_g = molecular_weight / self.AVOGADRO_CONSTANT
            total_mass_g += count * single_molecule_mass_g
        
        print(f"Total number of molecules: {sum(counts)}")
        print(f"Total system mass: {total_mass_g:.10f} g")
        
        # Unit conversion steps:
        # Convert total mass from grams to kilograms
        total_mass_kg = total_mass_g / 1000.0
        print(f"Total system mass: {total_mass_kg:.12f} kg")
        
        # Calculate volume (m³) - using formula: volume = mass / density
        volume_m3 = total_mass_kg / target_density_kg_per_m3
        print(f"Calculated box volume: {volume_m3:.12e} m³")
        
        # Calculate cube side length (m)
        side_length_m = volume_m3 ** (1/3)
        
        # Convert side length from meters to angstroms (1 m = 1e10 Å)
        side_length_angstrom = side_length_m * 1e10
        
        # Calculate volume in Å³ units for display
        volume_angstrom3 = volume_m3 * 1e30  # 1 m³ = 1e30 Å³
        
        print(f"Box volume: {volume_angstrom3:.2f} Å³")
        print(f"Calculated cube box side length: {side_length_angstrom:.2f} Å")
        
        # Verify if calculation is correct
        # Verification 1: Calculate density back from Å³ volume
        calculated_density_kg_per_m3 = total_mass_kg / (volume_angstrom3 / 1e30)
        print(f"Verification - calculated density: {calculated_density_kg_per_m3:.2f} kg/m³")
        
        # Verification 2: Calculate volume from Å side length
        calculated_volume_angstrom3 = side_length_angstrom ** 3
        print(f"Verification - volume calculated from side length: {calculated_volume_angstrom3:.2f} Å³")
        
        # Return box_size format [0.0, A, 0.0, A, 0.0, A]
        box_size = [0.0, side_length_angstrom, 0.0, side_length_angstrom, 0.0, side_length_angstrom]
        
        return box_size, side_length_angstrom
    
    def cleanup(self, keep_files=True):
        """Clean up working directory"""
        if not keep_files and self.workshop_dir and os.path.exists(self.workshop_dir):
            shutil.rmtree(self.workshop_dir)
            print("Cleaned up working directory")
        elif keep_files:
            print(f"Intermediate files saved at: {self.workshop_dir}")
    
    def smiles_to_xyz(self, smiles, output_filename, charge_method='mmff94'):
        """
        Convert SMILES to XYZ coordinate file using Open Babel
        """
        try:
            output_path = os.path.join(self.workshop_dir, output_filename)
            cmd = f'obabel -:"{smiles}" -o xyz -O {output_path} --gen3D --partialcharge {charge_method}'
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(f"Successfully generated XYZ file: {output_path}")
                # Check if the generated XYZ file is valid
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
    
    def create_packmol_input(self, xyz_files, counts, box_size, output_file="packed.xyz"):
        """
        Create Packmol input file
        """
        # Use relative paths within working directory
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
        print("Packmol input file content:")
        print(packmol_input)
        
        return input_filename
    
    def run_packmol(self, input_file):
        """Run Packmol for molecular packing"""
        try:
            input_path = os.path.join(self.workshop_dir, input_file)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                print(f"Packmol input file does not exist: {input_path}")
                return False
            
            # Check if XYZ files exist and are valid
            with open(input_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('structure'):
                        xyz_file = line.split()[1]
                        xyz_path = os.path.join(self.workshop_dir, xyz_file)
                        if not os.path.exists(xyz_path):
                            print(f"XYZ file does not exist: {xyz_path}")
                            return False
                        # Check XYZ file content
                        with open(xyz_path, 'r') as xyz_f:
                            xyz_content = xyz_f.read()
                            if len(xyz_content.strip().split('\n')) < 3:
                                print(f"XYZ file format error: {xyz_path}")
                                return False
            
            # Run packmol (execute within working directory)
            original_dir = os.getcwd()
            os.chdir(self.workshop_dir)
            
            cmd = f"packmol < {input_file}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=300)
            
            # Switch back to original directory
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print("Packmol execution successful")
                # Check output file
                output_file = os.path.join(self.workshop_dir, "packed.xyz")
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"Successfully generated packed file: {output_file}")
                    return True
                else:
                    print("Packmol output file is empty or does not exist")
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
            print(f"Standard output: {e.stdout}")
            return False
        except Exception as e:
            print(f"Packmol execution exception: {e}")
            return False
    
    def xyz_to_lammps_data(self, xyz_file, box_size, output_file="lammps.data"):
        """
        Convert Packmol output XYZ file to LAMMPS data format
        
        Args:
            xyz_file: Input XYZ filename
            box_size: Box dimensions [xlo, xhi, ylo, yhi, zlo, zhi]
            output_file: Output filename
        """
        try:
            xyz_path = os.path.join(self.workshop_dir, xyz_file)
            with open(xyz_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                print("XYZ file format error: insufficient lines")
                return False
            
            # Parse atom count
            try:
                num_atoms = int(lines[0].strip())
            except ValueError:
                print("XYZ file format error: first line is not atom count")
                return False
            
            # Skip first two lines (atom count and comment)
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
                        
                        # Simplified atom type mapping
                        if element.upper() == 'C':
                            atom_type = 1
                        elif element.upper() == 'H':
                            atom_type = 2  
                        elif element.upper() == 'O':
                            atom_type = 3
                        elif element.upper() == 'N':
                            atom_type = 4
                        else:
                            atom_type = 1  # Default type
                            
                        atom_types.add(atom_type)
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
            
            # Write LAMMPS data file (output to current directory)
            output_path = output_file
            with open(output_path, 'w') as f:
                # File header - use calculated box_size
                f.write("# ReaxFF data for LAMMPS\n")
                f.write("\n")
                f.write(f"{len(atoms)} atoms\n")
                f.write(f"{len(atom_types)} atom types\n")
                f.write(f"{box_size[0]:.6f} {box_size[1]:.6f} xlo xhi\n")
                f.write(f"{box_size[2]:.6f} {box_size[3]:.6f} ylo yhi\n")
                f.write(f"{box_size[4]:.6f} {box_size[5]:.6f} zlo zhi\n")
                f.write("\n")
                
                # Masses section
                f.write("Masses\n")
                f.write("\n")
                f.write("1 12.0  # C\n")
                f.write("2 1.0  # H\n")
                f.write("3 16.0  # O\n")
                f.write("\n")
                
                # Atoms section
                f.write("Atoms # full\n")
                f.write("\n")
                
                for atom in atoms:
                    f.write(f"{atom['id']} {atom['mol_id']} {atom['atom_type']} {atom['charge']:.6f} {atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
            
            print(f"Successfully generated LAMMPS data file: {output_path}")
            print(f"Box dimensions: xlo={box_size[0]:.6f}, xhi={box_size[1]:.6f}, ylo={box_size[2]:.6f}, yhi={box_size[3]:.6f}, zlo={box_size[4]:.6f}, zhi={box_size[5]:.6f}")
            return True
            
        except Exception as e:
            print(f"Failed to convert LAMMPS data file: {e}")
            return False
    
    def build_lammps_input(self, smiles_list, counts, target_density_kg_per_m3, output_file="lammps.data"):
        """
        Main function: Build LAMMPS input file from SMILES list
        
        Args:
            smiles_list: List of SMILES strings
            counts: List of molecular counts
            target_density_kg_per_m3: Target density (kg/m³), must be specified
            output_file: Output filename
        """
        try:
            # Check density parameter
            if target_density_kg_per_m3 is None:
                raise ValueError("Target density target_density_kg_per_m3 must be specified")
            
            if target_density_kg_per_m3 <= 0:
                raise ValueError("Target density must be greater than 0")
            
            # Set up working directory
            self.setup_workshop()
            
            # Calculate box dimensions
            print("=== Calculate Box Dimensions ===")
            box_size, side_length = self.calculate_box_size(smiles_list, counts, target_density_kg_per_m3)
            print(f"Using box dimensions: {box_size}")
            
            # Convert each SMILES to XYZ file
            print("\n=== Generate Molecular Coordinates ===")
            xyz_files = []
            for i, smiles in enumerate(smiles_list):
                xyz_file = f"mol_{i+1}.xyz"
                if self.smiles_to_xyz(smiles, xyz_file):
                    # Check generated XYZ file
                    xyz_path = os.path.join(self.workshop_dir, xyz_file)
                    if os.path.exists(xyz_path) and os.path.getsize(xyz_path) > 0:
                        xyz_files.append(xyz_file)
                    else:
                        raise Exception(f"Generated XYZ file is invalid: {xyz_path}")
                else:
                    raise Exception(f"Unable to convert SMILES: {smiles}")
            
            print(f"Successfully generated {len(xyz_files)} XYZ files")
            
            # Create and run Packmol
            print("\n=== Molecular Packing ===")
            packmol_input = self.create_packmol_input(xyz_files, counts, box_size)
            if not self.run_packmol(packmol_input):
                raise Exception("Packmol packing failed")
            
            # Convert to LAMMPS format - pass box_size parameter
            print("\n=== Format Conversion ===")
            packed_xyz = "packed.xyz"
            if not self.xyz_to_lammps_data(packed_xyz, box_size, output_file):
                raise Exception("LAMMPS format conversion failed")
            
            print("\n=== Completed ===")
            print("LAMMPS input file construction completed!")
            
            # Show intermediate file locations
            print(f"\nAll intermediate files saved at: {self.workshop_dir}")
            print("Includes the following files:")
            for file in os.listdir(self.workshop_dir):
                file_path = os.path.join(self.workshop_dir, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size} bytes)")
            
            return True
            
        except Exception as e:
            print(f"Build process failed: {e}")
            return False
        finally:
            # Keep intermediate files
            self.cleanup(keep_files=True)

# Usage example
if __name__ == "__main__":
    builder = SmilesToLammps(workspace_name="my_data")
    
    # Example: must specify density
    print("=== Build LAMMPS Input File ===")
    smiles_list = ["CCCCC(=O)OCC(C)(COC(=O)CCCC)COC(=O)CCCC", "O=O"]  # ester and O2
    counts = [15, 60]  # 50 esters, 2 O2 
    
    # Must specify target density
    target_density = 50.0  # kg/m³ (0.05 g/cm³) 50.0 kg/m³
    
    success = builder.build_lammps_input(
        smiles_list=smiles_list,
        counts=counts,
        target_density_kg_per_m3=target_density,
        output_file=f"{counts[0]}E_{counts[1]}O.data"
    )
    
    if success:
        print("Successfully generated REAXFF input file!")
        print(f"Final LAMMPS file: reaxff_input.data")
        print(f"Intermediate file location: tmp/my_data/")
    else:
        print("Generation failed!")