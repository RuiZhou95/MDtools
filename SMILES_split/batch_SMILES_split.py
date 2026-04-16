import pandas as pd
from rdkit import Chem
from rdkit import rdBase, RDLogger
import sys
import os
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from collections import defaultdict
import gc
import warnings

try:
    import SMILES_split
except ImportError:
    print("Error: 'SMILES_split.py' not found.")
    sys.exit(1)

def init_worker():
    """
    Called when each worker process initializes, disables RDKit warnings for that process.
    """
    import warnings
    from rdkit import RDLogger
    warnings.filterwarnings('ignore')
    RDLogger.DisableLog('rdApp.*')

def process_smiles_batch(args):
    """
    Process a batch of SMILES, return fragment and attachment point information.
    """
    batch_smiles, batch_id = args
    batch_results = defaultdict(list)
    
    for original_smiles in batch_smiles:
        if original_smiles is None or not isinstance(original_smiles, str) or not original_smiles.strip():
            continue
            
        try:
            # Get fragments
            fragments = SMILES_split.disassemble_molecule_all_strategies(original_smiles)
            
            # Directly process each fragment, avoiding repeated parsing later
            for frag_smi in fragments:
                if not frag_smi or not isinstance(frag_smi, str):
                    continue
                    
                # Use safer SMILES parsing options
                mol = Chem.MolFromSmiles(frag_smi, sanitize=False)
                if mol is None:
                    continue
                    
                try:
                    # Attempt sanitization, skip if it fails
                    Chem.SanitizeMol(mol)
                except:
                    continue
                
                # Check unsaturated sites
                for atom in mol.GetAtoms():
                    explicit_val = atom.GetExplicitValence()
                    total_val = atom.GetTotalValence()
                    
                    if total_val > explicit_val:
                        # Store necessary information, avoid duplicate SMILES storage
                        key = (frag_smi, atom.GetIdx(), atom.GetSymbol())
                        batch_results[key] = [explicit_val, total_val]
                        
        except Exception as e:
            continue
            
    return batch_results

def generate_fragments_dictionary_optimized(csv_file_path, output_file, batch_size=10000, n_processes=None):
    """
    Use batching, multiprocessing, and memory optimization.
    """
    print(f"Processing {csv_file_path}...")
    
    # Read CSV in chunks to avoid loading entire file into memory
    print("Reading CSV in chunks...")
    
    try:
        # First check file size and column names
        with open(csv_file_path, 'r') as f:
            first_line = f.readline()
        if 'smiles' not in first_line:
            print("Warning: 'smiles' column not found in first line. Trying to read without column specification.")
            chunks = pd.read_csv(csv_file_path, chunksize=batch_size)
        else:
            chunks = pd.read_csv(csv_file_path, usecols=['smiles'], chunksize=batch_size)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Configure multiprocessing
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one core for the system
    
    print(f"Using {n_processes} processes...")
    
    # 3. Collect all results
    all_results = defaultdict(list)
    global_index = 0
    fragment_counter = 0
    
    # 4. Process by chunk
    for chunk_idx, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {chunk_idx + 1}...")
        
        # Prepare batch tasks
        if 'smiles' not in chunk.columns:
            print(f"Warning: 'smiles' column not found in chunk {chunk_idx + 1}")
            # Try to find possible column names
            possible_cols = [col for col in chunk.columns if 'smile' in col.lower()]
            if possible_cols:
                smiles_list = chunk[possible_cols[0]].dropna().tolist()
            else:
                # Use the first column
                smiles_list = chunk.iloc[:, 0].dropna().tolist()
        else:
            smiles_list = chunk['smiles'].dropna().tolist()
            
        if not smiles_list:
            continue
            
        # Split into sub-batches for multiprocessing
        sub_batch_size = max(100, len(smiles_list) // (n_processes * 2))
        batches = []
        for i in range(0, len(smiles_list), sub_batch_size):
            batch = smiles_list[i:i + sub_batch_size]
            if batch:
                batches.append((batch, i))
        
        # Use multiprocessing with initializer function
        with Pool(processes=n_processes, initializer=init_worker) as pool:
            # Use imap to maintain order
            batch_results_list = list(tqdm(
                pool.imap(process_smiles_batch, batches),
                total=len(batches),
                desc=f"Processing chunk {chunk_idx + 1}",
                unit="batch",
                position=0,  # Fixed position to avoid multiple progress bars
                leave=True   # Keep progress bar after completion
            ))
        
        # Merge results
        for batch_results in batch_results_list:
            for (frag_smi, attach_idx, symbol), (explicit_val, total_val) in batch_results.items():
                # Check if identical fragment and attachment point already exist
                existing_key = (frag_smi, attach_idx, symbol)
                if existing_key not in all_results:
                    fragment_name = f"Frag_{global_index}_{symbol}"
                    all_results[existing_key] = {
                        'name': fragment_name,
                        'smiles': frag_smi,
                        'attach_idx': attach_idx,
                        'ExplicitValence': explicit_val,
                        'TotalValence': total_val,
                        'global_index': global_index
                    }
                    global_index += 1
                    fragment_counter += 1
        
        # Show progress
        print(f"  Processed {fragment_counter} unique fragments so far...")
        
        # Regular memory cleanup
        if (chunk_idx + 1) % 5 == 0:
            gc.collect()
    
    # Build final dictionary
    print(f"\nBuilding final dictionary with {len(all_results)} entries...")
    FRAGMENTS = {}
    for idx, (key, entry) in enumerate(all_results.items()):
        FRAGMENTS[idx] = {
            'name': entry['name'],
            'smiles': entry['smiles'],
            'attach_idx': entry['attach_idx'],
            'ExplicitValence': entry['ExplicitValence'],
            'TotalValence': entry['TotalValence']
        }
    
    # 6. Save to file
    print(f"Saving to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FRAGMENTS = {\n")
            lines = []
            for key, value in FRAGMENTS.items():
                lines.append(f"    {key}: {value},\n")
            f.writelines(lines)
            f.write("}\n")
        
        print(f"Done. Total fragments: {len(FRAGMENTS)}")
        
        # Also save a CSV version for easier viewing
        csv_output = output_file.replace('.txt', '.csv')
        df_output = pd.DataFrame.from_dict(FRAGMENTS, orient='index')
        df_output.to_csv(csv_output, index=False)
        print(f"Also saved CSV version to {csv_output}")
        
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    
    print("=" * 60)
    print("Running optimized version with multiprocessing...")
    print("=" * 60)
    generate_fragments_dictionary_optimized(
        'zinc250k.csv', 
        'mol_species_optimized.txt',
        batch_size=20000,
        n_processes=cpu_count() // 2
    )