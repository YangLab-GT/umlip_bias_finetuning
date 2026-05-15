import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import sys
import json
from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import torch
from mace.calculators import MACECalculator
from pymatgen.io.vasp.outputs import Vasprun
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from pymatgen.core import Structure
"""First command line argument = directory"""
"""Second CLA = model"""
def main():
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    calc = MACECalculator(sys.argv[2], device=device)
    pairs = get_respective_files(sys.argv[1])
    dft_matrix, mace_matrix, error_matrix, timesteps, mace_stress, dft_stress, dft_energy, mace_energy  = compute_all(pairs, calc)
    atoms = read(pairs[0][0])
    symbols = list(atoms.get_chemical_symbols())
    print(compute_stress_RMSE(dft_stress, mace_stress))
    print(compute_RMSE(dft_matrix, mace_matrix))
    print(compute_energy_RMSE(dft_energy, mace_energy))
    print(compute_relative_force_rmse(dft_matrix, mace_matrix))
    print(print_relative_force_rmse_by_element(compute_relative_force_rmse_by_element(dft_matrix, mace_matrix, symbols)))
def print_relative_force_rmse_by_element(results, model_name=None):
    """Pretty-print the dict returned by compute_relative_force_rmse_by_element."""
    if model_name:
        print(f"\nPer-element force errors: {model_name}")
    else:
        print("\nPer-element force errors")
    print("-" * 68)
    header = f"{'Element':<8}{'N_at':>6}{'RMSE':>10}{'<|F|>':>10}{'σ(F)':>10}{'rel/<|F|>':>12}"
    print(header)
    print(f"{'':<8}{'':>6}{'(eV/Å)':>10}{'(eV/Å)':>10}{'(eV/Å)':>10}{'(%)':>12}")
    print("-" * 68)
    for element, r in results.items():
        print(f"{element:<8}"
              f"{r['n_atoms']:>6d}"
              f"{r['rmse']:>10.4f}"
              f"{r['mean_abs_component']:>10.4f}"
              f"{r['std_component']:>10.4f}"
              f"{r['rel_rmse_mean_pct']:>11.1f}%")
    print("-" * 78)
def compute_relative_force_rmse_by_element(dft_matrix, mace_matrix, symbols):
    symbols = np.asarray(symbols)
    dft_forces  = dft_matrix[:, :, 3:]
    mace_forces = mace_matrix[:, :, 3:]
    error = dft_forces - mace_forces
    results = {}
    for element in sorted(set(symbols)):
        mask = (symbols == element)
        e_err = error[:, mask, :]
        e_dft = dft_forces[:, mask, :]
        rmse = float(np.sqrt(np.mean(e_err ** 2)))
        mean_abs = float(np.mean(np.abs(e_dft)))
        std = float(np.std(e_dft))
        results[element] = {'n_atoms': int(mask.sum()),
            'n_components': int(e_err.size),
            'rmse': rmse,
            'mean_abs_component': mean_abs,
            'std_component': std,
            'rel_rmse_mean': rmse / mean_abs,
            'rel_rmse_mean_pct': 100.0 * rmse / mean_abs,
        }
    return results
def compute_relative_force_rmse(dft_matrix, mace_matrix):
    dft_forces  = dft_matrix[:, :, 3:]
    mace_forces = mace_matrix[:, :, 3:]
    error = dft_forces - mace_forces
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mean_abs = float(np.mean(np.abs(dft_forces)))
    std = float(np.std(dft_forces))

    return {'rmse': rmse,
        'mean_abs_component': mean_abs,
        'std_component': std,
        'rel_rmse_mean': rmse / mean_abs,
        'rel_rmse_std': rmse / std,
        'rel_rmse_mean_pct': 100.0 * rmse / mean_abs,
        'rel_rmse_std_pct': 100.0 * rmse / std,
    }
def compute_energy_RMSE(dft, mace):
    return np.sqrt(np.mean((dft-mace)**2))
def compute_stress_RMSE(dft, mace):
    return np.sqrt(np.mean((dft - mace)**2))
def compute_RMSE(dft_matrix, mace_matrix):
    """Takes in DFT / Predicted Matrices of form
    Params:
        dft_matrix, pred matrix = [[[x1, y1, z1, fx1, fy1, fz1],
                                    [x2, y2, z2, fx2, fy2, fz2],
                                    ...
                                    [xN, yN, zN, fxN, fyN, fzN]],...]
    Returns:
        fx, fy, fz
    """
    dft_forces = dft_matrix[:, :, 3:]
    mace_forces = mace_matrix[:, :, 3:]
    rmse = {}
    directions = ['fx', 'fy', 'fz']
    for i, dir in enumerate(directions):
        error = dft_forces[:, :, i] - mace_forces[:, :, i]
        mse = np.mean(error ** 2)
        rmse[dir] = np.sqrt(mse)
    return rmse
def get_respective_files(root_dir):
    """Takes in directory, returns all pairs of POSCAR/vasprun.
    Params:
        Root directory
    Returns:
        matrix of form: [[.../POSCAR, .../vasprun.xml], ...]
    """
    poscars = {}
    vaspruns = {}
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in filenames:
            if filename == "POSCAR":
                parent = os.path.basename(dirpath)
                poscars[parent] = os.path.join(dirpath, filename)
            elif filename == "vasp_info.json":
                parent = os.path.basename(dirpath)
                # write_info_to_file(os.path.join(dirpath, filename), os.path.join(dirpath, "vasp_info.json"))
                vaspruns[parent] = os.path.join(dirpath, filename)
    paired = []
    for key in poscars:
        if key in vaspruns:
            paired.append((poscars[key], vaspruns[key]))
    return paired
def compute_all(pairs, mace_calc):
    """Takes in all pairs of POSCAR/vasprun, computes error
    Params:
        pairs: [POSCAR/vasprun] pairs
        mace_calc: MACECalculator object
    Returns:
        dft_matrix:
        mace_matrix:
        error_matrix: dft_matrix-mace_matrix
        dist_from_interesting:
    """
    dft_matrix, mace_matrix, error_matrix, mace_stress_arr, dft_stress_arr, mace_energies, dft_energies = [], [], [], [], [], [], []
    dist_from_interesting = []
    total = len(pairs)
    for i, pair in enumerate(pairs, start=1):
        timestep = extract_timestep(pair[0])
        dft, mace, error, converged, mace_stress, dft_stress, mace_energy, dft_energy = compute_error(pair, mace_calc)
        if(converged):
            dft_energies.append(dft_energy*1000/197)
            mace_energies.append(mace_energy*1000/197)
            dft_stress_gpa = dft_stress.flatten() / (1602.1766208 * -1)
            dft_stress_gpa_voigt = np.array([dft_stress_gpa[0], # xx
                                      dft_stress_gpa[4], # yy
                                      dft_stress_gpa[8], # zz
                                      dft_stress_gpa[1], # yz
                                      dft_stress_gpa[2], # xz
                                      dft_stress_gpa[3]]) # xy
            dist_from_interesting.append(timestep-0)
            dft_matrix.append(dft)
            mace_matrix.append(mace)
            # print(mace)
            error_matrix.append(error)
            mace_stress_arr.append(mace_stress)
            dft_stress_arr.append(dft_stress_gpa_voigt)
        percent_done = (i / total) * 100
        print(f"Progress: {percent_done:.2f}% ({i}/{total})")
    dft_energies = np.array(dft_energies)
    mace_energies = np.array(mace_energies)
    print(dft_energies-mace_energies)
    dft_matrix = np.array(dft_matrix)
    mace_matrix = np.array(mace_matrix)
    error_matrix = np.array(error_matrix)
    mace_stress_matrix = np.array(mace_stress_arr)
    dft_stress_matrix = np.array(dft_stress_arr)
    return dft_matrix, mace_matrix, error_matrix, dist_from_interesting, mace_stress_matrix, dft_stress_matrix, dft_energies, mace_energies
def compute_error(pair, mace_calc):
    """Takes in pair of POSCAR, vasprun, computes forces using MACE / collects forces from vasprun, computes error"""
    """If vr is unconverged, skip this"""
    poscar_path = pair[0]
    vr_path = pair[1]
    dft_force, converged, dft_stress, dft_energy = get_forces_from_dft(vr_path)
    if(converged == False):
        return None, None, None, False, None, None, None, None
    else:
        mace_force, mace_stress, mace_energy = compute_forces(poscar_path, mace_calc)
    return dft_force, mace_force, (dft_force-mace_force), converged, mace_stress, dft_stress, mace_energy, dft_energy
def compute_forces(poscar_file, mace_calc):
    """Takes in POSCAR, model, computes forces"""
    atoms = read(poscar_file)
    #print(atoms[79])
    atoms.calc = mace_calc
    mace_positions = atoms.get_positions()
    mace_forces = atoms.get_forces()
    mace_stresses = atoms.get_stress()
    mace_energy = atoms.get_potential_energy()
    return np.hstack((mace_positions, mace_forces)), mace_stresses, mace_energy
def get_forces_from_dft(json_path):
    """
    Takes in a JSON file from write_info_to_file / write_vasprun_json,
    returns (per_frame, success_flag, stress)
    
    per_frame: Nx6 array [[x,y,z,fx,fy,fz], ...]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    try:
        # Reconstruct structure
        structure = Structure.from_dict(data["structure"])

        # Convert lists to NumPy arrays
        forces = np.array(data["forces"])
        stresses = np.array(data["stress"])
        energy = data["energy"]
        per_frame = []
        for site, force in zip(structure.sites, forces):
            row = np.concatenate([site.coords, force])
            per_frame.append(row)
        per_frame = np.array(per_frame)
        return per_frame, True, stresses, energy

    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None, False, None, None
def plot_force_parity_components(dft_matrix, mace_matrix, dist_from_interesting, title_prefix="DFT vs MACE Force Parity"):
    """
    GPT Generated
    Creates three parity plots: fx, fy, fz.
    Colors each point by its structure's dist_from_interesting value.
    """
    # Extract force components
    dft_forces = dft_matrix[:, :, 3:]  # fx, fy, fz
    mace_forces = mace_matrix[:, :, 3:]

    # Flatten each component
    dft_fx = dft_forces[:, :, 0].flatten()
    dft_fy = dft_forces[:, :, 1].flatten()
    dft_fz = dft_forces[:, :, 2].flatten()

    mace_fx = mace_forces[:, :, 0].flatten()
    mace_fy = mace_forces[:, :, 1].flatten()
    mace_fz = mace_forces[:, :, 2].flatten()

    # Prepare color mapping
    cmap = get_cmap("seismic")
    norm = Normalize(vmin=min(dist_from_interesting), vmax=max(dist_from_interesting))

    # Repeat each structure's color for all its atoms
    n_atoms = dft_matrix.shape[1]
    color_values = np.repeat(dist_from_interesting, n_atoms)
    colors = cmap(norm(color_values))

    # Plot each component
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    components = ['fx', 'fy', 'fz']
    dft_all = [dft_fx, dft_fy, dft_fz]
    mace_all = [mace_fx, mace_fy, mace_fz]

    for i in range(3):
        ax = axes[i]
        sc = ax.scatter(dft_all[i], mace_all[i], c=color_values, cmap=cmap, norm=norm, s=5, alpha=0.6, edgecolors='none')
        min_val = min(dft_all[i].min(), mace_all[i].min())
        max_val = max(dft_all[i].max(), mace_all[i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal parity')
        ax.set_xlabel(f"DFT {components[i]} (eV/Å)")
        ax.set_ylabel(f"MACE {components[i]} (eV/Å)")
        ax.set_title(f"{title_prefix}: {components[i]}")
        ax.legend()
        ax.grid(True)

    # Add colorbar
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation='vertical')
    cbar.set_label("Distance from Interesting")

    # plt.tight_layout()
    plt.savefig("parity_plot_colored.png")
    plt.show()
def write_info_to_file(vasprun_path, output_file="vasp_info.json"):
    # if os.path.exists(output_file):
    #     print(f"JSON file already exists: {output_file}, skipping.")
    #     return
    vr = Vasprun(vasprun_path)
    if vr.as_dict().get('has_vasp_completed', False):
        step0 = vr.ionic_steps[0]
        structure = step0['structure']
        forces = step0['forces']
        stresses = step0['stress']
        energy = step0['e_0_energy']
        struct_dict = structure.as_dict()
        data = {
            "structure": struct_dict,
            "forces": forces.tolist(),
            "stress": stresses.tolist(),
            "energy": energy
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data written to {output_file}")
    else:
        print("VASP calculation did not complete successfully.")
def extract_timestep(path):
    """
    Extracts the timestep from a path like .../naive.x/POSCAR
    Returns it as an integer.
    """
    parent = os.path.basename(os.path.dirname(path))
    timestep = int(parent.split('.')[-1])
    return timestep



if __name__ == "__main__":
    main()
